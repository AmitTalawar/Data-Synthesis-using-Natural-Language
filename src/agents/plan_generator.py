from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

FILE_DIR = Path(__file__).resolve()
REPO_ROOT = FILE_DIR.parents[2]  # .../Data-Synthesis-using-Natural-Language
CONFIG_PATH = REPO_ROOT / "config" / "retail_sales_ir_v1.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"

try:
    ONE_SHOT_EXAMPLE = json.dumps(json.loads(CONFIG_PATH.read_text(encoding="utf-8")), indent=2)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: The example IR file '{CONFIG_PATH}' was not found.")

SYSTEM_PROMPT = f"""You are an expert database administrator and data engineer. Your task is to convert a user's natural language request for a synthetic dataset into a precise and structured JSON object, which we call the Intermediate Representation (IR).

You must adhere strictly to the provided JSON schema. Do not add any keys that are not in the schema. Do not explain your output. Only output the JSON object.

JSON SCHEMA (Shape contract - not a JSON Schema file):
{{
  "metadata": {{
	"projectName": "string",
	"version": "string",
	"description": "string"
  }},
  "schema": {{
	"fact_table": "string",
	"dimension_tables": ["string"],
	"tables": {{
	  "table_name": [
		{{"name": "column_name", "type": "SQL_TYPE", "special": "primary_key|foreign_key:table_name|null"}}
	  ]
	}}
  }},
  "generation_properties": {{
	"row_counts": {{
	  "table_name": "integer"
	}},
	"distributions": {{
	  "table_name.column_name": {{
		"in_sales_table": "distribution_name(param=value)",
		"base": "distribution_name",
		"pattern": "seasonal(peak=Month)"
	  }}
	}}
  }},
  "stress_goals": ["string"]
}}

High-quality example IR (for guidance only):
{ONE_SHOT_EXAMPLE}

Instructions:
- Produce ONLY the IR JSON object as final output; no prose, no surrounding code fences.
- Use the tools if helpful. Validate the JSON shape before finalizing. If validation fails, correct and try again.
"""

def _strip_code_fences(s: str) -> str:
	s = s.strip()
	if s.startswith("```"):
		# remove initial fence
		s = s.split("\n", 1)[1] if "\n" in s else s
	if s.endswith("```"):
		s = s.rsplit("```", 1)[0]
	# remove language tag like ```json
	s = s.replace("```json", "").replace("```JSON", "").strip()
	return s

@tool("validate_ir_schema", return_direct=False)
def validate_ir_schema(ir_json: str) -> str:
	"""Validate IR JSON string against the expected shape.

	Args:
		ir_json: The JSON string to validate.

	Returns:
		A JSON string with fields: {"valid": bool, "errors": [str]}.
	"""
	errors: List[str] = []
	try:
		payload = json.loads(_strip_code_fences(ir_json))
	except Exception as e:
		return json.dumps({"valid": False, "errors": [f"Not valid JSON: {e}"]})

	def req(obj: Dict[str, Any], key: str, typ: Tuple[type, ...] | type) -> None:
		if key not in obj:
			errors.append(f"Missing key: {key}")
			return
		if not isinstance(obj[key], typ):
			errors.append(f"Key '{key}' must be of type {typ} but got {type(obj[key]).__name__}")

	# Top-level keys
	if not isinstance(payload, dict):
		return json.dumps({"valid": False, "errors": ["Top-level IR must be an object"]})

	for top in ["metadata", "schema", "generation_properties", "stress_goals"]:
		if top not in payload:
			errors.append(f"Missing top-level key: {top}")

	# metadata
	md = payload.get("metadata", {})
	if isinstance(md, dict):
		for k in ["projectName", "version", "description"]:
			if k not in md or not isinstance(md[k], str) or not md[k].strip():
				errors.append(f"metadata.{k} must be a non-empty string")
	else:
		errors.append("metadata must be an object")

	# schema
	sch = payload.get("schema", {})
	if isinstance(sch, dict):
		if not isinstance(sch.get("fact_table"), str) or not sch.get("fact_table"):
			errors.append("schema.fact_table must be a non-empty string")
		if not isinstance(sch.get("dimension_tables"), list) or not all(
			isinstance(x, str) and x for x in sch.get("dimension_tables", [])
		):
			errors.append("schema.dimension_tables must be a list of non-empty strings")
		tables = sch.get("tables")
		if not isinstance(tables, dict):
			errors.append("schema.tables must be an object mapping table names to column arrays")
		else:
			for tname, cols in tables.items():
				if not isinstance(tname, str) or not tname:
					errors.append("schema.tables contains an invalid table name")
				if not isinstance(cols, list):
					errors.append(f"schema.tables.{tname} must be an array of columns")
					continue
				for i, col in enumerate(cols):
					if not isinstance(col, dict):
						errors.append(f"schema.tables.{tname}[{i}] must be an object")
						continue
					for ck in ["name", "type", "special"]:
						if ck not in col:
							errors.append(f"schema.tables.{tname}[{i}] missing '{ck}'")
					if "name" in col and not isinstance(col["name"], str):
						errors.append(f"schema.tables.{tname}[{i}].name must be string")
					if "type" in col and not isinstance(col["type"], str):
						errors.append(f"schema.tables.{tname}[{i}].type must be string")
					if "special" in col and not (isinstance(col["special"], str) or col["special"] is None):
						errors.append(f"schema.tables.{tname}[{i}].special must be string or null")
	else:
		errors.append("schema must be an object")

	# generation_properties
	gp = payload.get("generation_properties", {})
	if isinstance(gp, dict):
		rc = gp.get("row_counts")
		if not isinstance(rc, dict):
			errors.append("generation_properties.row_counts must be an object")
		else:
			for k, v in rc.items():
				if not isinstance(k, str) or not isinstance(v, int) or v < 0:
					errors.append("row_counts must map table_name (str) to non-negative int")
		dist = gp.get("distributions")
		if dist is not None and not isinstance(dist, dict):
			errors.append("generation_properties.distributions must be an object if present")
		elif isinstance(dist, dict):
			for key, dval in dist.items():
				if not isinstance(key, str) or "." not in key:
					errors.append("distributions keys must be 'table.column'")
				if not isinstance(dval, dict):
					errors.append("distributions values must be objects")
				else:
					# Optional string fields
					for fld in ["in_sales_table", "base", "pattern"]:
						if fld in dval and not (isinstance(dval[fld], str) and dval[fld].strip()):
							errors.append(f"distributions['{key}'].{fld} must be a non-empty string if provided")
	else:
		errors.append("generation_properties must be an object")

	# stress_goals
	sg = payload.get("stress_goals")
	if not isinstance(sg, list) or not all(isinstance(x, str) and x.strip() for x in sg or []):
		errors.append("stress_goals must be a list of non-empty strings")

	return json.dumps({"valid": len(errors) == 0, "errors": errors})

@tool("save_ir_plan", return_direct=True)
def save_ir_plan(ir_json: str, filename_stem: Optional[str] = None) -> str:
	"""Save the IR JSON string to the outputs directory with a timestamped name.

	Args:
		ir_json: The IR JSON string (may be fenced). Must be valid JSON.
		filename_stem: Optional file stem (without extension). Defaults to 'ir_plan'.

	Returns:
		The absolute path of the saved file.
	"""
	try:
		parsed = json.loads(_strip_code_fences(ir_json))
	except Exception as e:
		return json.dumps({"status": "error", "message": f"IR not valid JSON: {e}"})

	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	stem = (filename_stem or "ir_plan").strip() or "ir_plan"
	out_path = OUTPUTS_DIR / f"{stem}_{ts}.json"
	out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
	return json.dumps({"status": "ok", "path": str(out_path)})

def build_agent() -> Any:
	load_dotenv()
	tools = [validate_ir_schema, save_ir_plan]
	model = ChatGoogleGenerativeAI(
		model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro"),
		temperature=1.0,
        max_retries=2,
        google_api_key= os.getenv("GEMINI_API_KEY")
    )
	agent = create_react_agent(model,tools)
	return agent

def run_ir_generation(user_prompt: str, filename_stem: Optional[str] = None) -> Dict[str, Any]:
	"""Run the ReAct agent to produce an IR JSON.

	The agent is instructed to output only the IR JSON as the final message.
	This function returns a dict with keys: {"ir_plan": str | None, "error": str | None, "saved_path": str | None}.
	"""
	agent = build_agent()
	messages = [
		SystemMessage(content=SYSTEM_PROMPT),
		HumanMessage(content=(
			"Convert the following request into an IR JSON matching the schema above. "
			"If needed, use the validation tool to check shape, fix issues, then as final answer output ONLY the JSON.\n\n"
			f"User Prompt:\n\"{user_prompt}\"\n\n"
			+ (f"When correct, save it using filename_stem='{filename_stem}'." if filename_stem else "When correct, save it.")
		)),
	]

	try:
		result = agent.invoke({"messages": messages})
	except Exception as e:
		return {"ir_plan": None, "error": str(e), "saved_path": None}

	# Extract final message content
	final_messages = result.get("messages", [])
	final_text: Optional[str] = None
	if final_messages:
		# Find last AI message content
		for msg in reversed(final_messages):
			# Tool messages might be dicts; we look for string content
			content = getattr(msg, "content", None)
			if isinstance(content, str) and content.strip():
				final_text = content.strip()
				break

	if not final_text:
		return {"ir_plan": None, "error": "Agent did not return content", "saved_path": None}

	cleaned = _strip_code_fences(final_text)

	# Try to parse; if it isn't JSON (e.g., tool output), try to locate a JSON substring
	try:
		parsed = json.loads(cleaned)
		ir_plan_str = json.dumps(parsed, indent=2)
	except Exception:
		# Best effort: locate first/last braces
		start = cleaned.find("{")
		end = cleaned.rfind("}")
		if start != -1 and end != -1 and end > start:
			maybe = cleaned[start : end + 1]
			try:
				parsed = json.loads(maybe)
				ir_plan_str = json.dumps(parsed, indent=2)
			except Exception as e:
				return {"ir_plan": None, "error": f"Final output not valid JSON: {e}", "saved_path": None}
		else:
			return {"ir_plan": None, "error": "Final output did not contain JSON", "saved_path": None}

	# # Optionally save again here if the agent didn't call the save tool
	# saved_path: Optional[str] = None
	# try:
	# 	# If the agent called save_ir_plan, it would have produced a tool message. We can still save here.
	# 	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	# 	stem = (filename_stem or "ir_plan").strip() or "ir_plan"
	# 	out_path = OUTPUTS_DIR / f"{stem}_{ts}_enhanced.json"
	# 	out_path.write_text(ir_plan_str, encoding="utf-8")
	# 	saved_path = str(out_path)
	# except Exception:
	# 	pass

	# return {"ir_plan": ir_plan_str, "error": None, "saved_path": saved_path}

if __name__ == "__main__":

	demo_prompt = (
		"Generate a logistics dataset with one central fact table and four supporting "
		"dimension tables. The fact table should contain 75 million rows, representing "
		"individual shipment events. Shipment volumes should follow a Pareto distribution, "
		"where a small set of distribution hubs handle the majority of shipments, while many "
		"smaller hubs handle only a few. Delivery activity should show weekly seasonality, with "
		"clear peaks on Mondays and Fridays. The data should be designed to stress-test join "
		"performance and complex aggregations, with a mix of high-cardinality attributes like "
		"tracking numbers and low-cardinality attributes like shipping methods. The four dimension "
		"tables should represent Hubs, Carriers, Delivery Routes, and Time, each with rich descriptive "
		"attributes to support diverse analytical queries."
	)

	result = run_ir_generation(demo_prompt, filename_stem="ir_plan")

	if result["error"]:
		print("Agent failed to generate a plan.")
		print(f"Error: {result['error']}")
	else:
		if result.get("saved_path"):
			print(f"IR plan saved to: {result['saved_path']}")
		print("\n--- FINAL IR PLAN ---\n")
		print(result["ir_plan"])  # JSON string