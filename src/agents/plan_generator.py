"""
Plan Generator Agent using vLLM OpenAI-compatible API

This module generates structured JSON plans from natural language descriptions
for database schema and data generation. It uses vLLM's OpenAI-compatible API
instead of directly launching vLLM instances.

Configuration via Environment Variables:
- VLLM_API_BASE: Base URL for vLLM API (default: http://localhost:8000/v1)
- VLLM_API_KEY: API key (default: "EMPTY" - vLLM doesn't require real keys)
- VLLM_MODEL_NAME: Model name (default: meta-llama/Meta-Llama-3.1-8B-Instruct)

Make sure to start your vLLM server before running this script:
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --host 0.0.0.0

You can also customize the server configuration:
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --port 8000 \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9
"""

import json
import os
from datetime import datetime
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Configuration ---
# vLLM OpenAI-compatible API configuration
# You can override these with environment variables
VLLM_CONFIG = {
    "api_base": os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),  # vLLM doesn't require a real API key
    "model_name": os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "2048")),
    "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.5")),
    "top_p": float(os.getenv("VLLM_TOP_P", "1.0")),
}

# --- 1. Define the State for our Agent ---
# The state is a "snapshot" of our agent's memory at any given time.
# For now, it just needs to hold the user's prompt and the generated IR plan.
class AgentState(TypedDict):
    """
    Represents the state of our data generation agent.
    
    Attributes:
        user_prompt (str): The initial natural language prompt from the user.
        ir_plan (str): The generated Intermediate Representation (IR) JSON plan.
        error_message (str): To hold any potential error messages.
    """
    user_prompt: str
    ir_plan: str
    error_message: str

# --- 2. Define the System Prompt and One-Shot Example ---
try:
    with open("/data4/home/amittalawar/DBMS-project/nl_datagen/config/retail_sales_ir_v1.json", "r") as f:
        ONE_SHOT_EXAMPLE = json.dumps(json.load(f), indent=2)
except FileNotFoundError:
    print("Error: The example IR file 'config/retail_sales_ir_v1.json' was not found.")
    exit() # Exit if the example is not found, as the agent quality will be poor.

SYSTEM_PROMPT = f"""You are an expert database administrator and data engineer. Your task is to convert a user's natural language request for a synthetic dataset into a precise and structured JSON object, which we call the Intermediate Representation (IR).

You must adhere strictly to the provided JSON schema. Do not add any keys that are not in the schema. Do not explain your output. Only output the JSON object.

**JSON SCHEMA:**
```json
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
```

**HIGH-QUALITY EXAMPLE:**

**User Prompt:**
"Generate a retail sales dataset with one fact table and three dimension tables. The fact table should have 50 million rows. Product sales should follow a Zipf distribution (few items sold very frequently, many items rarely sold). Customer purchases should be seasonal with peaks in December. The data should stress-test group-by and join performance."

**Your Output (The IR JSON):**
```json
{ONE_SHOT_EXAMPLE}
```
"""


# --- 3. Helper Functions ---

def validate_vllm_connection() -> bool:
    """
    Validates connection to the vLLM server.
    
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        import requests
        response = requests.get(f"{VLLM_CONFIG['api_base'].replace('/v1', '')}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not validate vLLM connection: {e}")
        return False

# --- 4. Define the Agent's Core Logic (The Node) ---

def generate_ir_plan(state: AgentState) -> AgentState:
    """
    Node that invokes the LLM to generate the IR plan from the user prompt.
    """
    print("--- INVOKING LLM TO GENERATE PLAN ---")
    user_prompt = state['user_prompt']

    # Initialize ChatOpenAI with vLLM configuration
    llm = ChatOpenAI(
        base_url=VLLM_CONFIG["api_base"],
        api_key=VLLM_CONFIG["api_key"],
        model=VLLM_CONFIG["model_name"],
        max_tokens=VLLM_CONFIG["max_tokens"],
        temperature=VLLM_CONFIG["temperature"],
        top_p=VLLM_CONFIG["top_p"],
    )
    
    # Create messages for chat format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"**User Prompt:**\n\"{user_prompt}\"\n\n**Your Output (The IR JSON):**"}
    ]

    try:
        response = llm.invoke(messages)
        # Extract content from the response (ChatOpenAI returns an AIMessage object)
        response_content = response.content
        
        # Clean up the response to extract only the JSON
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].strip()
        if "```" in response_content:
            response_content = response_content.split("```")[0].strip()

        # Validate that the response is valid JSON
        parsed_json = json.loads(response_content)
        print("--- PLAN GENERATED SUCCESSFULLY ---")
        return {"ir_plan": json.dumps(parsed_json, indent=2), "error_message": None}

    except json.JSONDecodeError:
        print("--- ERROR: LLM OUTPUT WAS NOT VALID JSON ---")
        return {"ir_plan": None, "error_message": "Failed to decode LLM response into valid JSON."}
    except Exception as e:
        print(f"--- AN UNEXPECTED ERROR OCCURRED: {e} ---")
        return {"ir_plan": None, "error_message": str(e)}


# --- 5. Build the Graph ---
# For now, this is a very simple, linear graph with only one step.
workflow = StateGraph(AgentState)

# Add our single node to the graph
workflow.add_node("planner", generate_ir_plan)

# Set the entry point of the graph
workflow.set_entry_point("planner")

# The graph ends after the planner node is done.
workflow.add_edge("planner", END)

# Compile the graph into a runnable application
app = workflow.compile()


# --- 6. Run the Agent ---
if __name__ == "__main__":
    # Print configuration
    print(f"Using vLLM API at: {VLLM_CONFIG['api_base']}")
    print(f"Model: {VLLM_CONFIG['model_name']}")
    
    # Validate connection (optional - will continue even if validation fails)
    if validate_vllm_connection():
        print("✓ vLLM server connection validated")
    else:
        print("⚠ Could not validate vLLM server connection - proceeding anyway")
    
    prompt = "Generate a logistics dataset with one central fact table and four supporting dimension tables. The fact table should contain 75 million rows, representing individual shipment events. Shipment volumes should follow a Pareto distribution, where a small set of distribution hubs handle the majority of shipments, while many smaller hubs handle only a few. Delivery activity should show weekly seasonality, with clear peaks on Mondays and Fridays. The data should be designed to stress-test join performance and complex aggregations, with a mix of high-cardinality attributes like tracking numbers and low-cardinality attributes like shipping methods. The four dimension tables should represent Hubs, Carriers, Delivery Routes, and Time, each with rich descriptive attributes to support diverse analytical queries."

    inputs = {"user_prompt": prompt, "ir_plan": None, "error_message": None}

    result = app.invoke(inputs)

    # Save the final IR plan to outputs directory
    if result.get("ir_plan"):
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ir_plan_{timestamp}.json"
        output_path = os.path.join("/data4/home/amittalawar/DBMS-project/nl_datagen/outputs", output_filename)
        
        try:
            with open(output_path, "w") as f:
                f.write(result["ir_plan"])
            print(f"IR plan saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save IR plan: {e}")

    # Print the final IR plan
    print("\n--- FINAL IR PLAN ---\n")
    if result.get("ir_plan"):
        print(result["ir_plan"])
    else:
        print("Agent failed to generate a plan.")
        print(f"Error: {result.get('error_message')}")
