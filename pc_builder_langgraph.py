from langgraph.graph import StateGraph, END
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, List, Tuple, Optional
import re

# 1. Setup Ollama model with low temperature for consistency
llm = ChatOllama(model="llama2", temperature=0.1, confidence=0.95,top_p=0.9,
    num_ctx=4096)

# 2. Prompt template
system_prompt = """
You are a professional PC building assistant. Your job is to help users build the best PC for their needs based on their budget and primary use-case.

INSTRUCTIONS:
1. Query Evaluation

    - If the user query is about building or upgrading a PC, selecting components, or hardware-related topics, you MUST respond with a complete and valid JSON object for a PC build recommendation.

    - If the query is not about PC hardware, respond with a JSON object indicating:
      {{
        "status": "rejected",
        "message": "Sorry, I only provide PC building advice."
      }}

2. PC Build Recommendation:
   - Your ENTIRE response MUST be a valid JSON object adhering strictly to the following structure. Ensure ALL components listed below are included in the 'recommendations' array. Choose ONE specific value for 'price' and 'company' for each component.

   {{
     "status": "complete" or "pending",
     "region": "India",
     "recommendation_description": "A PC build recommendation for science studies within a budget of 2 Lakhs.",
     "recommendations": [
       {{
         "part": "CPU",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "GPU",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "Motherboard",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "RAM",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "Storage",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "Power Supply",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "Cooling",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "PC Case",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "OS",
         "description": "detailed_description",
         "price": "price_value",
         "company": "company_name",
         "product_launched_in_year": "year",
         "generation": "generation_info"
       }},
       {{
         "part": "Peripherals",
         "description": [
           {{
             "part_name": "peripheral_name",
             "price": "price_value",
             "company": "company_name",
             "product_launched_in_year": "year",
             "generation": "generation_info"
           }}
         ]
       }}
     ],
     "total_estimated_cost": "total_cost"
   }}

REQUIREMENTS:
- Your ENTIRE response MUST be valid JSON. Do not include any surrounding text, explanations, or commentary.
- Follow the JSON structure precisely, including all commas and quotation marks.
- The "status" should be "complete" if both budget and use-case are identified, otherwise "pending".
- If the user hasn't specified a budget, default to a mid-range build.
- Avoid newline escape characters like \\n or tab characters like \\t in JSON response.
- Ensure all JSON values are properly quoted strings.
- **YOU MUST INCLUDE ALL COMPONENTS LISTED IN THE JSON STRUCTURE IN THE 'recommendations' ARRAY, INCLUDING 'Peripherals'. For 'Peripherals', include at least one common item like 'Keyboard and Mouse Combo' with a placeholder price if needed.**
- For each component, provide a single, specific value for 'price' and 'company'. Do not use "or".
- Do not include null values - use appropriate default values instead.
- Do not include newlines in the JSON response.
- All prices should be in the same currency format (e.g., "INR 25000"). Do not use currency symbols like '₹' or '$'.
- The 'total_estimated_cost' MUST be a realistic sum of the individual 'price' values provided in the 'recommendations' array.
- The response must be valid JSON that can be parsed by json.loads().
- For each component, provide specific model numbers and current market prices (if possible).
- Ensure all components are compatible with each other.
- Consider the user's region for pricing and availability (Noida, Uttar Pradesh, India).
- If a specific component is not available, recommend the closest alternative.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 3. State
class PCBuildState(TypedDict):
    question: str
    history: List[Tuple[str, str]]
    response: Optional[dict]

# 4. Helpers
def detect_budget(text: str) -> Tuple[bool, Optional[str]]:
    if not isinstance(text, str): return False, None
    if "budget" in text.lower():
        match = re.findall(r'\$?\d+(?:,\d+)?(?:\.\d+)?', text)
        return True, match[0] if match else None
    return False, None

def detect_use_case(text: str) -> bool:
    if not isinstance(text, str): return False
    keywords = ["gaming", "stream", "edit", "render", "work", "office", "1080p", "1440p", "4k", "design"]
    return any(k in text.lower() for k in keywords)

# Create a relevance check prompt
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at determining if a user's request is relevant to building or upgrading a personal computer. A relevant query includes questions about PC hardware, components, upgrades, compatibility, and building guides.

Respond with a JSON object indicating 'is_relevant' (true/false). Ensure your response is ONLY a valid JSON object.

Example of relevant queries:
- "What's a good graphics card for gaming?"
- "Help me build a PC for video editing."
- "Is my CPU compatible with this motherboard?"
- "I want to upgrade my RAM."

Example of non-relevant queries:
- "What's the weather like?"
- "Can you write a poem?"
- "How do I fix my printer?"
"""),
    ("human", "{question}")
])

relevance_chain = relevance_prompt | llm | StrOutputParser()

def is_pc_build_related(text: str) -> bool:
    if not isinstance(text, str):
        return False

    try:
        response = relevance_chain.invoke({"question": text})
        result = json.loads(response)
        return result.get("is_relevant", False)
    except Exception as e:
        print(f"Error during relevance check: {e}")
        # Fallback to basic check if LLM check fails
        return any(word in text.lower() for word in ["pc", "computer", "build", "upgrade", "hardware"])

# 5. Agent node
def pc_build_agent_node(state: PCBuildState) -> PCBuildState:
    question = state.get("question", "").strip()
    history = state.get("history", [])

    # --- Relevance Check using LLM ---
    relevance_prompt_llm = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at determining if a user's request is relevant to building or upgrading a personal computer. Respond with a JSON object indicating 'is_relevant' (true/false). Ensure your response is ONLY a valid JSON object."""),
    ("human", "{question}")
    ])
    relevance_chain_llm = relevance_prompt_llm | llm | StrOutputParser()

    try:
        relevance_response = relevance_chain_llm.invoke({"question": question})
        relevance_result = json.loads(relevance_response)
        is_relevant = relevance_result.get("is_relevant", False)
    except json.JSONDecodeError as e:
        print(f"Error decoding relevance JSON: {e}")
        is_relevant = False
    except Exception as e:
        print(f"Error during LLM relevance check: {e}")
        is_relevant = False

    if not is_relevant:
        response_text = (
            "❌ Sorry, your query does not seem to be about building or upgrading a PC. "
            "Please ask something related to building or upgrading a computer."
        )
        structured_response = {
            "status": "rejected",
            "prompt": question,
            "message": response_text
        }
        if question:
            history.append(("human", question))
        history.append(("assistant", response_text))
        return {
            "question": "",
            "history": history,
            "response": structured_response
        }

    # --- Budget Detection (Keep this simple for now) ---
    budget_in_msg, budget_val = detect_budget(question)

    # --- Use Case Detection using LLM ---
    use_case_prompt_llm = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at identifying the primary use case from a user's request for building a PC. If the use case is clear, extract it concisely (e.g., "gaming", "video editing", "science studies"). If it's not clear or multiple use cases are mentioned, respond with "general use"."""),
        ("human", "{question}")
    ])
    use_case_chain_llm = use_case_prompt_llm | llm | StrOutputParser()

    try:
        use_case = use_case_chain_llm.invoke({"question": question}).strip().lower()
        use_case_provided = use_case != "general use"
    except Exception as e:
        print(f"Error during LLM use case detection: {e}")
        use_case = "general use"
        use_case_provided = False

    if not budget_val:
        for role, msg in history:
            if role == "human":
                _, b = detect_budget(msg)
                if b:
                    budget_val = b
                    break

    budget_provided = budget_in_msg or any(detect_budget(msg)[0] for r, msg in history if r == "human")

    full_context = "\n".join([f"{r}: {msg}" for r, msg in history])
    full_prompt = f"{full_context}\nUser: {question}" if history else question

    budget_requests = sum(1 for r, msg in history if r == "assistant" and "budget" in msg.lower())
    proceed = not history or (budget_requests >= 1 and use_case_provided)

    if not (budget_provided and use_case_provided) and not proceed:
        missing = []
        if not budget_provided: missing.append("budget")
        if not use_case_provided: missing.append("primary use-case")
        response_text = (
            f"Before I can recommend PC parts, I need your {' and '.join(missing)}. "
            "Could you provide this?"
        )
        structured_response = {
            "status": "pending",
            "prompt": full_prompt,
            "message": response_text
        }
    else:
        instructions = []
        if not history:
            instructions.append("Ask for budget and use case.")
        if budget_val:
            instructions.append(f"User has budget {budget_val}. Recommend parts optimized for '{use_case}'.")
        elif budget_provided:
            instructions.append("User mentioned budget, but no amount. Ask for amount.")
        elif budget_requests >= 1:
            instructions.append("No budget specified. Provide mid-range build optimized for '{use_case}'." if use_case_provided else "No budget or clear use case. Provide a general mid-range build.")
        if not budget_provided:
            instructions.append("Ask for the budget.")
        if not use_case_provided:
            instructions.append("Ask for the primary use case.")

        full_prompt += "\n\nInstructions: " + ". ".join(instructions)

        import ast

        raw_response = chain.invoke({"question": full_prompt}).strip()

        try:
            structured_response = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"JSONDecodeError: Attempting literal evaluation...")
            try:
                structured_response = ast.literal_eval(raw_response)
            except Exception as e:
                structured_response = {
                    "status": "error",
                    "message": f"Failed to parse JSON response: {raw_response}",
                    "error": str(e)
                }
                print(structured_response)
        except Exception as e:
            structured_response = {
                "status": "error",
                "message": f"An unexpected error occurred during JSON parsing: {raw_response}",
                "error": str(e)
            }
            print(structured_response)

        if "recommendations" in structured_response:
            for recommendation in structured_response["recommendations"]:
                if "company" not in recommendation:
                    recommendation["company"] = "Default Company"
                if "product_launched_in_year" not in recommendation:
                    recommendation["product_launched_in_year"] = "2023"
                if "generation" not in recommendation:
                    recommendation["generation"] = "Mid-range"

    if question:
        history.append(("human", question))
    history.append(("assistant", structured_response.get("message", "")))

    return {
        "question": "",
        "history": history,
        "response": structured_response
    }
    
    
# 6. LangGraph setup
builder = StateGraph(PCBuildState)
builder.add_node("PCBuildAgent", pc_build_agent_node)
builder.set_entry_point("PCBuildAgent")
builder.add_edge("PCBuildAgent", END)
graph = builder.compile()

# 7. CLI Runner
if __name__ == "__main__":
    print("\U0001F44B Welcome to the PC Build Assistant!")
    print("Type 'exit' to quit.\n")

    state: PCBuildState = {
        "question": "",
        "history": [],
        "response": None
    }

    while True:
        user_input = input("\U0001F9D1 You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("\U0001F44B Goodbye!")
            break

        state["question"] = user_input
        state = graph.invoke(state)

        print("\n\U0001F4AC Assistant:\n", state["response"], "\n")
