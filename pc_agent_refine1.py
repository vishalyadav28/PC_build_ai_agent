# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables import Runnable
# from langchain_ollama import OllamaLLM
# from pydantic import BaseModel, ValidationError
# import re
# import logging
# from typing import Tuple, Dict, List, Union, Optional
# import asyncio

# # Set up logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # --- LLM Relevance Chain Setup ---

# class RelevanceOutput(BaseModel):
#     is_relevant: bool
#     reason: str
#     category: str = "not_relevant"  # Default value

# prompt = PromptTemplate.from_template("""
# You are a PC Build Assistant that helps determine if user queries are about computer hardware and building PCs.

# Your task is to evaluate if the following query is SPECIFICALLY about:
# - Building/assembling physical computers or workstations
# - Upgrading or selecting physical computer hardware parts
# - Hardware compatibility, performance, or configuration
# - Purchasing decisions related to computer hardware

# Very important instructions:
# 1. A query like "I want to build a PC for programming" IS relevant (it's about PC building for a specific use case).
# 2. A query like "How does Python case syntax work?" is NOT relevant (it's only about programming).
# 3. A query like "Can I fit a PC in my car?" IS relevant (it's about PC hardware logistics) but DOESN'T need build recommendations.
# 4. Focus on whether the primary intent is hardware-related, not the end use of the computer.

# Query: {input}

# Respond ONLY in JSON format with these two keys:
# - is_relevant: true or false  
# - reason: a brief explanation (max 20 words)
# - category: one of ["build_recommendation", "transport", "comparison", "general_info", "not_relevant"]

# Example responses:
# For "What's a good graphics card under $300?":
# {{"is_relevant": true, "reason": "Directly asking about PC hardware component selection.", "category": "build_recommendation"}}

# For "Python syntax for switch statements":
# {{"is_relevant": false, "reason": "Purely about programming language syntax, not hardware.", "category": "not_relevant"}}

# For "Best PC build for Java development":
# {{"is_relevant": true, "reason": "Asking about PC hardware for a specific use case.", "category": "build_recommendation"}}

# For "Can I fit my PC in a Honda Civic?":
# {{"is_relevant": true, "reason": "Question about PC transport/logistics.", "category": "transport"}}
# """)

# parser = JsonOutputParser(pydantic_object=RelevanceOutput)

# llm = OllamaLLM(model="llama2")
# relevance_chain: Runnable = prompt | llm | parser

# # --- Relevance Checker Class ---

# class RelevanceChecker:
#     def __init__(self):
#         self.keywords = [
#             "pc", "computer", "build", "gpu", "cpu", "motherboard",
#             "hardware", "ram", "upgrade", "ssd", "power supply", "cabinet"
#         ]
        
#         # Database of PC parts - this would ideally come from an external database
#         self.parts_database = {
#             "CPU": [
#                 {"name": "AMD Ryzen 7 7800X3D", "description": "High-performance CPU for gaming", "price": "380", "company": "AMD", "year": "2023", "generation": "Ryzen 7000"},
#                 {"name": "Intel Core i7-14700K", "description": "Performance hybrid architecture CPU", "price": "400", "company": "Intel", "year": "2023", "generation": "14th Gen"},
#                 {"name": "AMD Ryzen 5 7600", "description": "Mid-range gaming CPU", "price": "220", "company": "AMD", "year": "2023", "generation": "Ryzen 7000"}
#             ],
#             "GPU": [
#                 {"name": "NVIDIA RTX 4070", "description": "Great for 1440p gaming", "price": "550", "company": "NVIDIA", "year": "2023", "generation": "RTX 40 Series"},
#                 {"name": "AMD RX 7800 XT", "description": "High-performance AMD GPU", "price": "500", "company": "AMD", "year": "2023", "generation": "RDNA 3"},
#                 {"name": "NVIDIA RTX 4060 Ti", "description": "Efficient 1080p/1440p gaming", "price": "400", "company": "NVIDIA", "year": "2023", "generation": "RTX 40 Series"}
#             ],
#             # Add more categories as needed
#         }

#     def keyword_check(self, text: str) -> bool:
#         """
#         Perform a context-aware keyword-based relevance check.
#         Determines if the text is more about PC hardware or programming.
#         """
#         text_lower = text.lower()
        
#         # Check for PC hardware terms
#         pc_terms = [
#             "pc", "computer", "build", "gpu", "cpu", "motherboard", "processor",
#             "hardware", "ram", "upgrade", "ssd", "power supply", "cabinet",
#             "cooling", "fan", "case", "desktop", "laptop", "gaming", "workstation",
#             "performance", "fps", "graphics", "card", "memory", "storage"
#         ]
        
#         # Context indicators that suggest the question is about building a PC FOR some purpose
#         pc_build_context = [
#             "build for", "pc for", "computer for", "system for", "setup for",
#             "build a pc", "building a pc", "assemble", "put together", "specs for",
#             "recommend", "suggestion", "budget for", "best for", "ideal for"
#         ]
        
#         # Patterns that strongly indicate a programming question rather than a PC build question
#         programming_only_patterns = [
#             r'how to (code|program|write|implement)',
#             r'(syntax|function|method|variable|class) in \w+',
#             r'(error|bug|exception|debug) in (code|program)',
#             r'(algorithm|data structure)',
#         ]
        
#         # Count occurrences of PC terms
#         pc_term_count = sum(1 for term in pc_terms if term in text_lower)
        
#         # Check for strong PC build context
#         has_build_context = any(context in text_lower for context in pc_build_context)
        
#         # Check if it's clearly a programming-only question
#         is_programming_only = any(re.search(pattern, text_lower) for pattern in programming_only_patterns)
        
#         # Decision logic:
#         # 1. If it contains build context phrases → likely relevant
#         # 2. If it has multiple PC terms → likely relevant
#         # 3. If it matches programming-only patterns → likely not relevant
#         # 4. Balance between PC terms and programming context
        
#         if has_build_context and pc_term_count >= 1:
#             return True  # "Build a PC for Python development" - relevant
#         elif pc_term_count >= 3:
#             return True  # Multiple PC terms suggest hardware focus
#         elif is_programming_only:
#             return False  # Clearly just about programming
#         else:
#             # For ambiguous cases, require at least some PC terminology
#             return pc_term_count >= 1

#     async def is_relevant(self, text: str) -> Tuple[bool, str, str]:
#         """
#         Check if the query is relevant to PC building and identify its category.
#         Returns (is_relevant, reason, category)
#         """
#         try:
#             # Create a new event loop for each invocation if needed
#             try:
#                 loop = asyncio.get_event_loop()
#                 if loop.is_closed():
#                     asyncio.set_event_loop(asyncio.new_event_loop())
#             except RuntimeError:
#                 asyncio.set_event_loop(asyncio.new_event_loop())
            
#             text_lower = text.lower()
            
#             # Quick check for common query categories
            
#             # Standard PC build intent keywords
#             build_intent_phrases = ["build a pc", "upgrade my pc", "new computer", 
#                                    "best gpu", "recommend cpu", "gaming pc"]
            
#             # Transportation/logistics keywords
#             transport_phrases = ["fit pc", "pc fit", "move pc", "transport pc", 
#                                "pc in car", "shipping pc", "pc dimensions"]
            
#             # Comparison keywords
#             comparison_phrases = ["vs", "better than", "compare", "difference between"]
            
#             # Fast path for common categories
#             if any(phrase in text_lower for phrase in build_intent_phrases):
#                 return True, "Direct PC building/hardware query detected.", "build_recommendation"
            
#             if any(phrase in text_lower for phrase in transport_phrases):
#                 return True, "PC transport/logistics question detected.", "transport"
            
#             if any(phrase in text_lower for phrase in comparison_phrases) and ("pc" in text_lower or "computer" in text_lower):
#                 return True, "PC comparison question detected.", "comparison"
            
#             # Programming syntax without PC context
#             programming_syntax_phrases = ["python syntax", "javascript function", 
#                                         "how to code", "programming language"]
#             if any(phrase in text_lower for phrase in programming_syntax_phrases) and "pc" not in text_lower and "computer" not in text_lower:
#                 return False, "Programming syntax question without PC hardware context.", "not_relevant"
            
#             # Check if query is about building a PC FOR programming
#             build_for_programming = re.search(r'(pc|computer|build|system).+(for|to).+(program|develop|coding|python|javascript)', text_lower)
#             programming_for_pc = re.search(r'(program|develop|coding|python|javascript).+(pc|computer|system)', text_lower)
            
#             if build_for_programming or programming_for_pc:
#                 return True, "Query is about PC hardware for programming use case.", "build_recommendation"
            
#             # For ambiguous cases, consult the LLM
#             logger.info("Using LLM for ambiguous query classification")
#             result = await relevance_chain.ainvoke({"input": text})
            
#             if isinstance(result, dict):
#                 try:
#                     parsed = RelevanceOutput(**result)
#                     logger.info(f"LLM relevance result: {parsed}")
                    
#                     # Return the full classification
#                     return parsed.is_relevant, parsed.reason, parsed.category
                    
#                 except ValidationError as ve:
#                     logger.warning(f"Pydantic parsing failed: {ve}")
#                     # Fall back to keyword check
#                     is_pc_related = self.keyword_check(text)
#                     category = "build_recommendation" if is_pc_related else "not_relevant"
#                     return is_pc_related, "Parsing error, using keyword analysis.", category
#             else:
#                 logger.info(f"LLM relevance result: {result}")
#                 return result.is_relevant, result.reason, getattr(result, 'category', 'general_info')

#         except Exception as e:
#             logger.warning(f"LLM relevance check failed: {e}")
#             # Fall back to keyword check in case of failure
#             is_pc_related = self.keyword_check(text)
#             category = "build_recommendation" if is_pc_related else "not_relevant"
#             return is_pc_related, "Error in processing. Using keyword analysis.", category

#     def get_recommendations_for_query(self, query: str) -> List[Dict]:
#         """
#         Generate relevant recommendations based on the query.
#         This is a placeholder - in a real system, this would use NLP to extract requirements
#         and filter the database accordingly.
#         """
#         # Just a simple example - in reality, you'd want more sophisticated parsing
#         results = []
#         query_lower = query.lower()
        
#         # Check for specific part mentions
#         for part_type, parts in self.parts_database.items():
#             if part_type.lower() in query_lower:
#                 # Add all parts of this type
#                 for part in parts:
#                     results.append({
#                         "part": part_type,
#                         "name": part["name"],
#                         "description": part["description"],
#                         "price": part["price"],
#                         "company": part["company"],
#                         "product_launched_in_year": part["year"],
#                         "generation": part["generation"]
#                     })
        
#         # If no specific parts mentioned, return a balanced selection
#         if not results:
#             # Get one item from each category
#             for part_type, parts in self.parts_database.items():
#                 if parts:  # Make sure the category has parts
#                     part = parts[0]  # Just get the first part as an example
#                     results.append({
#                         "part": part_type,
#                         "name": part["name"],
#                         "description": part["description"],
#                         "price": part["price"],
#                         "company": part["company"],
#                         "product_launched_in_year": part["year"],
#                         "generation": part["generation"]
#                     })
        
#         return results

#     async def format_response(self, query: str, region: str = "unknown") -> Dict:
#         """Process the query and return a formatted response with query categorization."""
#         is_relevant, reason = await self.is_relevant(query)
        
#         # Detect query intent to handle different types of relevant queries
#         query_lower = query.lower()
        
#         # For relevant queries, determine if they're asking for recommendations or something else
#         if is_relevant:
#             # Categories of PC-related queries that shouldn't return build recommendations
#             transport_patterns = [
#                 r'(fit|transport|move|carry|ship|pack|store).*?(pc|computer)',
#                 r'(pc|computer).*?(fit|transport|move|carry|ship|pack|store)',
#                 r'(dimensions|size|weight).*?(pc|computer)',
#                 r'(pc|computer).*?(dimensions|size|weight)',
#                 r'(car|vehicle|trunk|backseat).*?(pc|computer)',
#                 r'(pc|computer).*?(car|vehicle|trunk|backseat)'
#             ]
            
#             # Check for transportation/logistics questions
#             is_transport_question = any(re.search(pattern, query_lower) for pattern in transport_patterns)
            
#             # Check for comparison/opinion questions that don't need build recommendations
#             comparison_patterns = [
#                 r'(better|worse|vs|versus|compare|difference)',
#                 r'(opinion|think|review)',
#                 r'(pros|cons)'
#             ]
#             is_comparison_question = any(re.search(pattern, query_lower) for pattern in comparison_patterns)
            
#             # Don't provide build recommendations for these types of queries
#             if is_transport_question:
#                 return {
#                     "status": "complete",
#                     "region": region,
#                     "recommendation_description": "✅ This is a PC transport/logistics question, not a build recommendation request.",
#                     "answer": "Your question is about transporting or fitting a PC, not about building one. Most mid-tower PC cases are approximately 18-22 inches tall, 8-10 inches wide, and 18-20 inches deep, and should fit in most car trunks or back seats. Smaller form factors like mini-ITX are more portable.",
#                     "recommendations": [],
#                     "total_estimated_cost": "0"
#                 }
#             elif is_comparison_question:
#                 return {
#                     "status": "complete",
#                     "region": region,
#                     "recommendation_description": "✅ This is a PC comparison/opinion question, not a build recommendation request.",
#                     "answer": "Your question is asking for a comparison or opinion about PC hardware, not a specific build recommendation. For detailed comparisons, please specify which components you're interested in comparing.",
#                     "recommendations": [],
#                     "total_estimated_cost": "0"
#                 }
                
#             # Proceed with build recommendations for standard build queries
#             recommendations = self.get_recommendations_for_query(query)
#             total_cost = sum(
#                 int(p["price"]) if isinstance(p.get("price"), str) and p.get("price").isdigit() else 0
#                 for p in recommendations
#             )

#             return {
#                 "status": "complete",
#                 "region": region,
#                 "recommendation_description": f"✅ PC build recommendations for: '{query}'",
#                 "recommendations": recommendations,
#                 "total_estimated_cost": str(total_cost)
#             }
#         else:
#             # Not relevant to PCs at all
#             return {
#                 "status": "complete",
#                 "region": region,
#                 "recommendation_description": f"❌ Irrelevant query: {reason}",
#                 "recommendations": [],
#                 "total_estimated_cost": "0"
#             }


# # --- Main Function for PC Assistant Node ---

# async def pc_assistant_node(state: Dict) -> Dict:
#     """Process a single query and update the state."""
#     question = state.get("question", "")
#     region = state.get("region", "unknown")
    
#     # Create a new RelevanceChecker instance for each request
#     # This helps avoid event loop issues
#     checker = RelevanceChecker()
#     response = await checker.format_response(question, region)
#     state["response"] = response
#     return state


# # --- CLI Main Program ---

# async def main_loop():
#     """Run the main interaction loop."""
#     print("\U0001F44B Welcome to the PC Build Assistant!")
#     print("Type 'exit' to quit.\n")

#     state = {
#         "question": "",
#         "region": "unknown",
#         "response": None
#     }

#     while True:
#         user_input = input("\U0001F9D1 You: ").strip()
#         if user_input.lower() in ("exit", "quit"):
#             print("\U0001F44B Goodbye!")
#             break

#         state["question"] = user_input
        
#         # Use a new event loop for each interaction
#         try:
#             state = await pc_assistant_node(state)
#         except Exception as e:
#             logger.error(f"Error processing input: {e}")
#             state["response"] = {
#                 "status": "error",
#                 "region": state["region"],
#                 "recommendation_description": f"Error processing your request: {str(e)}",
#                 "recommendations": [],
#                 "total_estimated_cost": "0"
#             }

#         print("\n\U0001F4AC Assistant:\n", state["response"], "\n")


# if __name__ == "__main__":
#     asyncio.run(main_loop())




from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, ValidationError
import re
import logging
from typing import Tuple, Dict, List, Union, Optional
import asyncio

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- LLM Relevance Chain Setup ---

class RelevanceOutput(BaseModel):
    is_relevant: bool
    reason: str
    category: str = "not_relevant"  # Default value

prompt = PromptTemplate.from_template("""
You are a PC Build Assistant that helps determine if user queries are about computer hardware and building PCs.

Your task is to evaluate if the following query is SPECIFICALLY about:
- Building/assembling physical computers or workstations
- Upgrading or selecting physical computer hardware parts
- Hardware compatibility, performance, or configuration
- Purchasing decisions related to computer hardware

Very important instructions:
1. A query like "I want to build a PC for programming" IS relevant (it's about PC building for a specific use case).
2. A query like "How does Python case syntax work?" is NOT relevant (it's only about programming).
3. A query like "Can I fit a PC in my car?" IS relevant (it's about PC hardware logistics) but DOESN’T need build recommendations.
4. Focus on whether the primary intent is hardware-related, not the end use of the computer.

Query: {input}

Respond ONLY in JSON format with these two keys:
- is_relevant: true or false  
- reason: a brief explanation (max 20 words)
- category: one of ["build_recommendation", "transport", "comparison", "general_info", "not_relevant"]
""")

parser = JsonOutputParser(pydantic_object=RelevanceOutput)

llm = OllamaLLM(model="llama2")
relevance_chain: Runnable = prompt | llm | parser

# --- Relevance Checker Class ---

class RelevanceChecker:
    def __init__(self):
        self.keywords = [
            "pc", "computer", "build", "gpu", "cpu", "motherboard",
            "hardware", "ram", "upgrade", "ssd", "power supply", "cabinet"
        ]
        
        # Database of PC parts - this would ideally come from an external database
        self.parts_database = {
            "CPU": [
                {"name": "AMD Ryzen 7 7800X3D", "description": "High-performance CPU for gaming", "price": "380", "company": "AMD", "year": "2023", "generation": "Ryzen 7000"},
                {"name": "Intel Core i7-14700K", "description": "Performance hybrid architecture CPU", "price": "400", "company": "Intel", "year": "2023", "generation": "14th Gen"},
                {"name": "AMD Ryzen 5 7600", "description": "Mid-range gaming CPU", "price": "220", "company": "AMD", "year": "2023", "generation": "Ryzen 7000"}
            ],
            "GPU": [
                {"name": "NVIDIA RTX 4070", "description": "Great for 1440p gaming", "price": "550", "company": "NVIDIA", "year": "2023", "generation": "RTX 40 Series"},
                {"name": "AMD RX 7800 XT", "description": "High-performance AMD GPU", "price": "500", "company": "AMD", "year": "2023", "generation": "RDNA 3"},
                {"name": "NVIDIA RTX 4060 Ti", "description": "Efficient 1080p/1440p gaming", "price": "400", "company": "NVIDIA", "year": "2023", "generation": "RTX 40 Series"}
            ],
            # Add more categories as needed
        }

    def keyword_check(self, text: str) -> bool:
        """
        Perform a context-aware keyword-based relevance check.
        Determines if the text is more about PC hardware or programming.
        """
        text_lower = text.lower()
        
        # Check for PC hardware terms
        pc_terms = [
            "pc", "computer", "build", "gpu", "cpu", "motherboard", "processor",
            "hardware", "ram", "upgrade", "ssd", "power supply", "cabinet",
            "cooling", "fan", "case", "desktop", "laptop", "gaming", "workstation",
            "performance", "fps", "graphics", "card", "memory", "storage"
        ]
        
        # Context indicators that suggest the question is about building a PC FOR some purpose
        pc_build_context = [
            "build for", "pc for", "computer for", "system for", "setup for",
            "build a pc", "building a pc", "assemble", "put together", "specs for",
            "recommend", "suggestion", "budget for", "best for", "ideal for"
        ]
        
        # Patterns that strongly indicate a programming question rather than a PC build question
        programming_only_patterns = [
            r'how to (code|program|write|implement)',
            r'(syntax|function|method|variable|class) in \w+',
            r'(error|bug|exception|debug) in (code|program)',
            r'(algorithm|data structure)',
        ]
        
        # Count occurrences of PC terms
        pc_term_count = sum(1 for term in pc_terms if term in text_lower)
        
        # Check for strong PC build context
        has_build_context = any(context in text_lower for context in pc_build_context)
        
        # Check if it's clearly a programming-only question
        is_programming_only = any(re.search(pattern, text_lower) for pattern in programming_only_patterns)
        
        # Decision logic:
        # 1. If it contains build context phrases → likely relevant
        # 2. If it has multiple PC terms → likely relevant
        # 3. If it matches programming-only patterns → likely not relevant
        # 4. Balance between PC terms and programming context
        
        if has_build_context and pc_term_count >= 1:
            return True  # "Build a PC for Python development" - relevant
        elif pc_term_count >= 3:
            return True  # Multiple PC terms suggest hardware focus
        elif is_programming_only:
            return False  # Clearly just about programming
        else:
            # For ambiguous cases, require at least some PC terminology
            return pc_term_count >= 1

    async def is_relevant(self, text: str) -> Tuple[bool, str, str]:
        """
        Check if the query is relevant to PC building and identify its category.
        Returns (is_relevant, reason, category)
        """
        try:
            # Create a new event loop for each invocation if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    asyncio.set_event_loop(asyncio.new_event_loop())
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            
            text_lower = text.lower()
            
            # Quick check for common query categories
            
            # Standard PC build intent keywords
            build_intent_phrases = ["build a pc", "upgrade my pc", "new computer", 
                                   "best gpu", "recommend cpu", "gaming pc"]
            
            # Transportation/logistics keywords
            transport_phrases = ["fit pc", "pc fit", "move pc", "transport pc", 
                               "pc in car", "shipping pc", "pc dimensions"]
            
            # Comparison keywords
            comparison_phrases = ["vs", "better than", "compare", "difference between"]
            
            # Fast path for common categories
            if any(phrase in text_lower for phrase in build_intent_phrases):
                return True, "Direct PC building/hardware query detected.", "build_recommendation"
            
            if any(phrase in text_lower for phrase in transport_phrases):
                return True, "PC transport/logistics question detected.", "transport"
            
            if any(phrase in text_lower for phrase in comparison_phrases) and ("pc" in text_lower or "computer" in text_lower):
                return True, "PC comparison question detected.", "comparison"
            
            # Programming syntax without PC context
            programming_syntax_phrases = ["python syntax", "javascript function", 
                                        "how to code", "programming language"]
            if any(phrase in text_lower for phrase in programming_syntax_phrases) and "pc" not in text_lower and "computer" not in text_lower:
                return False, "Programming syntax question without PC hardware context.", "not_relevant"
            
            # Check if query is about building a PC FOR programming
            build_for_programming = re.search(r'(pc|computer|build|system).+(for|to).+(program|develop|coding|python|javascript)', text_lower)
            programming_for_pc = re.search(r'(program|develop|coding|python|javascript).+(pc|computer|system)', text_lower)
            
            if build_for_programming or programming_for_pc:
                return True, "Query is about PC hardware for programming use case.", "build_recommendation"
            
            # Check relevance using the keyword-based check
            if self.keyword_check(text):
                return True, "Hardware-related query detected.", "build_recommendation"
            
            # If nothing matches, check with the LLM-based method
            result = await relevance_chain.invoke({"input": text})
            
            # If LLM says it's relevant, return the result
            if result['is_relevant']:
                return True, result['reason'], result['category']
            
            # Otherwise, consider it not relevant
            return False, result['reason'], "not_relevant"
        
        except Exception as e:
            logger.error(f"Error during relevance check: {e}")
            return False, "Error processing relevance.", "not_relevant"

import asyncio

# Initialize the RelevanceChecker
relevance_checker = RelevanceChecker()

async def check_query_relevance(query: str):
    # Call the is_relevant function with the query input
    is_relevant, reason, category = await relevance_checker.is_relevant(query)
    
    # Print the output
    print("Is Relevance:", is_relevant)
    print("Reason:", reason)
    print("Category:", category)

# Example queries to test
queries = [
    "I want to build a PC for gaming",
    "How do I write a Python function?",
    "What’s the best GPU for 4K gaming?",
    "Can I fit a PC inside my car?",
    "What’s the difference between Ryzen 5 and Intel i5?",
    "Can fit a PC in a suitcase?",
    "Can fit a PC in a backpack?",
    "Python case syntax",
    "Case in python",
]

# Run checks for each query
async def run():
    for query in queries:
        print(f"Query: {query}")
        await check_query_relevance(query)
        print("-" * 50)

# Run the asyncio event loop
asyncio.run(run())
