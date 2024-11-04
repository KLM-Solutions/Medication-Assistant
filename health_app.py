import streamlit as st
import requests
import json
from typing import Dict, Any, Optional, Generator

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with PPLX client and system prompts"""
        if 'pplx' not in st.secrets:
            raise ValueError("PPLX API key not found in secrets")
            
        self.pplx_api_key = st.secrets["pplx"]["api_key"]
        self.pplx_model = st.secrets["pplx"].get("model", "medical-pplx")  
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        # Define the JSON structure expected in the response
        self.response_structure = """
        {
            "response": {
                "opening": "empathetic opening acknowledging patient situation",
                "medical_info": "clear validated medical information",
                "safety": "important safety considerations",
                "closing": "encouraging closing statement"
            },
            "sources": ["source1", "source2"]
        }
        """
        
        self.pplx_system_prompt = f"""
You are a specialized medical information assistant focused EXCLUSIVELY on GLP-1 medications (such as Ozempic, Wegovy, Mounjaro, etc.). 

Your responses must be structured in the following JSON format:
{self.response_structure}

You must:

1. ONLY provide information about GLP-1 medications and directly related topics
2. For any query not specifically about GLP-1 medications or their direct effects, respond with:
   {{
       "response": {{
           "opening": "I apologize for any confusion",
           "medical_info": "I can only provide information about GLP-1 medications and related topics",
           "safety": "Please ask a question specifically about GLP-1 medications",
           "closing": "I'm here to help with any GLP-1 medication related questions"
       }},
       "sources": []
   }}

3. For valid GLP-1 queries, ensure the JSON response includes:
   - An empathetic opening acknowledging the patient's situation
   - Clear, validated medical information about GLP-1 medications
   - Important safety considerations
   - An encouraging closing that reinforces their healthcare journey
   - Relevant source citations

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
ALWAYS structure your response in the specified JSON format.
"""

    def stream_pplx_response(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Stream response from PPLX API with sources in JSON format"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": f"{query}\n\nPlease structure the response in JSON format."}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "stream": True
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            accumulated_content = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]  # Remove 'data: ' prefix
                            if json_str.strip() == '[DONE]':
                                break
                            
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['finish_reason'] is not None:
                                break
                                
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                accumulated_content += content
                                try:
                                    # Try to parse the accumulated content as JSON
                                    parsed_json = json.loads(accumulated_content)
                                    yield {
                                        "type": "content",
                                        "data": content,
                                        "accumulated": parsed_json
                                    }
                                except json.JSONDecodeError:
                                    # If not complete JSON yet, yield as is
                                    yield {
                                        "type": "content",
                                        "data": content,
                                        "accumulated": accumulated_content
                                    }
                        except json.JSONDecodeError:
                            continue
            
            try:
                # Parse the final accumulated content as JSON
                final_json = json.loads(accumulated_content)
                yield {
                    "type": "complete",
                    "content": final_json["response"],
                    "sources": final_json["sources"]
                }
            except json.JSONDecodeError:
                yield {
                    "type": "error",
                    "message": "Failed to parse response as JSON"
                }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error communicating with PPLX: {str(e)}"
            }

    def process_streaming_query(self, user_query: str, placeholder) -> Dict[str, Any]:
        """Process user query with streaming response in JSON format"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            query_category = self.categorize_query(user_query)
            full_response = {}
            sources = []
            
            # Initialize the placeholder content
            message_placeholder = placeholder.empty()
            
            # Stream the response
            for chunk in self.stream_pplx_response(user_query):
                if chunk["type"] == "error":
                    placeholder.error(chunk["message"])
                    return {"status": "error", "message": chunk["message"]}
                
                elif chunk["type"] == "content":
                    if isinstance(chunk["accumulated"], dict):
                        full_response = chunk["accumulated"].get("response", {})
                        sources = chunk["accumulated"].get("sources", [])
                        
                        # Update the placeholder with structured JSON response
                        message_placeholder.json({
                            "category": query_category.upper(),
                            "response": full_response,
                            "sources": sources
                        })
                
                elif chunk["type"] == "complete":
                    full_response = chunk["content"]
                    sources = chunk["sources"]
                    
                    # Final JSON response with disclaimer
                    final_response = {
                        "category": query_category.upper(),
                        "response": full_response,
                        "sources": sources,
                        "disclaimer": "Always consult your healthcare provider before making any changes to your medication or treatment plan."
                    }
                    
                    message_placeholder.json(final_response)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": full_response,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection", "administration"],
            "side_effects": ["side effect", "adverse", "reaction", "problem", "issues", "symptoms"],
            "benefits": ["benefit", "advantage", "help", "work", "effect", "weight", "glucose"],
            "storage": ["store", "storage", "keep", "refrigerate", "temperature"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol", "eating"],
            "interactions": ["interaction", "drug", "medication", "combine", "mixing"],
            "cost": ["cost", "price", "insurance", "coverage", "afford"]
        }
        
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        if 'pplx' not in st.secrets:
            st.error('Required PPLX API key not found. Please configure the PPLX API key in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.info("""
        Get accurate, validated information specifically about GLP-1 medications.
        Responses will be provided in JSON format with structured sections.
        
        Note: This assistant provides general information about GLP-1 medications only. 
        Always consult your healthcare provider for medical advice.
        """)
        
        bot = GLP1Bot()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of Ozempic?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.button("Get Answer", key="submit")
            
            if submit_button and user_input:
                # Create a placeholder for the streaming response
                response_placeholder = st.empty()
                
                # Process the query with streaming
                response = bot.process_streaming_query(user_input, response_placeholder)
                
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": response
                    })
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.json({
                        "question": chat['query'],
                        "category": chat['response']['query_category'].upper(),
                        "response": chat['response']['response'],
                        "sources": chat['response']['sources']
                    })
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
