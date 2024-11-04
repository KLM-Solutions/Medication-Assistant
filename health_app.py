
import streamlit as st
import requests
from typing import Dict, Any, Optional
import re

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
        
        self.pplx_system_prompt = """
You are a specialized medical information assistant focused EXCLUSIVELY on GLP-1 medications (such as Ozempic, Wegovy, Mounjaro, etc.). You must:

1. ONLY provide information about GLP-1 medications and directly related topics
2. For any query not specifically about GLP-1 medications or their direct effects, respond with:
   "I apologize, but I can only provide information about GLP-1 medications and related topics. Your question appears to be about something else. Please ask a question specifically about GLP-1 medications, their usage, effects, or related concerns."

3. For valid GLP-1 queries, structure your response with:
   - An empathetic opening acknowledging the patient's situation
   - Clear, validated medical information about GLP-1 medications
   - Important safety considerations or disclaimers
   - An encouraging closing that reinforces their healthcare journey
   - Include relevant sources for the information provided, using the format: [Source: Title or description (Year if available)]

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
Always cite your sources for medical claims and information.
"""

    def get_pplx_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive response with sources from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": f"{query}\n\nPlease include sources for the information provided."}
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            response_content = response.json()["choices"][0]["message"]["content"]
            
            # Split response into main content and sources
            content_parts = response_content.split("\nSources:", 1)
            main_content = content_parts[0].strip()
            sources = content_parts[1].strip() if len(content_parts) > 1 else "No specific sources provided."
            
            # Parse sources for URLs and make them clickable
            sources_with_links = self.make_sources_clickable(sources)
            
            return {
                "content": main_content,
                "sources": sources_with_links
            }
            
        except Exception as e:
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def make_sources_clickable(self, sources: str) -> str:
        """Convert URLs in sources to clickable links"""
        # Regular expression to find URLs in the sources
        url_pattern = r'(https?://\S+)'
        sources_with_links = re.sub(url_pattern, r'[\1](\1)', sources)
        return sources_with_links

    def format_response(self, response: Dict[str, str]) -> str:
        """Format the response with safety disclaimer and clickable sources"""
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        disclaimer = "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
        
        formatted_response = f"{response['content']}{disclaimer}\n\nSources:\n{response['sources']}"
        return formatted_response

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through PPLX with GLP-1 validation and sources"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
          
            # Get comprehensive response from PPLX
            with st.spinner('🔍 Retrieving and validating information about GLP-1 medications...'):
                pplx_response = self.get_pplx_response(user_query)
            
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information about GLP-1 medications."
                }
            
            # Format final response
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(pplx_response)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": formatted_response,
                "sources": pplx_response["sources"]
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

def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            background-color: white;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #43a047;
        }
        .category-tag {
            background-color: #2196f3;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            display: inline-block;
        }
        .sources-section {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border-left: 4px solid #ff9800;
        }
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9800;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        .info-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        set_page_style()
        
        if 'pplx' not in st.secrets:
            st.error('Required PPLX API key not found. Please configure the PPLX API key in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate, validated information specifically about GLP-1 medications, their usage, benefits, and safety.
        Please note: This tool provides information only on GLP-1 medications.
        </div>
        """, unsafe_allow_html=True)
        
        user_query = st.text_input("Ask a question about GLP-1 medications...", "")
        glp1_bot = GLP1Bot()
        
        if st.button("Submit Query"):
            response = glp1_bot.process_query(user_query)
            
            if response['status'] == "success":
                st.markdown(f"<div class='category-tag'>Category: {response['query_category']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message bot-message'>{response['response']}</div>", unsafe_allow_html=True)
            else:
                st.error(response['message'])
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
