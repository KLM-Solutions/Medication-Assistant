import streamlit as st
import requests
from typing import Dict, Any, Optional

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
   - Clear, validated medical information about GLP-1 medications with citations
   - Important safety considerations or disclaimers
   - An encouraging closing that reinforces their healthcare journey

4. For each claim or statement about GLP-1 medications, include a citation to a reputable medical source.Always provide source citiations which is related to the generated response. Importantly only provide sources for about GLP-1 medications


Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
Ensure all medical claims are properly cited using reputable sources.
You are a medical content validator specialized in GLP-1 medications.
Review and enhance the information about GLP-1 medications only.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

    def get_pplx_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive response from PPLX API with citations"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "return_citations": True
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data["choices"][0]["message"]["content"]
            citations = response_data["choices"][0]["message"].get("citations", [])
            
            return {
                "content": content,
                "citations": citations
            }
            
        except Exception as e:
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def format_response(self, response_data: Dict[str, Any]) -> str:
        """Format the response with citations and safety disclaimer"""
        if not response_data:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        content = response_data["content"]
        citations = response_data["citations"]
        
        formatted_content = f"{content}\n\n"
        
        if citations:
            formatted_content += '<div class="citations-section">\n<h4>Sources:</h4>\n'
            for i, citation in enumerate(citations, 1):
                formatted_content += f"""
                <div class="citation-item">
                    <div class="citation-title">{i}. {citation['title']}</div>
                    <div class="citation-url">{citation['url']}</div>
                    {f'<div class="citation-publisher">{citation.get("publisher", "")}</div>' if citation.get("publisher") else ""}
                </div>
                """
            formatted_content += '</div>\n'
        
        formatted_content += '\n<div class="disclaimer">Disclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan.</div>'
        
        return formatted_content

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

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through PPLX with GLP-1 validation and citations"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
          
            with st.spinner('🔍 Retrieving and validating information about GLP-1 medications...'):
                response_data = self.get_pplx_response(user_query)
            
            if not response_data:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information about GLP-1 medications."
                }
            
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(response_data)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": formatted_response,
                "citations": response_data.get("citations", [])
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

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
        .stAlert {
            background-color: #ff5252;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .citations-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin-top: 1.5rem;
            border-left: 4px solid #9c27b0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .citations-section h4 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }
        .citation-item {
            margin: 1rem 0;
            padding: 0.8rem;
            background-color: white;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
        }
        .citation-item:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .citation-title {
            color: #1976d2;
            font-weight: bold;
            margin-bottom: 0.3rem;
            font-size: 1rem;
        }
        .citation-url {
            color: #666;
            font-size: 0.9rem;
            word-break: break-all;
            margin-bottom: 0.3rem;
        }
        .citation-publisher {
            color: #43a047;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
            border-left: 4px solid #ff9800;
            font-size: 0.9rem;
        }
        .info-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .processing-status {
            color: #1976d2;
            font-style: italic;
            margin: 0.5rem 0;
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
        Get accurate, validated information specifically about GLP-1 medications, their usage, benefits, and side effects.
        Our assistant specializes exclusively in GLP-1 medications and related topics.
        
        <em>Please note: This assistant provides general information about GLP-1 medications only. Always consult your healthcare provider for medical advice.</em>
        </div>
        """, unsafe_allow_html=True)
        
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
            
            if submit_button:
                if user_input:
                    response = bot.process_query(user_input)
                    
                    if response["status"] == "success":
                        st.session_state.chat_history.append({
                            "query": user_input,
                            "response": response
                        })
                        
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>Your Question:</b><br>{user_input}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                            <div class="category-tag">{response["query_category"].upper()}</div><br>
                            <b>Response:</b><br>{response["response"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(response["message"])
                else:
                    st.warning("Please enter a question about GLP-1 medications.")
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>Your Question:</b><br>{chat['query']}
                    </div>
                    <div class="chat-message bot-message">
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>Response:</b><br>{chat['response']['response']}
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
