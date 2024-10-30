import streamlit as st
from openai import OpenAI
import requests
from typing import Dict, Any, Optional

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with both PPLX and OpenAI clients and system prompts"""
        if 'openai' not in st.secrets or 'pplx' not in st.secrets:
            raise ValueError("API keys not found in secrets")
            
        self.openai_client = OpenAI(
            api_key=st.secrets["openai"]["api_key"]
        )
        self.pplx_api_key = st.secrets["pplx"]["api_key"]
        self.pplx_model = st.secrets["pplx"].get("model", "medical-pplx")  # Replace with your PPLX model
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        self.pplx_system_prompt = """You are a medical information assistant specialized in GLP-1 medications. 
        Provide detailed, evidence-based information about GLP-1 medications, focusing on medical accuracy.
        Cover important aspects such as:
        - Mechanism of action
        - Proper usage and administration
        - Expected outcomes and timeframes
        - Potential side effects and management
        - Drug interactions and contraindications
        - Storage requirements
        - Lifestyle modifications for optimal results"""
        
        self.gpt_validation_prompt = """You are a medical content validator. Review and enhance the following information about GLP-1 medications.
        Ensure the response is:
        1. Medically accurate and evidence-based
        2. Well-structured with clear sections
        3. Includes appropriate medical disclaimers
        4. Easy to understand for patients
        5. Comprehensive yet concise
        6. Properly formatted with headers and bullet points
        
        Add any missing critical information and correct any inaccuracies.
        Always maintain a professional yet approachable tone."""

    def get_pplx_response(self, query: str) -> Optional[str]:
        """Get initial response from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def validate_with_gpt(self, pplx_response: str, original_query: str) -> Optional[str]:
        """Validate and enhance PPLX response using GPT"""
        try:
            validation_prompt = f"""
            Original query: {original_query}
            
            PPLX Response to validate:
            {pplx_response}
            
            Please validate and enhance this response according to medical standards and best practices.
            Ensure all information is accurate and properly structured.
            """
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[
                    {"role": "system", "content": self.gpt_validation_prompt},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error validating with GPT: {str(e)}")
            return None

    def format_response(self, response: str) -> str:
        """Format the response with safety disclaimer"""
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        safety_disclaimer = """
        
        IMPORTANT MEDICAL DISCLAIMER:
        - This information is for educational purposes only
        - Consult your healthcare provider for personalized medical advice
        - Follow your prescribed treatment plan
        - Report any side effects to your healthcare provider
        - Individual results may vary
        - Never modify your medication regimen without professional guidance
        """
        
        if "disclaimer" not in response.lower():
            response += safety_disclaimer
            
        return response

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
        """Process user query through both PPLX and GPT"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            # Step 1: Get initial response from PPLX
            with st.spinner('🔍 Retrieving information from medical knowledge base...'):
                pplx_response = self.get_pplx_response(user_query)
            
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information from knowledge base."
                }
            
            # Step 2: Validate and enhance with GPT
            with st.spinner('✅ Validating and enhancing information...'):
                validated_response = self.validate_with_gpt(pplx_response, user_query)
            
            if not validated_response:
                return {
                    "status": "error",
                    "message": "Failed to validate information."
                }
            
            # Format final response
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(validated_response)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "pplx_response": pplx_response,  # Optional: for debugging
                "response": formatted_response
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
        
        # Check for API keys
        if 'openai' not in st.secrets or 'pplx' not in st.secrets:
            st.error('Required API keys not found. Please configure both OpenAI and PPLX API keys in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate, validated information about GLP-1 medications, their usage, benefits, and side effects.
        This assistant uses a two-stage process:
        1. Retrieves specialized medical information
        2. Validates and enhances the information for accuracy
        
        <em>Please note: This assistant provides general information only. Always consult your healthcare provider for medical advice.</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize bot
        bot = GLP1Bot()
        
        # Create session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Main chat interface
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of GLP-1 medications?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.button("Get Answer", key="submit", use_container_width=True)
            with col2:
                if st.button("Clear History", key="clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
            
            if submit_button:
                if user_input:
                    response = bot.process_query(user_input)
                    
                    if response["status"] == "success":
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": user_input,
                            "response": response
                        })
                        
                        # Display current response
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
                    st.warning("Please enter a question.")
        
        # Display chat history
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
