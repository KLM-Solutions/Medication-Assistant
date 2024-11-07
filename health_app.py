import streamlit as st
import requests
import json
import re 
from typing import Dict, Any, Optional, Generator, List

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

4. Always provide source citations which is related to the generated response. Importantly only provide sources for about GLP-1 medications
5. Provide response in a simple manner that is easy to understand at preferably a 11th grade literacy level with reduced pharmaceutical or medical jargon
6. Always Return sources in a hyperlink format

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
You are a medical content validator specialized in GLP-1 medications.
Review and enhance the information about GLP-1 medications only.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

        self.followup_system_prompt = """
You are a medical assistant specialized in GLP-1 medications. Your task is to generate 3-4 relevant follow-up questions based on the user's initial query and the response provided. The follow-up questions should:

1. Be directly related to GLP-1 medications and the context of the initial query
2. Help users better understand their medication journey
3. Cover different aspects that might be relevant to their situation
4. Be clear, concise, and focused on one aspect per question
5. Not repeat information already covered in the initial response

Format your response as a JSON array of strings, each string being a follow-up question.
Example format: ["Question 1?", "Question 2?", "Question 3?"]
"""
def format_sources_as_hyperlinks(self, sources_text: str) -> str:
        """Convert source text into formatted hyperlinks"""
        # Clean any existing HTML tags
        clean_text = re.sub(r'<[^>]+>', '', sources_text)
        
        # Common patterns for URLs in the text
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        
        # Find all URLs in the text
        urls = re.finditer(url_pattern, clean_text)
        formatted_text = clean_text
        
        # Replace each URL with a markdown hyperlink
        for url_match in urls:
            url = url_match.group(0)
            # Extract title if it appears before the URL (common format: "Title: URL")
            title_match = re.search(rf'([^.!?\n]+)(?=\s*{re.escape(url)})', formatted_text)
            title = title_match.group(1).strip() if title_match else url
            
            # Create markdown hyperlink
            hyperlink = f'[{title}]({url})'
            # Replace the URL and its title (if found) with the hyperlink
            if title_match:
                formatted_text = formatted_text.replace(f'{title_match.group(0)} {url}', hyperlink)
            else:
                formatted_text = formatted_text.replace(url, hyperlink)
        
        return formatted_text
 def generate_followup_questions(self, initial_query: str, initial_response: str) -> List[str]:
        """Generate follow-up questions based on the initial query and response"""
        try:
            # Enhanced prompt for better question generation
            enhanced_prompt = f"""Based on this GLP-1 medication query and response:

Query: {initial_query}
Response: {initial_response}

Generate 3-4 relevant follow-up questions that would help the user better understand GLP-1 medications. Questions should:
1. Be directly related to GLP-1 medications
2. Not repeat information already covered
3. Focus on practical aspects of medication use
4. Address potential concerns or common follow-up topics
5. Be clear and concise

Format your response as a valid JSON array of strings. Example:
["What are the long-term effects of this medication?", "How should I store this medication?"]"""

            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.followup_system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            try:
                questions_str = response_data['choices'][0]['message']['content']
                # Handle both JSON array and plain text formats
                try:
                    questions = json.loads(questions_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract questions using regex
                    questions = re.findall(r'"([^"]+)\?"', questions_str)
                    if not questions:
                        # Fallback to splitting by newlines and cleaning up
                        questions = [q.strip().strip('[]"') for q in questions_str.split('\n') 
                                   if '?' in q and len(q.strip()) > 10]
                
                # Validate and clean questions
                valid_questions = []
                for q in questions:
                    q = q.strip()
                    if q and q.endswith('?') and len(q) > 10:
                        valid_questions.append(q)
                
                return valid_questions[:4]  # Limit to 4 questions maximum
                
            except (KeyError, AttributeError):
                return []
                
        except Exception as e:
            st.error(f"Error generating follow-up questions: {str(e)}")
            return []

def stream_pplx_response(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Stream response from PPLX API with sources"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": f"{query}\n\nPlease include sources for the information provided, formatted as 'Title: URL' on separate lines."}
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
            found_sources = False
            sources_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['finish_reason'] is not None:
                                break
                                
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                # Check if we've hit the sources section
                                if "Sources:" in content:
                                    found_sources = True
                                    parts = content.split("Sources:", 1)
                                    if len(parts) > 1:
                                        accumulated_content += parts[0]
                                        sources_text += parts[1]
                                    else:
                                        accumulated_content += parts[0]
                                elif found_sources:
                                    sources_text += content
                                else:
                                    accumulated_content += content
                                
                                yield {
                                    "type": "content",
                                    "data": content,
                                    "accumulated": accumulated_content
                                }
                        except json.JSONDecodeError:
                            continue
            
            # Format sources as hyperlinks
            formatted_sources = self.format_sources_as_hyperlinks(sources_text.strip()) if sources_text.strip() else "No sources provided"
            
            yield {
                "type": "complete",
                "content": accumulated_content.strip(),
                "sources": formatted_sources
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error communicating with PPLX: {str(e)}"
            }

def process_streaming_query(self, user_query: str, placeholder, is_followup: bool = False) -> Dict[str, Any]:
        """Process user query with streaming response"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            query_category = self.categorize_query(user_query)
            full_response = ""
            sources = ""
            message_placeholder = placeholder.empty()
            followup_container = st.container()
            
            for chunk in self.stream_pplx_response(user_query):
                if chunk["type"] == "error":
                    placeholder.error(chunk["message"])
                    return {"status": "error", "message": chunk["message"]}
                
                elif chunk["type"] == "content":
                    full_response = chunk["accumulated"]
                    message_placeholder.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{query_category.upper()}</div><br>
                        <b>Response:</b><br>{full_response}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif chunk["type"] == "complete":
                    full_response = chunk["content"]
                    sources = chunk["sources"]
                    disclaimer = "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
                    
                    formatted_response = f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{query_category.upper()}</div><br>
                        <b>Response:</b><br>{full_response}
                        <div class="sources-section">
                            <b>Sources:</b><br>{sources}
                        </div>
                        <div class="disclaimer">{disclaimer}</div>
                    </div>
                    """
                    
                    message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                    
                    # Generate follow-up questions only for initial queries
                    if not is_followup:
                        with followup_container:
                            followup_questions = self.generate_followup_questions(user_query, full_response)
                            if followup_questions:
                                st.markdown("### Suggested Follow-up Questions")
                                cols = st.columns(2)  # Create two columns for better layout
                                for i, question in enumerate(followup_questions):
                                    col_idx = i % 2
                                    with cols[col_idx]:
                                        # Create a unique key for each button
                                        button_key = f"followup_{user_query[:10]}_{i}"
                                        if st.button(f"🔍 {question}", key=button_key):
                                            st.markdown("### Follow-up Response")
                                            followup_placeholder = st.empty()
                                            followup_response = self.process_streaming_query(
                                                question, 
                                                followup_placeholder, 
                                                is_followup=True
                                            )
                                            # Store follow-up in history
                                            if followup_response["status"] == "success":
                                                if 'followup_history' not in st.session_state:
                                                    st.session_state.followup_history = []
                                                st.session_state.followup_history.append({
                                                    "parent_query": user_query,
                                                    "query": question,
                                                    "response": followup_response
                                                })
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": full_response,
                "sources": sources,
                "disclaimer": disclaimer
            }
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
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
        .stButton button {
            background-color: #f0f8ff;
            border: 1px solid #1976d2;
            color: #1976d2;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            width: 100%;
            text-align: left;
        }
        .stButton button:hover {
            background-color: #e3f2fd;
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
        
        # Initialize session state for storing conversation history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'followup_history' not in st.session_state:
            st.session_state.followup_history = []

        bot = GLP1Bot()
        
        # Main input container
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
                # Display user question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Your Question:</b><br>{user_input}
                </div>
                """, unsafe_allow_html=True)
                
                # Create a placeholder for the streaming response
                response_placeholder = st.empty()
                
                # Process the query with streaming
                response = bot.process_streaming_query(user_input, response_placeholder)
                
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": response
                    })

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            
            # Display previous conversations in reverse order (newest first)
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    # Display user question
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>Your Question:</b><br>{chat['query']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display bot response
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>Response:</b><br>{chat['response']['response']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display follow-up questions if they exist in history
                    followups = [f for f in st.session_state.followup_history if f["parent_query"] == chat["query"]]
                    if followups:
                        st.markdown("#### Related Follow-up Questions")
                        for followup in followups:
                            with st.expander(f"Follow-up: {followup['query'][:50]}..."):
                                st.markdown(f"""
                                <div class="chat-message user-message">
                                    <b>Follow-up Question:</b><br>{followup['query']}
                                </div>
                                <div class="chat-message bot-message">
                                    <div class="category-tag">{followup['response']['query_category'].upper()}</div><br>
                                    <b>Response:</b><br>{followup['response']['response']}
                                </div>
                                """, unsafe_allow_html=True)

        # Footer with additional information
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <small>
        💡 Tips:
        - Ask specific questions about GLP-1 medications
        - Use follow-up questions to learn more
        - Review previous conversations in the history section
        - Always consult with your healthcare provider for medical advice
        </small>
        </div>
        """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
