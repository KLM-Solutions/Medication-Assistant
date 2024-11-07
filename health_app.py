import streamlit as st
import requests
import json
import re 
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
   - Generate 3 relevant follow-up questions based on the context
   
4. Always provide source citations which is related to the generated response. Importantly only provide sources for about GLP-1 medications
5. Provide response in a simple manner that is easy to understand at preferably a 11th grade literacy level with reduced pharmaceutical or medical jargon
6. Always Return sources in a hyperlink format
7. After the sources, provide a section titled "Follow-up Questions:" with 3 numbered, relevant questions that users might want to ask next

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
You are a medical content validator specialized in GLP-1 medications.
Review and enhance the information about GLP-1 medications only.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

    def format_sources_as_hyperlinks(self, sources_text: str) -> str:
        """Convert source text into formatted hyperlinks"""
        # Existing implementation remains the same
        clean_text = re.sub(r'<[^>]+>', '', sources_text)
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.finditer(url_pattern, clean_text)
        formatted_text = clean_text
        
        for url_match in urls:
            url = url_match.group(0)
            title_match = re.search(rf'([^.!?\n]+)(?=\s*{re.escape(url)})', formatted_text)
            title = title_match.group(1).strip() if title_match else url
            hyperlink = f'[{title}]({url})'
            if title_match:
                formatted_text = formatted_text.replace(f'{title_match.group(0)} {url}', hyperlink)
            else:
                formatted_text = formatted_text.replace(url, hyperlink)
        
        return formatted_text

    def extract_followup_questions(self, content: str) -> list:
        """Extract follow-up questions from the response content"""
        questions = []
        if "Follow-up Questions:" in content:
            questions_section = content.split("Follow-up Questions:")[-1]
            # Extract numbered questions using regex
            question_matches = re.findall(r'\d+\.\s*([^\n]+)', questions_section)
            questions = [q.strip() for q in question_matches]
        return questions

    def stream_pplx_response(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Stream response from PPLX API with sources and follow-up questions"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": f"{query}\n\nPlease include sources for the information provided, formatted as 'Title: URL' on separate lines, and suggest 3 relevant follow-up questions."}
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
            
            formatted_sources = self.format_sources_as_hyperlinks(sources_text.strip()) if sources_text.strip() else "No sources provided"
            followup_questions = self.extract_followup_questions(accumulated_content)
            
            yield {
                "type": "complete",
                "content": accumulated_content.strip(),
                "sources": formatted_sources,
                "followup_questions": followup_questions
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error communicating with PPLX: {str(e)}"
            }

    def process_streaming_query(self, user_query: str, placeholder) -> Dict[str, Any]:
        """Process user query with streaming response"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            query_category = self.categorize_query(user_query)
            full_response = ""
            followup_questions = []
            
            message_placeholder = placeholder.empty()
            
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
                    followup_questions = chunk["followup_questions"]
                    disclaimer = "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
                    
                    formatted_response = f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{query_category.upper()}</div><br>
                        <b>Response:</b><br>{full_response}{disclaimer}<br><br>
                        <div class="sources-section">
                            <b>Sources:</b><br>{chunk["sources"]}
                        </div>
                    </div>
                    """
                    
                    message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": f"{full_response}{disclaimer}",
                "sources": chunk.get("sources", ""),
                "followup_questions": followup_questions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    # Rest of the class implementation remains the same...

def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        /* Existing styles remain the same */
        .followup-question {
            background-color: #e8f5e9;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .followup-question:hover {
            background-color: #c8e6c9;
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
            
            if submit_button and user_input:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Your Question:</b><br>{user_input}
                </div>
                """, unsafe_allow_html=True)
                
                response_placeholder = st.empty()
                response = bot.process_streaming_query(user_input, response_placeholder)
                
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": response
                    })
                    
                    # Display follow-up questions if available
                    if response.get("followup_questions"):
                        st.markdown("### Follow-up Questions")
                        for question in response["followup_questions"]:
                            if st.button(question, key=f"followup_{hash(question)}"):
                                followup_response = bot.process_streaming_query(question, st.empty())
                                if followup_response["status"] == "success":
                                    st.session_state.chat_history.append({
                                        "query": question,
                                        "response": followup_response
                                    })
        
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
