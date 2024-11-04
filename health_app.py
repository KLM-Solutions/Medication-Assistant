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

4. Always provide source citiations which is related to the generated response. Importantly only provide sources for about GLP-1 medications

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
You are a medical content validator specialized in GLP-1 medications.
Review and enhance the information about GLP-1 medications only.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

    def stream_pplx_response(self, query: str) -> Generator[Dict[str, Any], None, None]:
     """Stream response from PPLX API with explicit citation handling"""
     try:
        payload = {
            "model": self.pplx_model,
            "messages": [
                {"role": "system", "content": self.pplx_system_prompt},
                {"role": "user", "content": f"{query}\n\nPlease include citations and references for all factual claims."}
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
            "stream": True,
            "extra_params": {
                "return_citations": True,  # Request citations in response
                "citation_format": "academic"  # Request academic-style citations
            }
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=self.pplx_headers,
            json=payload,
            stream=True
        )
        
        response.raise_for_status()
        accumulated_content = ""
        accumulated_citations = []
        
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
                            
                        # Extract content and any citations from the chunk
                        content = chunk['choices'][0]['delta'].get('content', '')
                        citations = self.extract_citations(content)
                        
                        if content:
                            accumulated_content += content
                            accumulated_citations.extend(citations)
                            
                            yield {
                                "type": "content",
                                "data": content,
                                "accumulated": accumulated_content,
                                "citations": accumulated_citations
                            }
                    except json.JSONDecodeError:
                        continue
        
        # Format final content and citations
        formatted_content = self.format_content_with_citations(accumulated_content)
        formatted_citations = self.format_citations(accumulated_citations)
        
        yield {
            "type": "complete",
            "content": formatted_content,
            "citations": formatted_citations
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Error communicating with PPLX: {str(e)}"
        }

def extract_citations(self, text: str) -> list[dict]:
    """Extract citations from text using regex patterns"""
    citations = []
    
    # Pattern for inline citations [n]
    inline_pattern = r'\[(\d+)\]'
    # Pattern for reference entries
    ref_pattern = r'\[(\d+)\]\s+(.*?)(?=\[\d+\]|\Z)'
    
    # Find all inline citations
    inline_citations = re.finditer(inline_pattern, text)
    for match in inline_citations:
        citations.append({
            "citation_number": match.group(1),
            "position": match.start()
        })
    
    # Find all reference entries
    references = re.finditer(ref_pattern, text, re.DOTALL)
    for match in references:
        citations.append({
            "citation_number": match.group(1),
            "reference": match.group(2).strip()
        })
    
    return citations
    def format_content_with_citations(self, content: str) -> str:
    """Format content with properly styled citation markers"""
    # Replace plain citation markers with styled ones
    formatted_content = re.sub(
        r'\[(\d+)\]',
        r'<sup class="citation-marker" data-citation="\1">[\1]</sup>',
        content
    )
    return formatted_content

   def format_citations(self, citations: list[dict]) -> str:
    """Format citations into HTML with proper styling"""
    # Group citations by number and combine inline/reference information
    citation_dict = {}
    for citation in citations:
        num = citation.get("citation_number")
        if num:
            if num not in citation_dict:
                citation_dict[num] = {}
            if "reference" in citation:
                citation_dict[num]["reference"] = citation["reference"]
            if "position" in citation:
                citation_dict[num]["position"] = citation["position"]
    
    # Generate HTML for citations
    html_citations = []
    for num in sorted(citation_dict.keys(), key=int):
        citation = citation_dict[num]
        if "reference" in citation:
            html_citations.append(f"""
                <div class="citation-entry" id="citation-{num}">
                    <span class="citation-number">[{num}]</span>
                    <span class="citation-text">{citation['reference']}</span>
                </div>
            """)
    
    return "\n".join(html_citations) if html_citations else "No citations provided"

   def process_streaming_query(self, user_query: str, placeholder) -> Dict[str, Any]:
    """Process user query with streaming response and enhanced citations"""
    try:
        if not user_query.strip():
            return {
                "status": "error",
                "message": "Please enter a valid question."
            }
        
        query_category = self.categorize_query(user_query)
        full_response = ""
        citations = ""
        
        message_placeholder = placeholder.empty()
        
        for chunk in self.stream_pplx_response(user_query):
            if chunk["type"] == "error":
                placeholder.error(chunk["message"])
                return {"status": "error", "message": chunk["message"]}
            
            elif chunk["type"] == "content":
                full_response = chunk["accumulated"]
                citations = self.format_citations(chunk.get("citations", []))
                message_placeholder.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="category-tag">{query_category.upper()}</div>
                    <div class="response-content">{self.format_content_with_citations(full_response)}</div>
                    <div class="citations-section">
                        <h4>References</h4>
                        {citations}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            elif chunk["type"] == "complete":
                full_response = chunk["content"]
                citations = chunk["citations"]
                
                disclaimer = "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
                message_placeholder.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="category-tag">{query_category.upper()}</div>
                    <div class="response-content">{self.format_content_with_citations(full_response)}{disclaimer}</div>
                    <div class="citations-section">
                        <h4>References</h4>
                        {citations}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        return {
            "status": "success",
            "query_category": query_category,
            "original_query": user_query,
            "response": full_response,
            "citations": citations
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
        
        set_page_style()  # Your existing style function remains the same
        
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
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>Your Question:</b><br>{chat['query']}
                    </div>
                    <div class="chat-message bot-message">
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>Response:</b><br>{chat['response']['response']}
                        <div class="sources-section">
                            <b>Sources:</b><br>{chat['response']['sources']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
