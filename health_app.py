import streamlit as st
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class Citation:
    """Data class for holding citation information"""
    title: str
    url: str
    publisher: Optional[str] = None
    year: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None

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

4. For each claim or statement about GLP-1 medications:
   - Include a citation using [Source X] notation for general references
   - Include a citation using [Citation X] notation for specific academic papers or studies
   After your response, list all sources and citations with their full details.

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
Ensure all medical claims are properly cited using reputable sources.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

    def extract_references(self, content: str) -> Tuple[str, List[Dict[str, str]], List[Citation]]:
        """Extract both source references and citations from content"""
        import re
        
        # Extract source references
        source_pattern = r'\[Source (\d+)\]'
        source_refs = set(re.findall(source_pattern, content))
        
        # Extract citation references
        citation_pattern = r'\[Citation (\d+)\]'
        citation_refs = set(re.findall(citation_pattern, content))
        
        # Extract sources section
        sources_section_pattern = r'Sources:(.*?)(?=(?:Citations:|$))'
        sources_match = re.search(sources_section_pattern, content, re.DOTALL)
        
        # Extract citations section
        citations_section_pattern = r'Citations:(.*?)(?=\n\n|$)'
        citations_match = re.search(citations_section_pattern, content, re.DOTALL)
        
        sources_list = []
        citations_list = []
        clean_content = content
        
        # Process sources
        if sources_match:
            sources_text = sources_match.group(1)
            clean_content = clean_content.replace(sources_match.group(0), '')
            
            source_lines = sources_text.strip().split('\n')
            for line in source_lines:
                if line.strip():
                    source_line_pattern = r'(\d+)\.(.*)'
                    source_match = re.match(source_line_pattern, line.strip())
                    if source_match:
                        source_num, source_details = source_match.groups()
                        if source_num in source_refs:
                            sources_list.append({
                                "number": source_num,
                                "details": source_details.strip(),
                                "reference_count": len(re.findall(f'\\[Source {source_num}\\]', content))
                            })
        
        # Process citations
        if citations_match:
            citations_text = citations_match.group(1)
            clean_content = clean_content.replace(citations_match.group(0), '')
            
            citation_lines = citations_text.strip().split('\n')
            current_citation = {}
            
            for line in citation_lines:
                if line.strip():
                    if line.startswith('[Citation'):
                        if current_citation:
                            citations_list.append(Citation(**current_citation))
                            current_citation = {}
                        
                        citation_num = re.search(r'\[Citation (\d+)\]', line)
                        if citation_num and citation_num.group(1) in citation_refs:
                            current_citation['number'] = citation_num.group(1)
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'title':
                            current_citation['title'] = value
                        elif key == 'url':
                            current_citation['url'] = value
                        elif key == 'publisher':
                            current_citation['publisher'] = value
                        elif key == 'year':
                            current_citation['year'] = value
                        elif key == 'authors':
                            current_citation['authors'] = [a.strip() for a in value.split(',')]
                        elif key == 'doi':
                            current_citation['doi'] = value
            
            if current_citation:
                citations_list.append(Citation(**current_citation))
        
        return clean_content.strip(), sources_list, citations_list

    def get_pplx_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive response from PPLX API with sources and citations"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
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
            response_data = response.json()
            
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract and process sources and citations
            clean_content, sources, citations = self.extract_references(content)
            
            return {
                "content": clean_content,
                "sources": sources,
                "citations": citations
            }
            
        except Exception as e:
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def format_response(self, response_data: Dict[str, Any]) -> str:
        """Format the response with sources, citations and safety disclaimer"""
        if not response_data:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        content = response_data["content"]
        sources = response_data.get("sources", [])
        citations = response_data.get("citations", [])
        
        formatted_content = f"{content}\n\n"
        
        if sources:
            formatted_content += '<div class="sources-section">\n<h4>General Sources:</h4>\n'
            for source in sources:
                formatted_content += f"""
                <div class="source-item">
                    <div class="source-number">Source {source['number']}</div>
                    <div class="source-details">{source['details']}</div>
                    <div class="reference-count">Referenced {source['reference_count']} time{'s' if source['reference_count'] > 1 else ''}</div>
                </div>
                """
            formatted_content += '</div>\n'
        
        if citations:
            formatted_content += '<div class="citations-section">\n<h4>Academic Citations:</h4>\n'
            for citation in citations:
                formatted_content += f"""
                <div class="citation-item">
                    <div class="citation-title">{citation.title}</div>
                    <div class="citation-url">{citation.url}</div>
                    """
                if citation.publisher:
                    formatted_content += f'<div class="citation-publisher">{citation.publisher}</div>'
                if citation.year:
                    formatted_content += f'<div class="citation-year">Year: {citation.year}</div>'
                if citation.authors:
                    formatted_content += f'<div class="citation-authors">Authors: {", ".join(citation.authors)}</div>'
                if citation.doi:
                    formatted_content += f'<div class="citation-doi">DOI: {citation.doi}</div>'
                formatted_content += '</div>'
            formatted_content += '</div>\n'
        
        formatted_content += '\n<div class="disclaimer">Disclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan.</div>'
        
        return formatted_content

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through PPLX with GLP-1 validation, sources, and citations"""
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
                "sources": response_data.get("sources", []),
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
        /* Previous styles remain the same */
        
        .sources-section, .citations-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin-top: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .sources-section {
            border-left: 4px solid #673ab7;
        }
        
        .citations-section {
            border-left: 4px solid #2196f3;
        }
        
        .sources-section h4, .citations-section h4 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }
        
        .source-item, .citation-item {
            margin: 1rem 0;
            padding: 0.8rem;
            background-color: white;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
        }
        
        .source-item:hover, .citation-item:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .source-number {
            color: #673ab7;
            font-weight: bold;
            margin-bottom: 0.3rem;
            font-size: 1rem;
        }
        
        .citation-title {
            color: #2196f3;
            font-weight: bold;
            margin-bottom: 0.3rem;
            font-size: 1rem;
        }
        
        .source-details, .citation-url, .citation-publisher,
        .citation-year, .citation-authors, .citation-doi {
            color: #333;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        
        .reference-count {
            color: #666;
            font-size: 0.8rem;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)

# The main() function remains the same
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
