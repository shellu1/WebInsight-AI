import os
import streamlit as st
import requests
import json
import numpy as np
from pypdf import PdfReader
import faiss
from dotenv import load_dotenv
import logging
import re
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define a simple class for direct API calls without using the OpenAI client
class DirectOpenAIAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is not provided")
        self.base_url = "https://api.openai.com/v1"
        
    def create_embedding(self, text, model="text-embedding-3-small"):
        """Create embedding using direct API call."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "input": text,
            "model": model
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return a zero vector as fallback
            dimension = 1536 if model == "text-embedding-3-small" else 3072
            return [0.0] * dimension
    
    def create_chat_completion(self, messages, model="gpt-3.5-turbo", temperature=0.3):
        """Create chat completion using direct API call."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error creating chat completion: {e}")
            return "I'm sorry, I encountered an error processing your request."

class SimpleRAGChatbot:
    def __init__(self, pdf_path):
        """Initialize the RAG chatbot with the website PDF content."""
        self.pdf_path = pdf_path
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.index = None
        self.chat_history = []
        
        # Initialize the API client
        self.api = DirectOpenAIAPI()
        
        # Process PDF and build the retrieval system
        self.process_document()
        self.build_index()
        
        logger.info("RAG chatbot initialized successfully")

    def process_document(self):
        """Load and process the PDF document."""
        # Load the PDF
        pdf_reader = PdfReader(self.pdf_path)
        
        # Extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                self.documents.append({
                    'content': text,
                    'source': f"{self.pdf_path}, page {i+1}"
                })
        
        # Split into chunks
        for doc in self.documents:
            # Simple text splitting by paragraphs
            paragraphs = re.split(r'\n\s*\n', doc['content'])
            for para in paragraphs:
                if len(para.strip()) > 40:  # Minimum length
                    # Split into chunks of ~1000 characters
                    if len(para) > 1000:
                        words = para.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk) + len(word) < 1000:
                                current_chunk += word + " "
                            else:
                                self.chunks.append({
                                    'content': current_chunk.strip(),
                                    'source': doc['source'],
                                    'embedding': None
                                })
                                current_chunk = word + " "
                        if current_chunk:
                            self.chunks.append({
                                'content': current_chunk.strip(),
                                'source': doc['source'],
                                'embedding': None
                            })
                    else:
                        self.chunks.append({
                            'content': para.strip(),
                            'source': doc['source'],
                            'embedding': None
                        })
        
        logger.info(f"Document processed: {len(self.documents)} pages, {len(self.chunks)} chunks")

    def build_index(self):
        """Build the vector index for retrieval."""
        # Get embeddings for each chunk
        for i, chunk in enumerate(self.chunks):
            embedding = self.api.create_embedding(chunk['content'])
            self.chunks[i]['embedding'] = embedding
            self.embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Get the dimension from the first embedding
        dimension = len(self.embeddings[0])
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        logger.info(f"Built index with {len(self.embeddings)} vectors of dimension {dimension}")

    def retrieve_relevant_chunks(self, query, k=5):
        """Retrieve the most relevant chunks for a query."""
        # Get query embedding
        query_embedding = self.api.create_embedding(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search in the index
        distances, indices = self.index.search(query_array, k)
        
        # Get the chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Ensure valid index
                results.append({
                    'content': self.chunks[idx]['content'],
                    'source': self.chunks[idx]['source'],
                    'distance': distances[0][i]
                })
        
        return results

    def ask_question(self, question):
        """Ask a question to the chatbot."""
        # Get relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question)
        
        # Format context
        context = "\n\n".join([
            f"Source: {chunk['source']}\nContent: {chunk['content']}" 
            for chunk in relevant_chunks
        ])
        
        # Format chat history
        history_text = ""
        if self.chat_history:
            history_text = "Chat History:\n" + "\n".join([
                f"User: {q}\nAssistant: {a}" for q, a in self.chat_history
            ])
        
        # Prepare the prompt
        system_message = "You are a helpful assistant that answers questions about website content."
        prompt = f"""
        Use the following retrieved information to answer the user's question.
        If you don't know the answer, say that you don't know.

        {history_text}

        Retrieved information:
        {context}

        User's question: {question}
        """
        
        # Make API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        answer = self.api.create_chat_completion(messages)
        
        # Update chat history
        self.chat_history.append((question, answer))
        
        return answer

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def main():
    st.set_page_config(
        page_title="SiteWhiz",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("SiteWhiz")
    st.subheader("Powered by RAG and OpenAI")
    
    # Session state initialization
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for uploading PDF
    with st.sidebar:
        st.header("Configuration")
        pdf_file = st.file_uploader("Upload Website PDF", type=["pdf"])
        
        if pdf_file and not st.session_state.chatbot:
            with st.spinner("Processing PDF and building knowledge base..."):
                # Save uploaded file temporarily
                temp_pdf_path = save_uploaded_file(pdf_file)
                
                # Initialize chatbot
                try:
                    st.session_state.chatbot = SimpleRAGChatbot(temp_pdf_path)
                    st.success("Knowledge base created! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error initializing chatbot: {str(e)}")
                    logger.error(f"Error initializing chatbot: {e}", exc_info=True)
        
        # About section
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This chatbot uses RAG (Retrieval Augmented Generation) to answer questions about website content.
        
        It first processes the PDF to extract text, then builds a vector database using OpenAI embeddings.
        
        When you ask a question, it retrieves the most relevant information and generates a coherent answer using OpenAI's language model.
        """)
    
    # Main area for chat interface
    if st.session_state.chatbot:
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            message = st.chat_message("user")
            message.write(question)
            
            message = st.chat_message("assistant")
            message.write(answer)
        
        # Input for new question
        user_question = st.chat_input("Ask a question about the website content...")
        
        if user_question:
            st.chat_message("user").write(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chatbot.ask_question(user_question)
                        st.write(response)
                        st.session_state.chat_history.append((user_question, response))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Error in chat: {e}", exc_info=True)
    else:
        st.info("Please upload a PDF to get started.")

if __name__ == "__main__":
    main()
