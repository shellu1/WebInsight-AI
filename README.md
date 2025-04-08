# WebInsight AI
A Retrieval Augmented Generation (RAG) chatbot that answers questions about website content using PDFs generated from web scraping.
# Overview
This project uses advanced AI techniques to create an intelligent assistant that can answer questions about website content. The system first scrapes website data to create a PDF, then uses that PDF as a knowledge base for an AI-powered question-answering system.
# Key Features
    •	Web Scraping: Extract structured content from any website
    •	Vector Search: Implement semantic search using FAISS and OpenAI embeddings
    •	RAG Architecture: Combine retrieval and generation for contextually aware responses
    •	Direct API Integration: Work with OpenAI's API without intermediary libraries
    •	User-friendly Interface: Clean Streamlit web interface for easy interaction
# Technology Stack
    •	Python: Core language for the entire application
    •	Beautiful Soup/Requests: Web scraping and content extraction
    •	PyPDF: PDF processing and text extraction
    •	FAISS: Efficient similarity search and clustering
    •	OpenAI API: Text embeddings and completion generation
    •	Streamlit: Web interface for user interaction
    •	Numpy: Numerical operations for vector processing
# Setup and Installation
1.	Clone the repository
2.	Create a virtual environment: 
python -m venv venv
venv\Scripts\activate
3.	Install dependencies: 
pip install -r requirements.txt
4.	Set up your API key in a .env file: 
OPENAI_API_KEY=your_api_key_here
# Usage
Step 1: Scrape Website Content
python scrap.py
This will generate a PDF with the website content.

Step 2: Run the RAG Chatbot
streamlit run simple_rag.py
Upload the generated PDF and start asking questions about the website content.
# How It Works
1.	Web Scraping: The system extracts structured content from websites, preserving text, images, and the document hierarchy.
2.	PDF Generation: The scraped content is organized into a comprehensive PDF for easy storage and processing.
3.	Text Processing: When loaded into the chatbot, the PDF is processed into smaller, semantically meaningful chunks.
4.	Vector Embedding: Each chunk is converted into a numerical representation using OpenAI's embedding models.
5.	Similarity Search: When a question is asked, the system finds the most relevant chunks using vector similarity.
6.	Response Generation: The system uses OpenAI's models to generate a coherent, accurate response based on the retrieved information.
# Future Improvements
•	Improved chunking strategies

•	PDF highlighting for source references

•	Local embedding model options
