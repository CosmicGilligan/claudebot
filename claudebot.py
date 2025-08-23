import anthropic
import streamlit as st
import os
import pandas as pd
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from typing import List, Tuple, Optional
import logging
from datetime import datetime
import time
import random
import base64
import hashlib
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import docx

def check_site_access():
    """Check site-wide access password"""
    if 'site_authenticated' not in st.session_state:
        st.session_state.site_authenticated = False
    
    if not st.session_state.site_authenticated:
        st.title("Access Required")
        st.write("This site is restricted to Professor Cox's history students.")
        
        password = st.text_input("Enter your section's access code:", type="password", key="site_password")
        
        if st.button("Access Site"):
            valid_passwords = [
                "History101A",  # Section 1
                "History101B",  # Section 2  
                "History101C"   # Section 3
            ]
            
            if password in valid_passwords:
                st.session_state.site_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect access code")
        st.stop()

def select_course():
    if 'selected_course' not in st.session_state:
        st.title("Select Your Course")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("World History", use_container_width=True):
                st.session_state.selected_course = "World History"
                st.session_state.data_path = "text/WorldHistory"
                st.rerun()
        
        with col2:
            if st.button("US History", use_container_width=True):
                st.session_state.selected_course = "US History"
                st.session_state.data_path = "text/USHistory"
                st.rerun()
        st.stop()

def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_admin_password():
    """Check if admin password is correct"""
    ADMIN_PASSWORD_HASH = hash_password("Pswd1Hell")
    
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        # Put admin login at bottom in an expander
        with st.sidebar.expander("Admin Access", expanded=False):
            password_input = st.text_input("Password:", type="password", key="admin_password")
            
            if st.button("Login", key="admin_login"):
                if hash_password(password_input) == ADMIN_PASSWORD_HASH:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
                    return False
        return False
    else:
        # Add logout button for admin at bottom
        with st.sidebar.expander("Admin Logout", expanded=False):
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
        return True

def create_admin_sidebar(data_path, max_tokens, top_k, similarity_threshold, model_options, selected_model, chatbot):
    """Create the admin-only sidebar controls"""
    st.sidebar.header("⚙️ Admin Configuration")
    
    # Configuration options (admin only)
    new_data_path = st.sidebar.text_input("📁 Documents Path", value=data_path)
    new_max_tokens = st.sidebar.slider("🔥 Max Tokens per Chunk", 100, 1000, max_tokens)
    new_top_k = st.sidebar.slider("📄 Documents to Retrieve", 1, 10, top_k)
    new_similarity_threshold = st.sidebar.slider("📊 Similarity Threshold", 0.0, 1.0, similarity_threshold, 0.05)
    
    # Model selection (admin only)
    new_selected_model = st.sidebar.selectbox("🤖 Claude Model", list(model_options.keys()), 
                                             index=list(model_options.keys()).index(selected_model))
    
    # API status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔌 API Status**")
    if 'last_api_error' in st.session_state:
        if "overloaded" in st.session_state.last_api_error.lower():
            st.sidebar.error("⚠️ API experiencing high traffic")
            st.sidebar.info("💡 Try again in 1-2 minutes")
        else:
            st.sidebar.warning("⚠️ API issue detected")
    else:
        st.sidebar.success("✅ API connection ready")
    
    # Admin controls
    st.sidebar.markdown("---")
    refresh_embeddings = st.sidebar.button("🔄 Refresh Embeddings")
    clear_chat = st.sidebar.button("🗑️ Clear Chat History")
    
    # Document statistics (admin only)
    if chatbot.df is not None and len(chatbot.df) > 0:
        with st.sidebar.expander("📊 Document Statistics"):
            df = chatbot.df
            st.write(f"**Total chunks:** {len(df)}")
            st.write(f"**Unique documents:** {df['filename'].nunique()}")
            st.write(f"**Average tokens per chunk:** {df['n_tokens'].mean():.0f}")
            
            # Show document breakdown by file type
            st.write("**Documents by type:**")
            file_types = {}
            for filename in df['filename'].unique():
                ext = os.path.splitext(filename)[1].lower()
                if ext in file_types:
                    file_types[ext] += 1
                else:
                    file_types[ext] = 1
            
            for ext, count in file_types.items():
                st.write(f"• {ext.upper() if ext else 'No extension'}: {count} files")
            
            # Show document breakdown
            st.write("**Chunks per document:**")
            doc_counts = df['filename'].value_counts()
            for doc, count in doc_counts.head(10).items():
                file_ext = os.path.splitext(doc)[1].upper()
                st.write(f"• {doc} ({file_ext}): {count} chunks")
    
    return (new_data_path, new_max_tokens, new_top_k, new_similarity_threshold, 
            new_selected_model, refresh_embeddings, clear_chat)

def create_student_sidebar():
    """Create a simplified sidebar for students"""
    st.sidebar.header("💡 How to Use This Chatbot")
    st.sidebar.markdown("""
    1. **Ask questions** about historical topics
    2. **Be specific** in your queries for better results
    3. **Reference documents** when possible
    4. **Explore different** historical periods and themes
    """)
    
    # Example questions for students
    with st.sidebar.expander("💬 Example Questions"):
        examples = [
            "What topics are covered in my documents?",
            "Tell me about the Civil War",
            "What happened in 1776?",
            "Summarize the main themes in my historical documents",
            "What can you tell me about the industrial revolution?"
        ]
        
        for example in examples:
            if st.sidebar.button(f"💬 {example}", key=f"example_{hash(example)}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

def get_base64_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

def create_custom_header():
    """Create a custom header with image and title"""
    image_base64 = get_base64_image("profile.png")  # or prof_cox.png
    
    st.markdown(f"""
    <style>
    .header-container {{
        display: flex;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #f0f0f0;
    }}
    .header-image {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-right: 1rem;
        object-fit: cover;
    }}
    .header-title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        margin: 0;
        line-height: 1.2;
    }}
    .header-subtitle {{
        font-size: 1rem;
        color: #666;
        font-style: italic;
        margin: 0;
        margin-top: 0.25rem;
    }}
    </style>
    <div class="header-container">
        <img src="data:image/jpeg;base64,{image_base64}" class="header-image" alt="Prof. Cox">
        <div>
            <h1 class="header-title">Prof. Cosmic History Chatbot</h1>
            <p class="header-subtitle">Ask questions about historical documents using local embeddings and Claude AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="📚 Historical Documents Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the embedding model"""
    return SentenceTransformer(model_name)

@st.cache_data
def load_api_key(key_file_path: str = "/home/drkeithcox/anthropic.key") -> str:
    """Load API key from file"""
    try:
        with open(key_file_path, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API key file is empty")
        logger.info(f"API key loaded from {key_file_path}")
        return api_key
    except FileNotFoundError:
        logger.error(f"API key file not found at {key_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading API key file: {e}")
        raise

class HistoricalChatbot:
    def __init__(self, anthropic_api_key: str, data_path: str = "text/Transcripts", 
                 max_tokens: int = 500, embedding_model=None):
        """
        Initialize the Historical Chatbot with local embeddings and Claude API
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.data_path = data_path
        self.max_tokens = max_tokens
        self.df = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = embedding_model
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text using local model"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.array([])
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts efficiently"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            return np.array([])
    
    def calculate_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and document embeddings"""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        return similarities
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into smaller chunks based on token limit"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if len(self.tokenizer.encode(paragraph)) <= self.max_tokens:
                if paragraph.strip():
                    chunks.append(paragraph.strip())
            else:
                sentences = paragraph.split('. ')
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    
                    if current_tokens + sentence_tokens > self.max_tokens:
                        if current_chunk:
                            chunks.append('. '.join(current_chunk) + '.')
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
                
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + ('.' if not current_chunk[-1].endswith('.') else ''))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing excess whitespace and newlines"""
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = ' '.join(text.split())
        return text
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text content from PDF file using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    print(f"Page {i+1}: '{page_text[:100] if page_text else 'None'}'")
                    if page_text:
                        text += page_text + "\n"
                if len(text.strip()) > 50:
                    return text
                
            # Otherwise, fall back to OCR
            st.info(f"Using OCR for image-based PDF: {os.path.basename(pdf_path)}")
            images = convert_from_path(pdf_path)
            ocr_text = ""
            
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    ocr_text += page_text + "\n"
            
            return ocr_text
        except Exception as e:
            logger.error(f"Error extracting PDF text from {pdf_path}: {e}")
            return ""
                # If we got meaningful text, return it

    def extract_docx_text(self, docx_path: str) -> str:
        """Extract text content from Word document"""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting docx text from {docx_path}: {e}")
            return ""        
              
    def crawl_documents(self) -> List[Tuple[str, str]]:
        """Crawl the document directory and extract text content"""

        documents = []
        
        if not os.path.exists(self.data_path):
            st.error(f"Data path {self.data_path} does not exist")
            return documents
        
        supported_extensions = ['.txt', '.lec', '.md', '.csv', '.docx']
        file_stats = {ext: 0 for ext in supported_extensions}
        
        progress_placeholder = st.empty()
        
        for root, dirs, files in os.walk(self.data_path):
            for i, file in enumerate(files):
                file_ext = None
                for ext in supported_extensions:
                    if file.lower().endswith(ext):
                        file_ext = ext
                        break
                
                if file_ext:
                    progress_placeholder.text(f"Loading documents... {file}")
                    file_path = os.path.join(root, file)
                    try:
                        if file_ext == '.docx':
                            content = self.extract_docx_text(file_path)
                        elif file_ext == '.pdf':
                            content = self.extract_pdf_text(file_path)
                        else:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                        
                        content = self.clean_text(content)
                        if content.strip() and len(content) > 50:
                            documents.append((file, content))
                            file_stats[file_ext] += 1
                            logger.info(f"Loaded {file_ext} file: {file} ({len(content)} characters)")
                    except Exception as e:
                        st.warning(f"Error reading {file_path}: {e}")
        
        progress_placeholder.empty()
        
        # Display file statistics
        if documents:
            st.success(f"📁 Loaded {len(documents)} documents:")
            cols = st.columns(len([k for k, v in file_stats.items() if v > 0]))
            col_idx = 0
            for ext, count in file_stats.items():
                if count > 0:
                    with cols[col_idx]:
                        st.metric(f"{ext.upper()} files", count)
                    col_idx += 1
        
        return documents
    
    def create_embeddings_dataframe(self, documents: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create a dataframe with embeddings for all document chunks"""
        all_chunks = []
        
        # Progress bar for chunking
        chunk_progress = st.progress(0)
        chunk_status = st.empty()
        
        for i, (filename, content) in enumerate(documents):
            chunk_status.text(f"Creating chunks for: {filename}")
            chunks = self.split_into_chunks(content)
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'filename': filename,
                    'chunk_id': j,
                    'text': chunk,
                    'n_tokens': len(self.tokenizer.encode(chunk))
                })
            chunk_progress.progress((i + 1) / len(documents))
        
        chunk_progress.empty()
        chunk_status.empty()
        
        df = pd.DataFrame(all_chunks)
        st.success(f"Created {len(df)} text chunks from {len(documents)} documents")
        
        if len(df) == 0:
            return df
        
        # Create embeddings with progress bar
        embed_progress = st.progress(0)
        embed_status = st.empty()
        embed_status.text("Creating embeddings using local model...")
        
        texts = df['text'].tolist()
        embeddings = self.get_embeddings_batch(texts)
        
        if len(embeddings) > 0:
            df['embeddings'] = list(embeddings)
        else:
            st.error("Failed to create embeddings")
            df['embeddings'] = [np.array([]) for _ in range(len(df))]
        
        embed_progress.progress(1.0)
        embed_progress.empty()
        embed_status.empty()
        
        st.success("Embeddings created successfully!")
        return df
    
    def get_embedding_filename(self):
        """Get course-specific embedding filename"""
        course = st.session_state.get('selected_course', 'World History')
        if course == "World History":
            return 'world_history_embeddings.pkl'
        elif course == "US History":
            return 'us_history_embeddings.pkl'
        else:
            return 'historical_embeddings.pkl'

    def save_embeddings(self, df: pd.DataFrame):
        """Save embeddings to course-specific pickle file"""
        filename = self.get_embedding_filename()
        try:
            df.to_pickle(filename)
            st.success(f"Embeddings saved to {filename}")
        except Exception as e:
            st.error(f"Error saving embeddings: {e}")

    def load_embeddings(self) -> Optional[pd.DataFrame]:
        """Load embeddings from course-specific pickle file"""
        filename = self.get_embedding_filename()
        try:
            df = pd.read_pickle(filename)
            st.success(f"Embeddings loaded from {filename}")
            return df
        except FileNotFoundError:
            st.info(f"No existing embeddings file found for this course. Will create new embeddings.")
            return None
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None
    '''
    def save_embeddings(self, df: pd.DataFrame, filename: str = 'historical_embeddings.pkl'):
        """Save embeddings to pickle file"""
        try:
            df.to_pickle(filename)
            st.success(f"Embeddings saved to {filename}")
        except Exception as e:
            st.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filename: str = 'historical_embeddings.pkl') -> Optional[pd.DataFrame]:
        """Load embeddings from pickle file"""
        try:
            df = pd.read_pickle(filename)
            st.success(f"Embeddings loaded from {filename}")
            return df
        except FileNotFoundError:
            st.info(f"No existing embeddings file found. Will create new embeddings.")
            return None
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None
    '''
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        with st.spinner("Loading embeddings..."):
            self.df = self.load_embeddings()
            
            if self.df is None or len(self.df) == 0:
                st.info("Creating new embeddings from your documents...")
                documents = self.crawl_documents()
                if documents:
                    self.df = self.create_embeddings_dataframe(documents)
                    if len(self.df) > 0:
                        self.save_embeddings(self.df)
                else:
                    st.error("No documents found to create embeddings")
                    self.df = pd.DataFrame()
            
            if self.df is not None and len(self.df) > 0:
                st.sidebar.success(f"📚 Ready with {len(self.df)} document chunks")
    
    def search_documents(self, query: str, top_k: int = 3, similarity_threshold: float = 0.1) -> pd.DataFrame:
        """Search for the most relevant documents based on query"""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        
        query_embedding = self.get_embedding(query)
        if len(query_embedding) == 0:
            return pd.DataFrame()
        
        doc_embeddings = np.stack(self.df['embeddings'].values)
        similarities = self.calculate_similarity(query_embedding, doc_embeddings)
        
        df_with_similarities = self.df.copy()
        df_with_similarities['similarity'] = similarities
        
        relevant_docs = df_with_similarities[df_with_similarities['similarity'] >= similarity_threshold]
        
        if len(relevant_docs) == 0:
            return pd.DataFrame()
        
        return relevant_docs.nlargest(top_k, 'similarity')
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for Claude"""
        return """You are a knowledgeable historical assistant with access to specific historical documents and transcripts. Your role is to:

1. FIRST: Check if the provided document context contains information relevant to the user's question
2. If relevant information is found in the documents:
   - Base your answer primarily on the document content
   - Cite which document(s) the information comes from
   - You may supplement with general historical knowledge if it adds helpful context
3. If the documents don't contain relevant information:
   - Clearly state that you didn't find relevant information in the provided documents
   - Provide a helpful answer using your general historical knowledge
   - Suggest what kinds of documents might contain better information

Always be conversational, engaging, and accurate. When citing documents, mention the filename and be specific about what information comes from where."""
    
    def generate_response_with_retry(self, query: str, conversation_history: List[dict], max_retries: int = 3) -> str:
        """Generate response with retry logic for API errors"""
        for attempt in range(max_retries):
            try:
                return self.generate_response(query, conversation_history)
            except Exception as e:
                error_str = str(e)
                
                # Check for specific error types
                if "overloaded" in error_str.lower() or "529" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                        st.warning(f"🔄 API temporarily overloaded. Retrying in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return """I apologize, but the Claude API is currently experiencing high traffic (Error 529: Overloaded). 
                        
Please try again in a few minutes. In the meantime, here are some suggestions:
- Try again in 1-2 minutes when traffic may be lower
- The system has successfully found relevant documents from your collection
- You can also try rephrasing your question

This is a temporary issue with Anthropic's servers, not with your documents or setup."""
                
                elif "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 60  # Wait longer for rate limits
                        st.warning(f"⏳ Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "I've hit the API rate limit. Please wait a moment and try again."
                
                elif "authentication" in error_str.lower() or "401" in error_str:
                    return "❌ Authentication error. Please check your API key in /home/drkeithcox/anthropic.key"
                
                else:
                    # For other errors, don't retry
                    return f"I encountered an error: {error_str}"
        
        return "Failed to get response after multiple attempts. Please try again later."
    
    def generate_response(self, query: str, conversation_history: List[dict]) -> str:
        """Generate a response using Claude API with document context"""
        relevant_docs = self.search_documents(query, top_k=3)
        
        # Check if this is a student (non-admin) session
        is_admin = st.session_state.get('admin_authenticated', False)
        
        # For students: only proceed if relevant documents are found
        if len(relevant_docs) == 0 and not is_admin:
            return """I can only help with questions related to the historical documents and topics covered in Professor Cox's course materials. 

    Please try asking about:
    - Topics covered in the assigned readings
    - Historical events, people, or themes from the course documents
    - Specific time periods or subjects we've studied

    If you have questions outside the course materials, please ask Professor Cox directly."""
        
        # Build context from relevant documents
        context = ""
        if len(relevant_docs) > 0:
            context = "=== RELEVANT DOCUMENTS ===\n\n"
            for _, doc in relevant_docs.iterrows():
                context += f"Document: {doc['filename']} (Relevance: {doc['similarity']:.3f})\n"
                context += f"Content: {doc['text']}\n\n"
            context += "=== END DOCUMENTS ===\n\n"
        else:
            context = "=== NO RELEVANT DOCUMENTS FOUND ===\n\n"
        
        # Prepare conversation history
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history[-6:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"
        
        # Use different system prompts for students vs admin
        if is_admin:
            system_prompt = self.get_system_prompt()
        else:
            system_prompt = """You are Professor Cox's historical assistant with access to specific course documents. Your role is to:

    1. Base your answer primarily on the provided course documents
    2. Always cite which document(s) the information comes from
    3. You may add relevant historical context or details that supplement the document content, but ONLY if they directly relate to the topics, events, people, or time periods discussed in the documents
    4. Do not answer questions about topics that are completely unrelated to the course materials

    Be educational and engaging while staying focused on the historical content covered in the course."""

        full_prompt = f"""{system_prompt}

    {context}{history_text}Current question: {query}

    Please provide a response based on the documents, supplementing with relevant historical knowledge only when it directly relates to the document content."""
        
        # Make API call
        response = self.client.messages.create(
            model=getattr(self, 'selected_model', "claude-sonnet-4-20250514"),
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text

def main():
    check_site_access()
    select_course()  # Add this after password check

    # Debug: Check what's in session state
    #st.write("DEBUG - Session State:")
    #st.write(f"selected_course: {st.session_state.get('selected_course', 'NOT SET')}")
    #st.write(f"data_path: {st.session_state.get('data_path', 'NOT SET')}")

    create_custom_header()
    
    # Load API key
    try:
        api_key = load_api_key("/home/drkeithcox/anthropic.key")
    except Exception as e:
        st.error(f"Failed to load API key: {e}")
        st.stop()
        
    # Get course-specific data path from session state
    data_path = st.session_state.get('data_path', 'text/WorldHistory')  # fallback to World History
    
    # Default configuration
    max_tokens = 500
    top_k = 3
    similarity_threshold = 0.1
    
    model_options = {
        "Claude Sonnet 4": "claude-sonnet-4-20250514",
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        "Claude 3 Haiku (Faster)": "claude-3-haiku-20240307",
        "Claude 3 Opus (Most Capable)": "claude-3-opus-20240229"
    }
    selected_model = "Claude Sonnet 4"  # Changed default
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
     # Initialize chatbot with course-specific path
    if 'chatbot' not in st.session_state or st.session_state.chatbot.data_path != data_path:
        st.session_state.chatbot = HistoricalChatbot(
            api_key, data_path, max_tokens, embedding_model
        )
        st.session_state.chatbot.selected_model = model_options[selected_model]
        st.session_state.chatbot.load_or_create_embeddings()
    
    # Always show student content first (unless admin is already authenticated)
    if not st.session_state.get('admin_authenticated', False):
        create_student_sidebar()
    
    # Check admin authentication (this will appear at bottom of sidebar)
    is_admin = check_admin_password()
    
    if is_admin:
        # Clear the sidebar and show admin controls
        st.sidebar.empty()
        (data_path, max_tokens, top_k, similarity_threshold, selected_model,
         refresh_embeddings, clear_chat) = create_admin_sidebar(
            data_path, max_tokens, top_k, similarity_threshold,
            model_options, selected_model, st.session_state.chatbot
        )
        
        # Handle admin actions
        if refresh_embeddings:
            st.session_state.chatbot.df = None
            st.session_state.chatbot.load_or_create_embeddings()
            st.rerun()
        
        if clear_chat:
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your historical documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Searching documents and generating response...")
            
            response = st.session_state.chatbot.generate_response_with_retry(
                prompt, st.session_state.messages[:-1]
            )
            
            message_placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
