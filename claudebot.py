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
    
    def crawl_documents(self) -> List[Tuple[str, str]]:
        """Crawl the document directory and extract text content"""
        documents = []
        
        if not os.path.exists(self.data_path):
            st.error(f"Data path {self.data_path} does not exist")
            return documents
        
        supported_extensions = ['.txt', '.lec', '.md', '.csv']
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
        
        context = ""
        if len(relevant_docs) > 0:
            context = "=== RELEVANT DOCUMENTS ===\n\n"
            for _, doc in relevant_docs.iterrows():
                context += f"Document: {doc['filename']} (Relevance: {doc['similarity']:.3f})\n"
                context += f"Content: {doc['text']}\n\n"
            context += "=== END DOCUMENTS ===\n\n"
        else:
            context = "=== NO RELEVANT DOCUMENTS FOUND ===\n\n"
        
        # Prepare conversation history (last 6 messages)
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history[-6:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"
        
        full_prompt = f"""{self.get_system_prompt()}

{context}{history_text}Current question: {query}

Please provide a helpful and accurate response based on the above information."""
        
        # Make API call with error handling
        response = self.client.messages.create(
            model=getattr(self, 'selected_model', "claude-3-5-sonnet-20241022"),
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text

def main():
    st.title("📚 Historical Documents Chatbot")
    st.markdown("*Ask questions about your historical documents using local embeddings and Claude AI*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load API key
    try:
        api_key = load_api_key("/home/drkeithcox/anthropic.key")
        st.sidebar.success("✅ API key loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load API key: {e}")
        st.error("Please ensure your Anthropic API key is in /home/drkeithcox/anthropic.key")
        st.stop()
    
    # Configuration options
    data_path = st.sidebar.text_input("📁 Documents Path", value="text/Transcripts")
    max_tokens = st.sidebar.slider("📄 Max Tokens per Chunk", 100, 1000, 500)
    top_k = st.sidebar.slider("🔍 Documents to Retrieve", 1, 10, 3)
    similarity_threshold = st.sidebar.slider("📊 Similarity Threshold", 0.0, 1.0, 0.1, 0.05)
    
    # Model selection
    model_options = {
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        "Claude 3 Haiku (Faster)": "claude-3-haiku-20240307",
        "Claude 3 Opus (Most Capable)": "claude-3-opus-20240229"
    }
    selected_model = st.sidebar.selectbox("🤖 Claude Model", list(model_options.keys()))
    
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
    
    # Load embedding model
    embedding_model = load_embedding_model()
    st.sidebar.success("✅ Embedding model loaded")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HistoricalChatbot(
            api_key, data_path, max_tokens, embedding_model
        )
        # Store selected model
        st.session_state.chatbot.selected_model = model_options[selected_model]
        st.session_state.chatbot.load_or_create_embeddings()
    
    # Update model if changed
    if hasattr(st.session_state.chatbot, 'selected_model'):
        if st.session_state.chatbot.selected_model != model_options[selected_model]:
            st.session_state.chatbot.selected_model = model_options[selected_model]
    else:
        st.session_state.chatbot.selected_model = model_options[selected_model]
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display document statistics
    if st.session_state.chatbot.df is not None and len(st.session_state.chatbot.df) > 0:
        with st.sidebar.expander("📊 Document Statistics"):
            df = st.session_state.chatbot.df
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
            
            # Show loading message
            message_placeholder.markdown("🔍 Searching documents and generating response...")
            
            # Generate response with retry logic
            response = st.session_state.chatbot.generate_response_with_retry(
                prompt, st.session_state.messages[:-1]
            )
            
            # Display final response
            message_placeholder.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar controls
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Embeddings"):
        st.session_state.chatbot.df = None
        st.session_state.chatbot.load_or_create_embeddings()
        st.experimental_rerun()
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Example questions
    with st.sidebar.expander("💡 Example Questions"):
        examples = [
            "What topics are covered in my documents?",
            "Tell me about the Civil War",
            "What happened in 1776?",
            "Summarize the main themes in my historical documents",
            "What can you tell me about the industrial revolution?"
        ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"example_{hash(example)}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.experimental_rerun()

if __name__ == "__main__":
    main()