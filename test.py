#!/usr/bin/env python3
"""
Test script to debug embedding issues
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

def test_embedding_model():
    """Test if the embedding model works"""
    print("Testing sentence transformer model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "This is a test sentence about American history."
        embedding = model.encode(test_text, convert_to_tensor=False)
        print(f"✅ Model loaded successfully")
        print(f"✅ Test embedding shape: {embedding.shape}")
        print(f"✅ Test embedding type: {type(embedding)}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_batch_embeddings(model):
    """Test batch embedding creation"""
    print("\nTesting batch embeddings...")
    try:
        test_texts = [
            "George Washington was the first president.",
            "The Civil War began in 1861.",
            "The Declaration of Independence was signed in 1776."
        ]
        embeddings = model.encode(test_texts, convert_to_tensor=False)
        print(f"✅ Batch embeddings shape: {embeddings.shape}")
        print(f"✅ Individual embedding shape: {embeddings[0].shape}")
        return embeddings
    except Exception as e:
        print(f"❌ Error creating batch embeddings: {e}")
        return None

def test_existing_pickle():
    """Test loading existing pickle file"""
    print("\nTesting existing embeddings file...")
    try:
        df = pd.read_pickle('historical_embeddings.pkl')
        print(f"✅ Pickle file loaded")
        print(f"✅ DataFrame shape: {df.shape}")
        print(f"✅ Columns: {df.columns.tolist()}")
        
        if 'embeddings' in df.columns:
            first_embedding = df['embeddings'].iloc[0]
            print(f"✅ First embedding type: {type(first_embedding)}")
            print(f"✅ First embedding shape: {first_embedding.shape}")
            
            # Check if embeddings are actually empty
            empty_count = sum(1 for emb in df['embeddings'] if len(emb) == 0)
            print(f"✅ Empty embeddings count: {empty_count}/{len(df)}")
        
        return df
    except FileNotFoundError:
        print("ℹ️  No pickle file found")
        return None
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return None

def main():
    print("🔍 Debugging Embedding Issues\n")
    
    # Test 1: Model loading
    model = test_embedding_model()
    if not model:
        return
    
    # Test 2: Batch processing
    
    embeddings = test_batch_embeddings(model)
    if embeddings is None:
        return
    
    # Test 3: Existing pickle file
    df = test_existing_pickle()
    
    print("\n🎯 Summary:")
    print("- Model loading: ✅")
    print("- Batch embeddings: ✅")
    if df is not None:
        empty_embeddings = sum(1 for emb in df['embeddings'] if len(emb) == 0)
        if empty_embeddings > 0:
            print(f"- Pickle file: ❌ Contains {empty_embeddings} empty embeddings")
            print("  🔧 Solution: Delete historical_embeddings.pkl and recreate")
        else:
            print("- Pickle file: ✅")
    else:
        print("- Pickle file: Not found (will be created)")

if __name__ == "__main__":
    main()