#!/usr/bin/env python3
"""
Vector Index Creation Module

This module creates FAISS indices from retrieval corpus and stores them along with
the corresponding text passages for later retrieval.
"""

import json
import argparse
import torch # type: ignore
from typing import List, Tuple
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import faiss # type: ignore
import os
from config import VECTOR_INDEX as CFG
# from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRContextEncoder, AutoTokenizer


class VectorIndexer:
    """Handles the creation and storage of vector indices for retrieval."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the VectorIndexer.
        
        Args:
            model_path: Path to the DPR context encoder model
            device: Device to use for encoding (cuda or cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load models
        self.ctx_encoder = DPRContextEncoder.from_pretrained(model_path).to(self.device)
        # self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def load_passages(self, filepath: str) -> List[str]:
        """
        Load passages from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing passages
            
        Returns:
            List of passages
        """
        knowledge_base = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                knowledge_base.append(json.loads(line)["passage"])
        return knowledge_base
    
    def chunk_passage(self, passage: str, chunk_size: int = 100) -> List[str]:
        """
        Split a passage into chunks of specified word size.
        
        Args:
            passage: Text passage to chunk
            chunk_size: Number of words per chunk
            
        Returns:
            List of chunked passages
        """
        words = passage.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    def chunk_all_passages(self, passages: List[str], chunk_size: int = 100) -> List[str]:
        """
        Chunk all passages into smaller segments.
        
        Args:
            passages: List of passages to chunk
            chunk_size: Number of words per chunk
            
        Returns:
            List of all chunked passages
        """
        chunks = []
        for passage in tqdm(passages, desc=f"Chunking into {chunk_size} words per doc"):
            chunks.extend(self.chunk_passage(passage, chunk_size))
        return chunks
    
    def encode_passages(self, passages: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode passages into embeddings using the DPR context encoder.
        
        Args:
            passages: List of passages to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
            batch = passages[i:i+batch_size]
            inputs = self.ctx_tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=256, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                reps = self.ctx_encoder(**inputs).pooler_output
            
            # Normalize for cosine similarity
            reps = torch.nn.functional.normalize(reps, dim=1)
            embeddings.append(reps.cpu().numpy())
        
        return np.vstack(embeddings)
    
    # def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
    #     """
    #     Create a FAISS index from embeddings.
        
    #     Args:
    #         embeddings: Numpy array of embeddings
            
    #     Returns:
    #         FAISS index
    #     """

    #     # Create a FAISS index with the embeddings  
        
    #     index = faiss.IndexFlatIP(embeddings.shape[1])
    #     index.add(embeddings.astype(np.float32))
    #     return index

    def create_faiss_index(self, 
                       embeddings: np.ndarray, 
                       M: int = 32, 
                       efConstruction: int = 200, 
                       efSearch: int = 64) -> faiss.Index:
        """
        Create a FAISS HNSW index from embeddings using float16 storage.

        Args:
            embeddings (np.ndarray): Numpy array of embeddings (normalized)
            M (int): Number of bi-directional links for HNSW (higher = better recall, more memory)
            efConstruction (int): Controls accuracy/speed during index construction
            efSearch (int): Controls accuracy/speed during search

        Returns:
            faiss.Index: HNSW FAISS index
        """
        dim = embeddings.shape[1]

        # Convert embeddings to float32 for memory efficiency
        embeddings = embeddings.astype(np.float32)

        # Create HNSW index with Inner Product (cosine if normalized)
        index = faiss.IndexHNSWFlat(dim, M)
        # Set to inner product (works as cosine if embeddings are normalized)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch

        # Add embeddings to the index
        index.add(embeddings)

        return index

    
    def save_index_and_passages(self, index: faiss.Index, passages: List[str], 
                               output_dir: str, index_name: str = "index") -> Tuple[str, str]:
        """
        Save the FAISS index and passages to files.
        
        Args:
            index: FAISS index to save
            passages: List of passages to save
            output_dir: Directory to save files
            index_name: Base name for the index files
            
        Returns:
            Tuple of (index_path, passages_path)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, f"{index_name}.faiss")
        faiss.write_index(index, index_path)
        
        # Save passages
        passages_path = os.path.join(output_dir, f"{index_name}.txt")
        with open(passages_path, "w", encoding="utf-8") as f:
            for passage in passages:
                f.write(passage.strip() + "\n")
        
        return index_path, passages_path
    
    def process_corpus(self, corpus_path: str, output_dir: str, 
                      chunk_size: int = 100, batch_size: int = 64,
                      index_name: str = "index") -> Tuple[str, str]:
        """
        Process the entire corpus and create indices.
        
        Args:
            corpus_path: Path to the corpus JSON file
            output_dir: Directory to save outputs
            chunk_size: Size of chunks for passages
            batch_size: Batch size for encoding
            index_name: Base name for the index files
            
        Returns:
            Tuple of (index_path, passages_path)
        """
        print(f"Loading passages from {corpus_path}...")
        knowledge_base = self.load_passages(corpus_path)
        
        print(f"Chunking {len(knowledge_base)} passages...")
        chunked_passages = self.chunk_all_passages(knowledge_base, chunk_size)
        
        print(f"Encoding {len(chunked_passages)} chunks...")
        embeddings = self.encode_passages(chunked_passages, batch_size)
        
        print("Creating FAISS index...")
        index = self.create_faiss_index(embeddings)
        
        print(f"Saving index and passages to {output_dir}...")
        index_path, passages_path = self.save_index_and_passages(
            index, chunked_passages, output_dir, index_name
        )
        
        print(f" Indexed {len(chunked_passages)} passages and saved to:")
        print(f"   Index: {index_path}")
        print(f"   Passages: {passages_path}")
        
        return index_path, passages_path


def main():
    """Main function to handle command line arguments and run the indexing process."""
    parser = argparse.ArgumentParser(description="Create vector indices from retrieval corpus")

    parser.add_argument(
        "--corpus_path", 
        type=str, 
        required=True,
        help="Path to the retrieval corpus JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save the index and passages files"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to the DPR context encoder model"
    )
    
    args = parser.parse_args()
    # Paths from CLI
    corpus_path = args.corpus_path
    output_dir = args.output_dir
    model_path = args.model_path

    # Non-path params from Python config
    chunk_size = CFG.get("chunk_size", 100)
    batch_size = CFG.get("batch_size", 64)
    index_name = CFG.get("index_name", "index")
    device = CFG.get("device")
    

    # Validate required inputs (paths must be provided via command line)
    if not corpus_path:
        raise ValueError("--corpus_path is required (must be provided via command line)")
    if not output_dir:
        raise ValueError("--output_dir is required (must be provided via command line)")
    if not model_path:
        model_path = "models/cxt_encoder"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Create indexer and process corpus
    indexer = VectorIndexer(model_path, device)
    indexer.process_corpus(
        corpus_path=corpus_path,
        output_dir=output_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        index_name=index_name
    )


if __name__ == "__main__":
    main()



# python build_index.py --corpus_path sample/retrieval_corpus.jsonl --output_dir "temporary" 