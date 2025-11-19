import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    document: Document
    score: float
    

class RAGSystem:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "home_credit_docs"
    ):
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized RAG system with collection: {collection_name}")
        self._document_cache = {}
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the RAG system
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        # Check for existing documents and filter them out
        ids = [doc.id for doc in documents]
        existing_ids = set()
        
        # Get existing documents from collection
        try:
            existing = self.collection.get(ids=ids)
            if existing and existing['ids']:
                existing_ids = set(existing['ids'])
        except Exception as e:
            logger.debug(f"Error checking existing documents: {e}")
        
        # Filter out documents that already exist
        new_documents = [doc for doc in documents if doc.id not in existing_ids]
        
        if not new_documents:
            logger.debug(f"All {len(documents)} documents already exist, skipping")
            return
        
        # Generate embeddings only for new documents
        contents = [doc.content for doc in new_documents]
        embeddings = self.embedder.encode(contents).tolist()
        
        # Prepare data for ChromaDB
        new_ids = [doc.id for doc in new_documents]
        metadatas = [doc.metadata for doc in new_documents]
        
        # Add to collection
        self.collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        # Update cache
        for doc, embedding in zip(new_documents, embeddings):
            doc.embedding = np.array(embedding)
            self._document_cache[doc.id] = doc
        
        if existing_ids:
            logger.info(f"Added {len(new_documents)} new documents to RAG system (skipped {len(existing_ids)} existing)")
        else:
            logger.info(f"Added {len(new_documents)} documents to RAG system")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata if filter_metadata else None
        )
        
        # Parse results
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                
                # Try to get from cache first
                if doc_id in self._document_cache:
                    doc = self._document_cache[doc_id]
                else:
                    doc = Document(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                
                score = 1 - results['distances'][0][i]  # Convert distance to similarity
                search_results.append(SearchResult(document=doc, score=score))
        
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the RAG system
        
        Args:
            document_ids: List of document IDs to delete
        """
        self.collection.delete(ids=document_ids)
        for doc_id in document_ids:
            if doc_id in self._document_cache:
                del self._document_cache[doc_id]
        logger.info(f"Deleted {len(document_ids)} documents from RAG system")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name
        }
    
    def clear_collection(self) -> None:
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._document_cache.clear()
        logger.info("Cleared RAG collection")


class DatabaseRAG(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_descriptions = {}
        self.column_descriptions = {}
    
    def index_database_schema(self, database_info: Dict[str, Any]) -> None:
        """
        Index database schema information
        
        Args:
            database_info: Database information including tables and columns
        """
        documents = []
        
        # Index table information
        for table_name in database_info.get('tables', []):
            table_details = database_info.get('table_details', {}).get(table_name, {})
            
            # Create document for table
            table_doc = Document(
                id=f"table_{table_name}",
                content=f"Table: {table_name}\n"
                       f"Columns: {', '.join(table_details.get('sample_columns', []))}\n"
                       f"Total columns: {table_details.get('columns', 0)}",
                metadata={
                    "type": "table",
                    "table_name": table_name,
                    "column_count": table_details.get('columns', 0)
                }
            )
            documents.append(table_doc)
        
        # Add documents to RAG
        if documents:
            self.add_documents(documents)
            logger.info(f"Indexed {len(documents)} database schema documents")
    
    def index_table_descriptions(self, descriptions_file: str) -> None:
        """
        Index table and column descriptions from a CSV file
        
        Args:
            descriptions_file: Path to CSV file with descriptions
        """
        try:
            df = pd.read_csv(descriptions_file, encoding='latin-1')
            documents = []
            
            for _, row in df.iterrows():
                # Create document for each column description
                doc = Document(
                    id=f"col_desc_{row.get('Table', 'unknown')}_{row.get('Row', 'unknown')}",
                    content=f"Table: {row.get('Table', '')}\n"
                           f"Column: {row.get('Row', '')}\n"
                           f"Description: {row.get('Description', '')}\n"
                           f"Special: {row.get('Special', '')}",
                    metadata={
                        "type": "column_description",
                        "table": row.get('Table', ''),
                        "column": row.get('Row', '')
                    }
                )
                documents.append(doc)
                
                # Store in cache for quick access
                table_key = row.get('Table', '').lower()
                column_key = row.get('Row', '').lower()
                
                if table_key not in self.table_descriptions:
                    self.table_descriptions[table_key] = {}
                
                self.column_descriptions[f"{table_key}.{column_key}"] = row.get('Description', '')
            
            # Add to RAG
            if documents:
                self.add_documents(documents)
                logger.info(f"Indexed {len(documents)} column descriptions")
                
        except Exception as e:
            logger.error(f"Failed to index descriptions: {e}")
    
    def get_relevant_schema_context(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        Get relevant schema context for a query
        
        Args:
            query: User query
            top_k: Number of relevant contexts to retrieve
            
        Returns:
            Schema context as string
        """
        # Search for relevant schema
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "No relevant schema found."
        
        # Compile context
        context_parts = []
        for result in results:
            context_parts.append(f"Relevance: {result.score:.2f}\n{result.document.content}")
        
        return "\n\n".join(context_parts)
    
    def index_sql_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Index SQL query examples for few-shot learning
        
        Args:
            examples: List of dictionaries with 'question' and 'sql' keys
        """
        documents = []
        
        for i, example in enumerate(examples):
            doc = Document(
                id=f"sql_example_{i}",
                content=f"Question: {example['question']}\nSQL: {example['sql']}",
                metadata={
                    "type": "sql_example",
                    "question": example['question'],
                    "sql": example['sql']
                }
            )
            documents.append(doc)
        
        if documents:
            self.add_documents(documents)
            logger.info(f"Indexed {len(documents)} SQL examples")
    
    def get_similar_queries(self, question: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Get similar SQL queries for few-shot learning
        
        Args:
            question: User question
            top_k: Number of examples to retrieve
            
        Returns:
            List of similar query examples
        """
        results = self.search(
            question,
            top_k=top_k,
            filter_metadata={"type": "sql_example"}
        )
        
        examples = []
        for result in results:
            if result.score > 0.5:  # Only include relevant examples
                examples.append({
                    "question": result.document.metadata.get("question", ""),
                    "sql": result.document.metadata.get("sql", ""),
                    "score": result.score
                })
        
        return examples
