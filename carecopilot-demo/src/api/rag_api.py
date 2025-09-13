"""
RAG Query Engine with lower threshold for better demo results
"""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
import time

from config.settings import settings
from src.models.database import SessionLocal

class RAGQueryEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("RAG Query Engine initialized")
    
    def query(self, question: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            start_time = time.time()
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(question, convert_to_tensor=False)
            embedding_list = query_embedding.tolist()
            
            print(f"Generated embedding for query: {question}")
            
            # Lower threshold to 0.05 for better demo results
            results = db.execute(text("""
                SELECT 
                    dc.content,
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    1 - (dc.embedding <=> CAST(:embedding AS vector)) as similarity_score
                FROM document_chunks dc
                WHERE 1 - (dc.embedding <=> CAST(:embedding AS vector)) > 0.05
                ORDER BY dc.embedding <=> CAST(:embedding AS vector)
                LIMIT 5
            """), {"embedding": embedding_list}).fetchall()
            
            search_time = (time.time() - start_time) * 1000
            
            # Process results
            search_results = []
            for row in results:
                similarity = float(row.similarity_score) if row.similarity_score else 0.0
                search_results.append({
                    'content': row.content,
                    'id': row.id,
                    'document_id': row.document_id,
                    'chunk_index': row.chunk_index,
                    'similarity': similarity
                })
            
            # Generate answer
            if search_results:
                top_content = search_results[0]['content'][:300]
                answer = f"Based on the medical records: {top_content}..."
            else:
                answer = "No relevant medical records found for this query."
            
            return {
                "answer": answer,
                "search_results": search_results,
                "query_time_ms": search_time,
                "total_results": len(search_results)
            }
            
        except Exception as e:
            print(f"RAG Query Error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "search_results": [],
                "query_time_ms": 0,
                "total_results": 0
            }
        finally:
            db.close()

# Initialize the query engine
query_engine = RAGQueryEngine()
