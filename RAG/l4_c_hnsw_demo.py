# pre-requisites: l4_a_prod_storage.py is executed
# CREATE INDEX documents_hnsw_idx
# ON documents
# USING hnsw (embedding vector_cosine_ops)
# WITH (m = 16, ef_construction = 64);


from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # free, runs locally, no API key needed
import psycopg
from pgvector.psycopg import register_vector
import hashlib
from psycopg.types.json import Json
import uuid
import time
from pgvector.psycopg import Vector
load_dotenv()
CONNECTION_STRING = "postgresql://langchain:langchain@localhost:6024/langchain"
# EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
# EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")  # requires allowlisted project
# sentence-transformers/all-MiniLM-L6-v2: free local model, 384 dims, no API key needed
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query = "What is BERT ?"
query_embedding = EMBEDDING_MODEL.embed_query(query)

# https://www.psycopg.org/docs/connection.html
# https://www.psycopg.org/docs/cursor.html

def run_hnsw_search(ef):
    conn = psycopg.connect(CONNECTION_STRING) 

    with conn:
        register_vector(conn)
        vector = Vector(query_embedding)
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {ef};")
            # Run the similarity search
            cur.execute(
                """
                EXPLAIN ANALYZE
                SELECT id, content, 
                embedding <=> %s::vector AS similarity
                FROM documents
                ORDER BY similarity 
                LIMIT 5
                """,
                (vector,)  # comma makes it a tuple — psycopg maps this single element to %s
            )
            results = cur.fetchall()
            print("\n\nQuery Plan:")
            for row in results:
                print(row[0])

for efs in [1, 5, 10, 20]:
    run_hnsw_search(efs)


