# pre-requisites: l4_a_prod_storage.py is executed
# CREATE INDEX IF NOT EXISTS documents_ivfflat_idx
# ON documents
# USING ivfflat (embedding vector_cosine_ops)
# WITH (lists = 25);


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

def run_search(probes):
    conn = psycopg.connect(CONNECTION_STRING) 

    with conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # nprobes: number of nearest clusters (lists) to search at query time; higher = more accurate but slower
            cur.execute(f"SET ivfflat.probes = {probes};")
            start_time = time.time()
            # Run the similarity search
            cur.execute(
                """
                SELECT id, content, 
                embedding <=> %s::vector AS similarity
                FROM documents
                ORDER BY similarity 
                LIMIT 5
                """,
                (query_embedding,)  # comma makes it a tuple — psycopg maps this single element to %s
            )
            results = cur.fetchall()
            end_time = time.time()
            duration = end_time - start_time
            return results, duration

for probe in [1, 5, 10, 20]:
    output = run_search(probe)
    print(f"Probes: {probe}, Duration: {output[1]:.2f} seconds")
    for result in output[0]:
        print(f"ID: {result[0]}, Content: {result[1]}, Similarity: {result[2]:.2f}")
        print("-" * 100)


