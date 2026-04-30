from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- Semantic Chunking Example ---
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv() # Make sure you have your GOOGLE_API_KEY in .env

p = Path(__file__).parent / "test_documents" / "02_legal_contract.txt"

text = p.read_text() # read the text file

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, #max number of characters in a chunk
                chunk_overlap=20, #number of characters to overlap between chunks
                separators=["\n\n", "\n", " ", ""]) 

chunks = text_splitter.split_text(text)
print("\n" + "="*50)
print("RECURSIVE CHARACTER TEXT SPLITTER RESULTS")
print("="*50)
for chunk in chunks:
    print(chunk)
    print("-" * 100)


# Initialize the embedding model (using Gemini as seen in your other file)
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Create the semantic chunker
#breakpoint_threshold_type can be "percentile" or "standard_deviation"
#percentile - Calculate all chunk embeddings, then find the median similarity score
#standard_deviation - Calculate all chunk embeddings, then find the standard deviation of similarity scores
semantic_chunker = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile")

# Split the text
semantic_chunks = semantic_chunker.split_text(text)

print("\n" + "="*50)
print("SEMANTIC CHUNKING RESULTS")
print("="*50)

for i, chunk in enumerate(semantic_chunks):
    print(f"Chunk {i+1}:\n{chunk}")
    print("-" * 100)
