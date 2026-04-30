#from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np
load_dotenv()

#embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

SENTENCES = [
    "The cat sat on the mat.",
    "The Felines are resting on the rug.",
    "I love eating pizza with friends.",
    "The dog barked at the mailman who was delivering packages.",
]

query = "Where is the cat?"

embeddings = embeddings_model.embed_documents(SENTENCES)
query_embedding = embeddings_model.embed_query(query)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i, sent1 in enumerate(SENTENCES):
        row = f"{sent1[:28]:<28} |"
        for j in range(len(SENTENCES)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row += f"  {sim:.3f}  |"
        print(row)
        print("-" * 60)

print("Similar sentences should have high similarity scores.")
print("Query: ", query)
print(f"Similarity between query and first sentence: {cosine_similarity(query_embedding, embeddings[0])}")
print(f"Similarity between query and second sentence: {cosine_similarity(query_embedding, embeddings[1])}")
print(f"Similarity between query and third sentence: {cosine_similarity(query_embedding, embeddings[2])}")
print(f"Similarity between query and fourth sentence: {cosine_similarity(query_embedding, embeddings[3])}") 