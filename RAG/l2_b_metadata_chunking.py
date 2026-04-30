from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import datetime
from langchain_core.documents import Document

p = Path(__file__).parent / "test_documents" / "02_legal_contract.txt"

text = p.read_text() # read the text file

def chunk_document_with_metadata(text, extraction_metadata, access_metadata) :
    """
    Chunk a document with metadata.
    Args:
        text: Extracted text
        extraction_metadata: From Layer 1 (source, pages, quality)
        access_metadata: For Layer 6 (department, access_level)
    """
    # Create a unique identifier for the document
    document_id = hashlib.sha256(text.encode()).hexdigest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, #max number of characters in a chunk
                chunk_overlap=20, #number of characters to overlap between chunks
                separators=["\n\n", "\n", " ", ""]) 
    chunks = splitter.split_text(text)

    documents = []
    for idx, chunk_text in enumerate(chunks):
        chunk_metadata = {
            **extraction_metadata,
            **access_metadata,
            "document_id": document_id,
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk_text),
            "created_at": datetime.datetime.now().isoformat(),
        }
        doc = Document(page_content=chunk_text, metadata=chunk_metadata)
        documents.append(doc)
    return documents

extraction_metadata = {
    "source": "02_legal_contract.txt",
    "pages": 1,
    "quality": 0.95,
    "extraction_method": "text_extraction",
}
access_metadata = {
    "department": "legal",
    "access_level": "confidential"
}
documents = chunk_document_with_metadata(text, extraction_metadata, access_metadata)
for document in documents:
    print(document.page_content)
    print(document.metadata)
    print("-" * 100)