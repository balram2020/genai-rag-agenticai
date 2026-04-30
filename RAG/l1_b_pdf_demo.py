from pathlib import Path
import PyPDF2
import pdfplumber
import re
# Get the path to the PDF file
pdf_path = Path(__file__).parent / "test_documents" / "financial_report.pdf"

def extract_pypdf2(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
print("pyPDF2 extracted text:")
print(extract_pypdf2(pdf_path))
print("-" * 100)

def extract_pdfplumber(pdf_path: Path) -> str:
    quality_score = 1.0
    with open(pdf_path, "rb") as f:
        pdf = pdfplumber.open(f)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        
        non_ascii = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii > 0.3:
            quality_score -= 0.2
        
        if len(text) < 100:
            quality_score -= 0.1
        
        if len(text) == 0:
            #return no text found
            quality_score = 0.0
        
        gibberish_pattern = r'^[a-zA-Z0-9]+$'
        if re.search(gibberish_pattern, text):
            quality_score -= 0.3
        
        if quality_score < 0.5:
            return "No text found or quality too low" # pass_to_human_fallback
        
        return text, quality_score
        
print("pdfplumber extracted text:")
print(extract_pdfplumber(pdf_path)[0])
print("Quality score:", extract_pdfplumber(pdf_path)[1])
print("-" * 100)