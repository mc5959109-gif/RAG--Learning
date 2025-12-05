import pdfplumber
import os

DATA_DIR = "data"
PDF_DIR = "pdfs"  # put your PDFs here

os.makedirs(DATA_DIR, exist_ok=True)

for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
        
        # Save as text file for vector store
        txt_file = os.path.join(DATA_DIR, pdf_file.replace(".pdf", ".txt"))
        with open(txt_file, "w", encoding="utf8") as f:
            f.write(full_text)
        print(f"Saved {txt_file}")

