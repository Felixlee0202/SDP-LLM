import os
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

PDF_FILE = "VantageSequenceProgrammingManual.pdf"
INDEX_FILE = "faiss.index"
META_FILE = "chunks.pkl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def main():
    reader = PdfReader(PDF_FILE)
    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    chunks = chunk_text(all_text)
    print(f"Total chunks: {len(chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index + chunks saved")

if __name__ == "__main__":
    main()
