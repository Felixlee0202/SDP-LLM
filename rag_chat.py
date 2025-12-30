import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

INDEX_FILE = "faiss.index"
META_FILE = "chunks.pkl"
TOP_K = 3

BASE_MODEL = "huggyllama/llama-7b"

def load_retriever():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        legacy=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model

def retrieve(query, index, chunks, embedder):
    q_emb = embedder.encode([query])
    distances, ids = index.search(q_emb, TOP_K)
    return [chunks[i] for i in ids[0]]

def generate(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    index, chunks, embedder = load_retriever()
    tokenizer, model = load_llm()

    print("\nUltrasound RAG (LLaMA-7B). Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        context_chunks = retrieve(query, index, chunks, embedder)
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an expert ultrasound assistant.

Answer ONLY using the context below.
If the answer is not present, say "I do not know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

        answer = generate(prompt, tokenizer, model)

        print(f"\nRAG:\n{answer}\n")
        print("Sources:")
        for i, c in enumerate(context_chunks, 1):
            print(f"{i}. {c[:200]}...\n")

if __name__ == "__main__":
    main()
