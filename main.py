from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from sympy.physics.units import temperature
from torch import while_loop

'''from pinecone import ServerlessSpec'''
import re
import pinecone

from google import genai

# === PDF Extraction ===
def extract_text_with_headings(pdf_path):
    reader = PdfReader(pdf_path)
    data = []
    for page in reader.pages:
        text = page.extract_text()
        lines = text.split("\n")
        for line in lines:
            if re.match(r"^[A-Z][A-Z\s]+$", line.strip()):  # Uppercase headings
                data.append({"heading": line.strip(), "content": ""})
            elif data:
                data[-1]["content"] += line.strip() + " "
    return data

# === Chunking ===
def chunk_by_headings(data, max_chunk_size=500):
    chunks = []
    for section in data:
        heading = section["heading"]
        content = section["content"].split()
        for i in range(0, len(content), max_chunk_size):
            chunk = " ".join(content[i:i + max_chunk_size])
            chunks.append(f"{heading}\n{chunk}")
    return chunks

# === Embeddings ===
def generate_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings

# === Pinecone Upsert ===
def store_in_pinecone(vectors_to_upsert):
    pc = Pinecone(api_key="pcsk_Rrnjs_CVyUkzMpFoNkkaUPpajF83W9ftkBcjAYANwwtsTpaaAqPJ8fPsfRwa1kqfymgLo")
    index = pc.Index(host="https://pdfreader-fzsu338.svc.aped-4627-b74a.pinecone.io")
    index.upsert(vectors_to_upsert)
    return index

# === Query Embedding ===
def query_embedding(query):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    return query_vector

# === Main Flow ===
def main():
    pdf_path = "Resume.pdf"

    # Step 1: Extract and Chunk
    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)

    # Step 2: Generate Embeddings
    embeddings = generate_embeddings(chunks)

    # Step 3: Format for Pinecone
    vectors_to_upsert = [
        {"id": f"embedding_{i}", "values": emb.tolist()}
        for i, emb in enumerate(embeddings)
    ]

    # Step 4: Store in Pinecone
    index = store_in_pinecone(vectors_to_upsert)

    # Step 5: Query
    while (True):
        print("input your query or type 'Exit' to exit:")
        query = input()
        if query == "Exit":
            break
        query_vector = query_embedding(query)
        query_results = index.query(vector=query_vector.tolist(), top_k=3, include_values=True)

    # Step 6: Retrieve Relevant Chunks
        numerical_ids = [int(match['id'].split('_')[-1]) for match in query_results['matches']]
        context = " ".join([chunks[i] for i in numerical_ids])

    # Step 7: Summarize with Gemini
        client = genai.Client(api_key="AIzaSyAEb5Sgx9x7NZqSjgq-iHK4JXvmx924cGo")
        response = client.models.generate_content(
           model="gemini-2.0-flash", contents=[query, context],config={"max_output_tokens": 500,'temperature':0.3}
        )
        print("\n--- Summary ---")
        print("ans", response.text)

if __name__ == "__main__":
    main()
