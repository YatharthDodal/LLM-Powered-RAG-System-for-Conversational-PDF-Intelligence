import re
from PyPDF2 import PdfReader
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from google import genai


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_with_headings(pdf_path):
    reader = PdfReader(pdf_path)
    data = []
    for page in reader.pages:
        text = page.extract_text()
        lines = text.split("\n")
        for line in lines:
            if re.match(r"^[A-Z][A-Z\s]+$", line.strip()):  # Detect uppercase headings
                data.append({"heading": line.strip(), "content": ""})
            elif data:
                data[-1]["content"] += line.strip() + " "
    return data

def chunk_by_headings(data, max_chunk_size=500):
    chunks = []
    for section in data:
        heading = section["heading"]
        content = section["content"].split()
        for i in range(0, len(content), max_chunk_size):
            chunk = " ".join(content[i:i + max_chunk_size])
            chunks.append(f"{heading}\n{chunk}")
    return chunks


from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings


def put_into_pinecone(embeddings):
    pc = Pinecone(api_key="pcsk_5TWkve_Tj2RyCEZdgZxmHNd7Cn7soZZHxVL6AFy7zHNqzyGSWRoabWeZHFtfvwcVHWfTK9")
    index = pc.Index(host="https://pdfreadder-xlallq1.svc.aped-4627-b74a.pinecone.io")
    index.upsert(embeddings)
    return index


def query_embedding(query):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vector = model.encode([query], normalize_embeddings=True)[0]  # Get the first (and only) vector
    return query_vector

def vector_to_upsert(embeddings):
    vector=[
        {"id": f"embedding_{i}", "values": embedding.tolist()}  # Create a unique ID for each embedding
        for i, embedding in enumerate(embeddings)
    ]
    return vector

def query_search(query_vector, index):
    query_results = index.query(
        vector=query_vector.tolist(),  # Convert to list for Pinecone
        top_k=3,  # Number of similar vectors to retrieve
        include_values=True
    )
    return query_results

def numerical_value(query_results):
    numerical_ids = []
    for match in query_results['matches']:
        query_id = match['id']  # Get the ID of the match
        score = match['score']  # Get the similarity score
        values = match['values']

        numerical_id = query_id.split('_')[-1]
        numerical_ids.append(int(numerical_id))

    return numerical_ids

def joining_chunks(chunks, numerical_ids):
    # Ensure indices are within the valid range
    valid_ids = [i for i in numerical_ids if 0 <= i < len(chunks)]

   # if len(valid_ids) < len(numerical_ids):
    #    print(f"Warning: Some indices were out of bounds and have been ignored.")

    # Create context from valid indices only
    context = [chunks[i] for i in valid_ids]
    return context


def calling_gemini(query,context,chat):

    chat.send_message(context)
    response = chat.send_message(query)
    return response

def main():
    pdf_path = "Resume.pdf"

    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)
    embeddings = generate_embeddings(chunks)
    vectors_to_upsert = vector_to_upsert(embeddings)
    index=put_into_pinecone(vectors_to_upsert)
    client = genai.Client(api_key="AIzaSyAEb5Sgx9x7NZqSjgq-iHK4JXvmx924cGo")
    chat = client.chats.create(model="gemini-2.0-flash")

    while(True):
        query=input("You :")
        if query=="exit":
            break

        query_vector = query_embedding(query)
        query_results = query_search(query_vector, index)


        numerical_ids = numerical_value(query_results)

        context=joining_chunks(chunks,numerical_ids)

        response = calling_gemini(query,context,chat)
        print("Chatbot:",response.text)


if __name__ == "__main__":
    main()