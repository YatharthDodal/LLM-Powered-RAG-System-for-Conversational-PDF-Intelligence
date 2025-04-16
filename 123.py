import re
from PyPDF2 import PdfReader
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


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




def main():
    pdf_path = "Resume.pdf"

    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)
    embeddings = generate_embeddings(chunks)
    vectors_to_upsert = [
        {"id": f"embedding_{i}", "values": embedding.tolist()}  # Create a unique ID for each embedding
        for i, embedding in enumerate(embeddings)
    ]
    index=put_into_pinecone(vectors_to_upsert)

    query = "explain Significance of Data Types"
    query_vector = query_embedding(query)
    query_results = index.query(
        vector=query_vector.tolist(),  # Convert to list for Pinecone
        top_k=3,  # Number of similar vectors to retrieve
        include_values=True
    )
    print("chunks", len(chunks))
    print("embedding", len(embeddings))

    numerical_ids = []
    for match in query_results['matches']:
        query_id = match['id']  # Get the ID of the match
        score = match['score']  # Get the similarity score
        values = match['values']


        numerical_id = query_id.split('_')[-1]
        numerical_ids.append(int(numerical_id))
        print(f"Query ID: {query_id}, Numerical ID: {numerical_id}")

    print(numerical_ids)
    context=[chunks[i] for i in numerical_ids]
    context=" ".join(context)
    print(context)

    client=genai.Client(api_key="AIzaSyAEb5Sgx9x7NZqSjgq-iHK4JXvmx924cGo")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["give me summary in 100 words of given context", context]
    )
    print("ans",response.text)


if __name__ == "__main__":
    main()