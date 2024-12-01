import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import pipeline

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# CONSTANTS
# == DATA ==
DATA_PATH = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
# == CHROMA ==
CHROMA_PATH = "chroma"
COLLECTION_NAME = "legal-docs"
# == MODELS ==
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANSWERING_MODEL = "deepset/roberta-base-squad2"

def main():
    docs = load_documents()
    chunks = split_documents(docs)
    collection = save(chunks)

    question = "What is the state language of the Republic of Kazakhstan"
    query_results = collection.query(query_texts=[question], n_results=1)
    context = query_results["documents"][0][0]

    response = ask(question, context)
    printResults(question, context, response)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

def save(chunks: list[Document]):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = get_embedding_function()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = collection.get(include=[])  # IDs are always included by default
    merge_chunks(existing_items, chunks_with_ids, collection)

    return collection
    

def merge_chunks(existing_items, chunks_with_ids, collection):
    # Add or Update the documents.
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        documents = [chunk.page_content for chunk in new_chunks]
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        metadatas = [chunk.metadata for chunk in new_chunks]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=new_chunk_ids
        )
        # dbclient.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def ask(question, context):
    generator = pipeline("question-answering", model=ANSWERING_MODEL, tokenizer=ANSWERING_MODEL)
    response = generator(question=question, context=context)

    return response


def printResults(question, context, response):
    print(f"Query: {question}")
    print("Retrieved Context:", context)

    if response is not None:
        print(f"Response '{response['answer']}'")
    else:
        print("Response is undefined")


if __name__ == "__main__":
    main()

# =========================================

# # Step 2: Create a Chroma collection for the knowledge base
# collection_name = "knowledge_base"
# collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedder)

# # Step 4: Add knowledge base entries
# knowledge_entries = [
#     {"id": "1", "text": "Python is a programming language known for its simplicity."},
#     {"id": "2", "text": "RAG stands for Retrieval-Augmented Generation, which enhances language models with external data."},
# ]

# # Add documents to the collection with embeddings
# documents = [entry["text"] for entry in knowledge_entries]
# doc_ids = [entry["id"] for entry in knowledge_entries]
# collection.add(documents=documents, ids=doc_ids)

# # Step 5: Query ChromaDB
# query_text = "What is RAG?"
# query_results = collection.query(query_texts=[query_text], n_results=1)

# # Step 6: Use a Hugging Face model to generate an answer using the retrieved knowledge
# retrieved_text = query_results["documents"][0][0]
# model_name = "deepset/roberta-base-squad2"
# generator = pipeline("question-answering", model=model_name, tokenizer=model_name)
# response = generator(question=query_text, context=retrieved_text)

# # Print the result
# print("Query:", query_text)
# print("Retrieved Context:", retrieved_text)
# print(f"Response '{response['answer']}'")
# # if response is not None:
# #     print("Generated Answer:", response[0]['generated_text'])
# # else:
# #     print("Response is undefined")
