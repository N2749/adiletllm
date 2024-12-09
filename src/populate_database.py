import chromadb
from chromadb.config import Settings
from transformers import pipeline

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from config_reader import get


# == getting constants from config file ==
CHROMA_PATH     = get("database", "chroma_path")
COLLECTION_NAME = get("database", "collection_name")
DATA_PATH       = get("database", "data_path")
CHUNK_SIZE      = get("database", "chunk_size")
CHUNK_OVERLAP   = get("database", "chunk_overlap")
EMBEDDING_MODEL = get("models", "embedder")


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    collection = save(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)


def save(chunks: list[Document]):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = get_embedding_function()
    collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedder
    )

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


if __name__ == "__main__":
    main()

