import chromadb
import os
import re
from chromadb.config import Settings
from transformers import pipeline

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from config_reader import get
from embedder import get_embedding_function
from pdf_txt_converter import convert_dir
from utils import simple_progress_bar


# == getting constants from config file ==
CHROMA_PATH     = get("database", "chroma_path")
COLLECTION_NAME = get("database", "collection_name")
DATA_PATH       = get("database", "data_path")
CHUNK_SIZE      = int(get("database", "chunk_size"))        # to avoid type err
CHUNK_OVERLAP   = int(get("database", "chunk_overlap"))     # to avoid type err
EMBEDDING_MODEL = get("models", "embedder")


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    collection = save(chunks)


def load_documents():
    convert_dir(DATA_PATH, DATA_PATH)
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        total = len(files)
        for i, file in enumerate(files):
            simple_progress_bar(i, total, prefix="Checking pdf files for conversion to txt")
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Create a Document with the content and metadata
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path}
            ))
    return documents
# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)


def process_in_batches(data, batch_size=5461):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def save(chunks: list[Document]):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = get_embedding_function(EMBEDDING_MODEL)
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

    # Only add documents that don"t exist in the DB.
    new_chunks = []
    total_chunks = len(chunks_with_ids)
    for id, chunk in enumerate(chunks_with_ids):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        documents = [chunk.page_content for chunk in new_chunks]
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        metadatas = [chunk.metadata for chunk in new_chunks]

        # print(f"new_chunks[0] {new_chunks[0]}")
        # print(f"metadatas[0]  {metadatas[0]}")

        process_in_batches(
            col=collection,
            docs=documents,
            metas=metadatas,
            ids=new_chunk_ids
        )

    else:
        print("No new documents to add")


def process_in_batches(col, docs, metas, ids, batch_size=5461):
    assert len(docs) == len(metas)
    assert len(docs) == len(ids)
    for i in range(0, len(docs), batch_size):
        print(f"Processing batch {i+1} ({i}, {i+batch_size})")
        col.add(
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )


def calculate_chunk_ids(chunks):

    current_chunk_index = 0
    total_chunks = len(chunks)
    source = None

    # For making legal reference
    last_law = None
    last_section = None
    last_chapter = None
    last_article = None
    last_clause = None

    for i, chunk in enumerate(chunks):

        simple_progress_bar(i + 1, total_chunks, prefix="Calculating chunk IDs")
        current_chunk_index += 1

        text = chunk.page_content
        # Indicates that file has changed
        if source != chunk.metadata.get('source'):
            source = chunk.metadata.get('source')

            # One document contains only one law, so we need to change the
            # `last_law` varibale only if the file changes
            last_law = last_law or get_law(text)

        last_chapter = last_chapter or get_chapter(text)
        last_section = last_section or get_section(text)
        last_article = last_article or get_article(text)
        last_clause  = last_clause  or get_clause(text)

        legal_ref = ""
        legal_ref = legal_ref + (       last_law     if last_law     else '')
        legal_ref = legal_ref + (', ' + last_chapter if last_chapter else '')
        legal_ref = legal_ref + (', ' + last_section if last_section else '')
        legal_ref = legal_ref + (', ' + last_article if last_article else '')
        legal_ref = legal_ref + (', ' + last_clause  if last_clause  else '')

        # Calculate metadata
        chunk.metadata["id"] = f"{chunk.metadata.get('source')}:{current_chunk_index}"
        chunk.metadata["legal_ref"] = legal_ref


    return chunks


def get_law(text):
    codex_match = re.search(r"^(.*?)(?:\n|$)", text, re.IGNORECASE | re.MULTILINE)
    return codex_match.group(1).strip() if codex_match else None


def get_section(text):
    section_match = re.search(r"^(?!.*Footnote\.).*?section\s+(\d+|[IVXLCDM]+)\.?\s+(.*?)\s+(?=Article|Chapter)", text, re.IGNORECASE | re.MULTILINE)
    section_title = section_match.group(1).strip() if section_match else None
    return f"Section {section_title}" if section_title else None


def get_chapter(text):
    chapter_match = re.search(r"^(?!.*Footnote\.).*?Chapter\s+(\d+|[IVXLCDM]+)\.?\s+(.*?)\s+(?=Article)", text)
    chapter_title = chapter_match.group(1).strip() if chapter_match else None
    return f"Chapter {chapter_title}" if chapter_title else None


def get_article(text):
    article_match = re.search(r"Article\s+(\d+)\.?\s*(.+)?", text, re.IGNORECASE | re.MULTILINE)
    if not article_match: return None

    article_number = article_match.group(1)
    article_title = article_match.group(2).strip() if article_match.group(2) else None
    return f"Article {article_number} {article_title}"


def get_clause(text):
    item_match = re.search(r"(\d+)\)\s+(.+)", text, re.IGNORECASE | re.MULTILINE)
    if not item_match: return None

    item_number = f"Item {item_match.group(1)}"
    item_title = item_match.group(2).strip()
    return f"Clause {item_number}"

if __name__ == "__main__":
    main()

