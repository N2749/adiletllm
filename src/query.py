import chromadb
from transformers import pipeline
from langchain.schema.document import Document

from embedder import get_embedding_function
from config_reader import get

# == getting constants from config file ==
CHROMA_PATH     = get("database", "chroma_path")
COLLECTION_NAME = get("database", "collection_name")
EMBEDDING_MODEL = get("models", "embedder")
ANSWERING_MODEL = get("models", "answerer")


def main():
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)
    question = "Is it legal to sell ice cream?"
    query_results = collection.query(query_texts=[question], n_results=1)
    context = query_results["documents"][0][0]

    response = ask(question, context)
    printResults(question, context, response)


def get_collection(chroma_path: str, collection_name: str):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = get_embedding_function(EMBEDDING_MODEL)
    return chroma_client .get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder
    )


def ask(question, context):
    generator = pipeline(
        "question-answering",
        model=ANSWERING_MODEL,
        tokenizer=ANSWERING_MODEL
    )
    return generator(question=question, context=context)


def printResults(question, context, response):
    print(f"Query: {question}")
    print("Retrieved Context:", context)

    if response is not None:
        print(f"Response '{response['answer']}'")
    else:
        print("Response is undefined")


if __name__ == "__main__":
    main()

