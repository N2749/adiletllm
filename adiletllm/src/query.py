import chromadb
from transformers import pipeline
from langchain.schema.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from src.embedder import get_embedding_function
from src.config_reader import get


# == getting constants from config file ==
CHROMA_PATH     = get("database", "chroma_path")
COLLECTION_NAME = get("database", "collection_name")
EMBEDDING_MODEL = get("models", "embedder")
OLLAMA_MODEL    = get("models", "ollama")


PROMPT_TEMPLATE = """
Answer the question based only on the following context.
If the context does not contain the answer, say 'Answer not found'.

{context}

Answer the question based on the above context: {question}
"""


def main():
    question = "Is Kazakhstan secular country?"
    ask(question)

def test():
    return "heh"

def get_collection(chroma_path: str, collection_name: str):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = get_embedding_function(EMBEDDING_MODEL)
    return chroma_client .get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder
    )


def ask(query_text: str):

    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)

    # result is a dict that contains list of strings documents and metadata
    # https://docs.trychroma.com/getting-started#6.-inspect-results
    results = collection.query(query_texts=[query_text], n_results=5)

    context_text = "\n".join(results["documents"][0])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model=OLLAMA_MODEL)
    response_text = model.invoke(prompt)

    sources = [metadata.get("id", None) for metadata in results["metadatas"][0]]
    
    formatted_response = f"Response: {response_text}\nSources: {sources}\n"
    print(formatted_response)
    return formatted_response


if __name__ == "__main__":
    main()

