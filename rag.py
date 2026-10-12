
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



from langchain_openai import OpenAIEmbeddings

load_dotenv()

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )



embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def build_rag(repo_path):
    documents = []
    root = Path(repo_path).resolve()

    ignored = {".venv", "venv", ".git", "__pycache__", "node_modules"}

    for path in root.rglob("*.py"):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue

        code = path.read_text(encoding="utf-8", errors="replace")

        documents.append(
            Document(
                page_content=code,
                metadata={"source": str(path.relative_to(root))}
            )
        )

    if not documents:
        raise ValueError("No Python files found")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="codescope"
    )

    return db


if __name__ == "__main__":
    db = build_rag("demo_repo")

    results = db.similarity_search(
        "Find the function responsible for user authentication",
        k=3
    )

    for doc in results:
        print("\nFILE:", doc.metadata["source"])
        print(doc.page_content)
