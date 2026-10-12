
import ast
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

IGNORED = {".venv", "venv", ".git", "__pycache__", "node_modules"}


def chunk_by_function(code, rel_path):
    """AST chunking: one chunk per function/method, with file + name added."""
    tree = ast.parse(code)
    lines = code.splitlines()
    chunks = []

    def add(node, name):
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        chunks.append(
            Document(
                page_content=f"# File: {rel_path}\n# Function: {name}\n{body}",
                metadata={
                    "source": rel_path,
                    "symbol": name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                },
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, f"{node.name}.{item.name}")

    return chunks


def build_rag(repo_path):
    root = Path(repo_path).resolve()
    chunks = []

    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in IGNORED for part in relative.parts):
            continue
        if path.name.startswith("test_"):
            continue

        code = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_by_function(code, relative.as_posix()))

    if not chunks:
        raise ValueError("No functions found")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="codescope",
    )


if __name__ == "__main__":
    db = build_rag("demo_repo")

    for doc in db.similarity_search("Find the function responsible for user authentication", k=3):
        m = doc.metadata
        print(f"\nFILE: {m['source']}  SYMBOL: {m['symbol']}  LINES: {m['start_line']}-{m['end_line']}")
        print(doc.page_content)