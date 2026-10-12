
from langchain_core.tools import tool


def create_tools(db, repo_path):
    @tool
    def search_code(query: str) -> str:
        """Search the repository for relevant Python functions."""
        results = db.similarity_search_with_score(query, k=3)

        return "\n\n".join(
            f"File: {doc.metadata['source']}\n"
            f"Function: {doc.metadata.get('symbol', 'unknown')}\n"
            f"Lines: {doc.metadata.get('start_line', '?')}-{doc.metadata.get('end_line', '?')}\n"
            f"Distance: {score:.4f}\n"
            f"{doc.page_content}"
            for doc, score in results
        )

    return [search_code]
