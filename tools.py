from langchain_core.tools import tool

def create_tools(db, repo_path):
    @tool
    def search_code(query: str) -> str:
        """Search the repository for relevant source code."""
        docs = db.similarity_search(query, k=3)
        return "\n\n".join(
            f"File: {d.metadata['source']}\n{d.page_content}"
            for d in docs
        )

    return [search_code]