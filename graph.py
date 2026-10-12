
from typing import Annotated, TypedDict
import ast
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents import (
    create_investigator,
    create_developer,
    create_evaluator,
)
from rag import build_rag
from tools import create_tools


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_count: int
    diff: str
    test_results: str
    test_passed: bool
    attempts: int
    investigation: str
    relevant_files: list[str]
    evidence_sufficient: bool
    evidence_reason: str
    next_query: str
    retrieval_attempts: int


def build_graph(repo_path):
    db = build_rag(repo_path)
    tools = create_tools(db, repo_path)

    investigator = create_investigator(tools)
    developer = create_developer()
    evaluator = create_evaluator()

    source = Path(repo_path).resolve()
    destination = source.parent / "repaired_repo"

    system_prompt = SystemMessage(
        content=(
            "You are a Python code investigator. "
            "Use search_code to find relevant source code. "
            "Identify the likely faulty function and affected files. "
            "Do not repeatedly search for the same information."
        )
    )

    def agent_node(state: AgentState):
        response = investigator.invoke(
            [system_prompt] + state["messages"]
        )
        return {"messages": [response]}

    def route_agent(state: AgentState):
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            if state["tool_count"] >= 3:
                return "evaluate"
            return "tools"

        return "evaluate"

    def tools_node(state: AgentState):
        results = ToolNode(tools).invoke(state)

        return {
            "messages": results["messages"],
            "tool_count": state["tool_count"] + len(
                state["messages"][-1].tool_calls
            )
        }

    def evaluate_node(state: AgentState):
        original_request = state["messages"][0].content

        retrieved = "\n\n".join(
            str(message.content)
            for message in state["messages"]
            if message.type == "tool"
        )

        if not retrieved.strip():
            return {
                "evidence_sufficient": False,
                "evidence_reason": "No code was retrieved.",
                "next_query": str(original_request),
                "relevant_files": [],
                "retrieval_attempts": state["retrieval_attempts"] + 1
            }

        evaluation = evaluator.invoke(
            f"""
You are evaluating retrieved Python code for a bug repair task.

User request:
{original_request}

Retrieved code:
{retrieved}

Determine whether the retrieved code is sufficient to
identify the likely faulty function and affected source files.

Mark sufficient=True only when the evidence supports
a concrete diagnosis.

Return relevant_files using exact file paths shown in
the retrieved code.

If evidence is insufficient, provide a different,
more specific search query in next_query.

Do not invent filenames.
"""
        )

        valid_files = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*.py")
        }

        relevant_files = [
            name for name in evaluation.relevant_files
            if name in valid_files
            and not Path(name).name.startswith("test_")
        ]

        print("\nEVIDENCE EVALUATION:")
        print("Sufficient:", evaluation.sufficient)
        print("Reason:", evaluation.reason)
        print("Relevant files:", relevant_files)

        return {
            "evidence_sufficient": evaluation.sufficient,
            "evidence_reason": evaluation.reason,
            "next_query": evaluation.next_query,
            "relevant_files": relevant_files,
            "retrieval_attempts": state["retrieval_attempts"] + 1
        }

    def route_evaluation(state: AgentState):
        if state["evidence_sufficient"] and state["relevant_files"]:
            return "developer"

        if state["retrieval_attempts"] >= 3:
            return "developer" if state["relevant_files"] else END

        if state["tool_count"] >= 3:
            return "developer" if state["relevant_files"] else END

        return "refine"

    def refine_node(state: AgentState):
        query = state["next_query"].strip()

        if not query:
            query = str(state["messages"][0].content)

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The retrieved evidence was insufficient. "
                        f"Reason: {state['evidence_reason']}\n"
                        f"Search again using this query: {query}\n"
                        "Use search_code to retrieve missing evidence."
                    )
                )
            ]
        }

    def developer_node(state: AgentState):
        attempt = state["attempts"] + 1

        # Copy the repository only on the first repair attempt
        if attempt == 1:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)

        all_files = {
            path.relative_to(destination).as_posix():
            path.read_text(encoding="utf-8")
            for path in destination.rglob("*.py")
        }

        # Only send RAG-selected source files to the Developer
        selected_files = {
            name: all_files[name]
            for name in state["relevant_files"]
            if name in all_files
        }

        if not selected_files:
            raise ValueError("No relevant source files selected for repair")

        original_request = state["messages"][0].content

        investigation = "\n".join(
            str(message.content)
            for message in state["messages"][1:]
            if message.type == "ai"
            and not getattr(message, "tool_calls", None)
        )

        has_tests = any(
            Path(name).name.startswith("test_")
            for name in all_files
        )

        code_context = "\n\n".join(
            f"FILE: {name}\n{code}"
            for name, code in selected_files.items()
        )

        fix = developer.invoke(
            f"""
You are a Python Developer Agent.

User request:
{original_request}

Investigator findings:
{investigation}

Evidence evaluation:
{state["evidence_reason"]}

Relevant repository code:
{code_context}

Previous test results:
{state["test_results"] if attempt > 1 else "First attempt"}

Attempt: {attempt}/3

Fix the reported bug.
Change only one existing Python source file.
Return its relative filename and COMPLETE corrected code.
Preserve existing behavior unless the fix requires changes.

Tests already exist: {has_tests}

If tests exist, return an empty test_code string.
If no tests exist, generate 2-3 minimal pytest tests
for the corrected function in test_code.
Use actual function names and expected behavior.
Do not use network calls or external services.
"""
        )

        if fix.filename not in selected_files:
            raise ValueError("Developer selected a file outside retrieved context")

        ast.parse(fix.code)

        target = destination / fix.filename
        old_code = selected_files[fix.filename]
        target.write_text(fix.code, encoding="utf-8")

        if not has_tests and fix.test_code.strip():
            ast.parse(fix.test_code)

            test_path = destination / f"test_{Path(fix.filename).stem}.py"
            test_path.write_text(fix.test_code, encoding="utf-8")

            print("Generated tests:", test_path)

        diff = "".join(
            difflib.unified_diff(
                old_code.splitlines(keepends=True),
                fix.code.splitlines(keepends=True),
                fromfile=f"before/{fix.filename}",
                tofile=f"after/{fix.filename}"
            )
        )

        print(f"\nDEVELOPER AGENT - Attempt {attempt}/3")
        print(fix.explanation)
        print("\nCODE CHANGES:")
        print(diff or "No changes")

        return {
            "messages": [
                HumanMessage(
                    content=f"Developer completed attempt {attempt}: {fix.explanation}"
                )
            ],
            "diff": diff if diff else state["diff"],
            "attempts": attempt,
            "investigation": investigation
        }

    def validator_node(state: AgentState):
        try:
            for file in destination.rglob("*.py"):
                ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError as error:
            return {
                "test_results": f"Syntax validation failed: {error}",
                "test_passed": False
            }

        test_files = list(destination.rglob("test_*.py"))

        if not test_files:
            return {
                "test_results": "No tests available. Validation incomplete.",
                "test_passed": False
            }

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=destination,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout + "\n" + result.stderr
        passed = result.returncode == 0

        print(f"\nVALIDATOR - Attempt {state['attempts']}/3")
        print(output)
        print("PASS" if passed else "FAIL")

        return {
            "test_results": output,
            "test_passed": passed
        }

    def route_validator(state: AgentState):
        if state["test_passed"] or state["attempts"] >= 3:
            return END
        return "developer"

    graph = StateGraph(AgentState)

    graph.add_node("investigator", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("developer", developer_node)
    graph.add_node("validator", validator_node)

    graph.add_edge(START, "investigator")
    graph.add_conditional_edges("investigator", route_agent)
    graph.add_edge("tools", "investigator")
    graph.add_conditional_edges("evaluate", route_evaluation)
    graph.add_edge("refine", "investigator")
    graph.add_edge("developer", "validator")
    graph.add_conditional_edges("validator", route_validator)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph("demo_repo")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Fix the authentication function so invalid credentials are handled safely."
                )
            ],
            "tool_count": 0,
            "diff": "",
            "test_results": "",
            "test_passed": False,
            "attempts": 0,
            "investigation": "",
            "relevant_files": [],
            "evidence_sufficient": False,
            "evidence_reason": "",
            "next_query": "",
            "retrieval_attempts": 0
        },
        config={"recursion_limit": 40}
    )

    print("\nWorkflow completed.")
    print("Retrieval attempts:", result["retrieval_attempts"])
    print("Evidence sufficient:", result["evidence_sufficient"])
    print("Relevant files:", result["relevant_files"])
    print("Repair attempts:", result["attempts"])
    print("Validation:", "PASS" if result["test_passed"] else "FAIL")
    print(result["test_results"])
