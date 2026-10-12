
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

from agents import create_investigator, create_final_agent, create_developer
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


def build_graph(repo_path):
    db = build_rag(repo_path)
    tools = create_tools(db, repo_path)

    investigator = create_investigator(tools)
    final_agent = create_final_agent()
    developer = create_developer()

    source = Path(repo_path).resolve()
    destination = source.parent / "repaired_repo"

    system_prompt = SystemMessage(
        content=(
            "You are a Python code investigator. "
            "Use search_code to find relevant source code. "
            "Explain the relevant function, likely bug, and affected file. "
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

        if not last_message.tool_calls:
            return "developer"

        if state["tool_count"] >= 3:
            return "final"

        return "tools"

    def tools_node(state: AgentState):
        results = ToolNode(tools).invoke(state)
        return {
            "messages": results["messages"],
            "tool_count": state["tool_count"] + len(
                state["messages"][-1].tool_calls
            )
        }

    def final_node(state: AgentState):
        messages = state["messages"]

        if messages[-1].type == "ai" and messages[-1].tool_calls:
            messages = messages[:-1]

        response = final_agent.invoke(
            [system_prompt] + messages + [
                HumanMessage(
                    content="Summarize the relevant code and likely bug."
                )
            ]
        )
        return {"messages": [response]}

    def developer_node(state: AgentState):
        attempt = state["attempts"] + 1

        # Copy original repository only on the first attempt
        if attempt == 1:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)

        files = {}
        for path in destination.rglob("*.py"):
            relative = path.relative_to(destination).as_posix()
            files[relative] = path.read_text(encoding="utf-8")

        original_request = state["messages"][0].content

        # Use investigator's findings and retrieved code
        investigation = state.get("investigation", "")

        if not investigation:
            investigation = "\n".join(
                str(message.content)
                for message in state["messages"][1:]
                if message.type in ("ai", "tool")
                and not getattr(message, "tool_calls", None)
            )

        has_tests = any(
            path.name.startswith("test_")
            and path.suffix == ".py"
            for path in destination.rglob("*.py")
        )

        code_context = "\n\n".join(
            f"FILE: {name}\n{code}"
            for name, code in files.items()
            if not name.startswith("test_")
        )

        fix = developer.invoke(
            f"""
You are a Python Developer Agent.

User request:
{original_request}

Investigator findings:
{investigation}

Repository code:
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

        if fix.filename not in files or fix.filename.startswith("test_"):
            raise ValueError("Developer selected an invalid source file")

        ast.parse(fix.code)

        target = destination / fix.filename
        old_code = files[fix.filename]
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
            "diff": diff,
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
        if state['attempts']==1:
            passed=False
            output += "\nSimulated failure for retry testing."

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
    graph.add_node("final", final_node)
    graph.add_node("developer", developer_node)
    graph.add_node("validator", validator_node)

    graph.add_edge(START, "investigator")
    graph.add_conditional_edges("investigator", route_agent)

    graph.add_edge("tools", "investigator")
    graph.add_edge("final", "developer")
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
            "investigation": ""
        },
        config={"recursion_limit": 30}
    )

    print("\nWorkflow completed.")
    print("Attempts:", result["attempts"])
    print("Validation:", "PASS" if result["test_passed"] else "FAIL")
    print(result["test_results"])
