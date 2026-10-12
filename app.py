
import streamlit as st
from langchain_core.messages import HumanMessage
from graph import build_graph

st.set_page_config(
    page_title="AI Code Repair Assistant",
    layout="wide"
)

st.title("AI Code Repair Assistant")
st.write("Agentic AI + RAG + Automated Code Repair")

bug = st.text_area(
    "Describe the bug",
    "Fix the authentication function to handle invalid credentials safely.",
    height=120
)

if st.button("Analyze and Fix", type="primary"):
    if bug.strip():
        with st.spinner("Analyzing, repairing and testing..."):
            graph = build_graph("demo_repo")

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=bug)],
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

        st.subheader("Agent Result")

        st.write(f"Retrieval evaluations: {result['retrieval_attempts']}/3")

        if result["evidence_sufficient"]:
            st.success("Retrieved evidence is sufficient.")
        else:
            st.warning("Retrieved evidence was insufficient.")

        st.write(result["evidence_reason"])

        st.write("Relevant files:", result["relevant_files"])
        st.write(f"Repair attempts: {result['attempts']}/3")

        st.subheader("Code Changes")
        if result["diff"]:
            st.code(result["diff"], language="diff")
        else:
            st.info("No code changes reported.")

        st.subheader("Validation Results")

        if result["attempts"] == 0:
            st.warning("No repair attempted because no relevant files were identified.")
        elif result["test_passed"]:
            st.success("All tests passed!")
        else:
            st.warning("Validation did not pass after repair attempts.")

        if result["test_results"]:
            st.code(result["test_results"], language="text")
