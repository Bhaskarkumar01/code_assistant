
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


class CodeFix(BaseModel):
    filename: str = Field(
        description="Relative Python filename to update, e.g. auth.py"
    )
    code: str = Field(
        description="Complete corrected Python source code"
    )
    explanation: str = Field(
        description="Short explanation of the fix"
    )
    test_code: str = Field(
        description="2-3 minimal pytest tests if no tests exist; otherwise an empty string"
    )


def create_investigator(tools):
    return llm.bind_tools(tools)


def create_final_agent():
    return llm


def create_developer():
    return llm.with_structured_output(CodeFix)
