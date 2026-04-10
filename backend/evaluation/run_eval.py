from langchain_core.documents import Document
from langsmith.evaluation import evaluate
from agents.research_agent import ResearchAgent

research_agent = ResearchAgent()

def run_research_agent(inputs: dict) -> dict:
    docs = [Document(page_content=inputs["context"])]
    return research_agent.generate(question=inputs["question"], documents=docs)

evaluate(
    run_research_agent,
    data="research_agent_dataset",
    experiment_prefix="research-agent-v1",
)

#NOTE run teh file by : python -m evaluation.run_eval