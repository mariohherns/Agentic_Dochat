"""Research agent that generates answers from document content using ChatOpenAI."""

import logging
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from llm.openai_llm import OPENAI_API_KEY, traceable

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Generate answers from document context using the ChatOpenAI model."""

    def __init__(self):
        """
        Initialize the research agent with the IBM WatsonX ModelInference.
        """
        # Initialize the OpenAI Model llm
        print("Initializing ResearchAgent with OPEN AI...")
        self.model = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=OPENAI_API_KEY,
            max_completion_tokens=80,  # Adjust based on desired response length
            temperature=0.2,  # Controls randomness; lower values make output more deterministic
        )

        print("ChatOpenAI LLM initialized successfully.")

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.
        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return a clear, concise, and factual short paragraph under 80 tokens from the context.
        
        **Question:** {question}
        **Context:**
        {context}
        **Provide your answer below:**
        """
        return prompt

    @traceable(run_type="llm", name="ResearchAgent.generate")
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(
            f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents."
        )
        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)

        print("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            print("Sending prompt to the model...")
            response = self.model.invoke([HumanMessage(content=prompt)])
            print("LLM response received.")
            draft_answer = response.content.strip() or "I cannot answer this question based on the provide"  # type: ignore
            logger.debug("LLM response: %s", draft_answer)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        print(f"Generated answer: {draft_answer}")
        return {"draft_answer": draft_answer, "context_used": context}
