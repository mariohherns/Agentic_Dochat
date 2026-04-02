"""Relevance checker agent that validates whether document content matches the user's question."""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from llm.openai_llm import OPENAI_API_KEY, traceable

logger = logging.getLogger(__name__)


class RelevanceChecker:
    """Classify whether retrieved document content is relevant to the user's question."""

    def __init__(self):
        # Initialize the OpenAI Model llm
        print("Initializing RelevanceChecker with OPEN AI...")

        self.model = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=OPENAI_API_KEY,
            max_completion_tokens=10,
            temperature=0,
        )

    def generate_prompt(self, question: str, document_content: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.
        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.
        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
        **Question:** {question}
        **Passages:** {document_content}
        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """
        return prompt

    @traceable(run_type="llm", name="RelevanceChecker.check")
    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.
        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        logger.debug(
            "RelevanceChecker.check called with question=%s and k=%s",
            question,
            k,
        )

        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)

        if not top_docs:
            logger.debug(
                "No documents returned from retriever.invoke(). Classifying as NO_MATCH."
            )
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, document_content)

        # Call the LLM (ChatOpenAI returns an AIMessage)
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            llm_response = response.content.strip().upper().upper()  # type: ignore
            logger.debug("LLM response: %s", llm_response)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Error during model inference: %s", e)
            return "NO_MATCH"

        print(f"Checker response: {llm_response}")

        # Validate the response
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug("Classification recognized as '%s'.", llm_response)
            classification = llm_response

        return classification
