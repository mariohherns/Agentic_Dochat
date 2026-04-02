"""
Create LangSmith datasets for all three agents:
  1. research_agent_dataset     – question + context → draft_answer
  2. verification_agent_dataset – answer + context  → verification_report
  3. relevance_checker_dataset  – question + context → classification label

Run once to seed your evaluation datasets, then use them with langsmith evaluate().
"""

import os
import logging
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── LangSmith client ──────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL") or os.getenv("LANGSMITH_ENDPOINT")

if not LANGSMITH_API_KEY:
    raise EnvironmentError("LANGSMITH_API_KEY is not set in your .env file.")

client = (
    Client(
        api_url=LANGSMITH_API_URL,
        api_key=LANGSMITH_API_KEY,
    )
    if LANGSMITH_API_URL
    else Client(api_key=LANGSMITH_API_KEY)
)


# ── Helper ────────────────────────────────────────────────────────────────────
def get_or_create_dataset(name: str, description: str):
    """Return existing dataset or create a fresh one."""
    if client.has_dataset(dataset_name=name):
        logger.info("[dataset] '%s' already exists – reusing it.", name)
        return client.read_dataset(dataset_name=name)
    logger.info("[dataset] Creating '%s' ...", name)
    return client.create_dataset(dataset_name=name, description=description)


def seed_examples(dataset, examples: list, label: str):
    """Add examples to a dataset and log progress."""
    for ex in examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata=ex.get("metadata", {}),
            dataset_id=dataset.id,
        )
    logger.info("[%s] Added %d examples.", label, len(examples))


# =============================================================================
# 1. ResearchAgent dataset
#    Agent  : ResearchAgent.generate(question, documents)
#    Input  : question (str) + context (str — pre-joined doc.page_content)
#    Output : draft_answer (str)
#
#    Test cases cover:
#      - Direct factual question with clear context
#      - Multi-sentence context requiring synthesis
#      - Numerical/date fact extraction
#      - Definition question
#      - Context that only partially answers the question
#      - Context in a domain-specific tone (legal, medical, technical)
#      - Question whose answer requires combining two context sentences
#      X Context that does NOT contain the answer (fallback expected)
# =============================================================================
RESEARCH_EXAMPLES = [
    # ── Direct factual ────────────────────────────────────────────────────────
    {
        "inputs": {
            "question": "What is the capital of France?",
            "context": (
                "France is a country in Western Europe. Its capital city is Paris, "
                "which is also its largest city and a major cultural centre."
            ),
        },
        "outputs": {"draft_answer": "The capital of France is Paris."},
        "metadata": {"category": "direct_fact", "difficulty": "easy"},
    },
    # ── Date / number extraction ──────────────────────────────────────────────
    {
        "inputs": {
            "question": "When was the Eiffel Tower built?",
            "context": (
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
                "It was constructed between 1887 and 1889 as the centerpiece of the 1889 World's Fair."
            ),
        },
        "outputs": {
            "draft_answer": "The Eiffel Tower was built between 1887 and 1889."
        },
        "metadata": {"category": "date_extraction", "difficulty": "easy"},
    },
    # ── Definition ────────────────────────────────────────────────────────────
    {
        "inputs": {
            "question": "What is photosynthesis?",
            "context": (
                "Photosynthesis is the process used by plants, algae, and some bacteria "
                "to convert sunlight, water, and carbon dioxide into glucose and oxygen."
            ),
        },
        "outputs": {
            "draft_answer": (
                "Photosynthesis is the process by which plants convert sunlight, "
                "water, and CO2 into glucose and oxygen."
            )
        },
        "metadata": {"category": "definition", "difficulty": "easy"},
    },
    # ── Multi-sentence synthesis ──────────────────────────────────────────────
    {
        "inputs": {
            "question": "Who invented the telephone and when?",
            "context": (
                "Alexander Graham Bell is widely credited with inventing the telephone in 1876. "
                "Bell was a Scottish-born inventor, scientist, and engineer. "
                "His work on the telegraph led to the development of the telephone."
            ),
        },
        "outputs": {
            "draft_answer": "Alexander Graham Bell invented the telephone in 1876."
        },
        "metadata": {"category": "synthesis", "difficulty": "medium"},
    },
    # ── Domain-specific: legal ────────────────────────────────────────────────
    {
        "inputs": {
            "question": "What does the contract say about termination notice?",
            "context": (
                "Section 12.3 of the Agreement states that either party may terminate this contract "
                "by providing thirty (30) days written notice to the other party. "
                "Termination does not affect any accrued rights or obligations."
            ),
        },
        "outputs": {
            "draft_answer": (
                "The contract requires thirty days written notice from either party to terminate, "
                "and termination does not affect accrued rights or obligations."
            )
        },
        "metadata": {"category": "domain_legal", "difficulty": "medium"},
    },
    # ── Domain-specific: medical ──────────────────────────────────────────────
    {
        "inputs": {
            "question": "What are the side effects of ibuprofen?",
            "context": (
                "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). "
                "Common side effects include stomach pain, heartburn, nausea, and headache. "
                "Serious side effects may include gastrointestinal bleeding and kidney problems."
            ),
        },
        "outputs": {
            "draft_answer": (
                "Common side effects of ibuprofen include stomach pain, heartburn, nausea, and headache. "
                "Serious side effects can include gastrointestinal bleeding and kidney problems."
            )
        },
        "metadata": {"category": "domain_medical", "difficulty": "medium"},
    },
    # ── Partial context (answer is inferable but not explicit) ────────────────
    {
        "inputs": {
            "question": "Is Python a compiled or interpreted language?",
            "context": (
                "Python is a high-level programming language known for its readability. "
                "Python code is executed line by line by the Python interpreter at runtime."
            ),
        },
        "outputs": {
            "draft_answer": "Python is an interpreted language, executed line by line at runtime."
        },
        "metadata": {"category": "inference_required", "difficulty": "medium"},
    },
    # ── Combining two context sentences ───────────────────────────────────────
    {
        "inputs": {
            "question": "What did Marie Curie discover and what awards did she receive?",
            "context": (
                "Marie Curie discovered two elements: polonium and radium. "
                "She was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911, "
                "making her the only person to win Nobel Prizes in two different sciences."
            ),
        },
        "outputs": {
            "draft_answer": (
                "Marie Curie discovered polonium and radium. She received the Nobel Prize in Physics in 1903 "
                "and the Nobel Prize in Chemistry in 1911."
            )
        },
        "metadata": {"category": "multi_fact", "difficulty": "hard"},
    },
    # ── Fallback: context does not contain the answer ─────────────────────────
    {
        "inputs": {
            "question": "What is the population of Tokyo?",
            "context": (
                "Tokyo is the capital of Japan and one of the most densely populated cities in the world. "
                "It is known for its technology industry, cuisine, and public transport system."
            ),
        },
        "outputs": {
            "draft_answer": "I cannot answer this question based on the provided documents."
        },
        "metadata": {"category": "fallback_no_answer", "difficulty": "hard"},
    },
]


# =============================================================================
# 2. VerificationAgent dataset
#    Agent  : VerificationAgent.check(answer, documents)
#    Input  : answer (str) + context (str)
#    Output : verification_report (str — exact format from format_verification_report)
#
#    Test cases cover:
#      - Fully supported, relevant answer
#      - Answer with a wrong date (contradiction)
#      - Answer that adds an unsupported claim
#      - Answer that is correct but off-topic (irrelevant)
#      - Partially supported answer
#      X Completely wrong answer (not supported, contradictions present)
#      X Empty/vague answer
# =============================================================================
VERIFICATION_EXAMPLES = [
    # ── Fully supported ───────────────────────────────────────────────────────
    {
        "inputs": {
            "answer": "The capital of France is Paris.",
            "context": "France is a country in Western Europe. Its capital city is Paris.",
        },
        "outputs": {
            "verification_report": (
                "**Supported:** YES\n"
                "**Unsupported Claims:** None\n"
                "**Contradictions:** None\n"
                "**Relevant:** YES\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "fully_supported", "difficulty": "easy"},
    },
    # ── Wrong date — contradiction ────────────────────────────────────────────
    {
        "inputs": {
            "answer": "Alexander Graham Bell invented the telephone in 1900.",
            "context": "Alexander Graham Bell invented the telephone in 1876.",
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** None\n"
                "**Contradictions:** The answer states 1900 but the context says 1876.\n"
                "**Relevant:** YES\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "contradiction_date", "difficulty": "easy"},
    },
    # ── Unsupported claim added ───────────────────────────────────────────────
    {
        "inputs": {
            "answer": "Photosynthesis converts sunlight into oxygen, glucose, and water vapour.",
            "context": (
                "Photosynthesis is the process used by plants to convert sunlight, "
                "water, and CO2 into glucose and oxygen."
            ),
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** water vapour as a product of photosynthesis\n"
                "**Contradictions:** None\n"
                "**Relevant:** YES\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "unsupported_claim", "difficulty": "medium"},
    },
    # ── Correct answer but off-topic context ──────────────────────────────────
    {
        "inputs": {
            "answer": "The Eiffel Tower is located in Paris.",
            "context": (
                "The Louvre is a historic palace in Paris that now serves as a world-renowned art museum. "
                "It houses thousands of works of art including the Mona Lisa."
            ),
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** Eiffel Tower location\n"
                "**Contradictions:** None\n"
                "**Relevant:** NO\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "irrelevant_context", "difficulty": "medium"},
    },
    # ── Partially supported ───────────────────────────────────────────────────
    {
        "inputs": {
            "answer": "Marie Curie won two Nobel Prizes and was born in Poland.",
            "context": (
                "Marie Curie was awarded the Nobel Prize in Physics in 1903 and "
                "the Nobel Prize in Chemistry in 1911."
            ),
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** born in Poland\n"
                "**Contradictions:** None\n"
                "**Relevant:** YES\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "partially_supported", "difficulty": "medium"},
    },
    # ── Completely wrong answer ───────────────────────────────────────────────
    {
        "inputs": {
            "answer": "Ibuprofen is an antibiotic used to treat bacterial infections.",
            "context": (
                "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce "
                "fever and treat pain or inflammation."
            ),
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** antibiotic, treats bacterial infections\n"
                "**Contradictions:** Context states ibuprofen is an NSAID, not an antibiotic.\n"
                "**Relevant:** YES\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "completely_wrong", "difficulty": "hard"},
    },
    # ── Vague / empty answer ──────────────────────────────────────────────────
    {
        "inputs": {
            "answer": "I cannot answer this question based on the provided documents.",
            "context": (
                "Tokyo is the capital of Japan and one of the most densely populated cities in the world."
            ),
        },
        "outputs": {
            "verification_report": (
                "**Supported:** NO\n"
                "**Unsupported Claims:** None\n"
                "**Contradictions:** None\n"
                "**Relevant:** NO\n"
                "**Additional Details:** None\n"
            )
        },
        "metadata": {"category": "fallback_answer", "difficulty": "easy"},
    },
]


# =============================================================================
# 3. RelevanceChecker dataset
#    Agent  : RelevanceChecker.check(question, retriever, k)
#    Input  : question (str) + context (str — simulates joined retriever output)
#    Output : classification (str) — one of: CAN_ANSWER, PARTIAL, NO_MATCH
#
#    Test cases cover:
#      - CAN_ANSWER — context fully addresses the question
#      - PARTIAL    — context mentions the topic but is incomplete
#      - NO_MATCH   — context is completely unrelated
#      - PARTIAL    — context is related but answers a different aspect
#      - CAN_ANSWER — technical/domain-specific context
#      X NO_MATCH   — misleadingly similar topic but wrong domain
# =============================================================================
RELEVANCE_EXAMPLES = [
    # ── CAN_ANSWER: clear, direct match ──────────────────────────────────────
    {
        "inputs": {
            "question": "What is the boiling point of water?",
            "context": (
                "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard "
                "atmospheric pressure (1 atm)."
            ),
        },
        "outputs": {"classification": "CAN_ANSWER"},
        "metadata": {"category": "can_answer_direct", "difficulty": "easy"},
    },
    # ── CAN_ANSWER: domain-specific ──────────────────────────────────────────
    {
        "inputs": {
            "question": "What does the contract say about liability?",
            "context": (
                "Section 9 of the Agreement limits liability to the total fees paid in the "
                "preceding twelve months. Neither party shall be liable for indirect or "
                "consequential damages."
            ),
        },
        "outputs": {"classification": "CAN_ANSWER"},
        "metadata": {"category": "can_answer_legal", "difficulty": "medium"},
    },
    # ── PARTIAL: topic mentioned but incomplete ───────────────────────────────
    {
        "inputs": {
            "question": "What are all the side effects of aspirin?",
            "context": (
                "Aspirin is commonly used to reduce fever and relieve pain. "
                "It can cause stomach irritation in some patients."
            ),
        },
        "outputs": {"classification": "PARTIAL"},
        "metadata": {"category": "partial_incomplete", "difficulty": "medium"},
    },
    # ── PARTIAL: related but answers a different aspect ───────────────────────
    {
        "inputs": {
            "question": "How does Python handle memory management?",
            "context": (
                "Python is a high-level interpreted programming language. "
                "It supports multiple programming paradigms and has a large standard library."
            ),
        },
        "outputs": {"classification": "PARTIAL"},
        "metadata": {"category": "partial_wrong_aspect", "difficulty": "medium"},
    },
    # ── PARTIAL: timeframe mentioned but details missing ──────────────────────
    {
        "inputs": {
            "question": "What were the key outcomes of the 2023 climate summit?",
            "context": (
                "The 2023 climate summit brought together world leaders to discuss emissions targets. "
                "Several nations pledged to increase renewable energy investment."
            ),
        },
        "outputs": {"classification": "PARTIAL"},
        "metadata": {"category": "partial_timeframe", "difficulty": "hard"},
    },
    # ── NO_MATCH: completely unrelated ────────────────────────────────────────
    {
        "inputs": {
            "question": "What is the GDP of Germany?",
            "context": (
                "The Amazon rainforest covers approximately 5.5 million square kilometres. "
                "It is home to an estimated 10% of all species on Earth."
            ),
        },
        "outputs": {"classification": "NO_MATCH"},
        "metadata": {"category": "no_match_unrelated", "difficulty": "easy"},
    },
    # ── NO_MATCH: similar topic, wrong domain ─────────────────────────────────
    {
        "inputs": {
            "question": "What is the interest rate set by the European Central Bank?",
            "context": (
                "The Bank of England raised interest rates to 5.25% in 2023 in response "
                "to elevated inflation levels in the United Kingdom."
            ),
        },
        "outputs": {"classification": "NO_MATCH"},
        "metadata": {"category": "no_match_wrong_entity", "difficulty": "hard"},
    },
    # ── CAN_ANSWER: numerical/statistical ────────────────────────────────────
    {
        "inputs": {
            "question": "How many employees does the company have?",
            "context": (
                "As of December 2023, the company employed 4,320 full-time staff across "
                "its offices in New York, London, and Singapore."
            ),
        },
        "outputs": {"classification": "CAN_ANSWER"},
        "metadata": {"category": "can_answer_numeric", "difficulty": "easy"},
    },
]


# =============================================================================
# Seed all datasets
# =============================================================================
research_ds = get_or_create_dataset(
    name="research_agent_dataset",
    description=(
        "Evaluation dataset for ResearchAgent.generate() — question + context → draft_answer. "
        "Covers: direct facts, date extraction, definitions, synthesis, domain-specific, "
        "partial context, multi-fact, and fallback cases."
    ),
)
seed_examples(research_ds, RESEARCH_EXAMPLES, "research_agent_dataset")

verification_ds = get_or_create_dataset(
    name="verification_agent_dataset",
    description=(
        "Evaluation dataset for VerificationAgent.check() — answer + context → verification_report. "
        "Covers: fully supported, contradictions, unsupported claims, irrelevant context, "
        "partial support, completely wrong, and fallback answer cases."
    ),
)
seed_examples(verification_ds, VERIFICATION_EXAMPLES, "verification_agent_dataset")

relevance_ds = get_or_create_dataset(
    name="relevance_checker_dataset",
    description=(
        "Evaluation dataset for RelevanceChecker.check() — question + context → CAN_ANSWER | PARTIAL | NO_MATCH. "
        "Covers: direct match, domain-specific, incomplete context, wrong aspect, "
        "unrelated context, and wrong entity cases."
    ),
)
seed_examples(relevance_ds, RELEVANCE_EXAMPLES, "relevance_checker_dataset")

logger.info("Done. Visit https://smith.langchain.com to inspect your datasets.")
