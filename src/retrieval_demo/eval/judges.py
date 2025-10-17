"""LLM judges for evaluating retrieval and answer quality."""

import logging
from typing import List, Dict, Any
from openevals import create_llm_as_judge
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT, RAG_GROUNDEDNESS_PROMPT

logger = logging.getLogger(__name__)


class RetrievalRelevanceJudge:
    """Judge for evaluating retrieval relevance using LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI model.

        Args:
            model: Model name to use for judging (e.g., "gpt-4o-mini", "gpt-5-mini")
        """
        # Create judge using openevals
        self.judge_fn = create_llm_as_judge(
            prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
            model=f"openai:{model}",
            feedback_key="retrieval_relevance",
        )

    def judge(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Judge the relevance of retrieved chunks to the question.

        Args:
            question: The question being asked
            retrieved_chunks: List of retrieved chunks with 'text' field

        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Format context from retrieved chunks
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

            # Call the judge with inputs and context
            result = self.judge_fn(inputs=question, context=context)

            # Extract score from result
            score = result.get("score", 0.0)

            # Ensure it's a float and clamp to 0-1 range
            return max(0.0, min(1.0, float(score)))

        except (ValueError, AttributeError, TypeError) as e:
            logger.error(f"Failed to parse relevance score: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error judging retrieval relevance: {e}")
            return 0.0


class GroundednessJudge:
    """Judge for evaluating answer groundedness using LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI model.

        Args:
            model: Model name to use for judging (e.g., "gpt-4o-mini", "gpt-5-mini")
        """
        # Create judge using openevals
        self.judge_fn = create_llm_as_judge(
            prompt=RAG_GROUNDEDNESS_PROMPT,
            model=f"openai:{model}",
            feedback_key="groundedness",
        )

    def judge(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Judge whether the answer is grounded in the retrieved chunks.

        Args:
            answer: The generated answer
            retrieved_chunks: List of retrieved chunks with 'text' field

        Returns:
            Groundedness score between 0.0 and 1.0
        """
        try:
            # Format context from retrieved chunks
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

            # Call the judge with outputs and context
            result = self.judge_fn(outputs=answer, context=context)

            # Extract score from result
            score = result.get("score", 0.0)

            # Ensure it's a float and clamp to 0-1 range
            return max(0.0, min(1.0, float(score)))

        except (ValueError, AttributeError, TypeError) as e:
            logger.error(f"Failed to parse groundedness score: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error judging groundedness: {e}")
            return 0.0
