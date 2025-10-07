"""LLM judges for evaluating retrieval and answer quality."""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT, RAG_GROUNDEDNESS_PROMPT

logger = logging.getLogger(__name__)


class RetrievalRelevanceJudge:
    """Judge for evaluating retrieval relevance using LLM."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize with OpenAI model.

        Args:
            model: OpenAI model to use for judging
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

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

            # Use openevals RAG_RETRIEVAL_RELEVANCE_PROMPT
            messages = [
                {"role": "system", "content": RAG_RETRIEVAL_RELEVANCE_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0
            )

            # Parse score from response
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)

            # Clamp to 0-1 range
            return max(0.0, min(1.0, score))

        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse relevance score: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error judging retrieval relevance: {e}")
            return 0.0


class GroundednessJudge:
    """Judge for evaluating answer groundedness using LLM."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize with OpenAI model.

        Args:
            model: OpenAI model to use for judging
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

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

            # Use openevals RAG_GROUNDEDNESS_PROMPT
            messages = [
                {"role": "system", "content": RAG_GROUNDEDNESS_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer: {answer}"},
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0
            )

            # Parse score from response
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)

            # Clamp to 0-1 range
            return max(0.0, min(1.0, score))

        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse groundedness score: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error judging groundedness: {e}")
            return 0.0
