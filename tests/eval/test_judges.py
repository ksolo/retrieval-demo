"""Tests for LLM judges."""

import pytest
from unittest.mock import MagicMock, patch
from retrieval_demo.eval.judges import RetrievalRelevanceJudge, GroundednessJudge


class TestRetrievalRelevanceJudge:
    """Tests for RetrievalRelevanceJudge."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch("retrieval_demo.eval.judges.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def retrieved_chunks(self):
        """Sample retrieved chunks."""
        return [
            {
                "text": "Paris is the capital of France.",
                "metadata": {"document_id": "rag12000_1"},
            },
            {
                "text": "The Eiffel Tower is in Paris.",
                "metadata": {"document_id": "rag12000_1"},
            },
            {
                "text": "Berlin is the capital of Germany.",
                "metadata": {"document_id": "rag12000_2"},
            },
        ]

    def test_judge_relevance_returns_score(self, mock_openai_client, retrieved_chunks):
        """Test that judge returns relevance score."""
        # Mock response with score
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.8"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        assert score == 0.8
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_judge_relevance_uses_correct_prompt(
        self, mock_openai_client, retrieved_chunks
    ):
        """Test that judge uses RAG_RETRIEVAL_RELEVANCE_PROMPT."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Verify prompt contains expected elements
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        # Should have system and user message
        assert len(messages) >= 2

        # Check that context chunks are included
        user_message = messages[-1]["content"]
        assert "Paris is the capital of France" in user_message
        assert "What is the capital of France?" in user_message

    def test_judge_handles_invalid_score(self, mock_openai_client, retrieved_chunks):
        """Test that judge handles invalid score responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not a number"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on parse error
        assert score == 0.0

    def test_judge_handles_api_error(self, mock_openai_client, retrieved_chunks):
        """Test that judge handles API errors gracefully."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on error
        assert score == 0.0


class TestGroundednessJudge:
    """Tests for GroundednessJudge."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch("retrieval_demo.eval.judges.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def retrieved_chunks(self):
        """Sample retrieved chunks."""
        return [
            {
                "text": "Paris is the capital of France.",
                "metadata": {"document_id": "rag12000_1"},
            },
            {
                "text": "The Eiffel Tower is in Paris.",
                "metadata": {"document_id": "rag12000_1"},
            },
        ]

    def test_judge_groundedness_returns_score(
        self, mock_openai_client, retrieved_chunks
    ):
        """Test that judge returns groundedness score."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.95"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        assert score == 0.95
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_judge_groundedness_uses_correct_prompt(
        self, mock_openai_client, retrieved_chunks
    ):
        """Test that judge uses RAG_GROUNDEDNESS_PROMPT."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1.0"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = GroundednessJudge(model="gpt-4o-mini")
        judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Verify prompt contains expected elements
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        # Should have system and user message
        assert len(messages) >= 2

        # Check that context and answer are included
        user_message = messages[-1]["content"]
        assert "Paris is the capital of France" in user_message

    def test_judge_handles_invalid_score(self, mock_openai_client, retrieved_chunks):
        """Test that judge handles invalid score responses."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "invalid"
        mock_openai_client.chat.completions.create.return_value = mock_response

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on parse error
        assert score == 0.0

    def test_judge_handles_api_error(self, mock_openai_client, retrieved_chunks):
        """Test that judge handles API errors gracefully."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on error
        assert score == 0.0

    def test_judge_with_optional_api_key(self, mock_openai_client, retrieved_chunks):
        """Test that judge can be initialized with optional API key."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Initialize with explicit API key
        judge = GroundednessJudge(model="gpt-4o-mini", api_key="test-key")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        assert score == 0.9
