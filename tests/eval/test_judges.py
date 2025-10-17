"""Tests for LLM judges."""

import pytest
from unittest.mock import MagicMock, patch
from retrieval_demo.eval.judges import RetrievalRelevanceJudge, GroundednessJudge


class TestRetrievalRelevanceJudge:
    """Tests for RetrievalRelevanceJudge."""

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

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_relevance_returns_score(
        self, mock_create_judge, retrieved_chunks
    ):
        """Test that judge returns relevance score."""
        # Mock the judge function to return a score
        mock_judge_fn = MagicMock(return_value={"score": 0.8})
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        assert score == 0.8
        # Verify judge function was called
        mock_judge_fn.assert_called_once()

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_relevance_uses_correct_parameters(
        self, mock_create_judge, retrieved_chunks
    ):
        """Test that judge calls judge_fn with correct parameters."""
        mock_judge_fn = MagicMock(return_value={"score": 0.9})
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Verify judge_fn was called with inputs and context
        call_kwargs = mock_judge_fn.call_args.kwargs
        assert call_kwargs["inputs"] == "What is the capital of France?"
        assert "Paris is the capital of France" in call_kwargs["context"]
        assert "Berlin is the capital of Germany" in call_kwargs["context"]

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_initializes_with_correct_model(self, mock_create_judge):
        """Test that judge initializes create_llm_as_judge with correct model."""
        mock_judge_fn = MagicMock(return_value={"score": 0.5})
        mock_create_judge.return_value = mock_judge_fn

        RetrievalRelevanceJudge(model="gpt-5-mini")

        # Verify create_llm_as_judge was called with correct model format
        call_kwargs = mock_create_judge.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-5-mini"
        assert call_kwargs["feedback_key"] == "retrieval_relevance"

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_invalid_score(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles invalid score responses."""
        mock_judge_fn = MagicMock(return_value={"score": "not a number"})
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on parse error
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_missing_score(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles missing score in response."""
        mock_judge_fn = MagicMock(return_value={})
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 when score is missing
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_api_error(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles API errors gracefully."""
        mock_judge_fn = MagicMock(side_effect=Exception("API Error"))
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on error
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_clamps_score_to_range(self, mock_create_judge, retrieved_chunks):
        """Test that judge clamps scores to 0-1 range."""
        # Test score > 1
        mock_judge_fn = MagicMock(return_value={"score": 1.5})
        mock_create_judge.return_value = mock_judge_fn

        judge = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )
        assert score == 1.0

        # Test score < 0 (need new judge instance)
        mock_judge_fn_negative = MagicMock(return_value={"score": -0.3})
        mock_create_judge.return_value = mock_judge_fn_negative

        judge2 = RetrievalRelevanceJudge(model="gpt-4o-mini")
        score = judge2.judge(
            question="What is the capital of France?", retrieved_chunks=retrieved_chunks
        )
        assert score == 0.0


class TestGroundednessJudge:
    """Tests for GroundednessJudge."""

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

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_groundedness_returns_score(
        self, mock_create_judge, retrieved_chunks
    ):
        """Test that judge returns groundedness score."""
        mock_judge_fn = MagicMock(return_value={"score": 0.95})
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        assert score == 0.95
        mock_judge_fn.assert_called_once()

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_groundedness_uses_correct_parameters(
        self, mock_create_judge, retrieved_chunks
    ):
        """Test that judge calls judge_fn with correct parameters."""
        mock_judge_fn = MagicMock(return_value={"score": 1.0})
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Verify judge_fn was called with outputs and context
        call_kwargs = mock_judge_fn.call_args.kwargs
        assert call_kwargs["outputs"] == "Paris is the capital of France."
        assert "Paris is the capital of France" in call_kwargs["context"]
        assert "Eiffel Tower" in call_kwargs["context"]

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_initializes_with_correct_model(self, mock_create_judge):
        """Test that judge initializes create_llm_as_judge with correct model."""
        mock_judge_fn = MagicMock(return_value={"score": 0.5})
        mock_create_judge.return_value = mock_judge_fn

        GroundednessJudge(model="gpt-5-mini")

        # Verify create_llm_as_judge was called with correct model format
        call_kwargs = mock_create_judge.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-5-mini"
        assert call_kwargs["feedback_key"] == "groundedness"

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_invalid_score(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles invalid score responses."""
        mock_judge_fn = MagicMock(return_value={"score": "invalid"})
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on parse error
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_missing_score(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles missing score in response."""
        mock_judge_fn = MagicMock(return_value={})
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 when score is missing
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_handles_api_error(self, mock_create_judge, retrieved_chunks):
        """Test that judge handles API errors gracefully."""
        mock_judge_fn = MagicMock(side_effect=Exception("API Error"))
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )

        # Should return 0.0 on error
        assert score == 0.0

    @patch("retrieval_demo.eval.judges.create_llm_as_judge")
    def test_judge_clamps_score_to_range(self, mock_create_judge, retrieved_chunks):
        """Test that judge clamps scores to 0-1 range."""
        # Test score > 1
        mock_judge_fn = MagicMock(return_value={"score": 1.8})
        mock_create_judge.return_value = mock_judge_fn

        judge = GroundednessJudge(model="gpt-4o-mini")
        score = judge.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )
        assert score == 1.0

        # Test score < 0 (need new judge instance)
        mock_judge_fn_negative = MagicMock(return_value={"score": -0.5})
        mock_create_judge.return_value = mock_judge_fn_negative

        judge2 = GroundednessJudge(model="gpt-4o-mini")
        score = judge2.judge(
            answer="Paris is the capital of France.", retrieved_chunks=retrieved_chunks
        )
        assert score == 0.0
