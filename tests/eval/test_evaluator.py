"""Tests for strategy evaluators."""

import pytest
from unittest.mock import Mock, MagicMock
from eval.evaluator import StrategyEvaluator
from eval.metrics import RetrievalMetrics, AggregateMetrics


class TestStrategyEvaluator:
    """Tests for StrategyEvaluator."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_relevance_judge(self):
        """Create a mock relevance judge."""
        mock = Mock()
        mock.judge.return_value = 0.9
        return mock

    @pytest.fixture
    def mock_groundedness_judge(self):
        """Create a mock groundedness judge."""
        mock = Mock()
        mock.judge.return_value = 0.85
        return mock

    @pytest.fixture
    def sample_eval_data(self):
        """Sample evaluation data."""
        return [
            {
                "eval_id": "rag12000_1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "Geography",
            },
            {
                "eval_id": "rag12000_2",
                "question": "What is 2+2?",
                "answer": "4",
                "category": "Math",
            },
        ]

    def test_evaluator_initialization(
        self, mock_graph, mock_relevance_judge, mock_groundedness_judge
    ):
        """Test evaluator initialization."""
        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        assert evaluator.graph == mock_graph
        assert evaluator.relevance_judge == mock_relevance_judge
        assert evaluator.groundedness_judge == mock_groundedness_judge

    def test_evaluate_sample_returns_metrics(
        self,
        mock_graph,
        mock_relevance_judge,
        mock_groundedness_judge,
        sample_eval_data,
    ):
        """Test evaluating a single sample returns RetrievalMetrics."""
        # Mock graph response
        mock_documents = [
            Mock(
                page_content="Paris is the capital of France.",
                metadata={"document_id": "rag12000_1"},
            ),
            Mock(
                page_content="The Eiffel Tower is in Paris.",
                metadata={"document_id": "rag12000_1"},
            ),
            Mock(
                page_content="Berlin is capital of Germany.",
                metadata={"document_id": "rag12000_2"},
            ),
        ]

        mock_response_message = Mock()
        mock_response_message.content = "Paris is the capital of France."

        mock_graph.invoke.return_value = {
            "documents": mock_documents,
            "messages": [
                Mock(),
                Mock(),
                mock_response_message,
            ],  # system, user, assistant
        }

        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        sample = sample_eval_data[0]
        metrics = evaluator.evaluate_sample(
            sample=sample,
            collection_name="chunks_recursive_500_50",
            retrieval_strategy="semantic",
            topk=5,
        )

        # Verify metrics structure
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.eval_id == "rag12000_1"
        assert metrics.retrieval_relevance == 0.9
        assert metrics.groundedness == 0.85
        assert isinstance(metrics.latency_ms, float)

        # Verify graph was invoked correctly
        mock_graph.invoke.assert_called_once()
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["query"] == "What is the capital of France?"
        assert call_args["collection"] == "chunks_recursive_500_50"
        assert call_args["retrieval_strategy"] == "semantic"
        assert call_args["topk"] == 5

    def test_evaluate_sample_calculates_recall_at_k(
        self,
        mock_graph,
        mock_relevance_judge,
        mock_groundedness_judge,
        sample_eval_data,
    ):
        """Test that evaluate_sample correctly calculates recall@k metrics."""
        # Mock: correct document at rank 2
        mock_documents = [
            Mock(
                page_content="Berlin", metadata={"document_id": "rag12000_10"}
            ),  # Rank 1: wrong
            Mock(
                page_content="Paris", metadata={"document_id": "rag12000_1"}
            ),  # Rank 2: correct!
            Mock(
                page_content="London", metadata={"document_id": "rag12000_15"}
            ),  # Rank 3: wrong
        ]

        mock_graph.invoke.return_value = {
            "documents": mock_documents,
            "messages": [Mock(), Mock(), Mock(content="Paris")],
        }

        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        sample = sample_eval_data[0]
        metrics = evaluator.evaluate_sample(
            sample=sample,
            collection_name="chunks_recursive_500_50",
            retrieval_strategy="semantic",
            topk=5,
        )

        # Correct doc at rank 2
        assert metrics.recall_at_1 is False  # Not at rank 1
        assert metrics.recall_at_3 is True  # In top 3
        assert metrics.recall_at_5 is True  # In top 5

    def test_evaluate_strategy_returns_aggregate_metrics(
        self,
        mock_graph,
        mock_relevance_judge,
        mock_groundedness_judge,
        sample_eval_data,
    ):
        """Test evaluating full strategy returns aggregate metrics."""
        # Mock graph responses
        mock_documents = [
            Mock(page_content="Paris", metadata={"document_id": "rag12000_1"}),
        ]

        mock_graph.invoke.return_value = {
            "documents": mock_documents,
            "messages": [Mock(), Mock(), Mock(content="Paris")],
        }

        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        per_sample_metrics, aggregate, _ = evaluator.evaluate_strategy(
            samples=sample_eval_data,
            collection_name="chunks_recursive_500_50",
            retrieval_strategy="semantic",
            topk=5,
        )

        # Verify per-sample metrics
        assert len(per_sample_metrics) == 2
        assert all(isinstance(m, RetrievalMetrics) for m in per_sample_metrics)

        # Verify aggregate metrics
        assert isinstance(aggregate, AggregateMetrics)
        assert aggregate.total_samples == 2
        assert aggregate.avg_retrieval_relevance == 0.9  # All mocked to 0.9
        assert aggregate.avg_groundedness == 0.85  # All mocked to 0.85

    def test_evaluate_strategy_by_category(
        self,
        mock_graph,
        mock_relevance_judge,
        mock_groundedness_judge,
        sample_eval_data,
    ):
        """Test evaluating strategy with category breakdown."""
        mock_documents = [
            Mock(page_content="Answer", metadata={"document_id": "rag12000_1"}),
        ]

        mock_graph.invoke.return_value = {
            "documents": mock_documents,
            "messages": [Mock(), Mock(), Mock(content="Answer")],
        }

        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        per_sample_metrics, aggregate, by_category = evaluator.evaluate_strategy(
            samples=sample_eval_data,
            collection_name="chunks_recursive_500_50",
            retrieval_strategy="semantic",
            topk=5,
            include_category_breakdown=True,
        )

        # Verify category breakdown
        assert "Geography" in by_category
        assert "Math" in by_category
        assert by_category["Geography"].total_samples == 1
        assert by_category["Math"].total_samples == 1

    def test_evaluate_sample_handles_judge_errors(
        self,
        mock_graph,
        mock_relevance_judge,
        mock_groundedness_judge,
        sample_eval_data,
    ):
        """Test that evaluator handles judge errors gracefully."""
        # Mock judge errors
        mock_relevance_judge.judge.side_effect = Exception("Judge error")
        mock_groundedness_judge.judge.side_effect = Exception("Judge error")

        mock_documents = [
            Mock(page_content="Paris", metadata={"document_id": "rag12000_1"}),
        ]

        mock_graph.invoke.return_value = {
            "documents": mock_documents,
            "messages": [Mock(), Mock(), Mock(content="Paris")],
        }

        evaluator = StrategyEvaluator(
            graph=mock_graph,
            relevance_judge=mock_relevance_judge,
            groundedness_judge=mock_groundedness_judge,
        )

        sample = sample_eval_data[0]
        metrics = evaluator.evaluate_sample(
            sample=sample,
            collection_name="chunks_recursive_500_50",
            retrieval_strategy="semantic",
            topk=5,
        )

        # Should return 0.0 scores on error
        assert metrics.retrieval_relevance == 0.0
        assert metrics.groundedness == 0.0
