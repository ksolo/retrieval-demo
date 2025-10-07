"""Tests for metrics calculation."""

import pytest
from eval.metrics import RetrievalMetrics, AggregateMetrics, MetricsCalculator


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_retrieval_metrics_creation(self):
        """Test creating a RetrievalMetrics instance."""
        metrics = RetrievalMetrics(
            eval_id="rag12000_1",
            retrieval_relevance=0.9,
            groundedness=0.85,
            recall_at_1=True,
            recall_at_3=True,
            recall_at_5=True,
            precision_at_k=0.8,
            latency_ms=150.5,
        )

        assert metrics.eval_id == "rag12000_1"
        assert metrics.retrieval_relevance == 0.9
        assert metrics.groundedness == 0.85
        assert metrics.recall_at_1 is True
        assert metrics.recall_at_3 is True
        assert metrics.recall_at_5 is True
        assert metrics.precision_at_k == 0.8
        assert metrics.latency_ms == 150.5


class TestAggregateMetrics:
    """Tests for AggregateMetrics dataclass."""

    def test_aggregate_metrics_creation(self):
        """Test creating an AggregateMetrics instance."""
        metrics = AggregateMetrics(
            avg_retrieval_relevance=0.85,
            avg_groundedness=0.80,
            avg_recall_at_1=0.75,
            avg_recall_at_3=0.90,
            avg_recall_at_5=0.95,
            avg_precision_at_k=0.78,
            avg_latency_ms=125.3,
            total_samples=100,
        )

        assert metrics.avg_retrieval_relevance == 0.85
        assert metrics.avg_groundedness == 0.80
        assert metrics.avg_recall_at_1 == 0.75
        assert metrics.avg_recall_at_3 == 0.90
        assert metrics.avg_recall_at_5 == 0.95
        assert metrics.avg_precision_at_k == 0.78
        assert metrics.avg_latency_ms == 125.3
        assert metrics.total_samples == 100


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return [
            RetrievalMetrics(
                eval_id="rag12000_1",
                retrieval_relevance=0.9,
                groundedness=0.85,
                recall_at_1=True,
                recall_at_3=True,
                recall_at_5=True,
                precision_at_k=0.8,
                latency_ms=100.0,
            ),
            RetrievalMetrics(
                eval_id="rag12000_2",
                retrieval_relevance=0.7,
                groundedness=0.75,
                recall_at_1=False,
                recall_at_3=True,
                recall_at_5=True,
                precision_at_k=0.6,
                latency_ms=150.0,
            ),
            RetrievalMetrics(
                eval_id="rag12000_3",
                retrieval_relevance=0.8,
                groundedness=0.70,
                recall_at_1=True,
                recall_at_3=True,
                recall_at_5=True,
                precision_at_k=0.9,
                latency_ms=200.0,
            ),
        ]

    def test_calculate_aggregate_metrics(self, sample_metrics):
        """Test calculating aggregate metrics from sample metrics."""
        calculator = MetricsCalculator()
        aggregate = calculator.calculate_aggregate(sample_metrics)

        # Check averages
        assert aggregate.avg_retrieval_relevance == pytest.approx(
            0.8
        )  # (0.9 + 0.7 + 0.8) / 3
        assert aggregate.avg_groundedness == pytest.approx(
            0.7667, rel=1e-3
        )  # (0.85 + 0.75 + 0.70) / 3
        assert aggregate.avg_recall_at_1 == pytest.approx(0.6667, rel=1e-3)  # 2/3
        assert aggregate.avg_recall_at_3 == 1.0  # 3/3
        assert aggregate.avg_recall_at_5 == 1.0  # 3/3
        assert aggregate.avg_precision_at_k == pytest.approx(
            0.7667, rel=1e-3
        )  # (0.8 + 0.6 + 0.9) / 3
        assert aggregate.avg_latency_ms == 150.0  # (100 + 150 + 200) / 3
        assert aggregate.total_samples == 3

    def test_calculate_aggregate_empty_list(self):
        """Test that empty list returns zeros."""
        calculator = MetricsCalculator()
        aggregate = calculator.calculate_aggregate([])

        assert aggregate.avg_retrieval_relevance == 0.0
        assert aggregate.avg_groundedness == 0.0
        assert aggregate.avg_recall_at_1 == 0.0
        assert aggregate.avg_recall_at_3 == 0.0
        assert aggregate.avg_recall_at_5 == 0.0
        assert aggregate.avg_precision_at_k == 0.0
        assert aggregate.avg_latency_ms == 0.0
        assert aggregate.total_samples == 0

    def test_calculate_by_category(self, sample_metrics):
        """Test calculating metrics grouped by category."""
        # Add category metadata
        categories = ["Science", "Math", "Science"]

        calculator = MetricsCalculator()
        by_category = calculator.calculate_by_category(sample_metrics, categories)

        # Should have 2 categories
        assert len(by_category) == 2
        assert "Science" in by_category
        assert "Math" in by_category

        # Science: samples 0 and 2
        science_metrics = by_category["Science"]
        assert science_metrics.total_samples == 2
        assert science_metrics.avg_retrieval_relevance == pytest.approx(
            0.85
        )  # (0.9 + 0.8) / 2
        assert science_metrics.avg_recall_at_1 == 1.0  # 2/2

        # Math: sample 1
        math_metrics = by_category["Math"]
        assert math_metrics.total_samples == 1
        assert math_metrics.avg_retrieval_relevance == 0.7
        assert math_metrics.avg_recall_at_1 == 0.0  # 0/1

    def test_calculate_by_category_mismatched_lengths(self, sample_metrics):
        """Test that mismatched lengths raises error."""
        calculator = MetricsCalculator()

        with pytest.raises(ValueError, match="must have same length"):
            calculator.calculate_by_category(
                sample_metrics, ["Science", "Math"]
            )  # Only 2 categories for 3 metrics

    def test_calculate_recall_at_k(self):
        """Test calculating recall@k from retrieved chunks."""
        calculator = MetricsCalculator()

        # Mock retrieved chunks with document_id metadata
        retrieved_chunks = [
            {"metadata": {"document_id": "rag12000_5"}},  # Rank 1: matches
            {"metadata": {"document_id": "rag12000_10"}},  # Rank 2: doesn't match
            {"metadata": {"document_id": "rag12000_15"}},  # Rank 3: doesn't match
            {"metadata": {"document_id": "rag12000_20"}},  # Rank 4: doesn't match
            {"metadata": {"document_id": "rag12000_25"}},  # Rank 5: doesn't match
        ]

        recall_1, recall_3, recall_5 = calculator.calculate_recall_at_k(
            retrieved_chunks=retrieved_chunks, correct_eval_id="rag12000_5"
        )

        assert recall_1 is True
        assert recall_3 is True
        assert recall_5 is True

    def test_calculate_recall_at_k_not_in_top_1(self):
        """Test recall when correct doc is not at rank 1."""
        calculator = MetricsCalculator()

        retrieved_chunks = [
            {"metadata": {"document_id": "rag12000_10"}},  # Rank 1: doesn't match
            {"metadata": {"document_id": "rag12000_5"}},  # Rank 2: matches
            {"metadata": {"document_id": "rag12000_15"}},
            {"metadata": {"document_id": "rag12000_20"}},
            {"metadata": {"document_id": "rag12000_25"}},
        ]

        recall_1, recall_3, recall_5 = calculator.calculate_recall_at_k(
            retrieved_chunks=retrieved_chunks, correct_eval_id="rag12000_5"
        )

        assert recall_1 is False
        assert recall_3 is True
        assert recall_5 is True

    def test_calculate_recall_at_k_not_in_top_3(self):
        """Test recall when correct doc is not in top 3."""
        calculator = MetricsCalculator()

        retrieved_chunks = [
            {"metadata": {"document_id": "rag12000_10"}},
            {"metadata": {"document_id": "rag12000_15"}},
            {"metadata": {"document_id": "rag12000_20"}},
            {"metadata": {"document_id": "rag12000_5"}},  # Rank 4: matches
            {"metadata": {"document_id": "rag12000_25"}},
        ]

        recall_1, recall_3, recall_5 = calculator.calculate_recall_at_k(
            retrieved_chunks=retrieved_chunks, correct_eval_id="rag12000_5"
        )

        assert recall_1 is False
        assert recall_3 is False
        assert recall_5 is True

    def test_calculate_recall_at_k_not_found(self):
        """Test recall when correct doc is not in results."""
        calculator = MetricsCalculator()

        retrieved_chunks = [
            {"metadata": {"document_id": "rag12000_10"}},
            {"metadata": {"document_id": "rag12000_15"}},
            {"metadata": {"document_id": "rag12000_20"}},
            {"metadata": {"document_id": "rag12000_25"}},
            {"metadata": {"document_id": "rag12000_30"}},
        ]

        recall_1, recall_3, recall_5 = calculator.calculate_recall_at_k(
            retrieved_chunks=retrieved_chunks, correct_eval_id="rag12000_5"
        )

        assert recall_1 is False
        assert recall_3 is False
        assert recall_5 is False

    def test_calculate_precision_at_k(self):
        """Test calculating precision@k."""
        calculator = MetricsCalculator()

        # 3 out of 5 chunks are relevant
        relevant_flags = [True, False, True, True, False]
        precision = calculator.calculate_precision_at_k(relevant_flags, k=5)

        assert precision == 0.6  # 3/5

    def test_calculate_precision_at_k_all_relevant(self):
        """Test precision when all chunks are relevant."""
        calculator = MetricsCalculator()

        relevant_flags = [True, True, True, True, True]
        precision = calculator.calculate_precision_at_k(relevant_flags, k=5)

        assert precision == 1.0

    def test_calculate_precision_at_k_none_relevant(self):
        """Test precision when no chunks are relevant."""
        calculator = MetricsCalculator()

        relevant_flags = [False, False, False, False, False]
        precision = calculator.calculate_precision_at_k(relevant_flags, k=5)

        assert precision == 0.0
