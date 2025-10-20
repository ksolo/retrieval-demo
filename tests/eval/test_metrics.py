"""Tests for metrics calculation."""

from retrieval_demo.eval.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

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
