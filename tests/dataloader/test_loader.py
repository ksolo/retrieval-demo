"""Tests for RAGDatasetLoader eval methods."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from retrieval_demo.dataloader.data.loader import RAGDatasetLoader


class TestRAGDatasetLoaderEvalMethods:
    """Tests for eval-specific methods in RAGDatasetLoader."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock HuggingFace dataset."""
        mock_ds = MagicMock()

        # Mock train split
        mock_train = MagicMock()

        # Store data
        data = [
            {"context": f"Context {i}", "question": f"Q{i}?", "answer": f"A{i}"}
            for i in range(20)
        ]

        # Support len()
        mock_train.__len__ = lambda self: len(data)

        # Support indexing: train_data[5]
        mock_train.__getitem__ = lambda self, idx: data[idx]

        # Support iteration
        mock_train.__iter__ = lambda self: iter(data)

        # Mock dataset info for versioning
        mock_info = MagicMock()
        mock_info.download_checksums = {
            'hf://datasets/test@abc123def456/data.parquet': {'num_bytes': 1000}
        }
        mock_info.dataset_size = 45472997
        mock_train.info = mock_info

        mock_ds.__getitem__ = lambda self, key: mock_train if key == 'train' else None

        return mock_ds

    @pytest.fixture
    def loader(self):
        """Create a RAGDatasetLoader instance."""
        return RAGDatasetLoader()

    def test_get_dataset_version_returns_commit_hash(self, loader, mock_dataset):
        """Test that get_dataset_version extracts git commit hash."""
        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            version = loader.get_dataset_version()
            assert version == "abc123def456"

    def test_get_dataset_version_falls_back_to_size(self, loader, mock_dataset):
        """Test fallback to dataset size when no checksums available."""
        # Remove checksums
        mock_dataset['train'].info.download_checksums = None

        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            version = loader.get_dataset_version()
            assert version == "45472997"

    def test_get_categorized_stratified_sample_returns_correct_structure(self, loader, mock_dataset):
        """Test that stratified sampling returns correct data structure."""
        # Mock categorized dataset
        mock_categorized_ds = Mock()
        categorized_samples = [
            {"original_index": 1, "context": "C1", "question": "Q1?", "answer": "A1", "category": "Cat1"},
            {"original_index": 2, "context": "C2", "question": "Q2?", "answer": "A2", "category": "Cat2"},
            {"original_index": 3, "context": "C3", "question": "Q3?", "answer": "A3", "category": "Cat1"},
            {"original_index": 4, "context": "C4", "question": "Q4?", "answer": "A4", "category": "Cat2"},
            {"original_index": 5, "context": "C5", "question": "Q5?", "answer": "A5", "category": "Cat1"},
        ]
        mock_categorized_ds.get_or_create_categories.return_value = categorized_samples

        # Mock categorizer
        mock_categorizer = Mock()

        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            with patch.object(loader, 'get_dataset_version', return_value="test_version"):
                with patch('retrieval_demo.dataloader.data.loader.CategorizedDataset', return_value=mock_categorized_ds):
                    result = loader.get_categorized_stratified_sample(
                        samples_per_category=2,
                        cache_path="/tmp/cache.json",
                        categorizer=mock_categorizer
                    )

        # Should have sampled 2 per category = 4 samples
        assert len(result) == 4

        # Check structure
        for item in result:
            assert "eval_id" in item
            assert "original_index" in item
            assert "category" in item
            assert "context" in item
            assert "question" in item
            assert "answer" in item

            # Check eval_id format
            assert item["eval_id"].startswith("rag12000_")

    def test_get_categorized_stratified_sample_distributes_evenly(self, loader, mock_dataset):
        """Test that stratified sampling distributes samples evenly across categories."""
        mock_categorized_ds = Mock()
        categorized_samples = [
            {"original_index": i, "context": f"C{i}", "question": f"Q{i}?", "answer": f"A{i}", "category": f"Cat{i % 3}"}
            for i in range(30)
        ]
        mock_categorized_ds.get_or_create_categories.return_value = categorized_samples
        mock_categorizer = Mock()

        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            with patch.object(loader, 'get_dataset_version', return_value="test_version"):
                with patch('retrieval_demo.dataloader.data.loader.CategorizedDataset', return_value=mock_categorized_ds):
                    result = loader.get_categorized_stratified_sample(
                        samples_per_category=5,
                        cache_path="/tmp/cache.json",
                        categorizer=mock_categorizer
                    )

        # Count samples per category
        category_counts = {}
        for item in result:
            cat = item["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Each category should have exactly 5 samples
        for count in category_counts.values():
            assert count == 5

    def test_get_categorized_stratified_sample_handles_insufficient_samples(self, loader, mock_dataset):
        """Test behavior when a category has fewer samples than requested."""
        mock_categorized_ds = Mock()
        categorized_samples = [
            {"original_index": 1, "context": "C1", "question": "Q1?", "answer": "A1", "category": "Cat1"},
            {"original_index": 2, "context": "C2", "question": "Q2?", "answer": "A2", "category": "Cat1"},
            {"original_index": 3, "context": "C3", "question": "Q3?", "answer": "A3", "category": "Cat2"},
            # Cat2 only has 1 sample, but we'll request 5
        ]
        mock_categorized_ds.get_or_create_categories.return_value = categorized_samples
        mock_categorizer = Mock()

        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            with patch.object(loader, 'get_dataset_version', return_value="test_version"):
                with patch('retrieval_demo.dataloader.data.loader.CategorizedDataset', return_value=mock_categorized_ds):
                    result = loader.get_categorized_stratified_sample(
                        samples_per_category=5,
                        cache_path="/tmp/cache.json",
                        categorizer=mock_categorizer
                    )

        # Should return what's available: 2 from Cat1, 1 from Cat2
        assert len(result) == 3

    def test_get_samples_by_ids_returns_correct_samples(self, loader, mock_dataset):
        """Test that get_samples_by_ids returns samples matching the requested IDs."""
        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            result = loader.get_samples_by_ids(["rag12000_5", "rag12000_10", "rag12000_15"])

        assert len(result) == 3

        # Check that we got the right samples
        assert result[0]["eval_id"] == "rag12000_5"
        assert result[0]["original_index"] == 5
        assert result[0]["context"] == "Context 5"

        assert result[1]["eval_id"] == "rag12000_10"
        assert result[1]["original_index"] == 10

        assert result[2]["eval_id"] == "rag12000_15"
        assert result[2]["original_index"] == 15

    def test_get_samples_by_ids_handles_invalid_ids(self, loader, mock_dataset):
        """Test that invalid IDs are skipped."""
        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            result = loader.get_samples_by_ids([
                "rag12000_5",
                "invalid_id",
                "rag12000_999",  # Out of range
                "rag12000_10"
            ])

        # Should only return valid IDs that exist
        assert len(result) == 2
        assert result[0]["eval_id"] == "rag12000_5"
        assert result[1]["eval_id"] == "rag12000_10"

    def test_get_samples_by_ids_preserves_order(self, loader, mock_dataset):
        """Test that returned samples maintain requested order."""
        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            result = loader.get_samples_by_ids(["rag12000_15", "rag12000_5", "rag12000_10"])

        # Order should match request order
        assert result[0]["original_index"] == 15
        assert result[1]["original_index"] == 5
        assert result[2]["original_index"] == 10

    def test_original_index_tracking(self, loader, mock_dataset):
        """Test that original_index is preserved correctly during categorization."""
        mock_categorized_ds = Mock()
        # Simulate categorization that skips some samples
        categorized_samples = [
            {"original_index": 2, "context": "C2", "question": "Q2?", "answer": "A2", "category": "Cat1"},
            {"original_index": 5, "context": "C5", "question": "Q5?", "answer": "A5", "category": "Cat1"},
            {"original_index": 7, "context": "C7", "question": "Q7?", "answer": "A7", "category": "Cat2"},
        ]
        mock_categorized_ds.get_or_create_categories.return_value = categorized_samples
        mock_categorizer = Mock()

        with patch.object(loader, 'load_dataset', return_value=mock_dataset):
            with patch.object(loader, 'get_dataset_version', return_value="test_version"):
                with patch('retrieval_demo.dataloader.data.loader.CategorizedDataset', return_value=mock_categorized_ds):
                    result = loader.get_categorized_stratified_sample(
                        samples_per_category=2,
                        cache_path="/tmp/cache.json",
                        categorizer=mock_categorizer
                    )

        # Each result should have original_index that can be used to retrieve it later
        for item in result:
            assert isinstance(item["original_index"], int)
            assert item["eval_id"] == f"rag12000_{item['original_index']}"
