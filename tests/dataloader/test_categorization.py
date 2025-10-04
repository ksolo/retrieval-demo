"""Tests for document categorization module."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from retrieval_demo.dataloader.categorization import Categorizer, CategorizedDataset


class TestCategorizer:
    """Tests for the Categorizer class."""

    def test_categorize_document_returns_valid_category(self, mock_openai_client):
        """Test that categorize returns a valid category string."""
        with mock_openai_client as mock:
            mock.set_response("Science")

            categorizer = Categorizer(model="gpt-5-mini")

            result = categorizer.categorize(
                context="The mitochondria is the powerhouse of the cell.",
                question="What is the function of mitochondria?"
            )

            assert result == "Science"
            assert mock.mock_client.chat.completions.create.called

    def test_categorize_document_handles_api_error(self, mock_openai_client):
        """Test that API errors return None to skip the document."""
        with mock_openai_client as mock:
            mock.set_error(Exception("API Error"))

            categorizer = Categorizer(model="gpt-5-mini")

            result = categorizer.categorize(
                context="Some context",
                question="Some question"
            )

            assert result is None

    def test_categorize_document_strips_whitespace(self, mock_openai_client):
        """Test that category response is stripped of whitespace."""
        with mock_openai_client as mock:
            mock.set_response("  Technology  \n")

            categorizer = Categorizer(model="gpt-5-mini")

            result = categorizer.categorize(
                context="AI and machine learning",
                question="What is AI?"
            )

            assert result == "Technology"

    def test_categorize_empty_response_returns_none(self, mock_openai_client):
        """Test that empty response returns None."""
        with mock_openai_client as mock:
            mock.set_response("   ")

            categorizer = Categorizer(model="gpt-5-mini")

            result = categorizer.categorize(
                context="Some context",
                question="Some question"
            )

            assert result is None


class TestCategorizedDataset:
    """Tests for the CategorizedDataset class."""

    @pytest.fixture
    def temp_cache_path(self, tmp_path):
        """Create a temporary cache file path."""
        return tmp_path / "categorization_cache.json"

    @pytest.fixture
    def mock_categorizer(self):
        """Create a mock categorizer."""
        categorizer = Mock(spec=Categorizer)
        return categorizer

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return [
            {"context": "Science content 1", "question": "Q1?", "answer": "A1"},
            {"context": "Tech content 1", "question": "Q2?", "answer": "A2"},
            {"context": "History content 1", "question": "Q3?", "answer": "A3"},
            {"context": "Science content 2", "question": "Q4?", "answer": "A4"},
            {"context": "Tech content 2", "question": "Q5?", "answer": "A5"},
        ]

    def test_get_or_create_categories_creates_new_on_empty_cache(
        self, temp_cache_path, mock_categorizer, sample_dataset
    ):
        """Test that categories are created when cache is empty."""
        mock_categorizer.categorize.side_effect = [
            "Science", "Technology", "History", "Science", "Technology"
        ]

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer
        )

        result = categorized_dataset.get_or_create_categories(
            dataset=sample_dataset,
            dataset_version="v1.0"
        )

        # Should have categorized all 5 samples
        assert len(result) == 5
        assert all("category" in item for item in result)
        assert result[0]["category"] == "Science"
        assert result[1]["category"] == "Technology"
        assert result[2]["category"] == "History"

        # Should have called categorizer 5 times
        assert mock_categorizer.categorize.call_count == 5

        # Cache should exist
        assert temp_cache_path.exists()

    def test_get_or_create_categories_uses_cache_on_version_match(
        self, temp_cache_path, mock_categorizer, sample_dataset
    ):
        """Test that cached categories are used when version matches."""
        # Pre-populate cache
        cache_data = {
            "dataset_version": {
                "id": "v1.0",
                "last_updated": "2025-10-04T12:00:00"
            },
            "samples": [
                {"index": 0, "category": "Science"},
                {"index": 1, "category": "Technology"},
                {"index": 2, "category": "History"},
                {"index": 3, "category": "Science"},
                {"index": 4, "category": "Technology"},
            ]
        }
        temp_cache_path.write_text(json.dumps(cache_data))

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer
        )

        result = categorized_dataset.get_or_create_categories(
            dataset=sample_dataset,
            dataset_version="v1.0"
        )

        # Should have used cache, not called categorizer
        assert mock_categorizer.categorize.call_count == 0

        # Should have correct categories
        assert len(result) == 5
        assert result[0]["category"] == "Science"
        assert result[1]["category"] == "Technology"

    def test_get_or_create_categories_recategorizes_on_version_mismatch(
        self, temp_cache_path, mock_categorizer, sample_dataset
    ):
        """Test that re-categorization happens when version doesn't match."""
        # Pre-populate cache with old version
        cache_data = {
            "dataset_version": {
                "id": "v0.9",
                "last_updated": "2025-10-03T12:00:00"
            },
            "samples": [
                {"index": 0, "category": "OldCategory"},
            ]
        }
        temp_cache_path.write_text(json.dumps(cache_data))

        mock_categorizer.categorize.side_effect = [
            "Science", "Technology", "History", "Science", "Technology"
        ]

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer
        )

        result = categorized_dataset.get_or_create_categories(
            dataset=sample_dataset,
            dataset_version="v1.0"  # Different version
        )

        # Should have re-categorized
        assert mock_categorizer.categorize.call_count == 5
        assert result[0]["category"] == "Science"

        # Cache should be updated with new version
        cache_content = json.loads(temp_cache_path.read_text())
        assert cache_content["dataset_version"]["id"] == "v1.0"

    def test_get_or_create_categories_skips_failed_categorizations(
        self, temp_cache_path, mock_categorizer, sample_dataset
    ):
        """Test that documents with None category are skipped."""
        mock_categorizer.categorize.side_effect = [
            "Science", None, "Technology", "Science", None
        ]

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer
        )

        result = categorized_dataset.get_or_create_categories(
            dataset=sample_dataset,
            dataset_version="v1.0"
        )

        # Should only have 3 samples (2 were skipped)
        assert len(result) == 3
        assert all("category" in item for item in result)
        assert None not in [item["category"] for item in result]

    def test_extracts_first_five_unique_categories(
        self, temp_cache_path, mock_categorizer
    ):
        """Test that only first 5 unique categories are kept."""
        # Dataset with 10 samples, 8 different categories
        large_dataset = [{"context": f"content {i}", "question": f"Q{i}?", "answer": f"A{i}"}
                        for i in range(10)]

        mock_categorizer.categorize.side_effect = [
            "Cat1", "Cat2", "Cat3", "Cat4", "Cat5",  # First 5 unique
            "Cat6", "Cat7", "Cat8",                   # Should be ignored
            "Cat1", "Cat2"                            # Duplicates of first 5, should keep
        ]

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer,
            max_categories=5
        )

        result = categorized_dataset.get_or_create_categories(
            dataset=large_dataset,
            dataset_version="v1.0"
        )

        # Get unique categories from result
        unique_categories = set(item["category"] for item in result)

        # Should only have first 5 unique categories
        assert len(unique_categories) <= 5
        assert unique_categories == {"Cat1", "Cat2", "Cat3", "Cat4", "Cat5"}

        # Samples with Cat6, Cat7, Cat8 should be excluded
        assert len(result) == 7  # 5 first unique + 2 duplicates of allowed categories

    def test_cache_saves_with_correct_structure(
        self, temp_cache_path, mock_categorizer, sample_dataset
    ):
        """Test that cache is saved with correct JSON structure."""
        mock_categorizer.categorize.side_effect = [
            "Science", "Technology", "History", "Science", "Technology"
        ]

        categorized_dataset = CategorizedDataset(
            cache_path=str(temp_cache_path),
            categorizer=mock_categorizer
        )

        categorized_dataset.get_or_create_categories(
            dataset=sample_dataset,
            dataset_version="v1.0"
        )

        # Read cache and verify structure
        cache_content = json.loads(temp_cache_path.read_text())

        assert "dataset_version" in cache_content
        assert "id" in cache_content["dataset_version"]
        assert "last_updated" in cache_content["dataset_version"]
        assert cache_content["dataset_version"]["id"] == "v1.0"

        assert "samples" in cache_content
        assert len(cache_content["samples"]) == 5

        for sample in cache_content["samples"]:
            assert "index" in sample
            assert "category" in sample
            assert isinstance(sample["index"], int)
            assert isinstance(sample["category"], str)
