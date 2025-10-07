"""Tests for EvalDatasetManager."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from eval.dataset import EvalDatasetManager


class TestEvalDatasetManager:
    """Tests for EvalDatasetManager."""

    @pytest.fixture
    def mock_langsmith_client(self):
        """Create a mock LangSmith client."""
        with patch("eval.dataset.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_data(self):
        """Sample evaluation data."""
        return [
            {
                "eval_id": "rag12000_1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "Geography",
                "original_index": 1,
            },
            {
                "eval_id": "rag12000_5",
                "question": "What is 2+2?",
                "answer": "4",
                "category": "Math",
                "original_index": 5,
            },
        ]

    def test_init_with_env_vars(self, mock_langsmith_client):
        """Test initialization with environment variables."""
        with patch.dict("os.environ", {"LANGSMITH_PROJECT": "test-project"}):
            manager = EvalDatasetManager()
            assert manager.project_name == "test-project"

    def test_init_with_explicit_project(self, mock_langsmith_client):
        """Test initialization with explicit project name."""
        manager = EvalDatasetManager(project_name="my-project")
        assert manager.project_name == "my-project"

    def test_init_raises_without_project(self, mock_langsmith_client):
        """Test that initialization fails without project name."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="project_name must be provided"):
                EvalDatasetManager()

    def test_create_or_update_dataset_creates_new(
        self, mock_langsmith_client, sample_data
    ):
        """Test creating a new dataset."""
        # Mock dataset doesn't exist
        mock_langsmith_client.read_dataset.side_effect = Exception("Not found")

        # Mock dataset creation
        mock_dataset = Mock()
        mock_dataset.id = "dataset_123"
        mock_langsmith_client.create_dataset.return_value = mock_dataset

        manager = EvalDatasetManager(project_name="test-project")
        dataset_id = manager.create_or_update_dataset(
            dataset_name="test-dataset", samples=sample_data, description="Test dataset"
        )

        # Verify dataset creation
        assert dataset_id == "dataset_123"
        mock_langsmith_client.create_dataset.assert_called_once_with(
            dataset_name="test-dataset", description="Test dataset"
        )

        # Verify examples creation
        mock_langsmith_client.create_examples.assert_called_once()
        call_args = mock_langsmith_client.create_examples.call_args

        # Check inputs
        inputs = call_args.kwargs["inputs"]
        assert len(inputs) == 2
        assert inputs[0] == {"question": "What is the capital of France?"}
        assert inputs[1] == {"question": "What is 2+2?"}

        # Check outputs
        outputs = call_args.kwargs["outputs"]
        assert outputs[0] == {"answer": "Paris"}
        assert outputs[1] == {"answer": "4"}

        # Check metadata
        metadata = call_args.kwargs["metadata"]
        assert metadata[0] == {
            "eval_id": "rag12000_1",
            "category": "Geography",
            "original_index": 1,
        }
        assert metadata[1] == {
            "eval_id": "rag12000_5",
            "category": "Math",
            "original_index": 5,
        }

    def test_create_or_update_dataset_updates_existing(
        self, mock_langsmith_client, sample_data
    ):
        """Test updating an existing dataset."""
        # Mock existing dataset
        mock_dataset = Mock()
        mock_dataset.id = "dataset_456"
        mock_langsmith_client.read_dataset.return_value = mock_dataset

        manager = EvalDatasetManager(project_name="test-project")
        dataset_id = manager.create_or_update_dataset(
            dataset_name="existing-dataset", samples=sample_data
        )

        # Verify no creation, just read
        assert dataset_id == "dataset_456"
        mock_langsmith_client.create_dataset.assert_not_called()
        mock_langsmith_client.create_examples.assert_called_once()

    def test_get_dataset_samples(self, mock_langsmith_client):
        """Test retrieving samples from dataset."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.id = "dataset_789"
        mock_langsmith_client.read_dataset.return_value = mock_dataset

        # Mock examples
        mock_example_1 = Mock()
        mock_example_1.inputs = {"question": "Q1?"}
        mock_example_1.outputs = {"answer": "A1"}
        mock_example_1.metadata = {
            "eval_id": "rag12000_10",
            "category": "Science",
            "original_index": 10,
        }

        mock_example_2 = Mock()
        mock_example_2.inputs = {"question": "Q2?"}
        mock_example_2.outputs = {"answer": "A2"}
        mock_example_2.metadata = {
            "eval_id": "rag12000_20",
            "category": "History",
            "original_index": 20,
        }

        mock_langsmith_client.list_examples.return_value = [
            mock_example_1,
            mock_example_2,
        ]

        manager = EvalDatasetManager(project_name="test-project")
        samples = manager.get_dataset_samples("test-dataset")

        # Verify results
        assert len(samples) == 2
        assert samples[0] == {
            "eval_id": "rag12000_10",
            "question": "Q1?",
            "answer": "A1",
            "category": "Science",
            "original_index": 10,
        }
        assert samples[1] == {
            "eval_id": "rag12000_20",
            "question": "Q2?",
            "answer": "A2",
            "category": "History",
            "original_index": 20,
        }

    def test_get_dataset_count(self, mock_langsmith_client):
        """Test getting dataset count."""
        mock_dataset = Mock()
        mock_dataset.id = "dataset_count"
        mock_langsmith_client.read_dataset.return_value = mock_dataset
        mock_langsmith_client.list_examples.return_value = [Mock(), Mock(), Mock()]

        manager = EvalDatasetManager(project_name="test-project")
        count = manager.get_dataset_count("test-dataset")

        assert count == 3

    def test_get_dataset_count_handles_error(self, mock_langsmith_client):
        """Test that count returns 0 on error."""
        mock_langsmith_client.read_dataset.side_effect = Exception("Error")

        manager = EvalDatasetManager(project_name="test-project")
        count = manager.get_dataset_count("nonexistent")

        assert count == 0

    def test_delete_dataset(self, mock_langsmith_client):
        """Test deleting a dataset."""
        mock_dataset = Mock()
        mock_dataset.id = "dataset_to_delete"
        mock_langsmith_client.read_dataset.return_value = mock_dataset

        manager = EvalDatasetManager(project_name="test-project")
        manager.delete_dataset("test-dataset")

        mock_langsmith_client.delete_dataset.assert_called_once_with(
            dataset_id="dataset_to_delete"
        )
