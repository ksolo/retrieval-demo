"""Manager for creating and managing LangSmith evaluation datasets."""

import logging
import os
from typing import List, Dict, Any, Optional
from langsmith import Client

logger = logging.getLogger(__name__)


class EvalDatasetManager:
    """Manages LangSmith datasets for evaluation."""

    def __init__(
        self, api_key: Optional[str] = None, project_name: Optional[str] = None
    ):
        """Initialize with LangSmith client.

        Args:
            api_key: LangSmith API key. If None, uses LANGSMITH_API_KEY env var.
            project_name: LangSmith project name. If None, uses LANGSMITH_PROJECT env var.
        """
        self.client = Client(api_key=api_key)  # Falls back to LANGSMITH_API_KEY env var
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT")

        if not self.project_name:
            raise ValueError(
                "project_name must be provided or LANGSMITH_PROJECT env var must be set"
            )

    def create_or_update_dataset(
        self,
        dataset_name: str,
        samples: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> str:
        """Create or update a LangSmith dataset with samples.

        Args:
            dataset_name: Name of the dataset in LangSmith
            samples: List of eval samples with eval_id, question, answer, category
            description: Optional dataset description

        Returns:
            Dataset ID from LangSmith
        """
        logger.info(
            f"Creating/updating LangSmith dataset '{dataset_name}' with {len(samples)} samples"
        )

        # Check if dataset exists
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Found existing dataset '{dataset_name}', will update")
            dataset_id = str(dataset.id)
        except Exception:
            # Dataset doesn't exist, create it
            logger.info(f"Creating new dataset '{dataset_name}'")
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=description
                or f"Evaluation dataset with {len(samples)} samples",
            )
            dataset_id = str(dataset.id)

        # Convert samples to LangSmith format
        examples = []
        for sample in samples:
            example = {
                "inputs": {"question": sample["question"]},
                "outputs": {"answer": sample["answer"]},
                "metadata": {
                    "eval_id": sample["eval_id"],
                    "category": sample.get("category"),
                    "original_index": sample.get("original_index"),
                },
            }
            examples.append(example)

        # Add examples to dataset
        self.client.create_examples(
            inputs=[ex["inputs"] for ex in examples],
            outputs=[ex["outputs"] for ex in examples],
            metadata=[ex["metadata"] for ex in examples],
            dataset_id=dataset_id,
        )

        logger.info(
            f"Successfully created/updated dataset '{dataset_name}' with {len(samples)} examples"
        )
        return dataset_id

    def get_dataset_samples(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Retrieve all samples from a LangSmith dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of samples with eval_id, question, answer, category, original_index
        """
        logger.info(f"Retrieving samples from dataset '{dataset_name}'")

        dataset = self.client.read_dataset(dataset_name=dataset_name)
        examples = list(self.client.list_examples(dataset_id=str(dataset.id)))

        samples = []
        for example in examples:
            sample = {
                "eval_id": example.metadata.get("eval_id"),
                "question": example.inputs["question"],
                "answer": example.outputs["answer"],
                "category": example.metadata.get("category"),
                "original_index": example.metadata.get("original_index"),
            }
            samples.append(sample)

        logger.info(f"Retrieved {len(samples)} samples from '{dataset_name}'")
        return samples

    def get_dataset_count(self, dataset_name: str) -> int:
        """Get count of examples in dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Number of examples in the dataset
        """
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=str(dataset.id)))
            return len(examples)
        except Exception as e:
            logger.error(f"Error getting count for dataset '{dataset_name}': {e}")
            return 0

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete a LangSmith dataset.

        Args:
            dataset_name: Name of the dataset to delete
        """
        logger.info(f"Deleting dataset '{dataset_name}'")
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        self.client.delete_dataset(dataset_id=str(dataset.id))
        logger.info(f"Deleted dataset '{dataset_name}'")
