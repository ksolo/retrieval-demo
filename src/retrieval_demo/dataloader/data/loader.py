"""Dataset loader for the RAG evaluation dataset."""

import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
from collections import defaultdict

from ..categorization import CategorizedDataset, Categorizer

logger = logging.getLogger(__name__)

DATASET_NAME = "neural-bridge/rag-dataset-12000"


class RAGDatasetLoader:
    """Loader for the neural-bridge/rag-dataset-12000 dataset."""

    def __init__(self):
        """Initialize the dataset loader."""
        self._dataset = None

    def load_dataset(self) -> Dataset:
        """Load the full dataset from HuggingFace."""
        if self._dataset is None:
            logger.info(f"Loading dataset: {DATASET_NAME}")
            self._dataset = load_dataset(DATASET_NAME)
            logger.info(
                f"Dataset loaded with {len(self._dataset['train'])} training examples"
            )

        return self._dataset

    def get_train_subset(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get a subset of the training data."""
        dataset = self.load_dataset()
        train_data = dataset["train"]

        # Take the first 'limit' items
        subset = train_data.select(range(min(limit, len(train_data))))

        # Convert to list of dictionaries with our expected keys
        result = []
        for i, item in enumerate(subset):
            # Ensure we have the expected keys: context, question, answer
            if all(key in item for key in ["context", "question", "answer"]):
                result.append(
                    {
                        "context": item["context"],
                        "question": item["question"],
                        "answer": item["answer"],
                    }
                )
            else:
                logger.warning(f"Item {i} missing required keys: {item.keys()}")

        logger.info(f"Returning {len(result)} items from training set")
        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        dataset = self.load_dataset()

        return {
            "dataset_name": DATASET_NAME,
            "splits": list(dataset.keys()),
            "train_size": len(dataset["train"]) if "train" in dataset else 0,
            "features": dataset["train"].features if "train" in dataset else {},
        }

    def get_dataset_version(self) -> str:
        """Get dataset version identifier from HuggingFace.

        Returns:
            Version string (commit hash or dataset size)
        """
        dataset = self.load_dataset()
        info = dataset["train"].info

        # Try git commit hash first
        commit_hash = self._extract_commit_hash(info)
        if commit_hash:
            return commit_hash

        # Fallback to dataset size
        return str(info.dataset_size)

    def _extract_commit_hash(self, info) -> Optional[str]:
        """Extract git commit hash from dataset info.

        Args:
            info: Dataset info object

        Returns:
            Commit hash string or None if not found
        """
        if not info.download_checksums:
            return None

        checksum_key = list(info.download_checksums.keys())[0]

        # Format: hf://datasets/name@COMMIT_HASH/path
        if "@" not in checksum_key:
            return None

        return checksum_key.split("@")[1].split("/")[0]

    def _convert_to_dict_with_index(self, train_data) -> List[Dict[str, Any]]:
        """Convert HuggingFace dataset to list of dicts with original indices.

        Args:
            train_data: HuggingFace dataset train split

        Returns:
            List of dicts with original_index added
        """
        result = []
        for i, item in enumerate(train_data):
            if all(key in item for key in ["context", "question", "answer"]):
                result.append(
                    {
                        "original_index": i,
                        "context": item["context"],
                        "question": item["question"],
                        "answer": item["answer"],
                    }
                )
        return result

    def _group_by_category(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group samples by category.

        Args:
            samples: List of samples with 'category' field

        Returns:
            Dictionary mapping category to list of samples
        """
        by_category = defaultdict(list)
        for sample in samples:
            by_category[sample["category"]].append(sample)
        return dict(by_category)

    def _stratified_sample(
        self, by_category: Dict[str, List[Dict[str, Any]]], samples_per_category: int
    ) -> List[Dict[str, Any]]:
        """Perform stratified sampling across categories.

        Args:
            by_category: Dictionary mapping category to samples
            samples_per_category: Number of samples to take per category

        Returns:
            List of selected samples with eval_id added
        """
        result = []
        for category, samples in by_category.items():
            selected = samples[:samples_per_category]

            # Add eval_id to each sample
            for sample in selected:
                sample["eval_id"] = f"rag12000_{sample['original_index']}"

            result.extend(selected)
            logger.info(
                f"Category '{category}': selected {len(selected)}/{len(samples)} samples"
            )

        return result

    def get_categorized_stratified_sample(
        self,
        samples_per_category: int = 100,
        cache_path: str = "eval/data/categorization_cache.json",
        max_categories: int = 5,
        categorizer: Optional[Categorizer] = None,
    ) -> List[Dict[str, Any]]:
        """Get stratified sample of categorized dataset.

        Args:
            samples_per_category: Number of samples to take per category
            cache_path: Path to categorization cache file
            max_categories: Maximum number of unique categories
            categorizer: Categorizer instance (optional, creates one if not provided)

        Returns:
            List of samples with eval_id, original_index, category, context, question, answer
        """
        dataset = self.load_dataset()
        train_data = dataset["train"]

        # Convert to indexed dicts
        full_dataset = self._convert_to_dict_with_index(train_data)
        logger.info(f"Loaded {len(full_dataset)} samples for categorization")

        # Categorize
        version = self.get_dataset_version()
        if categorizer is None:
            categorizer = Categorizer()

        categorized_dataset = CategorizedDataset(
            cache_path=cache_path,
            categorizer=categorizer,
            max_categories=max_categories,
        )
        categorized = categorized_dataset.get_or_create_categories(
            dataset=full_dataset, dataset_version=version
        )
        logger.info(f"Categorized dataset has {len(categorized)} samples")

        # Stratified sampling
        by_category = self._group_by_category(categorized)
        result = self._stratified_sample(by_category, samples_per_category)

        logger.info(f"Stratified sample complete: {len(result)} total samples")
        return result

    def _parse_eval_id(self, eval_id: str) -> Optional[int]:
        """Parse eval_id to extract dataset index.

        Args:
            eval_id: Eval ID string (e.g., "rag12000_42")

        Returns:
            Index integer or None if invalid
        """
        try:
            if not eval_id.startswith("rag12000_"):
                logger.warning(f"Invalid eval_id format: {eval_id}")
                return None

            index = int(eval_id.split("_")[1])
            return index
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse eval_id {eval_id}: {e}")
            return None

    def _fetch_sample_by_index(
        self, train_data, eval_id: str, index: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single sample by index.

        Args:
            train_data: HuggingFace dataset train split
            eval_id: Eval ID string
            index: Dataset index

        Returns:
            Sample dict or None if out of range
        """
        if index < 0 or index >= len(train_data):
            logger.warning(f"Index {index} out of range for eval_id {eval_id}")
            return None

        item = train_data[index]
        return {
            "eval_id": eval_id,
            "original_index": index,
            "context": item["context"],
            "question": item["question"],
            "answer": item["answer"],
        }

    def get_samples_by_ids(self, eval_ids: List[str]) -> List[Dict[str, Any]]:
        """Get specific samples by their eval_ids.

        Args:
            eval_ids: List of eval_id strings (e.g., ["rag12000_5", "rag12000_42"])

        Returns:
            List of samples matching the requested IDs, in the same order
        """
        dataset = self.load_dataset()
        train_data = dataset["train"]

        result = []
        for eval_id in eval_ids:
            index = self._parse_eval_id(eval_id)
            if index is not None:
                sample = self._fetch_sample_by_index(train_data, eval_id, index)
                if sample is not None:
                    result.append(sample)

        logger.info(f"Retrieved {len(result)}/{len(eval_ids)} samples by ID")
        return result
