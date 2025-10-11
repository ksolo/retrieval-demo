"""Document categorization using LLM."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

CATEGORIZATION_PROMPT = """You are a categorization assistant. Given a document context and question, determine the broad category this content belongs to.

Respond with ONLY the category name (e.g., "Science", "Technology", "History", "Business", etc.).
Be concise and use standard category names.

Context: {context}
Question: {question}

Category:"""


class Categorizer:
    """Categorizes documents using LLM."""

    def __init__(self, model: str = "gpt-5-mini", api_key: Optional[str] = None):
        """Initialize categorizer with model.

        Args:
            model: OpenAI model to use for categorization
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def categorize(self, context: str, question: str) -> Optional[str]:
        """Categorize a single document.

        Args:
            context: Document context
            question: Related question

        Returns:
            Category string, or None if categorization fails
        """
        try:
            prompt = CATEGORIZATION_PROMPT.format(
                context=context[:1000], question=question
            )

            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )

            category = response.choices[0].message.content.strip()

            if not category:
                logger.warning("Empty category response")
                return None

            return category

        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return None


class CategorizedDataset:
    """Manages categorization with caching."""

    def __init__(
        self, cache_path: str, categorizer: Categorizer, max_categories: int = 5
    ):
        """Initialize with cache path and categorizer.

        Args:
            cache_path: Path to cache file
            categorizer: Categorizer instance (dependency injection)
            max_categories: Maximum number of unique categories to keep
        """
        self.cache_path = Path(cache_path)
        self.categorizer = categorizer
        self.max_categories = max_categories

    def get_or_create_categories(
        self, dataset: list[dict], dataset_version: str
    ) -> list[dict]:
        """Get cached categories or create new ones.

        Args:
            dataset: List of documents with 'context', 'question', 'answer' keys
            dataset_version: Version identifier for the dataset

        Returns:
            Dataset with 'category' field added. Documents that fail
            categorization or exceed max_categories are excluded.
        """
        # Try to load from cache
        cache = self._load_cache()

        if cache and cache.get("dataset_version", {}).get("id") == dataset_version:
            logger.info(f"Using cached categories for version {dataset_version}")
            return self._apply_cached_categories(dataset, cache)

        # Cache miss or version mismatch - categorize
        logger.info(f"Categorizing dataset (version: {dataset_version})")
        categorized = self._categorize_dataset(dataset)

        # Save to cache
        self._save_cache(categorized, dataset_version)

        return categorized

    def _load_cache(self) -> Optional[dict]:
        """Load cache from file."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_cache(
        self, categorized_dataset: list[dict], dataset_version: str
    ) -> None:
        """Save categories to cache."""
        cache_data = {
            "dataset_version": {
                "id": dataset_version,
                "last_updated": datetime.now().isoformat(),
            },
            "samples": [
                {"index": i, "category": item["category"]}
                for i, item in enumerate(categorized_dataset)
            ],
        }

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Saved {len(categorized_dataset)} categories to cache")

    def _apply_cached_categories(self, dataset: list[dict], cache: dict) -> list[dict]:
        """Apply cached categories to dataset."""
        # Build index -> category map
        category_map = {
            sample["index"]: sample["category"] for sample in cache.get("samples", [])
        }

        # Apply categories
        result = []
        for i, item in enumerate(dataset):
            if i in category_map:
                categorized_item = item.copy()
                categorized_item["category"] = category_map[i]
                result.append(categorized_item)

        return result

    def _categorize_dataset(self, dataset: list[dict]) -> list[dict]:
        """Categorize entire dataset, respecting max_categories limit."""
        result = []
        seen_categories = set()
        allowed_categories = set()

        for i, item in enumerate(dataset):
            # Log progress every 10 samples
            if (i + 1) % 10 == 0:
                logger.info(f"Categorized {i + 1}/{len(dataset)} samples")

            category = self.categorizer.categorize(
                context=item["context"], question=item["question"]
            )

            # Skip if categorization failed
            if category is None:
                logger.debug(f"Skipping sample {i} (categorization failed)")
                continue

            # Track first N unique categories
            if category not in seen_categories:
                if len(allowed_categories) < self.max_categories:
                    allowed_categories.add(category)
                seen_categories.add(category)

            # Only include if category is in allowed set
            if category in allowed_categories:
                categorized_item = item.copy()
                categorized_item["category"] = category
                result.append(categorized_item)
            else:
                logger.debug(
                    f"Skipping sample {i} (category '{category}' not in first {self.max_categories})"
                )

        logger.info(
            f"Categorization complete: {len(result)} samples across "
            f"{len(allowed_categories)} categories: {sorted(allowed_categories)}"
        )

        return result
