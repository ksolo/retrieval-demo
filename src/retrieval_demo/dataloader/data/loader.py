"""Dataset loader for the RAG evaluation dataset."""

import logging
from typing import List, Dict, Any
from datasets import load_dataset, Dataset

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
            logger.info(f"Dataset loaded with {len(self._dataset['train'])} training examples")
        
        return self._dataset
    
    def get_train_subset(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get a subset of the training data."""
        dataset = self.load_dataset()
        train_data = dataset['train']
        
        # Take the first 'limit' items
        subset = train_data.select(range(min(limit, len(train_data))))
        
        # Convert to list of dictionaries with our expected keys
        result = []
        for i, item in enumerate(subset):
            # Ensure we have the expected keys: context, question, answer
            if all(key in item for key in ['context', 'question', 'answer']):
                result.append({
                    'context': item['context'],
                    'question': item['question'], 
                    'answer': item['answer']
                })
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
            "train_size": len(dataset['train']) if 'train' in dataset else 0,
            "features": dataset['train'].features if 'train' in dataset else {},
        }