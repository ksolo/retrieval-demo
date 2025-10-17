"""Multi-query retriever using LLM query expansion and reciprocal rank fusion."""

import logging
import re
from typing import List, Dict, Any
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ...vectorstore.client import WeaviateClient
from .base import Document

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """
    Retriever that expands queries using LLM and fuses results with RRF.

    This retriever implements a multi-query approach:
    1. Generate 3 alternative query phrasings using an LLM
    2. Retrieve documents for original + 3 variants (4 queries total)
    3. Apply Reciprocal Rank Fusion (RRF) to combine results
    4. Return top-k documents by fused score
    """

    def __init__(
        self, client: WeaviateClient, collection_name: str, openai_api_key: str
    ):
        """
        Initialize multi-query retriever with injected dependencies.

        Args:
            client: WeaviateClient instance for performing searches
            collection_name: Name of the collection to search in
            openai_api_key: API key for OpenAI LLM service
        """
        self.client = client
        self.collection_name = collection_name
        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=openai_api_key,
        )
        self.num_variants = 3  # Number of alternative queries to generate
        self.rrf_k = 60  # Standard RRF constant

    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve documents using multi-query expansion and RRF fusion.

        This method coordinates the retrieval pipeline:
        1. Generate query variants using LLM
        2. Retrieve documents for all queries (2x limit per query)
        3. Fuse results using Reciprocal Rank Fusion
        4. Convert fused results to Document objects

        Args:
            query: The search query text
            limit: Maximum number of documents to return

        Returns:
            List of Document objects ranked by RRF fusion score
        """
        # Step 1: Generate query variants
        variant_queries = self._generate_query_variants(query)
        all_queries = [query] + variant_queries

        logger.info(
            f"Expanded query into {len(all_queries)} variants for multi-query retrieval"
        )

        # Step 2: Retrieve for all queries
        query_results = self._retrieve_for_queries(all_queries, limit)

        # Step 3: Fuse results with RRF
        fused_results = self._fuse_results_rrf(query_results, limit)

        # Step 4: Build Document objects
        documents = self._build_documents(fused_results)

        logger.info(
            f"Returning {len(documents)} fused documents for query: {query[:50]}..."
        )

        return documents

    def _generate_query_variants(self, query: str) -> List[str]:
        """
        Generate alternative query phrasings using LLM.

        Uses gpt-5-mini to create query variations that expand the search surface area
        for better retrieval coverage.

        Args:
            query: Original search query

        Returns:
            List of alternative query strings (length = self.num_variants)
        """
        prompt = f"""You are a helpful assistant for a retrieval system. Generate {self.num_variants} alternative search queries that would help find the same information as the original query. These variations will expand our search area to improve retrieval results.

Return only the queries, one per line, without numbering or explanations.

Original query: {query}"""

        logger.info(f"Generating {self.num_variants} query variants using gpt-5-mini")

        response = self.llm.invoke([HumanMessage(content=prompt)])
        variants_text = response.content.strip()

        # Parse response - split by newlines and clean
        variants = []
        for line in variants_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip common numbering patterns: "1. ", "2. ", etc.
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^-\s*", "", line)

            if line:
                variants.append(line)

        # Ensure we have exactly num_variants (truncate or pad if needed)
        variants = variants[: self.num_variants]

        if len(variants) < self.num_variants:
            logger.warning(
                f"Generated only {len(variants)} variants instead of {self.num_variants}"
            )

        logger.info(f"Generated variants: {variants}")

        return variants

    def _retrieve_for_queries(
        self, queries: List[str], limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents for each query variant.

        Retrieves 2x the requested limit for each query to provide more
        candidates for the fusion algorithm.

        Args:
            queries: List of query strings (original + variants)
            limit: Final number of results desired (will retrieve 2x per query)

        Returns:
            Dictionary mapping query string to list of Weaviate results
        """
        retrieval_limit = 2 * limit
        query_results = {}

        for query_text in queries:
            results = self.client.semantic_search(
                collection_name=self.collection_name,
                query=query_text,
                limit=retrieval_limit,
            )
            query_results[query_text] = results

            logger.info(
                f"Retrieved {len(results)} results for variant query: {query_text[:50]}..."
            )

        return query_results

    def _fuse_results_rrf(
        self, query_results: Dict[str, List[Dict[str, Any]]], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple queries using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) for each query where doc appears
        Standard k value is 60.

        Args:
            query_results: Dictionary mapping queries to their result lists
            limit: Number of top results to return

        Returns:
            List of results (dicts) sorted by RRF score, limited to top-k
        """
        # Track RRF scores by UUID
        rrf_scores = defaultdict(float)
        # Track original result objects by UUID
        uuid_to_result = {}

        # Calculate RRF scores
        for query_text, results in query_results.items():
            for rank, result in enumerate(results):
                uuid = result["metadata"]["uuid"]
                # RRF formula: 1 / (k + rank)
                # rank is 0-indexed, so rank 0 gets score 1/(60+0) = 0.0167
                rrf_score = 1.0 / (self.rrf_k + rank)
                rrf_scores[uuid] += rrf_score

                # Store the result object (use first occurrence)
                if uuid not in uuid_to_result:
                    uuid_to_result[uuid] = result

        logger.info(f"Calculated RRF scores for {len(rrf_scores)} unique documents")

        # Sort by RRF score descending and take top-k
        sorted_uuids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        # Build result list with RRF scores in metadata
        fused_results = []
        for uuid, score in sorted_uuids:
            result = uuid_to_result[uuid].copy()
            result["metadata"]["rrf_score"] = score
            fused_results.append(result)

        logger.info(f"RRF fusion complete, returning top {len(fused_results)} results")

        return fused_results

    def _build_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert Weaviate results to Document objects.

        Args:
            results: List of Weaviate result dictionaries with RRF scores

        Returns:
            List of Document objects with metadata including RRF scores
        """
        documents = []

        for result in results:
            properties = result["properties"]
            metadata = result["metadata"].copy()

            # Add properties to metadata
            metadata.update(
                {
                    "document_id": properties.get("document_id"),
                    "chunk_index": properties.get("chunk_index"),
                    "chunk_size": properties.get("chunk_size"),
                }
            )

            doc = Document(page_content=properties.get("text", ""), metadata=metadata)
            documents.append(doc)

        return documents
