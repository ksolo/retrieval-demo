"""Shared test fixtures and utilities."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing.

    Returns a context manager that patches OpenAI and provides
    a configured mock client with helper methods.
    """
    class MockOpenAIContext:
        def __init__(self):
            self.patcher = None
            self.mock_client = None

        def __enter__(self):
            # Patch at the module where it's imported
            self.patcher = patch('retrieval_demo.dataloader.categorization.OpenAI')
            mock_openai_class = self.patcher.__enter__()
            self.mock_client = MagicMock()
            mock_openai_class.return_value = self.mock_client
            return self

        def __exit__(self, *args):
            self.patcher.__exit__(*args)

        def set_response(self, content: str):
            """Set the chat completion response content."""
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = content
            self.mock_client.chat.completions.create.return_value = mock_response

        def set_error(self, error: Exception):
            """Set the chat completion to raise an error."""
            self.mock_client.chat.completions.create.side_effect = error

        def set_responses(self, contents: list[str]):
            """Set multiple responses for sequential calls."""
            responses = []
            for content in contents:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = content
                responses.append(mock_response)
            self.mock_client.chat.completions.create.side_effect = responses

    return MockOpenAIContext()
