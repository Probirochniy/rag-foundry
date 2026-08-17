import pytest
from pydantic import ValidationError

from src.api.v1.models.requests.query.query_request import QueryRequest


def test_query_request_valid() -> None:
    req = QueryRequest(query="как задеплоить кубер", top_k=5)
    assert req.query == "как задеплоить кубер"
    assert req.top_k == 5


def test_query_request_empty_query_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="", top_k=3)


def test_query_request_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="тест", top_k=21)
