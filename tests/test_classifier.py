import pytest
from pydantic import ValidationError

from northstar_rag.config import settings
from northstar_rag.llm import GUIDES, DomainResult, domain_to_source


def test_guide_files_exist():
    for file_name in GUIDES.values():
        assert (settings.data_dir / file_name).exists()


def test_domain_to_source():
    assert domain_to_source("credit_card") == GUIDES["credit_card"]
    assert domain_to_source("all") is None


def test_bad_domain_rejected():
    with pytest.raises(ValidationError):
        DomainResult(domain="mortgage")
