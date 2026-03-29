"""
Basic unit tests for MARS components.

These tests validate the non-LLM-dependent code (tools, config, etc.)
without requiring actual API keys.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Config / settings tests
# ---------------------------------------------------------------------------

class TestSettings:
    def test_settings_importable(self) -> None:
        from mars.config import settings
        assert hasattr(settings, "DASHSCOPE_API_KEY")
        assert hasattr(settings, "DEEPSEEK_API_KEY")
        assert hasattr(settings, "MOONSHOT_API_KEY")
        assert hasattr(settings, "ZHIPU_API_KEY")

    def test_default_llm_provider(self) -> None:
        from mars.config import settings
        assert settings.DEFAULT_LLM_PROVIDER in ("qwen", "deepseek", "kimi", "glm")

    def test_max_papers_positive(self) -> None:
        from mars.config import settings
        assert settings.MAX_PAPERS_PER_SEARCH > 0
        assert settings.MAX_PAPERS_FOR_ANALYSIS > 0


# ---------------------------------------------------------------------------
# LLM Factory tests
# ---------------------------------------------------------------------------

class TestLLMFactory:
    def test_invalid_provider_raises(self) -> None:
        from mars.utils.llm_factory import get_llm
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(provider="nonexistent_provider")

    def test_get_llm_returns_object(self) -> None:
        """get_llm should return an object without API calls."""
        from mars.utils.llm_factory import get_llm
        # This should not raise even with empty API key (no actual call is made)
        llm = get_llm(provider="qwen")
        assert llm is not None

    def test_get_deepseek_llm(self) -> None:
        from mars.utils.llm_factory import get_deepseek_llm
        llm = get_deepseek_llm()
        assert llm is not None

    def test_get_kimi_llm(self) -> None:
        from mars.utils.llm_factory import get_kimi_llm
        llm = get_kimi_llm()
        assert llm is not None

    def test_get_glm_llm(self) -> None:
        from mars.utils.llm_factory import get_glm_llm
        llm = get_glm_llm()
        assert llm is not None


# ---------------------------------------------------------------------------
# CCF Database tool tests
# ---------------------------------------------------------------------------

class TestCCFDatabaseTool:
    def setup_method(self) -> None:
        from mars.tools.ccf_database import CCFDatabaseQueryTool
        self.tool = CCFDatabaseQueryTool()

    def test_query_machine_learning(self) -> None:
        result = self.tool._run("machine learning")
        assert "ICML" in result or "NeurIPS" in result or "Found" in result

    def test_query_security(self) -> None:
        result = self.tool._run("security")
        assert "CCS" in result or "USENIX" in result or "Found" in result

    def test_query_no_results(self) -> None:
        result = self.tool._run("xyzzy_nonexistent_domain_12345")
        assert "No CCF venues found" in result

    def test_query_returns_ccf_rank(self) -> None:
        result = self.tool._run("machine learning")
        assert "[A]" in result or "[B]" in result or "[C]" in result

    def test_tool_name(self) -> None:
        assert self.tool.name == "ccf_database_query"


# ---------------------------------------------------------------------------
# DBLP Search tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestDBLPSearchTool:
    def setup_method(self) -> None:
        from mars.tools.dblp_search import DBLPSearchTool
        self.tool = DBLPSearchTool()

    def test_tool_name(self) -> None:
        assert self.tool.name == "dblp_search"

    def test_empty_query_returns_error(self) -> None:
        result = self.tool._run('{"query": ""}')
        assert "Error" in result

    def test_plain_string_query(self) -> None:
        """Plain string (non-JSON) should be treated as query."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"hits": {"hit": []}}}
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.dblp_search.requests.get", return_value=mock_resp):
            result = self.tool._run("federated learning")
        assert "No results found" in result

    def test_parses_results_correctly(self) -> None:
        mock_hit = {
            "info": {
                "title": "Test Paper on Federated Learning",
                "authors": {"author": [{"text": "Alice"}, {"text": "Bob"}]},
                "venue": "ICML",
                "year": "2023",
                "doi": "10.1234/test",
                "url": "https://dblp.org/rec/test",
                "type": "Conference and Workshop Papers",
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"hits": {"hit": [mock_hit]}}}
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.dblp_search.requests.get", return_value=mock_resp):
            result = self.tool._run('{"query": "federated learning"}')

        assert "Test Paper on Federated Learning" in result
        assert "Alice" in result
        assert "ICML" in result


# ---------------------------------------------------------------------------
# Semantic Scholar tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestSemanticScholarTool:
    def setup_method(self) -> None:
        from mars.tools.semantic_scholar import SemanticScholarSearchTool
        self.tool = SemanticScholarSearchTool()

    def test_tool_name(self) -> None:
        assert self.tool.name == "semantic_scholar_search"

    def test_empty_query_returns_error(self) -> None:
        result = self.tool._run('{"query": ""}')
        assert "Error" in result

    def test_parses_results_with_citation_filter(self) -> None:
        mock_paper = {
            "paperId": "abc123",
            "title": "Differentially Private Federated Learning",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2023,
            "venue": "NeurIPS",
            "citationCount": 200,
            "abstract": "We propose a method for privacy-preserving federated learning.",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "url": "https://semanticscholar.org/paper/abc123",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [mock_paper]}
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.semantic_scholar.requests.get", return_value=mock_resp):
            result = self.tool._run(
                '{"query": "federated learning", "min_citations": 100}'
            )

        assert "Differentially Private Federated Learning" in result
        assert "NeurIPS" in result

    def test_min_citations_filter(self) -> None:
        mock_paper = {
            "paperId": "xyz789",
            "title": "Low Citation Paper",
            "authors": [],
            "year": 2023,
            "venue": "Workshop",
            "citationCount": 5,
            "abstract": None,
            "openAccessPdf": None,
            "url": "",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [mock_paper]}
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.semantic_scholar.requests.get", return_value=mock_resp):
            result = self.tool._run(
                '{"query": "some topic", "min_citations": 100}'
            )

        assert "No results matched filters" in result


# ---------------------------------------------------------------------------
# arXiv tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestArXivSearchTool:
    def setup_method(self) -> None:
        from mars.tools.arxiv_api import ArXivSearchTool
        self.tool = ArXivSearchTool()

    def test_tool_name(self) -> None:
        assert self.tool.name == "arxiv_search"

    def test_empty_query_returns_error(self) -> None:
        result = self.tool._run('{"query": ""}')
        assert "Error" in result

    def test_parses_atom_xml(self) -> None:
        atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2301.12345v1</id>
    <title>Federated Learning Survey</title>
    <author><name>Alice Smith</name></author>
    <published>2023-01-15T00:00:00Z</published>
    <summary>A comprehensive survey of federated learning methods.</summary>
  </entry>
</feed>"""
        mock_resp = MagicMock()
        mock_resp.text = atom_xml
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.arxiv_api.requests.get", return_value=mock_resp):
            result = self.tool._run('{"query": "federated learning"}')

        assert "Federated Learning Survey" in result
        assert "Alice Smith" in result
        assert "2301.12345" in result


# ---------------------------------------------------------------------------
# PDF Parser tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestPDFParserTool:
    def setup_method(self) -> None:
        from mars.tools.pdf_parser import PDFParserTool
        self.tool = PDFParserTool()

    def test_tool_name(self) -> None:
        assert self.tool.name == "pdf_parser"

    def test_empty_url_returns_error(self) -> None:
        result = self.tool._run('{"url": ""}')
        assert "Error" in result

    def test_download_failure(self) -> None:
        import requests as req
        with patch(
            "mars.tools.pdf_parser.requests.get",
            side_effect=req.RequestException("Connection refused"),
        ):
            result = self.tool._run('{"url": "https://example.com/paper.pdf"}')
        assert "Failed to download PDF" in result


# ---------------------------------------------------------------------------
# File Manager tool tests
# ---------------------------------------------------------------------------

class TestFileManagerTools:
    def test_file_writer_tool_name(self) -> None:
        from mars.tools.file_manager import FileWriterTool
        tool = FileWriterTool()
        assert tool.name == "file_writer"

    def test_file_reader_tool_name(self) -> None:
        from mars.tools.file_manager import FileReaderTool
        tool = FileReaderTool()
        assert tool.name == "file_reader"

    def test_write_and_read(self, tmp_path: "Path") -> None:
        from mars.tools.file_manager import FileReaderTool, FileWriterTool
        from mars.config import settings

        # Temporarily override output dir
        original_dir = settings.OUTPUT_DIR
        settings.OUTPUT_DIR = tmp_path
        try:
            writer = FileWriterTool()
            reader = FileReaderTool()

            write_result = writer._run(
                json.dumps({"filename": "test_output.txt", "content": "Hello MARS!"})
            )
            assert "written successfully" in write_result

            read_result = reader._run(json.dumps({"filename": "test_output.txt"}))
            assert "Hello MARS!" in read_result
        finally:
            settings.OUTPUT_DIR = original_dir

    def test_file_reader_missing_file(self, tmp_path: "Path") -> None:
        from mars.tools.file_manager import FileReaderTool
        from mars.config import settings

        original_dir = settings.OUTPUT_DIR
        settings.OUTPUT_DIR = tmp_path
        try:
            reader = FileReaderTool()
            result = reader._run(json.dumps({"filename": "nonexistent.txt"}))
            assert "not found" in result
        finally:
            settings.OUTPUT_DIR = original_dir

    def test_path_traversal_prevention(self, tmp_path: "Path") -> None:
        """Ensure path traversal attempts are neutralized."""
        from mars.tools.file_manager import FileWriterTool
        from mars.config import settings

        original_dir = settings.OUTPUT_DIR
        settings.OUTPUT_DIR = tmp_path
        try:
            writer = FileWriterTool()
            result = writer._run(
                json.dumps(
                    {
                        "filename": "../../etc/passwd",
                        "content": "malicious content",
                    }
                )
            )
            # Should write to tmp_path/passwd, not /etc/passwd
            assert (tmp_path / "passwd").exists()
            assert not (tmp_path / "../../etc" / "passwd").exists()
        finally:
            settings.OUTPUT_DIR = original_dir


# ---------------------------------------------------------------------------
# Citation Network tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestCitationNetworkTool:
    def setup_method(self) -> None:
        from mars.tools.citation_network import CitationNetworkTool
        self.tool = CitationNetworkTool()

    def test_tool_name(self) -> None:
        assert self.tool.name == "citation_network_builder"

    def test_empty_paper_ids(self) -> None:
        result = self.tool._run('{"paper_ids": []}')
        assert "Error" in result

    def test_invalid_json(self) -> None:
        result = self.tool._run("not valid json")
        assert "Error" in result

    def test_builds_network_from_api(self) -> None:
        mock_data = {
            "title": "Seed Paper",
            "references": [
                {"paperId": "ref001", "title": "Reference 1", "year": 2022, "citationCount": 50},
                {"paperId": "ref002", "title": "Reference 2", "year": 2021, "citationCount": 30},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status.return_value = None

        with patch("mars.tools.citation_network.requests.get", return_value=mock_resp):
            result = self.tool._run('{"paper_ids": ["seed001"]}')

        assert "Total nodes" in result
        assert "Total edges" in result


# ---------------------------------------------------------------------------
# Crew imports (smoke test - no API calls)
# ---------------------------------------------------------------------------

class TestCrewImports:
    def test_search_crew_importable(self) -> None:
        from mars.crews.search_crew import create_search_crew
        assert callable(create_search_crew)

    def test_analysis_crew_importable(self) -> None:
        from mars.crews.analysis_crew import create_analysis_crew
        assert callable(create_analysis_crew)

    def test_connection_crew_importable(self) -> None:
        from mars.crews.connection_crew import create_connection_crew
        assert callable(create_connection_crew)

    def test_full_research_crew_importable(self) -> None:
        from mars.crews.full_research_crew import create_full_research_crew
        assert callable(create_full_research_crew)


# ---------------------------------------------------------------------------
# Agent imports (smoke test - no API calls)
# ---------------------------------------------------------------------------

class TestAgentImports:
    def test_researcher_agent_importable(self) -> None:
        from mars.agents.researcher import create_researcher_agent
        assert callable(create_researcher_agent)

    def test_searcher_agent_importable(self) -> None:
        from mars.agents.searcher import create_searcher_agent
        assert callable(create_searcher_agent)

    def test_analyzer_agent_importable(self) -> None:
        from mars.agents.analyzer import create_analyzer_agent
        assert callable(create_analyzer_agent)

    def test_connector_agent_importable(self) -> None:
        from mars.agents.connector import create_connector_agent
        assert callable(create_connector_agent)

    def test_summarizer_agent_importable(self) -> None:
        from mars.agents.summarizer import create_summarizer_agent
        assert callable(create_summarizer_agent)

    def test_evaluator_agent_importable(self) -> None:
        from mars.agents.evaluator import create_evaluator_agent
        assert callable(create_evaluator_agent)
