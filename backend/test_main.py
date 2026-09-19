"""
Offline Test Suite — Quantic AI Career Assistant
==================================================
These tests run WITHOUT an API key or database. They validate:
- Health endpoint behaviour
- Request validation (Pydantic model enforcement)
- Error handling for missing AI client
- Fixture file integrity

These tests run on every CI push — no secrets required.

Usage:
    cd backend && pytest test_main.py -v
"""

import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Resolve imports: main.py lives at the project root, one level above backend/
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from main import app

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client (no real HTTP server needed)."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------
class TestHealthCheck:
    """Tests for the root / health endpoint."""

    def test_health_endpoint_returns_200(self, client):
        """GET / should always return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint_has_status(self, client):
        """Response must include a 'status' field."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert data["status"] == "online"

    def test_health_endpoint_reports_db_status(self, client):
        """Response must include a 'db' boolean field."""
        response = client.get("/")
        data = response.json()
        assert "db" in data
        assert isinstance(data["db"], bool)


# ---------------------------------------------------------------------------
# Request Validation Tests
# ---------------------------------------------------------------------------
class TestRequestValidation:
    """Tests for Pydantic request model enforcement."""

    def test_analyze_rejects_empty_body(self, client):
        """POST /analyze with no body should return 422 (validation error)."""
        response = client.post("/analyze")
        assert response.status_code == 422

    def test_analyze_rejects_missing_cv(self, client):
        """POST /analyze without cv_text should return 422."""
        response = client.post("/analyze", json={
            "job_description": "Some JD text"
        })
        assert response.status_code == 422

    def test_analyze_rejects_missing_jd(self, client):
        """POST /analyze without job_description should return 422."""
        response = client.post("/analyze", json={
            "cv_text": "Some CV text"
        })
        assert response.status_code == 422

    def test_cover_letter_rejects_empty_body(self, client):
        """POST /generate-cover-letter with no body should return 422."""
        response = client.post("/generate-cover-letter")
        assert response.status_code == 422

    def test_convert_pdf_rejects_no_file(self, client):
        """POST /convert-pdf-to-docx with no file should return 422."""
        response = client.post("/convert-pdf-to-docx")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for graceful error handling when the AI client is unavailable."""

    def test_analyze_returns_500_without_ai_key(self, client):
        """
        If GROQ_API_KEY is not set, the AI client is None.
        /analyze should return 500 with a descriptive error, not crash.
        """
        # This test is only meaningful when running without a key
        import main
        if main.client is not None:
            pytest.skip("GROQ_API_KEY is set — skipping missing-client test")

        response = client.post("/analyze", json={
            "cv_text": "Test CV content",
            "job_description": "Test JD content",
        })
        assert response.status_code == 500
        assert "AI Client" in response.json().get("detail", "")

    def test_cover_letter_returns_500_without_ai_key(self, client):
        """
        /generate-cover-letter should also return 500 when client is None.
        """
        import main
        if main.client is not None:
            pytest.skip("GROQ_API_KEY is set — skipping missing-client test")

        response = client.post("/generate-cover-letter", json={
            "cv_text": "Test CV",
            "job_description": "Test JD",
        })
        assert response.status_code == 500
        assert "AI Client" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# Fixture Integrity Tests
# ---------------------------------------------------------------------------
class TestFixtureIntegrity:
    """Verify that the benchmark test fixtures exist and are well-formed."""

    def test_fixture_directory_exists(self):
        """The fixtures/ directory must exist."""
        assert os.path.isdir(FIXTURES_DIR), \
            f"Fixtures directory not found: {FIXTURES_DIR}"

    def test_test_cv_exists_and_not_empty(self):
        """test_cv.txt must exist and contain substantial text."""
        path = os.path.join(FIXTURES_DIR, "test_cv.txt")
        assert os.path.isfile(path), f"test_cv.txt not found at {path}"
        with open(path, "r") as f:
            content = f.read()
        assert len(content) > 500, \
            f"test_cv.txt is too short ({len(content)} chars) — expected ≥500"

    def test_test_jd_exists_and_not_empty(self):
        """test_jd.txt must exist and contain substantial text."""
        path = os.path.join(FIXTURES_DIR, "test_jd.txt")
        assert os.path.isfile(path), f"test_jd.txt not found at {path}"
        with open(path, "r") as f:
            content = f.read()
        assert len(content) > 200, \
            f"test_jd.txt is too short ({len(content)} chars) — expected ≥200"

    def test_test_cv_contains_key_sections(self):
        """The benchmark CV must contain expected section markers."""
        path = os.path.join(FIXTURES_DIR, "test_cv.txt")
        with open(path, "r") as f:
            content = f.read().upper()
        for section in ["EXPERIENCE", "EDUCATION", "SKILLS"]:
            assert section in content, \
                f"Benchmark CV missing section: {section}"

    def test_test_jd_contains_key_sections(self):
        """The benchmark JD must contain expected section markers."""
        path = os.path.join(FIXTURES_DIR, "test_jd.txt")
        with open(path, "r") as f:
            content = f.read().upper()
        for section in ["RESPONSIBILITIES", "QUALIFICATIONS"]:
            assert section in content, \
                f"Benchmark JD missing section: {section}"

# ===========================================================================
# 5. GROUNDING — suggestions and cover letters must not invent skills
# ===========================================================================

import json
from types import SimpleNamespace

import main


class FakeGroq:
    """Returns canned chat completions in order, so grounding can be tested offline."""

    def __init__(self, *contents):
        self.contents, self.requests = list(contents), []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        if isinstance(content, Exception):
            raise content
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _grounded(*texts):
    return json.dumps({"results": [{"index": i, "unsupported": [], "text": t} for i, t in enumerate(texts)]})


class TestDropUngroundedSuggestions:
    """The code backstop that runs after the fact-check."""

    CV = "Senior Cloud Engineer. Built CI/CD with GitHub Actions and Terraform, reducing MTTR by 40%. Grafana, Prometheus."

    def _run(self, *texts, missing=("PagerDuty", "GCP (GKE, Cloud Build)")):
        result = {
            "missing_skills": list(missing),
            "suggestions": [{"id": f"s{i}", "replacement_text": text} for i, text in enumerate(texts)],
        }
        return [s["replacement_text"] for s in main.drop_ungrounded_suggestions(result, self.CV)["suggestions"]]

    def test_keeps_reworded_grounded_text(self):
        """Rewording facts the CV states is allowed."""
        text = "Cut MTTR by 40% by building Grafana and Prometheus alerting with Terraform."
        assert self._run(text) == [text]

    def test_drops_missing_skill(self):
        """A skill listed as missing must not appear in a suggestion."""
        assert self._run("Integrated Grafana alerts with PagerDuty for escalation.") == []

    def test_drops_skill_listed_inside_parentheses(self):
        """Skills inside a grouped entry such as 'GCP (GKE, ...)' are checked too."""
        assert self._run("Designed clusters with concepts applicable to GKE.") == []

    def test_drops_product_name_inside_longer_entry(self):
        """'FinOps frameworks' still catches a bare 'FinOps'."""
        assert self._run("Built FinOps dashboards.", missing=["FinOps frameworks"]) == []

    def test_ignores_plain_words_inside_longer_entry(self):
        """Ordinary words like 'security' are not treated as missing skills."""
        text = "Hardened security across GitHub Actions."
        assert self._run(text, missing=["Security clearance (SC/DV)"]) == [text]

    def test_drops_invented_metric(self):
        """Numbers the CV does not contain are invented metrics."""
        assert self._run("Reduced MTTR by 20% through better alerting.") == []


class TestGrounding:
    """The fact-check pass on suggestions and cover letters, with the AI replaced by canned replies."""

    CV = "Built CI/CD with GitHub Actions, reducing MTTR by 40%."
    JD = "Needs Jenkins and Pulumi."

    def _analysis(self, *replacements):
        return json.dumps({
            "is_cv": True,
            "missing_skills": ["Jenkins"],
            "suggestions": [
                {"id": f"s{i}", "original_text": "old", "replacement_text": text} for i, text in enumerate(replacements)
            ],
        })

    def test_analysis_uses_corrected_suggestions(self, client, monkeypatch):
        """Corrected texts replace the originals, and emptied suggestions are removed."""
        fake = FakeGroq(
            self._analysis("Built CI/CD with GitHub Actions and Jenkins.", "Invented Vault rollout."),
            _grounded("Built CI/CD with GitHub Actions.", ""),
        )
        monkeypatch.setattr(main, "client", fake)
        monkeypatch.setattr(main, "log_usage", lambda ip, text: None)
        response = client.post("/analyze", json={"cv_text": self.CV, "job_description": self.JD})
        assert response.status_code == 200
        assert [s["replacement_text"] for s in response.json()["suggestions"]] == ["Built CI/CD with GitHub Actions."]
        assert fake.requests[1]["model"] == main.GROUNDING_MODEL

    def test_analysis_falls_back_to_code_check_when_grounding_fails(self, client, monkeypatch):
        """If the fact-check call fails, the code backstop still removes missing skills."""
        fake = FakeGroq(
            self._analysis("Built CI/CD with GitHub Actions and Jenkins.", "Cut MTTR by 40% with GitHub Actions."),
            RuntimeError("rate limited"),
        )
        monkeypatch.setattr(main, "client", fake)
        monkeypatch.setattr(main, "log_usage", lambda ip, text: None)
        response = client.post("/analyze", json={"cv_text": self.CV, "job_description": self.JD})
        assert [s["replacement_text"] for s in response.json()["suggestions"]] == ["Cut MTTR by 40% with GitHub Actions."]

    def test_cover_letter_is_grounded_line_by_line(self, client, monkeypatch):
        """Each line of the letter is checked, and removed lines leave no gaps."""
        letter = "Dear Team,\n\nI use GitHub Actions.\n\nI am proficient in Pulumi.\n\nBest,\nAlex"
        fake = FakeGroq(letter, _grounded("Dear Team,", "I use GitHub Actions.", "", "Best,", "Alex"))
        monkeypatch.setattr(main, "client", fake)
        response = client.post("/generate-cover-letter", json={"cv_text": self.CV, "job_description": self.JD})
        assert response.json()["cover_letter"] == "Dear Team,\n\nI use GitHub Actions.\n\nBest,\nAlex"

    def test_empty_cover_letter_returns_500(self, client, monkeypatch):
        """A letter that comes back empty is an error, not a blank page."""
        monkeypatch.setattr(main, "client", FakeGroq(""))
        response = client.post("/generate-cover-letter", json={"cv_text": self.CV, "job_description": self.JD})
        assert response.status_code == 500
