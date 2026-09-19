"""
Benchmark Test Suite — Quantic AI Career Assistant
====================================================
Validates the /analyze endpoint against known test fixtures (test CV + test JD).

This test sends real requests to the Groq API, so it requires a valid
GROQ_API_KEY environment variable. If the key is not set, these tests
are automatically skipped — they will NOT block CI.

Usage:
    # Run the benchmark locally (requires .env with GROQ_API_KEY):
    cd backend && pytest test_benchmark.py -v

    # Run ONLY the benchmark (skip other tests):
    cd backend && pytest test_benchmark.py -v -m benchmark
"""

import os
import sys
import json
import pytest

# ---------------------------------------------------------------------------
# Resolve imports: main.py lives at the project root, one level above backend/
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Skip the entire module if the AI key is missing (CI without secrets)
GROQ_KEY = os.getenv("GROQ_API_KEY")
pytestmark = pytest.mark.skipif(
    not GROQ_KEY,
    reason="GROQ_API_KEY not set — skipping live benchmark tests"
)

from fastapi.testclient import TestClient
from main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def test_cv():
    """Load the benchmark CV text from fixtures."""
    with open(os.path.join(FIXTURES_DIR, "test_cv.txt"), "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def test_jd():
    """Load the benchmark JD text from fixtures."""
    with open(os.path.join(FIXTURES_DIR, "test_jd.txt"), "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client (no real HTTP server needed)."""
    return TestClient(app)


@pytest.fixture(scope="module")
def analysis_result(client, test_cv, test_jd):
    """
    Run the benchmark analysis ONCE for the entire module.

    This avoids burning multiple API calls — every test in this file
    validates a different aspect of the same response.
    """
    response = client.post("/analyze", json={
        "cv_text": test_cv,
        "job_description": test_jd,
    })
    assert response.status_code == 200, f"Analysis failed: {response.text}"
    return response.json()


# ---------------------------------------------------------------------------
# Benchmark Tests — Response Structure
# ---------------------------------------------------------------------------
class TestBenchmarkResponseStructure:
    """Validate that the AI response conforms to the expected JSON schema."""

    @pytest.mark.benchmark
    def test_top_level_keys_present(self, analysis_result):
        """All required top-level keys must exist in the response."""
        required_keys = {
            "is_cv", "score", "match_status",
            "matched_skills", "missing_skills",
            "extraction", "suggestions", "skill_gap_courses"
        }
        actual_keys = set(analysis_result.keys())
        missing = required_keys - actual_keys
        assert not missing, f"Missing top-level keys: {missing}"

    @pytest.mark.benchmark
    def test_is_valid_cv(self, analysis_result):
        """The benchmark CV must be recognised as a valid CV."""
        assert analysis_result["is_cv"] is True, \
            "Benchmark CV was not recognised as a CV"

    @pytest.mark.benchmark
    def test_score_in_valid_range(self, analysis_result):
        """Score must be an integer between 0 and 100."""
        score = analysis_result["score"]
        assert isinstance(score, int), f"Score is not an int: {type(score)}"
        assert 0 <= score <= 100, f"Score out of range: {score}"

    @pytest.mark.benchmark
    def test_match_status_is_string(self, analysis_result):
        """Match status must be a non-empty string."""
        status = analysis_result["match_status"]
        assert isinstance(status, str) and len(status) > 0

    @pytest.mark.benchmark
    def test_skills_are_lists(self, analysis_result):
        """Matched and missing skills must be lists of strings."""
        for key in ["matched_skills", "missing_skills"]:
            skills = analysis_result[key]
            assert isinstance(skills, list), f"{key} is not a list"
            for skill in skills:
                assert isinstance(skill, str), f"Skill in {key} is not a str: {skill}"


# ---------------------------------------------------------------------------
# Benchmark Tests — Extraction Quality
# ---------------------------------------------------------------------------
class TestBenchmarkExtraction:
    """Validate that the AI correctly extracts structured data from the CV/JD."""

    @pytest.mark.benchmark
    def test_extraction_has_cv_and_jd(self, analysis_result):
        """Extraction must contain both cv_data and jd_data."""
        extraction = analysis_result["extraction"]
        assert "cv_data" in extraction, "Missing cv_data in extraction"
        assert "jd_data" in extraction, "Missing jd_data in extraction"

    @pytest.mark.benchmark
    def test_cv_personal_info_extracted(self, analysis_result):
        """Personal info fields must be present and non-empty for key fields."""
        personal = analysis_result["extraction"]["cv_data"]["personal_info"]
        assert personal["name"].strip(), "Name not extracted"
        assert personal["email"].strip(), "Email not extracted"

    @pytest.mark.benchmark
    def test_cv_experience_extracted(self, analysis_result):
        """At least 2 experience records should be extracted (the CV has 3)."""
        experience = analysis_result["extraction"]["cv_data"]["experience"]
        assert isinstance(experience, list)
        assert len(experience) >= 2, \
            f"Expected ≥2 experience records, got {len(experience)}"

    @pytest.mark.benchmark
    def test_cv_experience_has_bullets(self, analysis_result):
        """Each experience record should have at least 1 bullet point."""
        experience = analysis_result["extraction"]["cv_data"]["experience"]
        for exp in experience:
            assert "bullets" in exp, f"Experience '{exp.get('title')}' missing bullets"
            assert len(exp["bullets"]) >= 1, \
                f"Experience '{exp.get('title')}' has no bullets"

    @pytest.mark.benchmark
    def test_cv_experience_has_ids(self, analysis_result):
        """Experience records and bullets must have IDs for cross-referencing."""
        experience = analysis_result["extraction"]["cv_data"]["experience"]
        for exp in experience:
            assert "id" in exp and exp["id"], \
                f"Experience '{exp.get('title')}' missing ID"
            for bullet in exp.get("bullets", []):
                assert "id" in bullet and bullet["id"], \
                    "Experience bullet missing ID"

    @pytest.mark.benchmark
    def test_cv_skills_extracted(self, analysis_result):
        """At least 5 skills should be extracted (the CV lists 20+)."""
        skills = analysis_result["extraction"]["cv_data"]["skills"]
        assert isinstance(skills, list)
        assert len(skills) >= 5, f"Expected ≥5 skills, got {len(skills)}"

    @pytest.mark.benchmark
    def test_cv_education_extracted(self, analysis_result):
        """At least 1 education record should be extracted."""
        education = analysis_result["extraction"]["cv_data"]["education"]
        assert isinstance(education, list)
        assert len(education) >= 1, "No education records extracted"

    @pytest.mark.benchmark
    def test_cv_certifications_extracted(self, analysis_result):
        """At least 2 certifications should be extracted (the CV has 4)."""
        certs = analysis_result["extraction"]["cv_data"]["certifications"]
        assert isinstance(certs, list)
        assert len(certs) >= 2, f"Expected ≥2 certifications, got {len(certs)}"

    @pytest.mark.benchmark
    def test_cv_projects_extracted(self, analysis_result):
        """At least 1 project should be extracted (the CV has 2)."""
        projects = analysis_result["extraction"]["cv_data"]["projects"]
        assert isinstance(projects, list)
        assert len(projects) >= 1, f"Expected ≥1 projects, got {len(projects)}"

    @pytest.mark.benchmark
    def test_jd_role_extracted(self, analysis_result):
        """JD role must be extracted as a non-empty string."""
        jd_data = analysis_result["extraction"]["jd_data"]
        assert "role" in jd_data
        assert isinstance(jd_data["role"], str) and len(jd_data["role"]) > 0

    @pytest.mark.benchmark
    def test_jd_skills_extracted(self, analysis_result):
        """At least 3 JD skills should be extracted."""
        jd_skills = analysis_result["extraction"]["jd_data"]["skills"]
        assert isinstance(jd_skills, list)
        assert len(jd_skills) >= 3, f"Expected ≥3 JD skills, got {len(jd_skills)}"


# ---------------------------------------------------------------------------
# Benchmark Tests — Suggestions Quality
# ---------------------------------------------------------------------------
class TestBenchmarkSuggestions:
    """Validate the AI-generated improvement suggestions."""

    @pytest.mark.benchmark
    def test_suggestions_count(self, analysis_result):
        """Up to 5 suggestions: the prompt asks for fewer rather than padding, and grounding drops any it empties."""
        suggestions = analysis_result["suggestions"]
        assert isinstance(suggestions, list)
        assert 1 <= len(suggestions) <= 5, \
            f"Expected 1-5 suggestions, got {len(suggestions)}"

    @pytest.mark.benchmark
    def test_suggestion_structure(self, analysis_result):
        """Each suggestion must have all required fields."""
        required_fields = {
            "id", "target_id", "type",
            "issue", "original_text", "replacement_text", "reason"
        }
        for i, suggestion in enumerate(analysis_result["suggestions"]):
            actual = set(suggestion.keys())
            missing = required_fields - actual
            assert not missing, \
                f"Suggestion {i} missing fields: {missing}"

    @pytest.mark.benchmark
    def test_suggestion_types_valid(self, analysis_result):
        """Each suggestion type must be one of the allowed values."""
        valid_types = {
            "summary", "experience_bullet", "project_description",
            "education_detail", "add_experience_bullet", "add_project"
        }
        for suggestion in analysis_result["suggestions"]:
            assert suggestion["type"] in valid_types, \
                f"Invalid suggestion type: '{suggestion['type']}'"

    @pytest.mark.benchmark
    def test_suggestion_has_replacement_text(self, analysis_result):
        """Each suggestion must have non-empty replacement text."""
        for suggestion in analysis_result["suggestions"]:
            assert suggestion["replacement_text"].strip(), \
                f"Suggestion '{suggestion['id']}' has empty replacement_text"

    @pytest.mark.benchmark
    def test_suggestion_ids_are_unique(self, analysis_result):
        """All suggestion IDs must be unique."""
        ids = [s["id"] for s in analysis_result["suggestions"]]
        assert len(ids) == len(set(ids)), f"Duplicate suggestion IDs: {ids}"


# ---------------------------------------------------------------------------
# Benchmark Tests — Skill Gap Courses
# ---------------------------------------------------------------------------
class TestBenchmarkSkillGaps:
    """Validate the skill gap course recommendations."""

    @pytest.mark.benchmark
    def test_skill_gap_courses_present(self, analysis_result):
        """At least 2 skill gap courses should be suggested."""
        courses = analysis_result["skill_gap_courses"]
        assert isinstance(courses, list)
        assert len(courses) >= 2, \
            f"Expected ≥2 courses, got {len(courses)}"

    @pytest.mark.benchmark
    def test_skill_gap_course_structure(self, analysis_result):
        """Each course must have topic and description fields."""
        for course in analysis_result["skill_gap_courses"]:
            assert "topic" in course and course["topic"].strip(), \
                "Course missing or empty topic"
            assert "description" in course and course["description"].strip(), \
                "Course missing or empty description"


# ---------------------------------------------------------------------------
# Benchmark Tests — Semantic Expectations (Known CV–JD Pair)
# ---------------------------------------------------------------------------
class TestBenchmarkSemanticExpectations:
    """
    Validate semantic expectations specific to the known benchmark fixtures.
    
    Because the test CV is a Cloud/DevOps engineer and the test JD is a
    Senior DevOps role, we can assert that certain high-confidence skills
    are matched. These are "relaxed" checks — at least SOME of these
    must appear, accounting for LLM variation across runs.
    """

    @pytest.mark.benchmark
    def test_expected_matched_skills(self, analysis_result):
        """
        The CV and JD both mention AWS, Kubernetes, Terraform, Docker, and Python.
        At least 3 of these should appear in matched_skills.
        """
        expected_core_skills = {"aws", "kubernetes", "terraform", "docker", "python"}
        matched_lower = {s.lower() for s in analysis_result["matched_skills"]}
        overlap = expected_core_skills & matched_lower
        assert len(overlap) >= 3, \
            f"Expected ≥3 core skills matched, got {len(overlap)}: {overlap}"

    @pytest.mark.benchmark
    def test_expected_missing_skills(self, analysis_result):
        """
        The JD mentions GCP/Pulumi/Vault which the CV lacks.
        At least 1 should appear in missing_skills.
        """
        expected_gaps = {"gcp", "pulumi", "vault", "datadog", "jaeger", "jenkins"}
        missing_lower = {s.lower() for s in analysis_result["missing_skills"]}
        overlap = expected_gaps & missing_lower
        assert len(overlap) >= 1, \
            f"Expected at least 1 gap skill identified, got: {missing_lower}"

    @pytest.mark.benchmark
    def test_score_in_realistic_range(self, analysis_result):
        """
        Given strong overlap (AWS, K8s, Terraform, CI/CD) but clear gaps
        (GCP, Pulumi, Vault, FinServices), the score should land in 55–95.
        """
        score = analysis_result["score"]
        assert 55 <= score <= 95, \
            f"Score {score} is outside the expected 55-95 range for this CV/JD pair"

    @pytest.mark.benchmark
    def test_candidate_name_extracted_correctly(self, analysis_result):
        """The candidate's name should be 'Alex Morgan'."""
        name = analysis_result["extraction"]["cv_data"]["personal_info"]["name"]
        assert "alex" in name.lower() and "morgan" in name.lower(), \
            f"Expected 'Alex Morgan', got '{name}'"



# ===========================================================================
# GROUNDING — suggestions are pasted into the CV and the letter is signed,
# so neither may invent skills
# ===========================================================================

import re

# Asked for by the benchmark JD but absent from the benchmark CV (checked by grep).
JD_ONLY_TERMS = [
    "pagerduty", "pulumi", "datadog", "jaeger", "vault", "gcp", "gke", "bigquery", "cloud build",
    "finops", "clearance", "jenkins", "network segmentation", "secrets management", "disaster recovery",
]


# The letter may call real cost-saving work "FinOps": that frames what the CV shows rather than claiming
# a tool the candidate never used. Suggestions stay strict, because they are pasted into the CV itself.
LETTER_TERMS = [term for term in JD_ONLY_TERMS if term != "finops"]


def _claimed_terms(text, terms=JD_ONLY_TERMS):
    return [term for term in terms if re.search(r"(?<!\w)" + re.escape(term), text.lower())]


class TestBenchmarkGrounding:
    """Validate that generated text only claims what the CV shows."""

    @pytest.mark.benchmark
    def test_suggestions_do_not_claim_missing_skills(self, analysis_result):
        """No suggestion may mention a JD skill the CV lacks."""
        for suggestion in analysis_result["suggestions"]:
            claimed = _claimed_terms(suggestion["replacement_text"])
            assert not claimed, f"Suggestion '{suggestion['id']}' claims {claimed}: {suggestion['replacement_text']}"

    @pytest.mark.benchmark
    def test_cover_letter_does_not_claim_missing_skills(self, client, test_cv, test_jd):
        """The letter may only mention a missing JD skill to say the candidate has not used it yet."""
        response = client.post("/generate-cover-letter", json={"cv_text": test_cv, "job_description": test_jd})
        assert response.status_code == 200, f"Cover letter failed: {response.text}"
        letter = response.json()["cover_letter"]
        honest = re.compile(r"not yet|haven't|have not|no direct|new to|keen to|eager to|look forward to", re.I)
        for sentence in re.split(r"(?<=[.!?])\s+", letter):
            claimed = _claimed_terms(sentence, LETTER_TERMS)
            assert not claimed or honest.search(sentence), f"Letter claims {claimed}: {sentence}"
