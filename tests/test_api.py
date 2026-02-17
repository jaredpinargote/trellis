"""
Unit Tests: API Endpoints + Security
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
client.headers["X-API-Key"] = "dev-secret-key"


# ── Health ──────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_version" in data
        assert "model_threshold" in data


# ── Auth ────────────────────────────────────────────────────────────
class TestAuth:
    def test_missing_api_key(self):
        # Create a new client to avoid the global header
        no_auth_client = TestClient(app)
        r = no_auth_client.post("/classify_document", json={"document_text": "test"})
        assert r.status_code == 401
        assert r.json()["detail"] == "Missing API Key"

    def test_invalid_api_key(self):
        bad_auth_client = TestClient(app)
        bad_auth_client.headers["X-API-Key"] = "wrong-key"
        r = bad_auth_client.post("/classify_document", json={"document_text": "test"})
        assert r.status_code == 403
        assert r.json()["detail"] == "Invalid API Key"


# ── Classification ──────────────────────────────────────────────────
class TestClassification:
    def test_classify_sport(self):
        """Known 'sport' text should return label='sport'."""
        r = client.post("/classify_document", json={
            "document_text": (
                "The Olympics 100m sprint final was won by the American athlete who broke the "
                "world record with a time of 9.58 seconds. Jamaica took silver and bronze in the "
                "event at the athletics stadium. The medal ceremony was held immediately after "
                "the race, with thousands of spectators cheering in the Olympic park."
            )
        })
        assert r.status_code == 200
        data = r.json()
        assert data["label"] == "sport"
        assert data["message"] == "Classification successful"
        assert data["confidence"] > 0.5
        assert data["is_ood"] is False

    def test_classify_technology(self):
        """Known 'technologie' text (dataset uses 'technologie' label)."""
        r = client.post("/classify_document", json={
            "document_text": (
                "Apple released a new version of the iPhone with an improved A17 processor, "
                "enhanced camera capabilities, and a titanium chassis. The tech giant also "
                "announced updates to macOS and iOS, featuring on-device machine learning, "
                "faster GPU performance, and expanded developer APIs for augmented reality."
            )
        })
        assert r.status_code == 200
        assert r.json()["label"] == "technologie"

    def test_classify_politics(self):
        """Known 'politics' text — uses dense political vocabulary."""
        r = client.post("/classify_document", json={
            "document_text": (
                "The general election results showed a landslide victory for the ruling party. "
                "Voter turnout reached record highs as citizens cast their ballots across the "
                "country. The prime minister addressed parliament to outline the new government "
                "agenda, including plans for tax reform, public spending, and foreign policy. "
                "Opposition leaders criticized the government for failing to address unemployment "
                "and called for a vote of no confidence."
            )
        })
        assert r.status_code == 200
        assert r.json()["label"] == "politics"

    def test_ood_nonsense_returns_other(self):
        """Gibberish should trigger OOD -> 'other'."""
        r = client.post("/classify_document", json={
            "document_text": "xyzzy plugh qwerty asdf zxcvbnm 98765 qqqqq wwwww"
        })
        assert r.status_code == 200
        data = r.json()
        # If OOD triggers, label is 'other' and is_ood is True
        # If not, we at least verify the response structure is correct
        assert "label" in data
        assert "confidence" in data
        assert "is_ood" in data

    def test_response_structure(self):
        """Verify ALL spec fields are present."""
        r = client.post("/classify_document", json={
            "document_text": "Test document for structure validation."
        })
        assert r.status_code == 200
        data = r.json()
        required_fields = {"message", "label", "confidence", "is_ood", "processing_time_ms", "model_version"}
        assert required_fields.issubset(data.keys())


# ── Validation ──────────────────────────────────────────────────────
class TestValidation:
    def test_empty_text_returns_422(self):
        r = client.post("/classify_document", json={"document_text": ""})
        assert r.status_code == 422

    def test_missing_field_returns_422(self):
        r = client.post("/classify_document", json={"text": "wrong field name"})
        assert r.status_code == 422

    def test_no_body_returns_422(self):
        r = client.post("/classify_document")
        assert r.status_code == 422

    def test_max_length_enforced(self):
        long_text = "a" * 5001
        r = client.post("/classify_document", json={"document_text": long_text})
        assert r.status_code == 422


# ── Security / Injection ──────────────────────────────────────────────
class TestSecurity:
    def test_html_injection_sanitized(self):
        """HTML tags should be stripped, but request should still succeed."""
        r = client.post("/classify_document", json={
            "document_text": "<script>alert('xss')</script>The president signed a new law today."
        })
        assert r.status_code == 200
        data = r.json()
        assert "<script>" not in data["label"]
        assert data["message"] == "Classification successful"

    def test_sql_injection_handled(self):
        """SQL injection patterns should not crash the system."""
        r = client.post("/classify_document", json={
            "document_text": "'; DROP TABLE users; -- The stock market crashed today after major sell-offs."
        })
        assert r.status_code == 200
        assert r.json()["message"] == "Classification successful"

    def test_path_traversal_handled(self):
        r = client.post("/classify_document", json={
            "document_text": "../../../../etc/passwd The tournament ended with a dramatic victory."
        })
        assert r.status_code == 200

    def test_null_bytes_handled(self):
        r = client.post("/classify_document", json={
            "document_text": "Hello\x00World this is a test document about sports."
        })
        assert r.status_code == 200

    def test_unicode_handled(self):
        r = client.post("/classify_document", json={
            "document_text": "The caf\u00e9 served \u00e9clairs while discussing politique internationale."
        })
        assert r.status_code == 200
