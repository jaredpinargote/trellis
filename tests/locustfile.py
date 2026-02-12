"""
Locust Load Test: High-Throughput API Stress Testing
====================================================
Simulates realistic traffic patterns including:
- Standard classification requests
- Cache-hit patterns (repeated text)
- Large document payloads (near max limit)
- Special character / unicode documents
- Programmatically generated bulk text
"""
import string
import random
from locust import HttpUser, task, between, events


def _random_text(length=500):
    """Generate random text of given character length."""
    words = []
    while len(' '.join(words)) < length:
        word_len = random.randint(3, 12)
        words.append(''.join(random.choices(string.ascii_lowercase, k=word_len)))
    return ' '.join(words)[:length]


# Pre-generate some large documents for reuse
_LARGE_DOC_1 = "The legal framework establishes " * 150  # ~4650 chars
_LARGE_DOC_2 = "International trade agreements stipulate that all parties must comply with " * 60
_LARGE_DOC_3 = "The court hereby rules in favor of the plaintiff based on the following evidence " * 55

# Category-representative texts for realistic load
_SAMPLE_TEXTS = [
    "The quarterback threw a touchdown pass in the final seconds of the championship game.",
    "NASA launched a new satellite to study dark matter and gravitational waves in deep space.",
    "Apple released new MacBook Pro with M4 chip featuring improved GPU and neural engine performance.",
    "The president signed a new executive order addressing immigration policy and national security.",
    "Researchers published findings showing a new treatment reduces cancer cell growth by 60 percent.",
    "The Renaissance period saw unprecedented advances in art, science, and political thought across Europe.",
    "Adobe released Photoshop 2026 with AI-powered tools that automate complex image editing workflows.",
    "The new Italian restaurant downtown serves authentic Neapolitan pizza with imported ingredients.",
    "The blockbuster movie earned over 500 million dollars worldwide in its opening weekend.",
    "Wall Street saw significant gains as tech stocks rallied following better than expected earnings.",
]


class HighThroughputUser(HttpUser):
    """Simulates high-volume API traffic with diverse request patterns."""
    wait_time = between(0.1, 0.5)  # Aggressive: 2-10 requests per second per user

    @task(5)
    def classify_standard(self):
        """Standard classification with category-representative text."""
        text = random.choice(_SAMPLE_TEXTS)
        self.client.post("/classify_document", json={"document_text": text})

    @task(3)
    def classify_cached(self):
        """Repeated text to test cache-hit performance."""
        self.client.post("/classify_document", json={
            "document_text": "This exact text is sent repeatedly to measure cache hit performance under load."
        })

    @task(2)
    def classify_large_document(self):
        """Near-maximum-length documents (~4500-5000 chars)."""
        doc = random.choice([_LARGE_DOC_1, _LARGE_DOC_2, _LARGE_DOC_3])
        self.client.post("/classify_document", json={"document_text": doc[:4900]})

    @task(2)
    def classify_generated_bulk(self):
        """Programmatically generated random text (unique each time, no cache hit)."""
        text = _random_text(random.randint(200, 4000))
        self.client.post("/classify_document", json={"document_text": text})

    @task(1)
    def classify_unicode(self):
        """Unicode / special character documents."""
        self.client.post("/classify_document", json={
            "document_text": "El tribunal dictamin\u00f3 que el acusado es inocente. \u00c9tude compl\u00e8te du march\u00e9 financier europ\u00e9en."
        })

    @task(1)
    def classify_mixed_injection(self):
        """Requests with injection patterns (should be sanitized, not crash)."""
        self.client.post("/classify_document", json={
            "document_text": "<b>Bold</b> text about SELECT * FROM sports WHERE score > 100; The game was exciting."
        })

    @task(1)
    def health_check(self):
        """Periodic health checks mixed into traffic."""
        self.client.get("/health")
