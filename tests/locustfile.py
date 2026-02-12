from locust import HttpUser, task, between
import json

class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def classify_standard(self):
        self.client.post("/classify_document", json={
            "text": "The contract shall be valid for a period of five years from the date of signature."
        })

    @task(1)
    def classify_cached(self):
        # Sending same text repeatedly to test cache hits
        self.client.post("/classify_document", json={
            "text": "This is a repeated document to test the redis caching layer efficiency."
        })

    @task(1)
    def classify_long(self):
        text = "legal " * 200
        self.client.post("/classify_document", json={"text": text})
