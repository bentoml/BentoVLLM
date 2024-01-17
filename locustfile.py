from locust import HttpUser
from locust import task


class VLLMHttpUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:

            locust --class-picker -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={
                "prompt": "Hi there!",
                "tokens": [],
            },
        )
