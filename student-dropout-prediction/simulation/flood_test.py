from locust import HttpUser, task, between

class FloodTestUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_endpoint(self):
        with open("smile.jpg", "rb") as f:
            self.client.post("/predict", files={"file": f})

