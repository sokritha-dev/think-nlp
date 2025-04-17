from locust import HttpUser, task, between
import io
import random


def generate_csv():
    review_count = random.randint(10, 100)
    rows = [f"review,rating\n"]
    rows += [f"Review {i},5\n" for i in range(review_count)]
    return io.BytesIO("".join(rows).encode("utf-8"))


class UploadUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def upload_csv(self):
        files = {"file": ("test.csv", generate_csv(), "text/csv")}
        self.client.post("/api/upload/", files=files)
