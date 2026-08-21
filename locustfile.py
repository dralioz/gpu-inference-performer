import io

from locust import HttpUser, between, task
from PIL import Image


def create_test_image() -> bytes:
    image = Image.new("RGB", (640, 640), color=(120, 160, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class InferenceUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.image = create_test_image()

    @task
    def predict(self) -> None:
        self.client.post(
            "/predict",
            files={"file": ("load-test.jpg", self.image, "image/jpeg")},
            name="POST /predict",
        )
