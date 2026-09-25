from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from galaxeye_ml.api import create_app
from galaxeye_ml.train import train


def _make_dataset_zip(path: Path) -> None:
    classes = ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"]
    with zipfile.ZipFile(path, "w") as archive:
        for class_index, class_name in enumerate(classes):
            for sample in range(2):
                image = Image.new("RGB", (24, 24), (class_index * 30, sample * 20, 255 - class_index * 20))
                content = io.BytesIO()
                image.save(content, format="PNG")
                archive.writestr(
                    f"dataset/candidate_tiles/{class_name}/{class_name}_{sample}.png", content.getvalue()
                )


def test_classify_and_query_round_trip(tmp_path: Path) -> None:
    dataset_zip = tmp_path / "tiles.zip"
    model_path = tmp_path / "model.joblib"
    _make_dataset_zip(dataset_zip)
    train(dataset_zip, model_path)
    app = create_app(
        model_path=model_path,
        database_path=tmp_path / "predictions.sqlite3",
        tile_directory=tmp_path / "tiles",
        uncertainty_threshold=0.0,
    )
    image = Image.new("RGB", (24, 24), (30, 20, 225))
    payload = io.BytesIO()
    image.save(payload, format="PNG")

    with TestClient(app) as client:
        response = client.post("/tiles", files={"file": ("tile.png", payload.getvalue(), "image/png")})
        assert response.status_code == 201
        saved = response.json()
        assert saved["predicted_class"] in {
            "AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"
        }
        assert saved["status"] == "classified"
        assert (tmp_path / "tiles" / f"{saved['image_sha256']}.png").is_file()

        queried = client.get("/predictions", params={"class": saved["predicted_class"]})
        assert queried.status_code == 200
        assert queried.json()["count"] == 1
        assert queried.json()["items"][0]["id"] == saved["id"]

        duplicate = client.post("/tiles", files={"file": ("again.png", payload.getvalue(), "image/png")})
        assert duplicate.status_code == 201
        assert duplicate.json()["id"] == saved["id"]


def test_rejects_non_image_upload(tmp_path: Path) -> None:
    dataset_zip = tmp_path / "tiles.zip"
    model_path = tmp_path / "model.joblib"
    _make_dataset_zip(dataset_zip)
    train(dataset_zip, model_path)
    app = create_app(
        model_path=model_path,
        database_path=tmp_path / "predictions.sqlite3",
        tile_directory=tmp_path / "tiles",
    )
    with TestClient(app) as client:
        response = client.post("/tiles", files={"file": ("fake.png", b"not an image", "image/png")})
        assert response.status_code == 400
