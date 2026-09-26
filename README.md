# GalaxEye offline tile classifier (v1)

A small, CPU-only service for classifying the supplied RGB satellite tiles. It accepts a PNG/JPEG tile, classifies it with a local model, saves the normalized image locally, and writes prediction metadata to SQLite. It does not call hosted services or download model weights at runtime.

The assignment's 1–2 page approach and trade-off discussion is in [DESIGN.md](DESIGN.md). The Part 3 answers are below.

## Quick start

Requires Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync --python 3.12 --extra dev
uv run --offline uvicorn galaxeye_ml.api:app --host 127.0.0.1 --port 8000
```

The trained `artifacts/landcover_model.joblib` is included, so the API starts without the dataset ZIP. To reproduce training with the supplied ZIP, run:

```bash
uv run --offline python -m galaxeye_ml.train --dataset-zip Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip
```

Training reads only the labelled `candidate_tiles/` folders. The first `uv sync` needs package access unless dependencies were prepared on the machine beforehand. After installation, the API and training command run without internet access.

To package the same API and bundled model in a container, build the image while package access is available, then run it with a persistent data directory:

```bash
docker build -t galaxeye-ml .
mkdir -p var
docker run --rm -p 8000:8000 -v "$(pwd)/var:/app/var" galaxeye-ml
```

The container needs no network access for inference. The mounted `var/` directory keeps predictions and uploaded tiles across restarts. For an isolated machine, transfer the built image with `docker save` and `docker load`.

## API

OpenAPI docs: <http://127.0.0.1:8000/docs>

Upload a tile:

```bash
curl -F "file=@path/to/tile.png" http://127.0.0.1:8000/tiles
```

List predictions or filter by class/status:

```bash
curl "http://127.0.0.1:8000/predictions?class=Forest&status=classified"
curl "http://127.0.0.1:8000/predictions?image_sha256=<hash>"
curl http://127.0.0.1:8000/predictions/<prediction-id>
curl http://127.0.0.1:8000/health
```

The upload response includes the predicted class, confidence, status, content hash, model version, ID, and timestamp. The model version is the SHA-256 of the exact saved model artifact. A new tile/model-version pair returns HTTP 201; re-uploading identical bytes with the same artifact returns the existing record with HTTP 200. After replacing the artifact, the same tile gets a new prediction while the old one remains available by ID. The uploaded image is stored under `var/tiles/`; SQLite metadata is stored under `var/predictions.sqlite3`. Existing databases using hash-only uniqueness are migrated on startup.

## Design and limits

The service uses a local CPU Random Forest, flags low-scoring predictions for review, and stores normalized tiles on disk with queryable metadata in SQLite. It accepts PNG/JPEG images up to 8 MiB and 4096 pixels per side. The design note covers the trade-offs, assumptions, and items left outside this thin slice.

## Check performance and run the tests

The provided eval labels are for measurement only; training ignores `eval_set/` and `eval_labels.csv`.

```bash
uv run --offline python -m galaxeye_ml.evaluate --dataset-zip Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip --report EVALUATION.md
uv run --offline pytest
```

The detailed [evaluation report](EVALUATION.md) includes precision, recall, F1, the confusion matrix, and the review-threshold trade-off. Overall, **148/210 (70.5%)** eval tiles were correct. At the selected threshold, 143/210 were classified automatically, with 120/143 (83.9%) correct. This small split is a check on this implementation, not a general accuracy guarantee.

Dataset attribution: the supplied tiles are a subset of EuroSAT (Helber et al.) using Sentinel-2 imagery (Copernicus), as noted in the dataset ZIP's README.

## Part 3: problem-solving answers

1. **If the classifier is wrong 30% of the time:** first define the decision it supports and the cost of false positives versus missed detections. Measure per-class precision/recall and confusion on representative, held-out labelled data, not just overall accuracy. Compare that with a simple baseline and the analyst's current process. Use a review/uncertain path for low-confidence predictions, and only call the model useful if it improves the workflow at an acceptable error cost.
2. **Checking an unattended offline deployment:** keep local logs and counters for received tiles, inference errors/latency, storage failures, and class/confidence distributions. Run periodic known-answer canary images and check that their outputs and model version remain stable. When analysts later label samples, compare them against saved predictions to detect drift; provide a way to export these diagnostics during maintenance.
3. **Finding wrong stored results:** reproduce with the tile ID and stored image; compare its hash and decoded pixels with the source. Check request parsing and image transforms, then run the exact stored model artifact and compare raw output. Check model/version metadata and class-index mapping. Finally inspect the persistence path, SQLite row, and any duplicate/retry behavior. This separates bad input, inference, and storage issues in order.
4. **Weakest part:** the baseline model uses simple RGB features and its confidence is an uncalibrated class score. Similar-looking land classes or different sensors/seasons may confuse it first. A representative labelled validation set and a stronger, validated model would be the first improvements.
