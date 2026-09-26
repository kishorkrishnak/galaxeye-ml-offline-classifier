# Design note: offline land-use tile classification

## Goal and proposed system

An analyst should be able to submit a satellite image tile, get a land-use prediction, and find that prediction later without an internet connection. I would package one local API service, a model artifact, a directory for image files, and a SQLite database. The model and its preprocessing code load from disk when the service starts. The API has no dependency on hosted inference or a remote database.

The tile flow is:

1. `POST /tiles` accepts a PNG or JPEG. Reject an empty, invalid, oversized, or unsupported image before inference.
2. Hash the uploaded bytes. If that hash already has a result from the current model version, return its saved record. Decode the image and apply the same RGB preprocessing used during training.
3. Run the local classifier and record its best class, score, model version, and whether the score is below the review threshold.
4. Save a normalized PNG on local disk and the queryable metadata in SQLite. Return the prediction ID. A failed database write should not leave a new image behind.
5. `GET /predictions/{id}` retrieves one record. `GET /predictions` lists recent records and filters by predicted class, review status, or image hash. Filtering by hash shows predictions from different model versions for the same tile.

The working slice implements this path synchronously. Its Random Forest learns from the 1,050 labelled `candidate_tiles` only. The supplied `eval_set` and `eval_labels.csv` are used for measurement, never training. The model uses a low-resolution RGB grid, color histograms, and coarse spatial averages. I chose this small baseline because it trains quickly on CPU, is easy to reproduce and explain, and is sufficient to exercise the backend path. On the supplied evaluation split it classified 148 of 210 tiles correctly; Highway was the weakest class at 9 of 30. That result is a warning against treating the model as reliable for every class, not a claim about new regions or seasons.

## Decisions and trade-offs

**Uncertain predictions.** The API saves the best class even when its score is low, but marks the record `uncertain` below a configurable threshold. This keeps the raw prediction available for investigation while giving analysts a review queue. Another option is to suppress the class entirely; I would choose that if analysts are likely to mistake a low-score label for a fact. Five-fold validation on candidate tiles selected the current `0.40` threshold under an illustrative policy: at least 80% accuracy among automatic predictions while keeping at least 50% of tiles automatic. The [evaluation report](EVALUATION.md) shows the trade-off and final eval result. A Random Forest score is not a calibrated probability; the real threshold depends on error costs and review capacity.

**What to store.** SQLite holds ID, SHA-256 of uploaded bytes, image path, predicted class, score, status, model version, and timestamp. The model version is a SHA-256 of the serialized artifact, so changing that file changes the duplicate key even when the training dataset is unchanged. The image itself is a PNG on disk so SQLite stays small and analysts can inspect the input. The current code saves normalized pixels; if exact input provenance matters, I would also retain the original bytes, content type, and source metadata. For auditability, a future record should include preprocessing version and possibly all class scores. Predictions are unique by `(image hash, model version)`, so reprocessing a tile with a new model keeps both results. The normalized image is shared by hash.

**What a query means.** The implemented API supports lookup by ID and recent-result filters for class and review status. For an analyst workflow, I would first confirm whether they need spatial bounds, acquisition date, confidence range, or source/sensor filters. Those need metadata that the supplied tiles do not contain. I would add the relevant indexed fields once their input format and query patterns are known, rather than invent coordinates from file names.

**Processing and storage scale.** A synchronous request and SQLite are simple for a local, low-volume slice. If inference becomes slow or submissions arrive in bursts, I would enqueue tiles, return a job ID, and make processing resumable. For multiple workers or machines, I would use a database and shared image store designed for that deployment. The desired throughput and failure recovery policy decide when that complexity is justified. An offline deployment bundle must include pinned dependencies and the model artifact; package installation can happen before the machine is isolated.

## Assumptions and questions

I assume each tile is an RGB image with one dominant class from the seven supplied labels, that a single local machine has persistent storage, and that modest request volume makes synchronous CPU inference acceptable. I also assume an analyst may review uncertain results. These are choices for the slice, not facts established by the prompt.

I would ask: Are source tiles always RGB, or must we handle multispectral bands? Do tiles have coordinates, capture times, or sensor IDs? Can a tile contain multiple land-use classes? Which mistakes have the highest cost, and what review capacity exists? How many tiles arrive per minute, how long may a response take, and must the service survive power loss mid-upload? How should analysts export or correct results? The answers would shape the model, schema, thresholds, and processing mode.

The thin slice leaves out authentication, an analyst review interface, spatial queries, a job queue, automatic model retraining, and deployment monitoring. The README answers the assignment's operational and failure-investigation questions.
