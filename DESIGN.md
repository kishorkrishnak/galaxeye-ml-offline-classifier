# Design note: offline land-use tile classification

## Goal and proposed system

An analyst should be able to submit a tile, get a land-use prediction, and find it later without internet access. I would package a local API, model artifact, image directory, and SQLite database. The model loads from disk at startup; inference and storage have no hosted dependencies.

The tile flow is:

1. `POST /tiles` rejects empty, invalid, oversized, or unsupported files; it accepts PNG and JPEG.
2. Hash the uploaded bytes. Return an existing result for the same hash and model version; otherwise apply the same RGB preprocessing used during training.
3. Run the local classifier and record its best class, score, model version, and review status.
4. Save an oriented RGB PNG and queryable metadata in SQLite. Return the prediction ID; on database failure, attempt to remove a newly created image.
5. `GET /predictions/{id}` retrieves one result. `GET /predictions` lists recent results and filters by class, status, or image hash. Hash filtering can show predictions from different model versions.

The working slice implements this path synchronously. Its Random Forest trains on the 1,050 labelled `candidate_tiles`; `eval_set` and `eval_labels.csv` are used only for measurement. Features include a small RGB grid, color histograms, and coarse spatial averages. This CPU baseline is quick to train and easy to explain. It classified 148/210 supplied evaluation tiles correctly; Highway was weakest at 9/30. This does not establish performance in new regions or seasons.

## Decisions and trade-offs

**Uncertain predictions.** The API saves the best class but marks a result `uncertain` below a configurable threshold, so analysts can list it for review. I would suppress the class if analysts might mistake a low-score label for a fact. Five-fold validation on candidate tiles selected `0.40` under an illustrative policy: at least 80% accuracy among automatic predictions and at least 50% automatic coverage. The [evaluation report](EVALUATION.md) shows the trade-off. Forest scores are not calibrated probabilities; a real threshold depends on error costs and review capacity.

**What to store.** SQLite holds the ID, upload-byte SHA-256, image path, class, score, status, model version, and timestamp; PNGs stay on disk. The model version hashes the exact artifact. Uniqueness on `(image hash, model version)` prevents duplicate retries while preserving results from later models. The saved PNG is normalized, so exact provenance would also require original bytes and source metadata. For fuller audits I would add preprocessing version and all class scores. Old model files are not archived here; replaying old predictions requires retaining them separately.

**What a query means.** The API supports lookup by ID and filters for class, status, and image hash. Analysts may also need spatial bounds, acquisition date, score range, or sensor filters. The supplied tiles lack that metadata, so I would confirm input formats and query needs before adding indexed fields.

**Processing and storage scale.** Synchronous requests and SQLite suit one machine at modest volume. For bursts or slow inference, I would queue resumable jobs and return job IDs. Multiple machines would need a shared image store and suitable database. An offline deployment bundle needs pinned dependencies and the model artifact; installation can happen before isolation.

## Assumptions and questions

I assume RGB tiles with one dominant class from the seven supplied labels, one machine with persistent storage, modest request volume, and an analyst who can review uncertain results. The prompt does not establish these assumptions.

I would ask: Must we handle multispectral bands? Are coordinates, capture times, and sensor IDs available? Can one tile have several classes? Which errors cost most, and how much review capacity exists? What are the throughput, latency, and crash-recovery requirements? How will analysts export or correct results? These answers would shape the model, schema, threshold, and processing mode.

The slice leaves out authentication, a review interface, spatial queries, a job queue, retraining, and monitoring. The README answers the assignment's operational and debugging questions.
