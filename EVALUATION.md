# Model evaluation

Model: `rf-rgb-grid-v2-80913b6e0af7`. Training and threshold selection used only the 1050 labelled `candidate_tiles`. The 210 labelled `eval_set` tiles were used for reporting, not model fitting or threshold selection.

## Review threshold from candidate tiles

The training script used 5-fold stratified out-of-fold candidate predictions (1050 examples). Its overall accuracy was 72.5%. The illustrative policy selects the lowest threshold that reaches at least 80% accuracy among automatically classified tiles while keeping at least 50% of tiles automatic. If no threshold meets both targets, the script selects the highest accepted accuracy at the required minimum coverage and reports that the target was missed.

| Score threshold | Automatic tiles | Coverage | Accuracy of automatic tiles |
|---:|---:|---:|---:|
| 0.00 | 1050/1050 | 100.0% | 72.5% |
| 0.35 | 896/1050 | 85.3% | 78.2% |
| 0.40 **selected** | 773/1050 | 73.6% | 81.6% |
| 0.45 | 653/1050 | 62.2% | 85.1% |
| 0.50 | 571/1050 | 54.4% | 88.8% |
| 0.60 | 411/1050 | 39.1% | 94.2% |

Chosen threshold: **0.40**, with 773/1050 (73.6%) candidate validation tiles classified automatically at 81.6% accuracy. Target met: **yes**. This is a review policy chosen from observed scores, not a calibrated probability or a business requirement supplied by GalaxEye.

## Final evaluation set

Overall: **148/210 correct (70.5%)**. At the chosen threshold, 143/210 tiles (68.1%) were classified automatically; 120/143 of those were correct (83.9%).

| True class | Precision | Recall | F1 | Tiles |
|---|---:|---:|---:|---:|
| AnnualCrop | 76.9% | 66.7% | 71.4% | 30 |
| Forest | 78.1% | 83.3% | 80.6% | 30 |
| Highway | 47.4% | 30.0% | 36.7% | 30 |
| Industrial | 73.7% | 93.3% | 82.4% | 30 |
| Residential | 62.9% | 73.3% | 67.7% | 30 |
| River | 63.6% | 70.0% | 66.7% | 30 |
| SeaLake | 85.2% | 76.7% | 80.7% | 30 |

### Confusion matrix

Rows are true classes; columns are predicted classes. The diagonal contains correct predictions.

| True \ Predicted | AnnualCrop | Forest | Highway | Industrial | Residential | River | SeaLake |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnnualCrop | 20 | 0 | 4 | 2 | 1 | 3 | 0 |
| Forest | 0 | 25 | 1 | 0 | 0 | 1 | 3 |
| Highway | 2 | 0 | 9 | 3 | 8 | 7 | 1 |
| Industrial | 0 | 0 | 0 | 28 | 2 | 0 | 0 |
| Residential | 2 | 0 | 2 | 3 | 22 | 1 | 0 |
| River | 1 | 1 | 3 | 2 | 2 | 21 | 0 |
| SeaLake | 1 | 6 | 0 | 0 | 0 | 0 | 23 |

Most common mistakes: Highway predicted as Residential (8); Highway predicted as River (7); SeaLake predicted as Forest (6).

This evaluation split was inspected during v1 development, so it is not a blind external test. The tiles lack location metadata, and the result cannot establish performance in new geographic regions or seasons. Reconsider the threshold once the costs of wrong labels and analyst review capacity are known.
