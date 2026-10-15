# CrewSignal - Local ML & Dashboard

This is a self-contained pipeline to compute the requested outputs locally (no external APIs). It trains Random Forest and SVM on local data, computes performance metrics (accuracy, precision, recall, F1, CV), confusion matrices, and saves plots and a JSON results file. A static dashboard reads those files for display and performs simple browser connectivity checks.

## Usage

1. Add your CSV dataset at `data/health_data.csv`. Required columns:
   - heart_rate
   - temperature
   - spo2
   - timestamp (optional)
   - label (optional) — if absent, the script will create labels using a simple rule.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run training & evaluation:
   ```
   python train_and_evaluate.py --data data/health_data.csv
   ```

   This writes `outputs/results.json` and images to `outputs/`.

4. Serve the repo locally (so the dashboard can fetch results.json):
   ```
   python -m http.server 8000
   ```
   Open: http://localhost:8000/dashboard.html

## What you get
- Model metrics for Random Forest and SVM: accuracy, precision, recall, f1_score, cv_score, confusion matrices.
- Confusion matrix heatmaps and trend charts saved to `outputs/`.
- `outputs/results.json` contains all numeric outputs that the dashboard reads.
- Dashboard performs connectivity checks using `navigator.onLine`, attempts to fetch `outputs/results.json` with timeout, and uses `localStorage` as a fallback cache.

## Note
- No external APIs are invoked. Everything is computed locally.
- If you'd like, I can add these files directly to your repository and open a pull request; say "Add files" and confirm the repository to write to.
