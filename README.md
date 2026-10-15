# CrewSignal

CrewSignal — Personal Health Trend Support for Long-Duration Missions

Quick start (local):

1. Create a Python virtual environment and install dependencies:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Start the Flask app locally:

   export FLASK_APP=app.py
   python app.py

   The app will be available on http://0.0.0.0:5000/

3. Use the web UI at `/` to train models (uses `Health data.csv` in repository root) and to enter measurements for predictions.

Render deployment:

- The repository includes a `Procfile` and `requirements.txt`. Create a new Web Service on Render from this GitHub repository, set the Environment to "Python", and the Start Command is:

  gunicorn app:app

Notes:

- The app reads the training data directly from `Health data.csv` in the repository root.
- No external datasets are required or generated.
- Offline support is provided via a service worker and IndexedDB in the browser.

