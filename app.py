from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__, static_folder='static')
DATA_PATH = 'Health data.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
rf_model = None
svm_model = None


def normalize_name(name):
    return str(name).strip().lower().replace(' ', '_')


def resolve_columns(df):
    columns = {}
    for col in df.columns:
        key = normalize_name(col)
        if 'astronaut' in key and 'id' in key:
            columns['astronaut_id'] = col
        elif 'astronaut' in key and 'name' in key:
            columns['astronaut_name'] = col
        elif 'age' in key:
            columns['age'] = col
        elif 'sex' in key:
            columns['sex'] = col
        elif 'time' in key:
            columns['time'] = col
        elif 'pulse' in key:
            columns['pulse'] = col
        elif 'temperature' in key or 'temp' in key or 'body' in key:
            columns['body_temperature'] = col
        elif 'spo2' in key or 'spo2' in key or 'oxygen' in key:
            columns['SpO2'] = col
        elif 'status' in key or 'label' in key or 'health' in key:
            columns['status'] = col
    return columns


def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def encode_target(series):
    s = series.astype(str).str.strip()
    unique = list(dict.fromkeys(s.dropna().tolist()))
    mapping = {v: idx for idx, v in enumerate(unique)}
    encoded = s.map(mapping)
    return encoded.astype(float), mapping


def build_models():
    global rf_model, svm_model
    if not os.path.exists(DATA_PATH):
        return None, {'error': 'data_not_found', 'path': DATA_PATH}

    df = pd.read_csv(DATA_PATH)
    cols = resolve_columns(df)
    required = ['pulse', 'body_temperature', 'SpO2', 'status']
    missing = [name for name in required if name not in cols]
    if missing:
        return None, {'error': 'missing_columns', 'missing': missing, 'available': list(df.columns)}

    X = df[[cols['pulse'], cols['body_temperature'], cols['SpO2']]].copy()
    X = X.apply(safe_to_numeric)
    y_raw = df[cols['status']].copy()
    y, label_mapping = encode_target(y_raw)

    X = X[(X.notna().all(axis=1)) & y.notna()]
    y = y[(X.notna().all(axis=1)) & y.notna()].astype(int)
    X = X.loc[y.index]

    if len(X) < 10:
        return None, {'error': 'insufficient_data', 'rows': len(X)}
    if len(np.unique(y)) < 2:
        return None, {'error': 'insufficient_classes', 'classes': sorted(np.unique(y).tolist())}

    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=2, random_state=42))
    ])

    svm = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42, max_iter=3000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    cv_splits = min(3, min(np.bincount(y).min(), 3))
    if cv_splits < 2:
        cv_splits = 2

    metrics = {
        'rf': {
            'accuracy': round(float(accuracy_score(y_test, rf_pred)), 6),
            'precision': round(float(precision_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(rf, X, y, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)))), 6),
            'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist()
        },
        'svm': {
            'accuracy': round(float(accuracy_score(y_test, svm_pred)), 6),
            'precision': round(float(precision_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(svm, X, y, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)))), 6),
            'confusion_matrix': confusion_matrix(y_test, svm_pred).tolist()
        }
    }

    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.joblib'))
    joblib.dump(svm, os.path.join(MODEL_DIR, 'svm_model.joblib'))
    rf_model = rf
    svm_model = svm
    return metrics, None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/healthcheck')
def healthcheck():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})


@app.route('/api/train', methods=['POST'])
def train_models():
    metrics, error = build_models()
    if error:
        return jsonify(error), 400
    return jsonify({'status': 'trained', 'metrics': metrics})


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    required = ['pulse', 'body_temperature', 'SpO2']
    for key in required:
        if key not in payload:
            return jsonify({'error': 'missing_field', 'field': key}), 400

    global rf_model, svm_model
    if rf_model is None:
        rf_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
        else:
            return jsonify({'error': 'models_not_trained'}), 400
    if svm_model is None:
        svm_path = os.path.join(MODEL_DIR, 'svm_model.joblib')
        if os.path.exists(svm_path):
            svm_model = joblib.load(svm_path)
        else:
            return jsonify({'error': 'models_not_trained'}), 400

    pulse = float(payload['pulse'])
    body_temperature = float(payload['body_temperature'])
    spo2 = float(payload['SpO2'])
    feature = np.array([[pulse, body_temperature, spo2]], dtype=float)

    rf_pred = int(rf_model.predict(feature)[0])
    svm_pred = int(svm_model.predict(feature)[0])
    rf_prob = rf_model.predict_proba(feature)[0].tolist()
    svm_prob = svm_model.predict_proba(feature)[0].tolist()
    svm_decision = svm_model.decision_function(feature)[0].tolist()
    agreement = 'agreement' if rf_pred == svm_pred else 'disagreement'

    if pulse > 100 or spo2 < 90 or body_temperature > 38:
        risk = 'Critical'
        next_step = 'Immediate clinical escalation and repeat measurement within 5 minutes.'
    elif pulse > 90 or spo2 < 95 or body_temperature > 37.5:
        risk = 'Moderate'
        next_step = 'Repeat the measurement, log symptoms, and escalate through the approved protocol.'
    else:
        risk = 'Low'
        next_step = 'Continue routine monitoring and log any new symptoms.'

    record = {
        'astronaut_id': payload.get('astronaut_id', ''),
        'astronaut_name': payload.get('astronaut_name', ''),
        'age': payload.get('age', ''),
        'sex': payload.get('sex', ''),
        'time': payload.get('time', ''),
        'pulse': pulse,
        'body_temperature': body_temperature,
        'SpO2': spo2,
        'rf_pred': rf_pred,
        'svm_pred': svm_pred,
        'agreement_status': agreement,
        'overall_risk_category': risk,
        'recommended_next_step': next_step,
        'timestamp': datetime.utcnow().isoformat(),
        'pulse_change': payload.get('pulse_change', 0),
        'body_temperature_change': payload.get('body_temperature_change', 0),
        'spo2_change': payload.get('spo2_change', 0)
    }

    if os.path.exists('records.csv'):
        history = pd.read_csv('records.csv')
        history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
        history.to_csv('records.csv', index=False)
    else:
        pd.DataFrame([record]).to_csv('records.csv', index=False)

    return jsonify({
        'rf_predicted_health_status': rf_pred,
        'svm_predicted_health_status': svm_pred,
        'model_agreement_status': agreement,
        'random_forest_probability': rf_prob,
        'svm_probability': svm_prob,
        'svm_decision_function': svm_decision,
        'overall_risk_category': risk,
        'recommended_next_step': next_step,
        'rule_based_risk': risk
    })


@app.route('/api/records', methods=['GET'])
def records():
    if os.path.exists('records.csv'):
        return jsonify(pd.read_csv('records.csv').to_dict(orient='records'))
    return jsonify([])


@app.route('/emergency_voice.mp3')
def voice():
    return send_from_directory('.', 'emergency_voice.mp3')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

