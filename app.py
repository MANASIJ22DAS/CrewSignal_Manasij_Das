from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import numpy as np
import pandas as pd
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


def resolve_columns(df):
    resolved = {}
    for column in df.columns:
        name = str(column).strip()
        key = name.lower()
        if 'pulse' in key:
            resolved['pulse'] = name
        elif 'temperature' in key or 'temp' in key or 'body' in key:
            resolved['body_temperature'] = name
        elif 'spo2' in key or 'oxygen' in key:
            resolved['SpO2'] = name
        elif 'status' in key or 'label' in key or 'health' in key:
            resolved['status'] = name
    return resolved


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

    feature_columns = [cols['pulse'], cols['body_temperature'], cols['SpO2']]
    target_column = cols['status']

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid].astype(int)

    if len(X) < 10:
        return None, {'error': 'insufficient_data', 'rows': len(X)}

    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=40, max_depth=7, random_state=42))
    ])

    svm = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True, random_state=42, max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    metrics = {
        'rf': {
            'accuracy': round(float(accuracy_score(y_test, rf_pred)), 6),
            'precision': round(float(precision_score(y_test, rf_pred, zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, rf_pred, zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, rf_pred, zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(rf, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)))), 6),
            'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist()
        },
        'svm': {
            'accuracy': round(float(accuracy_score(y_test, svm_pred)), 6),
            'precision': round(float(precision_score(y_test, svm_pred, zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, svm_pred, zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, svm_pred, zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(svm, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)))), 6),
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
    SpO2 = float(payload['SpO2'])

    feature = np.array([[pulse, body_temperature, SpO2]])
    rf_prob = rf_model.predict_proba(feature)[0].tolist()
    rf_pred = int(rf_model.predict(feature)[0])
    svm_prob = svm_model.predict_proba(feature)[0].tolist()
    svm_pred = int(svm_model.predict(feature)[0])
    svm_decision = svm_model.decision_function(feature)[0].tolist()
    agreement = 'agreement' if rf_pred == svm_pred else 'disagreement'

    if pulse > 100 or SpO2 < 90 or body_temperature > 38:
        risk = 'Critical'
        next_step = 'Immediate clinical escalation and repeat measurement within 5 minutes.'
    elif pulse > 90 or SpO2 < 95 or body_temperature > 37.5:
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
        'SpO2': SpO2,
        'rf_pred': rf_pred,
        'svm_pred': svm_pred,
        'agreement_status': agreement,
        'timestamp': datetime.utcnow().isoformat()
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
