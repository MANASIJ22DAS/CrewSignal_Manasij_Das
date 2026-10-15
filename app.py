from flask import Flask, request, jsonify, send_from_directory
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import threading
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='static')
DATA_PATH = os.getenv('DATA_PATH', 'Health data.csv')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config.json')
os.makedirs(MODEL_DIR, exist_ok=True)
rf_model = None
svm_model = None
label_encoder = None
MODEL_METRICS = None
TRAINING = False
TRAIN_THREAD = None


def normalize_name(value):
    return str(value).strip().lower().replace(' ', '_')


def resolve_columns(df):
    cols = {}
    for col in df.columns:
        key = normalize_name(col)
        if 'astronaut' in key and 'id' in key:
            cols['astronaut_id'] = col
        elif 'astronaut' in key and 'name' in key:
            cols['astronaut_name'] = col
        elif 'age' in key:
            cols['age'] = col
        elif 'sex' in key:
            cols['sex'] = col
        elif 'time' in key:
            cols['time'] = col
        elif 'pulse' in key:
            cols['pulse'] = col
        elif 'body' in key or 'temperature' in key or 'temp' in key:
            cols['body_temperature'] = col
        elif 'spo2' in key or 'oxygen' in key:
            cols['SpO2'] = col
        elif 'status' in key or 'label' in key or 'health' in key:
            cols['status'] = col
    return cols


def coerce_numeric(frame, columns):
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame


def prepare_dataframe(df):
    cols = resolve_columns(df)
    required = ['astronaut_id', 'astronaut_name', 'age', 'sex', 'time', 'pulse', 'body_temperature', 'SpO2', 'status']
    missing = [name for name in required if name not in cols]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    result = df[[cols['astronaut_id'], cols['astronaut_name'], cols['age'], cols['sex'], cols['time'], cols['pulse'], cols['body_temperature'], cols['SpO2'], cols['status']]].copy()
    result.columns = ['astronaut_id', 'astronaut_name', 'age', 'sex', 'time', 'pulse', 'body_temperature', 'SpO2', 'status']
    result['astronaut_id'] = result['astronaut_id'].astype(str).str.strip()
    result['astronaut_name'] = result['astronaut_name'].astype(str).str.strip()
    result['time'] = pd.to_datetime(result['time'], errors='coerce')
    result = coerce_numeric(result, ['age', 'pulse', 'body_temperature', 'SpO2'])
    result['sex'] = result['sex'].astype(str).str.strip().str.upper()
    result['status'] = result['status'].astype(str).str.strip()
    result = result.dropna(subset=['astronaut_id', 'astronaut_name', 'age', 'sex', 'time', 'pulse', 'body_temperature', 'SpO2', 'status'])
    return result.reset_index(drop=True)


def load_thresholds():
    config = {}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
    except Exception:
        config = {}
    def get_float(key, default):
        val = os.getenv(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                return float(default)
        return float(config.get(key, default))
    thresholds = {
        'PULSE_CRITICAL': get_float('PULSE_CRITICAL', 100),
        'SPO2_CRITICAL': get_float('SPO2_CRITICAL', 90),
        'TEMP_CRITICAL': get_float('TEMP_CRITICAL', 38),
        'PULSE_MODERATE': get_float('PULSE_MODERATE', 90),
        'SPO2_MODERATE': get_float('SPO2_MODERATE', 95),
        'TEMP_MODERATE': get_float('TEMP_MODERATE', 37.5)
    }
    return thresholds


THRESHOLDS = load_thresholds()


def save_model_bundle(bundle, metrics):
    bundle['metrics'] = metrics
    path = os.path.join(MODEL_DIR, 'crew_models.joblib')
    joblib.dump(bundle, path)
    logging.info('Model bundle saved to %s', path)


def load_models_from_disk():
    global rf_model, svm_model, label_encoder, MODEL_METRICS
    path = os.path.join(MODEL_DIR, 'crew_models.joblib')
    if os.path.exists(path):
        try:
            bundle = joblib.load(path)
            rf_model = bundle.get('rf')
            svm_model = bundle.get('svm')
            label_encoder = bundle.get('label_encoder')
            MODEL_METRICS = bundle.get('metrics')
            logging.info('Loaded models from %s', path)
            return True
        except Exception as e:
            logging.exception('Failed to load models from disk: %s', e)
            return False
    return False


def _make_synthetic_dataset(n_samples=120):
    rng = np.random.default_rng(42)
    ids = [f'A{(i%10)+1}' for i in range(n_samples)]
    names = [f'Crew_{(i%10)+1}' for i in range(n_samples)]
    ages = rng.integers(22, 60, size=n_samples)
    sexes = rng.choice(['M','F'], size=n_samples)
    start = datetime.utcnow() - timedelta(hours=1)
    times = [start + timedelta(seconds=int(i*30)) for i in range(n_samples)]
    pulse = (rng.normal(75,10,size=n_samples)).clip(40,160)
    temp = (rng.normal(36.6,0.6,size=n_samples)).clip(34,41)
    spo2 = (rng.normal(97,2,size=n_samples)).clip(80,100)
    status = []
    for p,t,s in zip(pulse,temp,spo2):
        st = 'healthy'
        if (s < THRESHOLDS['SPO2_CRITICAL']) or (t > THRESHOLDS['TEMP_CRITICAL']) or (p > THRESHOLDS['PULSE_CRITICAL']):
            st = 'unhealthy'
        elif (s < THRESHOLDS['SPO2_MODERATE']) or (t > THRESHOLDS['TEMP_MODERATE']) or (p > THRESHOLDS['PULSE_MODERATE']):
            st = rng.choice(['healthy','unhealthy'], p=[0.6,0.4])
        status.append(st)
    df = pd.DataFrame({
        'astronaut_id': ids,
        'astronaut_name': names,
        'age': ages,
        'sex': sexes,
        'time': times,
        'pulse': pulse,
        'body_temperature': temp,
        'SpO2': spo2,
        'status': status
    })
    return df


def build_models():
    """
    Build models from DATA_PATH if available. If DATA_PATH is missing or the CSV doesn't contain
    the required columns, generate a small synthetic dataset so the app works out-of-the-box.
    Returns (metrics, error) where error is None on success.
    """
    global rf_model, svm_model, label_encoder, MODEL_METRICS

    use_synthetic = False
    if not os.path.exists(DATA_PATH):
        logging.warning('Data path %s not found — generating synthetic dataset for demo/training', DATA_PATH)
        df = _make_synthetic_dataset()
        use_synthetic = True
    else:
        df = pd.read_csv(DATA_PATH)
        # check whether required columns exist; if not, fallback to synthetic
        try:
            df = prepare_dataframe(df)
        except ValueError as exc:
            logging.warning('Input CSV missing required columns (%s). Falling back to synthetic dataset for training.', str(exc))
            df = _make_synthetic_dataset()
            use_synthetic = True

    # at this point df should be prepared with expected columns
    # if the prepared frame doesn't have expected row count, fallback to synthetic
    try:
        df = prepare_dataframe(df)
    except Exception as e:
        # last-resort: synthetic
        logging.warning('Preparing dataframe failed unexpectedly (%s). Using synthetic dataset.', str(e))
        df = _make_synthetic_dataset()
        use_synthetic = True

    if len(df) < 12:
        logging.warning('Dataset too small (%d rows). Generating synthetic dataset.', len(df))
        df = _make_synthetic_dataset()
        use_synthetic = True

    df['time_elapsed'] = (df['time'] - df['time'].min()).dt.total_seconds() / 60.0
    feature_cols = ['astronaut_id', 'astronaut_name', 'age', 'sex', 'time_elapsed', 'pulse', 'body_temperature', 'SpO2']
    X = df[feature_cols].copy()
    y = df['status'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_transformer, ['astronaut_id', 'astronaut_name', 'sex']),
        ('num', numeric_transformer, ['age', 'time_elapsed', 'pulse', 'body_temperature', 'SpO2'])
    ])

    rf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=2, random_state=42))
    ])

    svm = Pipeline([
        ('preprocessor', preprocessor),
        ('model', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42, max_iter=3000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    cv_splits = min(3, min(np.bincount(y_encoded).min(), 3))
    if cv_splits < 2:
        cv_splits = 2

    metrics = {
        'rf': {
            'accuracy': round(float(accuracy_score(y_test, rf_pred)), 6),
            'precision': round(float(precision_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, rf_pred, average='weighted', zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(rf, X, y_encoded, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)))), 6),
            'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist()
        },
        'svm': {
            'accuracy': round(float(accuracy_score(y_test, svm_pred)), 6),
            'precision': round(float(precision_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'recall': round(float(recall_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'f1_score': round(float(f1_score(y_test, svm_pred, average='weighted', zero_division=0)), 6),
            'cv_score': round(float(np.mean(cross_val_score(svm, X, y_encoded, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)))), 6),
            'confusion_matrix': confusion_matrix(y_test, svm_pred).tolist()
        }
    }

    model_bundle = {'rf': rf, 'svm': svm, 'label_encoder': label_encoder, 'feature_columns': feature_cols, 'synthetic': use_synthetic}
    save_model_bundle(model_bundle, metrics)
    rf_model = rf
    svm_model = svm
    label_encoder = label_encoder
    MODEL_METRICS = metrics
    return metrics, None


@app.route('/')
def index():
    # serve the simple web UI
    return send_from_directory('static', 'index.html')


@app.route('/api/healthcheck')
def healthcheck():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat(), 'model_loaded': bool(MODEL_METRICS)})


@app.route('/api/train', methods=['POST'])
def train_models_endpoint():
    global TRAINING, TRAIN_THREAD
    blocking = request.args.get('blocking', 'false').lower() == 'true'
    if TRAINING:
        return jsonify({'status': 'training_already_running'}), 202

    def _train():
        global TRAINING, MODEL_METRICS
        TRAINING = True
        logging.info('Background training started')
        try:
            metrics, error = build_models()
            if error:
                logging.error('Training error: %s', error)
            else:
                MODEL_METRICS = metrics
                logging.info('Training finished')
        except Exception as e:
            logging.exception('Unexpected error during training: %s', e)
        finally:
            TRAINING = False

    if blocking:
        _train()
        return jsonify({'status': 'trained', 'metrics': MODEL_METRICS})
    else:
        TRAIN_THREAD = threading.Thread(target=_train, daemon=True)
        TRAIN_THREAD.start()
        return jsonify({'status': 'training_started'}), 202


@app.route('/api/train/status')
def train_status():
    return jsonify({'training': TRAINING})


@app.route('/api/metrics')
def model_metrics_endpoint():
    if MODEL_METRICS:
        return jsonify({'metrics': MODEL_METRICS})
    return jsonify({'metrics': None}), 204


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    required = ['astronaut_id', 'astronaut_name', 'age', 'sex', 'time', 'pulse', 'body_temperature', 'SpO2']
    for key in required:
        if key not in payload:
            return jsonify({'error': 'missing_field', 'field': key}), 400

    model_bundle_path = os.path.join(MODEL_DIR, 'crew_models.joblib')
    if not os.path.exists(model_bundle_path):
        return jsonify({'error': 'models_not_trained', 'message': 'No trained model found. Trigger /api/train'}), 400

    bundle = joblib.load(model_bundle_path)
    rf_model_local = bundle['rf']
    svm_model_local = bundle['svm']
    label_encoder_local = bundle['label_encoder']

    record = {
        'astronaut_id': str(payload['astronaut_id']).strip(),
        'astronaut_name': str(payload['astronaut_name']).strip(),
        'age': float(payload['age']),
        'sex': str(payload['sex']).strip().upper(),
        'time': pd.to_datetime(payload['time'], errors='coerce'),
        'pulse': float(payload['pulse']),
        'body_temperature': float(payload['body_temperature']),
        'SpO2': float(payload['SpO2'])
    }

    if pd.isna(record['time']):
        return jsonify({'error': 'invalid_time'}), 400

    feature = pd.DataFrame([{
        'astronaut_id': record['astronaut_id'],
        'astronaut_name': record['astronaut_name'],
        'age': record['age'],
        'sex': record['sex'],
        'time_elapsed': 0.0,
        'pulse': record['pulse'],
        'body_temperature': record['body_temperature'],
        'SpO2': record['SpO2']
    }])

    rf_pred = int(rf_model_local.predict(feature)[0])
    svm_pred = int(svm_model_local.predict(feature)[0])
    rf_prob = rf_model_local.predict_proba(feature)[0].tolist()
    svm_prob = svm_model_local.predict_proba(feature)[0].tolist()
    svm_decision = svm_model_local.decision_function(feature)[0].tolist()
    agreement = 'agreement' if rf_pred == svm_pred else 'disagreement'

    label_map = {idx: label for idx, label in enumerate(label_encoder_local.classes_)}
    rf_label = label_map.get(rf_pred, str(rf_pred))
    svm_label = label_map.get(svm_pred, str(svm_pred))

    pulse_crit = THRESHOLDS['PULSE_CRITICAL']
    spo2_crit = THRESHOLDS['SPO2_CRITICAL']
    temp_crit = THRESHOLDS['TEMP_CRITICAL']
    pulse_mod = THRESHOLDS['PULSE_MODERATE']
    spo2_mod = THRESHOLDS['SPO2_MODERATE']
    temp_mod = THRESHOLDS['TEMP_MODERATE']

    if record['pulse'] > pulse_crit or record['SpO2'] < spo2_crit or record['body_temperature'] > temp_crit:
        risk = 'Critical'
        next_step = 'Immediate clinical escalation and repeat measurement within 5 minutes.'
    elif record['pulse'] > pulse_mod or record['SpO2'] < spo2_mod or record['body_temperature'] > temp_mod:
        risk = 'Moderate'
        next_step = 'Repeat the measurement, log symptoms, and escalate through the approved protocol.'
    else:
        risk = 'Low'
        next_step = 'Continue routine monitoring and log any new symptoms.'

    return jsonify({
        'rf_predicted_health_status': rf_label,
        'svm_predicted_health_status': svm_label,
        'model_agreement_status': agreement,
        'random_forest_probability': rf_prob,
        'svm_probability': svm_prob,
        'svm_decision_function': svm_decision,
        'overall_risk_category': risk,
        'recommended_next_step': next_step,
        'rule_based_risk': risk,
        'worst_condition_label': str(label_encoder_local.classes_[-1]) if len(label_encoder_local.classes_) else ''
    })


@app.route('/api/records', methods=['GET'])
def records():
    if os.path.exists('records.csv'):
        return jsonify(pd.read_csv('records.csv').to_dict(orient='records'))
    return jsonify([])


@app.route('/emergency_voice.mp3')
def voice():
    return send_from_directory('.', 'emergency_voice.mp3')


# Try to load models at startup (non-blocking); if not present, start background training
try:
    loaded = load_models_from_disk()
    if loaded:
        logging.info('Models available at startup')
    else:
        logging.info('No pre-trained model found at startup — starting background training')
        def _startup_train():
            global MODEL_METRICS
            try:
                metrics, error = build_models()
                if error:
                    logging.error('Startup training error: %s', error)
                else:
                    MODEL_METRICS = metrics
                    logging.info('Startup training finished and models saved')
            except Exception:
                logging.exception('Unexpected error during startup training')
        t = threading.Thread(target=_startup_train, daemon=True)
        t.start()
except Exception:
    logging.exception('Error while loading models at startup')


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)
