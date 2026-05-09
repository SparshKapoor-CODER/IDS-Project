import os
import json
import pickle
import csv
import io
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# Use tflite_runtime on Pi, fallback to TensorFlow
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: No TensorFlow or tflite_runtime found. IDS model will not load.")

app = Flask(__name__)
latest_csv_results = None

# ---------------------- Load artefacts ----------------------
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
model_path = os.path.join(artifacts_dir, 'models', 'IDSmodel.tflite')

if TFLITE_AVAILABLE and os.path.exists(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
else:
    interpreter = None
    input_details = output_details = None
    print("Model not loaded. Check artifacts/models/IDSmodel.tflite")

with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)
with open(os.path.join(artifacts_dir, 'selected_features.json'), 'r') as f:
    selected_features = json.load(f)

if len(selected_features) != 36:
    raise ValueError(f"Expected 36 selected features, got {len(selected_features)}")

# ── The 42 raw input columns (order as they appear after dropping id & attack_cat) ──
RAW_FEATURE_NAMES = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit',
    'djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean',
    'dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl',
    'ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
    'is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst',
    'is_sm_ips_ports'
]

# ── UI grouping ──────────────────────────────────────────
FEATURE_GROUPS = {
    "Basic": ['dur', 'proto', 'service', 'state'],
    "Packet": ['spkts', 'dpkts', 'sbytes', 'dbytes'],
    "Load & Loss": ['sload', 'dload', 'sloss', 'dloss'],
    "TTL & Flags": ['sttl', 'dttl', 'tcprtt', 'synack', 'ackdat'],
    "Windows": ['swin', 'dwin', 'stcpb', 'dtcpb'],
    "Jitter & Interpkt": ['sjit', 'djit', 'sinpkt', 'dinpkt'],
    "Means": ['smean', 'dmean'],
    "ct_* features": [
        'ct_srv_src','ct_state_ttl','ct_dst_ltm',
        'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
        'ct_src_ltm','ct_srv_dst'
    ],
    "HTTP/FTP": ['trans_depth','response_body_len',
                 'is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd'],
    "Other": ['rate', 'is_sm_ips_ports']
}

# ---------------------- Preprocessing pipeline ----------------------
def preprocess_and_predict_from_dict(data_dict):
    """data_dict: keys = RAW_FEATURE_NAMES, values = strings."""
    if interpreter is None:
        raise RuntimeError("Model not loaded. Check artifacts/models/IDSmodel.tflite")

    # 1. Clean and encode categoricals
    data = {k: v.strip() for k, v in data_dict.items()}
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        val = data[col]
        le = encoders.get(col)
        if le:
            try:
                data[col] = le.transform([val])[0]
            except ValueError:
                data[col] = -1   # unseen label → -1
        else:
            data[col] = -1

    # 2. Build full 42‑feature numeric row
    full_row = []
    for feat in RAW_FEATURE_NAMES:
        try:
            full_row.append(float(data[feat]))
        except ValueError:
            raise ValueError(f"Could not convert feature '{feat}' to float. Value was '{data[feat]}'")
    X_raw = np.array(full_row, dtype=np.float32).reshape(1, -1)

    # 3. Scale with the fitted StandardScaler (fitted on 42 columns)
    X_scaled = scaler.transform(X_raw)   # shape (1,42)

    # 4. Keep only the 36 selected features
    idx_map = {name: i for i, name in enumerate(RAW_FEATURE_NAMES)}
    X_reduced = np.array([X_scaled[0, idx_map[f]] for f in selected_features],
                         dtype=np.float32).reshape(1, -1)   # (1,36)

    # 5. Reshape if the model expects 3D input, else flat
    if len(input_details['shape']) == 2:    # (1,36)
        inp = X_reduced
    else:                                   # (1,36,1)
        inp = X_reduced.reshape(1, 36, 1)

    # 6. Quantise input for INT8 models
    if input_details['dtype'] == np.int8:
        in_scale, in_zero = input_details['quantization']
        inp = (inp / in_scale + in_zero).clip(-128, 127).astype(np.int8)

    # 7. Run TFLite inference
    interpreter.set_tensor(input_details['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details['index'])[0][0]

    # 8. Dequantise output and apply sigmoid if needed
    if output_details['dtype'] == np.int8:
        out_scale, out_zero = output_details['quantization']
        prob = (float(out) - out_zero) * out_scale
    else:
        prob = float(out)
    if not (0 <= prob <= 1):
        prob = 1.0 / (1.0 + np.exp(-prob))

    pred = 'Attack' if prob >= 0.5 else 'Normal'
    return pred, prob


# ---------------------- Handle CSV upload ----------------------
def process_csv_file(file_storage):
    """
    Reads an uploaded CSV file, expects:
    - First row = header (must contain id, attack_cat, label plus the 42 features)
    - Each data row = exactly 46 columns (id, 42 features, attack_cat, label)
    Returns: list of dicts with row_index, prediction, confidence, error (if any)
    """
    stream = io.StringIO(file_storage.stream.read().decode("UTF8"), newline=None)
    reader = csv.reader(stream)
    header = next(reader, None)
    if not header:
        raise ValueError("CSV file is empty")

    # Create a mapping from column name to index
    col_map = {col.strip(): idx for idx, col in enumerate(header)}

    # Ensure required columns exist
    missing = [name for name in RAW_FEATURE_NAMES if name not in col_map]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    results = []
    for row_idx, row in enumerate(reader, start=1):   # 1-based data row (skip header)
        if not row or all(v.strip() == '' for v in row):
            continue
        # Build data_dict with only the 42 features
        data_dict = {}
        for feature_name in RAW_FEATURE_NAMES:
            col_idx = col_map[feature_name]
            data_dict[feature_name] = row[col_idx].strip()

        try:
            pred, prob = preprocess_and_predict_from_dict(data_dict)
            results.append({
                'row': row_idx,
                'prediction': pred,
                'confidence': float(prob)
            })
        except Exception as e:
            results.append({
                'row': row_idx,
                'error': str(e)
            })
    return results


# ---------------------- Routes ----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    global latest_csv_results
    result = None          # for single prediction display
    csv_results = None     # for CSV upload results

    if request.method == 'POST':
        # --- Check if a file was uploaded ---
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            file = request.files['csv_file']
            try:
                csv_results = process_csv_file(file)
                latest_csv_results = csv_results
            except Exception as e:
                csv_results = [{'error': str(e)}]
                latest_csv_results = None

        else:
            # Manual entry mode or paste mode
            paste_data = request.form.get('paste_csv_row', '').strip()
            if paste_data:
                values = [v.strip() for v in paste_data.split(',')]
                if len(values) != len(RAW_FEATURE_NAMES):
                    result = f"Error: Expected {len(RAW_FEATURE_NAMES)} comma-separated values, got {len(values)}."
                else:
                    data_dict = dict(zip(RAW_FEATURE_NAMES, values))
                    try:
                        pred, prob = preprocess_and_predict_from_dict(data_dict)
                        result = f'{pred} (confidence: {prob:.4f})'
                    except Exception as e:
                        result = f'Error: {str(e)}'
            else:
                # Manual entry from individual fields
                data_dict = {}
                for f in RAW_FEATURE_NAMES:
                    val = request.form.get(f, '0').strip()
                    if val == '':
                        val = '0'
                    data_dict[f] = val
                try:
                    pred, prob = preprocess_and_predict_from_dict(data_dict)
                    result = f'{pred} (confidence: {prob:.4f})'
                except Exception as e:
                    result = f'Error: {str(e)}'

    return render_template('index.html',
                           feature_names=RAW_FEATURE_NAMES,
                           feature_groups=FEATURE_GROUPS,
                           result=result,
                           csv_results=csv_results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API – expects {'features': 'comma-separated string of 42 values'}."""
    data = request.get_json()
    raw_line = data.get('features', '')
    values = [v.strip() for v in raw_line.split(',')]
    if len(values) != len(RAW_FEATURE_NAMES):
        return jsonify({'error': f'Expected {len(RAW_FEATURE_NAMES)} features, got {len(values)}'}), 400
    data_dict = dict(zip(RAW_FEATURE_NAMES, values))
    try:
        pred, prob = preprocess_and_predict_from_dict(data_dict)
        return jsonify({'prediction': pred, 'confidence': prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv', methods=['GET', 'POST'])
def download_csv():
    """Download the latest CSV prediction results, or process an uploaded file."""
    global latest_csv_results

    # Optional POST mode: upload file directly to this endpoint.
    if request.method == 'POST' and 'csv_file' in request.files and request.files['csv_file'].filename != '':
        file = request.files['csv_file']
        try:
            latest_csv_results = process_csv_file(file)
        except Exception as e:
            return f"Error: {e}", 500

    if not latest_csv_results:
        return "No prediction results available. Upload and predict a CSV first.", 400

    results = latest_csv_results

    # Create an in-memory CSV output
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['row', 'prediction', 'confidence'])
    for res in results:
        if 'error' in res:
            writer.writerow([res['row'], 'Error', res['error']])
        else:
            writer.writerow([res['row'], res['prediction'], f"{res['confidence']:.4f}"])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ids_predictions.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)