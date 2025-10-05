from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure project root is available for imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
    
from data_processing import DataProcessor  # noqa: E402
from model_manager import ModelManager  # noqa: E402
from ml.train_models import run_training  # noqa: E402

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
MODELS_FOLDER = BASE_DIR / 'saved_models'

for folder in (UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)


data_processor = DataProcessor()
model_manager = ModelManager(MODELS_FOLDER, data_processor.feature_columns)
model_manager.load_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_not_ready_response():
    return jsonify({'error': 'No model trained yet. Please train a model first.'}), 400


def _serialize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in results:
        payload.append({
            'name': item['name'],
            'accuracy': float(item['accuracy']),
            'recall': float(item['recall']),
            'best_threshold': float(item['best_threshold']),
            'report_path': item.get('report_path'),
            'scaler_path': item.get('scaler_path')
        })
    return payload


# ---------------------------------------------------------------------------
# Classification endpoints
# ---------------------------------------------------------------------------
@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        if not model_manager.is_loaded():
            return _model_not_ready_response()

        data = request.json or {}
        features = {col: data.get(col) for col in data_processor.feature_columns}

        processed = data_processor.preprocess_single(features)
        results = model_manager.predict_single(processed)

        return jsonify({
            'models': results
        })

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 500


@app.route('/api/classify/jsonl', methods=['POST'])
def classify_jsonl():
    try:
        if not model_manager.is_loaded():
            return _model_not_ready_response()

        payload = request.json or {}
        json_input = payload.get('jsonl')

        if json_input is None:
            return jsonify({'error': 'No JSON payload provided.'}), 400

        raw_objects: List[Dict[str, Any]] = []
        result_entries: List[Dict[str, Any]] = []

        if isinstance(json_input, str):
            lines = json_input.splitlines()
            if not lines:
                return jsonify({'error': 'JSON content is empty.', 'results': []}), 400

            for idx, raw_line in enumerate(lines, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    result_entries.append({
                        'line': idx,
                        'status': 'error',
                        'error': 'Empty line'
                    })
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    result_entries.append({
                        'line': idx,
                        'status': 'error',
                        'error': f'Invalid JSON: {exc.msg}'
                    })
                    continue

                if not isinstance(record, dict):
                    result_entries.append({
                        'line': idx,
                        'status': 'error',
                        'error': 'Line does not contain a JSON object'
                    })
                    continue

                raw_objects.append(record)
                result_entries.append({
                    'line': idx,
                    'status': 'ok',
                    'predictions': {}
                })
        else:
            json_obj = json_input
            if isinstance(json_obj, dict):
                raw_objects.append(json_obj)
                result_entries.append({
                    'line': 1,
                    'status': 'ok',
                    'predictions': {}
                })
            elif isinstance(json_obj, list):
                if not json_obj:
                    return jsonify({'error': 'JSON array is empty.', 'results': []}), 400

                for idx, record in enumerate(json_obj, start=1):
                    line_entry = {
                        'line': idx,
                        'status': 'ok',
                        'predictions': {}
                    }

                    if not isinstance(record, dict):
                        line_entry['status'] = 'error'
                        line_entry['error'] = 'Array item is not a JSON object'
                    else:
                        raw_objects.append(record)

                    result_entries.append(line_entry)
            else:
                return jsonify({'error': 'Unsupported JSON payload type. Provide JSON lines, a list of objects, or a single object.'}), 400

        total_records = len(result_entries)
        processed_indices: List[int] = [idx for idx, entry in enumerate(result_entries) if entry.get('status') == 'ok']

        required_features = data_processor.feature_columns

        if raw_objects:
            processed = data_processor.preprocess_records(raw_objects)
            batch_outputs = model_manager.predict_batch(processed)

            for model_name, output in batch_outputs.items():
                predictions = output['predictions']
                confidences = output['confidences']
                probabilities = output['probabilities']

                for pos, entry_index in enumerate(processed_indices):
                    entry = result_entries[entry_index]
                    entry.setdefault('predictions', {})[model_name] = {
                        'prediction': predictions[pos],
                        'confidence': float(confidences[pos]),
                        'probability': float(probabilities[pos]),
                        'threshold': float(output['threshold'])
                    }

        failed_count = sum(1 for entry in result_entries if entry.get('status') == 'error')
        processed_count = len(raw_objects)

        for entry in result_entries:
            if entry.get('status') != 'ok':
                entry.pop('predictions', None)

        response_payload = {
            'results': result_entries,
            'total': total_records,
            'processed': processed_count,
            'failed': failed_count,
            'required_features': required_features
        }

        if processed_count == 0:
            response_payload['error'] = 'No valid JSON records provided.'
            return jsonify(response_payload), 400

        return jsonify(response_payload)

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 500


@app.route('/api/classify/batch', methods=['POST'])
def classify_batch():
    try:
        if not model_manager.is_loaded():
            return _model_not_ready_response()

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)

        df = pd.read_csv(upload_path)
        processed_df = data_processor.preprocess_batch(df)

        batch_outputs = model_manager.predict_batch(processed_df.values)

        summary: Dict[str, Dict[str, Any]] = {}

        for model_name, output in batch_outputs.items():
            predictions = output['predictions']
            confidences = output['confidences']
            probabilities = output['probabilities']
            threshold = output['threshold']

            slug = model_name.lower().replace(' ', '_')
            df[f'prediction_{slug}'] = predictions
            df[f'confidence_{slug}'] = confidences
            df[f'probability_candidate_{slug}'] = probabilities

            candidates = sum(1 for label in predictions if label == model_manager.positive_label)
            likely_false = len(predictions) - candidates

            summary[model_name] = {
                'candidates': int(candidates),
                'likely_false_positives': int(likely_false),
                'threshold': float(threshold)
            }

        result_filename = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        result_path = RESULTS_FOLDER / result_filename
        df.to_csv(result_path, index=False)

        return jsonify({
            'total': len(df),
            'models': summary,
            'positive_label': model_manager.positive_label,
            'negative_label': model_manager.negative_label,
            'result_file': result_filename
        })

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_results(filename: str):
    try:
        filepath = RESULTS_FOLDER / filename
        if not filepath.exists():
            return jsonify({'error': 'Result file not found'}), 404
        return send_file(filepath, as_attachment=True)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 404


# ---------------------------------------------------------------------------
# Training & model management
# ---------------------------------------------------------------------------
@app.route('/api/train', methods=['POST'])
def train():
    try:
        data = request.json or {}
        dataset = data.get('dataset', 'combined')
        train_split = float(data.get('train_split', 0.8))
        scale = bool(data.get('scale', False))
        test_size = max(0.05, min(0.5, 1 - train_split))

        training_df = data_processor.build_training_dataframe(dataset)
        if training_df is None or training_df.empty:
            return jsonify({'error': 'Failed to build training dataset'}), 400

        start_time = datetime.now()
        results = run_training(
            data=training_df,
            feature_columns=data_processor.feature_columns,
            target='label',
            out_dir=MODELS_FOLDER,
            scale=scale,
            test_size=test_size,
            classes=[model_manager.positive_label, model_manager.negative_label],
            pos_label=model_manager.positive_label
        )
        training_time = datetime.now() - start_time

        model_manager.load_models()
        stats = model_manager.get_stats()

        return jsonify({
            'message': 'Training completed successfully',
            'results': _serialize_results(results),
            'models': stats.get('models'),
            'selected_model': stats.get('best_model'),
            'threshold': stats.get('best_model_metrics', {}).get('threshold') if stats.get('best_model_metrics') else None,
            'training_time': str(training_time).split('.')[0]
        })

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = model_manager.get_stats()
    status_code = 200 if stats.get('model_loaded') else 404
    return jsonify(stats), status_code


@app.route('/api/model/threshold', methods=['POST'])
def update_threshold():
    try:
        if not model_manager.is_loaded():
            return _model_not_ready_response()

        data = request.json or {}
        threshold = data.get('threshold')
        model_name = data.get('model')
        if threshold is None or not model_name:
            return jsonify({'error': 'Model name and threshold value are required.'}), 400

        model_manager.set_threshold(model_name, float(threshold))
        return jsonify(model_manager.get_stats())

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({'error': str(exc)}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.is_loaded()
    })


if __name__ == '__main__':  # pragma: no cover - manual launch helper
    print('Starting ExoDetect AI Backend Server...')
    print('Server running on http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
