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
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from data_processing import DataProcessor  # noqa: E402
from model_manager import ModelManager  # noqa: E402
from ml.train_models import run_training  # noqa: E402

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
MODELS_FOLDER = BASE_DIR / 'ml_models'

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
        jsonl_text = payload.get('jsonl')

        if jsonl_text is None:
            return jsonify({'error': 'No JSONL payload provided.'}), 400
        if not isinstance(jsonl_text, str):
            return jsonify({'error': 'JSONL payload must be a string.'}), 400

        lines = jsonl_text.splitlines()
        if not lines:
            return jsonify({'error': 'JSONL payload is empty.', 'results': []}), 400

        valid_records: List[Dict[str, Any]] = []
        valid_line_numbers: List[int] = []
        line_results: Dict[int, Dict[str, Any]] = {}

        for idx, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                line_results[idx] = {
                    'line': idx,
                    'status': 'error',
                    'error': 'Empty line'
                }
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                line_results[idx] = {
                    'line': idx,
                    'status': 'error',
                    'error': f'Invalid JSON: {exc.msg}'
                }
                continue

            if not isinstance(record, dict):
                line_results[idx] = {
                    'line': idx,
                    'status': 'error',
                    'error': 'Line does not contain a JSON object'
                }
                continue

            line_results[idx] = {
                'line': idx,
                'status': 'ok',
                'predictions': {}
            }
            valid_records.append(record)
            valid_line_numbers.append(idx)

        if valid_records:
            processed = data_processor.preprocess_records(valid_records)
            batch_outputs = model_manager.predict_batch(processed)

            for model_name, output in batch_outputs.items():
                predictions = output['predictions']
                confidences = output['confidences']
                probabilities = output['probabilities']

                for idx, line_no in enumerate(valid_line_numbers):
                    entry = line_results.get(line_no)
                    if not entry or entry.get('status') != 'ok':
                        continue
                    entry.setdefault('predictions', {})[model_name] = {
                        'prediction': predictions[idx],
                        'confidence': float(confidences[idx]),
                        'probability': float(probabilities[idx]),
                        'threshold': float(output['threshold'])
                    }

        results = [line_results[idx] for idx in sorted(line_results.keys())]
        total = len(lines)
        processed_count = len(valid_records)
        failed_count = sum(1 for item in results if item.get('status') == 'error')

        if processed_count == 0:
            return jsonify({
                'error': 'No valid JSON lines provided.',
                'results': results,
                'total': total,
                'processed': processed_count,
                'failed': failed_count
            }), 400

        return jsonify({
            'results': results,
            'total': total,
            'processed': processed_count,
            'failed': failed_count
        })

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
