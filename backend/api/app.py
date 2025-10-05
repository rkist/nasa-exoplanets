from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
from models import ExoplanetClassifier
from data_processing import DataProcessor

app = Flask(__name__)
CORS(app)

# Global variables
classifier = ExoplanetClassifier()
data_processor = DataProcessor()
current_model = None
model_stats = {}

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'saved_models'

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a single exoplanet candidate"""
    try:
        data = request.json or {}
        
        # Extract features
        features = {col: data.get(col) for col in data_processor.feature_columns}
        
        # Check if model is trained
        if current_model is None:
            return jsonify({
                'error': 'No model trained yet. Please train a model first.'
            }), 400
        
        # Preprocess and predict
        processed_features = data_processor.preprocess_single(features)
        prediction, confidence = classifier.predict(processed_features)
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'algorithm': model_stats.get('algorithm', 'unknown')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify/jsonl', methods=['POST'])
def classify_jsonl():
    """Classify multiple samples provided as JSON Lines"""
    try:
        if current_model is None:
            return jsonify({
                'error': 'No model trained yet. Please train a model first.'
            }), 400

        payload = request.json or {}
        jsonl_text = payload.get('jsonl')

        if jsonl_text is None:
            return jsonify({'error': 'No JSONL payload provided.'}), 400

        if not isinstance(jsonl_text, str):
            return jsonify({'error': 'JSONL payload must be a string.'}), 400

        lines = jsonl_text.splitlines()
        if not lines:
            return jsonify({'error': 'JSONL payload is empty.', 'results': []}), 400

        valid_records = []
        valid_line_numbers = []
        line_results = {}

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

            valid_records.append(record)
            valid_line_numbers.append(idx)

        if valid_records:
            processed = data_processor.preprocess_records(valid_records)
            predictions, confidences = classifier.predict_batch(processed)

            for line_no, prediction, confidence in zip(valid_line_numbers, predictions, confidences):
                line_results[line_no] = {
                    'line': line_no,
                    'status': 'ok',
                    'prediction': prediction,
                    'confidence': float(confidence)
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify/batch', methods=['POST'])
def classify_batch():
    """Classify multiple exoplanet candidates from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if model is trained
        if current_model is None:
            return jsonify({
                'error': 'No model trained yet. Please train a model first.'
            }), 400
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Read and process CSV
        df = pd.read_csv(filepath)
        processed_df = data_processor.preprocess_batch(df)
        
        # Make predictions
        predictions, confidences = classifier.predict_batch(processed_df)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['confidence'] = confidences
        
        # Count results
        confirmed = (predictions == 'Confirmed').sum()
        candidates = (predictions == 'Candidate').sum()
        false_positives = (predictions == 'False Positive').sum()
        
        # Save results
        result_filename = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        df.to_csv(result_path, index=False)
        
        return jsonify({
            'total': len(df),
            'confirmed': int(confirmed),
            'candidates': int(candidates),
            'false_positives': int(false_positives),
            'result_file': result_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_results(filename):
    """Download batch classification results"""
    try:
        filepath = os.path.join(RESULTS_FOLDER, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/train', methods=['POST'])
def train():
    """Train a new model"""
    global current_model, model_stats
    
    try:
        data = request.json
        dataset = data.get('dataset', 'combined')
        algorithm = data.get('algorithm', 'xgboost')
        train_split = data.get('train_split', 0.8)
        epochs = data.get('epochs', 100)
        hyperparameters = data.get('hyperparameters', {})
        
        print(f"Training {algorithm} on {dataset} dataset...")
        
        # Load dataset
        X, y = data_processor.load_dataset(dataset)
        
        if X is None or y is None:
            return jsonify({'error': 'Failed to load dataset'}), 400
        
        # Train model
        start_time = datetime.now()
        metrics = classifier.train(
            X, y,
            algorithm=algorithm,
            train_split=train_split,
            epochs=epochs,
            hyperparameters=hyperparameters
        )
        training_time = datetime.now() - start_time
        
        # Save model
        current_model = classifier.model
        model_path = os.path.join(MODELS_FOLDER, f'{algorithm}_model.pkl')
        joblib.dump(current_model, model_path)
        
        # Update stats
        model_stats = {
            'algorithm': algorithm,
            'dataset': dataset,
            'total_samples': len(X),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix'],
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_time': str(training_time).split('.')[0]
        }
        
        print(f"Training completed! Accuracy: {metrics['accuracy']:.4f}")
        
        return jsonify({
            'message': 'Training completed successfully',
            'accuracy': float(metrics['accuracy']) * 100,
            'f1_score': float(metrics['f1_score'])
        })
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current model statistics"""
    if not model_stats:
        return jsonify({
            'message': 'No model trained yet'
        })
    
    return jsonify(model_stats)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_model is not None
    })

if __name__ == '__main__':
    print("Starting ExoDetect AI Backend Server...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
