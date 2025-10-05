import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ExoplanetClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.algorithm = None
        
    def train(self, X, y, algorithm='xgboost', train_split=0.8, epochs=100, hyperparameters=None):
        """Train the model with specified algorithm"""
        self.algorithm = algorithm
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train based on algorithm
        if algorithm == 'xgboost':
            self.model = self._train_xgboost(X_train_scaled, y_train, hyperparameters)
        elif algorithm == 'catboost':
            self.model == self._train_catboost(X_train_scaled, y_train, hyperparameters)
        elif algorithm == 'lightgbm':
            self.model == self._train_lightgbm(X_train_scaled, y_train, hyperparameters)    
        elif algorithm == 'random-forest':
            self.model = self._train_random_forest(X_train_scaled, y_train, hyperparameters)
        elif algorithm == 'neural-net':
            self.model = self._train_neural_network(
                X_train_scaled, y_train, 
                X_test_scaled, y_test,
                epochs, hyperparameters
            )
        elif algorithm == 'svm':
            self.model = self._train_svm(X_train_scaled, y_train, hyperparameters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # For neural networks, convert probabilities to classes
        if algorithm == 'neural-net':
            y_pred = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': self._format_confusion_matrix(confusion_matrix(y_test, y_pred))
        }
        
        return metrics
    
    def _train_xgboost(self, X_train, y_train, hyperparams):
        """Train XGBoost model"""
        params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42
        }
        
        if hyperparams:
            params.update(hyperparams)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model
            
    def _train_catboost(self, X_train, y_train, hyperparams):
        """Train CatBoost model"""
        params = {
            'loss_function': 'MultiClass',
            'learning_rate': 0.1,
            'depth': 6,
            'iterations': 500,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False
        }
        if hyperparams:
            params.update(hyperparams)

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def _train_lightgbm(self, X_train, y_train, hyperparams):
        """Train LightGBM model"""
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'max_depth': -1,           # let LightGBM choose
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        if hyperparams:
            params.update(hyperparams)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_random_forest(self, X_train, y_train, hyperparams):
        """Train Random Forest model"""
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
        
        if hyperparams:
            if hyperparams.get('max_features') == 'none':
                hyperparams['max_features'] = None
            params.update(hyperparams)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, epochs, hyperparams):
        """Train Neural Network model"""
        # Default hyperparameters
        learning_rate = 0.001
        batch_size = 32
        hidden_layers = 5
        dropout = 0.2
        optimizer_name = 'adam'
        activation = 'relu'
        
        if hyperparams:
            learning_rate = hyperparams.get('learning_rate', learning_rate)
            batch_size = hyperparams.get('batch_size', batch_size)
            hidden_layers = hyperparams.get('hidden_layers', hidden_layers)
            dropout = hyperparams.get('dropout', dropout)
            optimizer_name = hyperparams.get('optimizer', optimizer_name)
            activation = hyperparams.get('activation', activation)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=3)
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        # Hidden layers
        neurons = [128, 64, 32, 16, 8]
        for i in range(hidden_layers):
            model.add(layers.Dense(neurons[i] if i < len(neurons) else 16, activation=activation))
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(3, activation='softmax'))
        
        # Compile
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            optimizer = 'adam'
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        return model
    
    def _train_svm(self, X_train, y_train, hyperparams):
        """Train SVM model"""
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
        
        if hyperparams:
            params.update(hyperparams)
        
        model = SVC(**params, probability=True)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, X):
        """Predict single sample"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.algorithm == 'neural-net':
            prediction_probs = self.model.predict(X_scaled, verbose=0)
            prediction = np.argmax(prediction_probs, axis=1)[0]
            confidence = np.max(prediction_probs)
        else:
            prediction = self.model.predict(X_scaled)[0]
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(X_scaled))
            else:
                confidence = 0.9
        
        # Map prediction to label
        label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
        return label_map[prediction], confidence
    
    def predict_batch(self, X):
        """Predict multiple samples"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.algorithm == 'neural-net':
            prediction_probs = self.model.predict(X_scaled, verbose=0)
            predictions = np.argmax(prediction_probs, axis=1)
            confidences = np.max(prediction_probs, axis=1)
        else:
            predictions = self.model.predict(X_scaled)
            if hasattr(self.model, 'predict_proba'):
                confidences = np.max(self.model.predict_proba(X_scaled), axis=1)
            else:
                confidences = np.full(len(predictions), 0.9)
        
        # Map predictions to labels
        label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
        labels = [label_map[p] for p in predictions]
        
        return labels, confidences
    
    def _format_confusion_matrix(self, cm):
        """Format confusion matrix for API response"""
        # Assuming binary classification or simplified view
        # For 3-class: [False Positive, Candidate, Confirmed]
        return {
            'tp': int(cm[2, 2]) if cm.shape[0] > 2 else int(cm[1, 1]),  # Confirmed correctly predicted
            'fp': int(np.sum(cm[:, 2]) - cm[2, 2]) if cm.shape[0] > 2 else int(cm[0, 1]),
            'fn': int(np.sum(cm[2, :]) - cm[2, 2]) if cm.shape[0] > 2 else int(cm[1, 0])
        }
