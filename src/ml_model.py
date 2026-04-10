
# Machine learning module for predicting avoidance maneuvers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time

from .dataset_generator import DatasetGenerator


class ManeuverPredictor:

    #  Classe pour prédire les manœuvres d'évitement avec ML    
    def __init__(self, model_type: str = 'random_forest'):

        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.training_history = {}
        
        # Créer le modèle
        self._create_model()
    
    def _create_model(self):
        """Crée le modèle selon le type spécifié"""
        if self.model_type == 'random_forest':
            self.model = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        elif self.model_type == 'gradient_boosting':
            self.model = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            )
        elif self.model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:

       # Entraîne le modèle
        print(f"Entraînement du modèle {self.model_type}...")
        print(f"  Samples train: {len(X_train)}")
        if X_val is not None:
            print(f"  Samples val: {len(X_val)}")
        
        # Normalisation
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Entraînement
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_scaled)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        # Évaluation sur train
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled)
        
        metrics = {
            'training_time': training_time,
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred)
        }
        
        # Évaluation sur validation si fournie
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_pred_scaled = self.model.predict(X_val_scaled)
            y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled)
            
            metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
            metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
            metrics['val_r2'] = r2_score(y_val, y_val_pred)
        
        self.training_history = metrics
        
        print(f"\nEntraînement terminé en {training_time:.2f}s")
        print(f"  Train MAE: {metrics['train_mae']*1000:.2f} m/s")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        if 'val_mae' in metrics:
            print(f"  Val MAE: {metrics['val_mae']*1000:.2f} m/s")
            print(f"  Val R²: {metrics['val_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        # Prédit les manœuvres
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné!")
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def predict_single(self, features: np.ndarray) -> np.ndarray:
        
        # Prédit une seule manœuvre

        # Prédictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Métriques globales
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Erreur par composante
        mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        
        # Erreur sur magnitude
        y_test_mag = np.linalg.norm(y_test, axis=1)
        y_pred_mag = np.linalg.norm(y_pred, axis=1)
        mae_magnitude = mean_absolute_error(y_test_mag, y_pred_mag)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'mae_magnitude': mae_magnitude,
            'avg_prediction_time': prediction_time / len(X_test),
            'n_samples': len(X_test)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:

        if self.model_type == 'random_forest' or self.model_type == 'gradient_boosting':
            importances = []
            for estimator in self.model.estimators_:
                importances.append(estimator.feature_importances_)
            return np.mean(importances, axis=0)
        else:
            return None
    
    def save(self, filepath: str):
        # Sauvegarde le modèle
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Charge un modèle sauvegardé"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler_X = model_data['scaler_X']
        predictor.scaler_y = model_data['scaler_y']
        predictor.is_trained = model_data['is_trained']
        predictor.training_history = model_data['training_history']
        
        return predictor


def compare_models(dataset: Dict,
                   test_size: float = 0.2,
                   random_state: int = 42) -> pd.DataFrame:

    # Compare différents modèles de ML
    X = dataset['X']
    y = dataset['y']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Modèles à comparer
    model_types = ['random_forest', 'gradient_boosting', 'mlp']
    
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Test du modèle: {model_type}")
        print(f"{'='*60}")
        
        # Créer et entraîner
        predictor = ManeuverPredictor(model_type=model_type)
        train_metrics = predictor.train(X_train, y_train)
        
        # Évaluer
        test_metrics = predictor.evaluate(X_test, y_test)
        
        # Compiler résultats
        result = {
            'model': model_type,
            'train_mae': train_metrics['train_mae'] * 1000,  # m/s
            'train_r2': train_metrics['train_r2'],
            'test_mae': test_metrics['mae'] * 1000,  # m/s
            'test_rmse': test_metrics['rmse'] * 1000,  # m/s
            'test_r2': test_metrics['r2'],
            'test_mae_magnitude': test_metrics['mae_magnitude'] * 1000,  # m/s
            'training_time': train_metrics['training_time'],
            'avg_prediction_time': test_metrics['avg_prediction_time'] * 1000  # ms
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_predictions(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_path: Optional[str] = None):
    
   # Analyse et visualise les prédictions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Scatter Δvx
    ax = axes[0, 0]
    ax.scatter(y_true[:, 0]*1000, y_pred[:, 0]*1000, alpha=0.5, s=10)
    lim = max(abs(y_true[:, 0]).max(), abs(y_pred[:, 0]).max()) * 1000
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Idéal')
    ax.set_xlabel('Δvₓ vrai [m/s]')
    ax.set_ylabel('Δvₓ prédit [m/s]')
    ax.set_title('Prédiction Δvₓ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter Δvy
    ax = axes[0, 1]
    ax.scatter(y_true[:, 1]*1000, y_pred[:, 1]*1000, alpha=0.5, s=10)
    lim = max(abs(y_true[:, 1]).max(), abs(y_pred[:, 1]).max()) * 1000
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Idéal')
    ax.set_xlabel('Δvᵧ vrai [m/s]')
    ax.set_ylabel('Δvᵧ prédit [m/s]')
    ax.set_title('Prédiction Δvᵧ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution des erreurs
    ax = axes[1, 0]
    errors = np.linalg.norm(y_pred - y_true, axis=1) * 1000  # m/s
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, 
               label=f'Moyenne: {np.mean(errors):.2f} m/s')
    ax.axvline(np.median(errors), color='g', linestyle='--', linewidth=2,
               label=f'Médiane: {np.median(errors):.2f} m/s')
    ax.set_xlabel('Erreur ||Δv|| [m/s]')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des erreurs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Erreur vs magnitude
    ax = axes[1, 1]
    magnitudes_true = np.linalg.norm(y_true, axis=1) * 1000
    ax.scatter(magnitudes_true, errors, alpha=0.5, s=10)
    ax.set_xlabel('||Δv|| vrai [m/s]')
    ax.set_ylabel('Erreur [m/s]')
    ax.set_title('Erreur vs Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardée: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE DE MACHINE LEARNING")
    print("=" * 60)
    
    # Charger un dataset de test
    print("\nGénération d'un dataset de test...")
    
    generator = DatasetGenerator(random_seed=42)
    dataset = generator.generate_dataset(
        n_scenarios=200,
        verbose=True
    )
    
    # Split train/test
    X = dataset['X']
    y = dataset['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nSplit train/test:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Entraîner un modèle Random Forest
    print("\n" + "-" * 60)
    print("Entraînement Random Forest")
    print("-" * 60)
    
    predictor = ManeuverPredictor(model_type='random_forest')
    train_metrics = predictor.train(X_train, y_train, X_test, y_test)
    
    # Évaluation détaillée
    print("\n" + "-" * 60)
    print("Évaluation sur test set")
    print("-" * 60)
    
    test_metrics = predictor.evaluate(X_test, y_test)
    
    print(f"\nMétriques de test:")
    print(f"  MAE: {test_metrics['mae']*1000:.2f} m/s")
    print(f"  RMSE: {test_metrics['rmse']*1000:.2f} m/s")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  MAE magnitude: {test_metrics['mae_magnitude']*1000:.2f} m/s")
    print(f"  Temps prédiction moyen: {test_metrics['avg_prediction_time']*1000:.2f} ms")
    
    # Importance des features
    importances = predictor.get_feature_importance()
    if importances is not None:
        print(f"\nImportance des features:")
        feature_names = dataset['feature_names']
        indices = np.argsort(importances)[::-1]
        
        for i in indices[:5]:
            print(f"  {feature_names[i]:15s}: {importances[i]:.4f}")
    
    # Analyse des prédictions
    print("\n" + "-" * 60)
    print("Analyse des prédictions")
    print("-" * 60)
    
    y_pred = predictor.predict(X_test)
    
    fig = analyze_predictions(
        y_test, 
        y_pred,
        save_path='/home/claude/orbital_avoidance_ml/results/test_ml_predictions.png'
    )
    
    # Sauvegarder le modèle
    model_path = '/home/claude/orbital_avoidance_ml/models/test_model.pkl'
    predictor.save(model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    # Test de chargement
    predictor_loaded = ManeuverPredictor.load(model_path)
    print(f"Modèle chargé avec succès: {predictor_loaded.is_trained}")
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)
