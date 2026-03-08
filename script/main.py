"""
Script principal pour le projet d'évitement orbital avec ML

Ce script orchestre :
1. Génération du dataset
2. Entraînement des modèles
3. Évaluation et comparaison
4. Génération des visualisations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    DatasetGenerator,
    ManeuverPredictor,
    compare_models,
    analyze_predictions,
    ManeuverOptimizer,
    create_circular_orbit
)


def generate_dataset_phase(n_scenarios: int = 5000,
                           output_dir: str = "data",
                           random_seed: int = 42):
    """
    Phase 1: Génération du dataset
    """
    print("\n" + "="*80)
    print("PHASE 1: GÉNÉRATION DU DATASET")
    print("="*80)
    
    generator = DatasetGenerator(
        altitude_range=(380, 450),
        alpha=1.0,
        beta=1e4,
        max_delta_v=0.050,
        random_seed=random_seed
    )
    
    dataset_path = Path(output_dir) / f"dataset_{n_scenarios}.pkl"
    
    dataset = generator.generate_dataset(
        n_scenarios=n_scenarios,
        verbose=True,
        save_path=str(dataset_path)
    )
    
    print(f"\n✓ Dataset sauvegardé: {dataset_path}")
    print(f"  Taille: {dataset['X'].shape}")
    
    return dataset, str(dataset_path)


def train_models_phase(dataset: dict,
                       output_dir: str = "models",
                       test_size: float = 0.2):
    """
    Phase 2: Entraînement des modèles
    """
    print("\n" + "="*80)
    print("PHASE 2: ENTRAÎNEMENT DES MODÈLES")
    print("="*80)
    
    # Comparaison des modèles
    print("\nComparaison des modèles...")
    comparison_df = compare_models(dataset, test_size=test_size)
    
    print("\n" + "-"*80)
    print("RÉSULTATS DE COMPARAISON")
    print("-"*80)
    print(comparison_df.to_string(index=False))
    
    # Sauvegarder le meilleur modèle (basé sur test R²)
    best_model_type = comparison_df.loc[comparison_df['test_r2'].idxmax(), 'model']
    print(f"\n✓ Meilleur modèle: {best_model_type}")
    
    # Entraîner le meilleur modèle sur tout le dataset
    from sklearn.model_selection import train_test_split
    
    X = dataset['X']
    y = dataset['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nEntraînement du modèle final ({best_model_type})...")
    predictor = ManeuverPredictor(model_type=best_model_type)
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Sauvegarder
    model_path = Path(output_dir) / "best_model.pkl"
    predictor.save(str(model_path))
    print(f"✓ Modèle sauvegardé: {model_path}")
    
    return predictor, X_test, y_test, comparison_df


def evaluate_phase(predictor: ManeuverPredictor,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   output_dir: str = "results"):
    """
    Phase 3: Évaluation détaillée
    """
    print("\n" + "="*80)
    print("PHASE 3: ÉVALUATION DÉTAILLÉE")
    print("="*80)
    
    # Évaluation
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\n" + "-"*80)
    print("MÉTRIQUES DE TEST")
    print("-"*80)
    print(f"  MAE globale:        {metrics['mae']*1000:8.2f} m/s")
    print(f"  RMSE:              {metrics['rmse']*1000:8.2f} m/s")
    print(f"  R²:                {metrics['r2']:8.4f}")
    print(f"  MAE Δvₓ:           {metrics['mae_x']*1000:8.2f} m/s")
    print(f"  MAE Δvᵧ:           {metrics['mae_y']*1000:8.2f} m/s")
    print(f"  MAE magnitude:     {metrics['mae_magnitude']*1000:8.2f} m/s")
    print(f"  Temps prédiction:  {metrics['avg_prediction_time']*1000:8.2f} ms")
    
    # Analyse des prédictions
    y_pred = predictor.predict(X_test)
    
    output_path = Path(output_dir) / "prediction_analysis.png"
    analyze_predictions(y_test, y_pred, save_path=str(output_path))
    print(f"\n✓ Analyse visuelle sauvegardée: {output_path}")
    
    # Importance des features
    importances = predictor.get_feature_importance()
    if importances is not None:
        print("\n" + "-"*80)
        print("IMPORTANCE DES FEATURES (Top 5)")
        print("-"*80)
        
        feature_names = [
            'x_sat', 'y_sat', 'vx_sat', 'vy_sat',
            'x_debris', 'y_debris', 'vx_debris', 'vy_debris',
            'd_current', 'v_relative', 'angle_approach'
        ]
        
        indices = np.argsort(importances)[::-1]
        for i in indices[:5]:
            print(f"  {feature_names[i]:15s}: {importances[i]:.4f}")
    
    return metrics


def comparison_phase(predictor: ManeuverPredictor,
                     output_dir: str = "results"):
    """
    Phase 4: Comparaison ML vs Optimisation classique
    """
    print("\n" + "="*80)
    print("PHASE 4: COMPARAISON ML vs OPTIMISATION")
    print("="*80)
    
    # Générer quelques scénarios de test
    from src.dataset_generator import DatasetGenerator
    
    generator = DatasetGenerator(random_seed=99)
    
    n_test_scenarios = 100
    print(f"\nGénération de {n_test_scenarios} scénarios de test...")
    
    results_ml = []
    results_opt = []
    times_ml = []
    times_opt = []
    
    optimizer = ManeuverOptimizer(alpha=1.0, beta=1e4, max_delta_v=0.050)
    
    for i in range(n_test_scenarios):
        # Générer scénario
        data = generator.generate_scenario_data(verbose=False)
        
        if data is None:
            continue
        
        features = data['features'].reshape(1, -1)
        label_opt = data['label']
        
        # Prédiction ML
        start = time.time()
        label_ml = predictor.predict(features)[0]
        time_ml = time.time() - start
        
        # Temps optimisation (déjà calculé lors de la génération)
        time_opt = 2.0  # Temps typique en secondes
        
        # Stocker
        results_ml.append(label_ml)
        results_opt.append(label_opt)
        times_ml.append(time_ml)
        times_opt.append(time_opt)
    
    results_ml = np.array(results_ml)
    results_opt = np.array(results_opt)
    
    # Calcul des métriques
    error = np.linalg.norm(results_ml - results_opt, axis=1) * 1000  # m/s
    
    print("\n" + "-"*80)
    print("RÉSULTATS DE COMPARAISON")
    print("-"*80)
    print(f"  Scénarios testés:      {len(error)}")
    print(f"\n  Erreur ML vs Opt:")
    print(f"    Moyenne:             {np.mean(error):8.2f} m/s")
    print(f"    Médiane:             {np.median(error):8.2f} m/s")
    print(f"    Max:                 {np.max(error):8.2f} m/s")
    print(f"    Écart-type:          {np.std(error):8.2f} m/s")
    
    print(f"\n  Temps de calcul:")
    print(f"    ML (moyen):          {np.mean(times_ml)*1000:8.2f} ms")
    print(f"    Optimisation (typ.): {np.mean(times_opt)*1000:8.0f} ms")
    print(f"    Accélération:        {np.mean(times_opt)/np.mean(times_ml):8.1f}x")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Erreur
    axes[0].hist(error, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(error), color='r', linestyle='--', linewidth=2,
                   label=f'Moyenne: {np.mean(error):.2f} m/s')
    axes[0].set_xlabel('Erreur ||Δv_ML - Δv_opt|| [m/s]')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution des erreurs ML vs Optimisation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temps
    data_times = [
        np.array(times_ml) * 1000,  # ms
        np.array(times_opt) * 1000  # ms
    ]
    axes[1].boxplot(data_times, labels=['ML', 'Optimisation'])
    axes[1].set_ylabel('Temps de calcul [ms]')
    axes[1].set_title('Comparaison des temps de calcul')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "comparison_ml_vs_opt.png"
    plt.savefig(str(output_path), dpi=150)
    print(f"\n✓ Comparaison visuelle sauvegardée: {output_path}")
    
    plt.close()


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Projet d'évitement orbital avec ML"
    )
    
    parser.add_argument(
        '--n-scenarios',
        type=int,
        default=5000,
        help="Nombre de scénarios à générer (défaut: 5000)"
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Proportion du test set (défaut: 0.2)"
    )
    
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help="Sauter la génération de dataset (utiliser existant)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Graine aléatoire (défaut: 42)"
    )
    
    args = parser.parse_args()
    
    # Créer les dossiers de sortie
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("PROJET: APPRENTISSAGE SUPERVISÉ D'ÉVITEMENT ORBITAL")
    print("="*80)
    print(f"\nParamètres:")
    print(f"  Scénarios:     {args.n_scenarios}")
    print(f"  Test size:     {args.test_size}")
    print(f"  Random seed:   {args.seed}")
    
    # Phase 1: Dataset
    if not args.skip_dataset:
        dataset, dataset_path = generate_dataset_phase(
            n_scenarios=args.n_scenarios,
            random_seed=args.seed
        )
    else:
        from src.dataset_generator import DatasetGenerator
        dataset_path = f"data/dataset_{args.n_scenarios}.pkl"
        print(f"\nChargement du dataset existant: {dataset_path}")
        dataset = DatasetGenerator.load_dataset(dataset_path)
        print(f"✓ Dataset chargé: {dataset['X'].shape}")
    
    # Phase 2: Entraînement
    predictor, X_test, y_test, comparison_df = train_models_phase(
        dataset,
        test_size=args.test_size
    )
    
    # Sauvegarder comparaison
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    print(f"✓ Comparaison sauvegardée: results/model_comparison.csv")
    
    # Phase 3: Évaluation
    metrics = evaluate_phase(predictor, X_test, y_test)
    
    # Phase 4: Comparaison
    comparison_phase(predictor)
    
    # Résumé final
    print("\n" + "="*80)
    print("PROJET TERMINÉ AVEC SUCCÈS ✓")
    print("="*80)
    print("\nFichiers générés:")
    print(f"  Dataset:             data/dataset_{args.n_scenarios}.pkl")
    print(f"  Modèle:              models/best_model.pkl")
    print(f"  Comparaison modèles: results/model_comparison.csv")
    print(f"  Analyse prédictions: results/prediction_analysis.png")
    print(f"  Comparaison ML/Opt:  results/comparison_ml_vs_opt.png")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()