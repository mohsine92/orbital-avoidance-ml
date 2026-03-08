# Apprentissage Supervisé pour Évitement Orbital

Projet de machine learning pour prédire des manœuvres d'évitement orbital optimales sous incertitude de détection optique.

## Table des matières

- [Objectif](#objectif)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)

---

## Objectif

Développer un système autonome capable d'apprendre une stratégie optimale d'évitement orbital en combinant :

1. **Mécanique orbitale** : propagation d'orbites 2D (problème à deux corps)
2. **Capteur optique bruité** : détection de débris avec incertitude
3. **Évaluation du risque** : probabilité de collision
4. **Optimisation** : manœuvre Δv minimisant carburant et risque
5. **Machine Learning** : prédiction rapide de la manœuvre optimale

---

## Architecture

### Modules principaux

| Module | Description |
|--------|-------------|
| `orbital_mechanics.py` | Propagation orbitale, équations du mouvement |
| `sensor_model.py` | Simulation capteur optique avec bruit gaussien |
| `collision_risk.py` | Évaluation risque, distance minimale d'approche |
| `optimizer.py` | Optimisation de la manœuvre (fonction coût J = α·Δv + β·Risk) |
| `dataset_generator.py` | Génération de scénarios et dataset pour ML |
| `ml_model.py` | Entraînement et prédiction (Random Forest, Gradient Boosting, MLP) |

### Pipeline complet

```
┌─────────────────┐
│  Scénario       │  État satellite + débris
│  aléatoire      │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Capteur        │  Position/vitesse observées (bruitées)
│  optique        │  σ_pos = 0.5 km, σ_vel = 5 m/s
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimisation   │  Recherche Δv optimal
│  (SLSQP/DE)     │  J = α·||Δv|| + β·Risk
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dataset        │  Features [11 dim] → Label [Δvₓ, Δvᵧ]
│  (5000+)        │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Training    │  Random Forest / GB / MLP
│  (RF, GB, MLP)  │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prédiction     │  Δv en <10 ms (vs 2s pour optimisation)
│  temps réel     │  
└─────────────────┘
```

---

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install numpy scipy scikit-learn pandas matplotlib tqdm
```

---

## Utilisation

### 1. Pipeline complet (recommandé)

Génère dataset, entraîne modèles, et évalue :

```bash
python main.py --n-scenarios 5000
```

Options :
- `--n-scenarios N` : Nombre de scénarios à générer (défaut: 5000)
- `--test-size X` : Proportion du test set (défaut: 0.2)
- `--seed N` : Graine aléatoire (défaut: 42)
- `--skip-dataset` : Utiliser dataset existant

### 2. Utilisation modulaire

#### Générer un dataset

```python
from src import DatasetGenerator

generator = DatasetGenerator(
    altitude_range=(380, 450),  # km
    alpha=1.0,
    beta=1e4,
    max_delta_v=0.050  # km/s
)

dataset = generator.generate_dataset(
    n_scenarios=5000,
    save_path="data/my_dataset.pkl"
)
```

#### Entraîner un modèle

```python
from src import ManeuverPredictor

predictor = ManeuverPredictor(model_type='random_forest')
predictor.train(X_train, y_train, X_val, y_val)
predictor.save("models/my_model.pkl")
```

#### Prédire une manœuvre

```python
from src import ManeuverPredictor

predictor = ManeuverPredictor.load("models/best_model.pkl")

# Features: [x_sat, y_sat, vx_sat, vy_sat, x_debris, y_debris, vx_debris, vy_debris, d_current, v_rel, angle]
features = np.array([7000, 0, 0, 7.5, 7010, 50, 0.1, 7.4, 50.1, 0.14, 1.57])

delta_v = predictor.predict_single(features)
print(f"Manœuvre optimale: Δvₓ={delta_v[0]*1000:.2f} m/s, Δvᵧ={delta_v[1]*1000:.2f} m/s")
```

### 3. Tests unitaires

Chaque module peut être testé indépendamment :

```bash
python -m src.orbital_mechanics
python -m src.sensor_model
python -m src.collision_risk
python -m src.optimizer
python -m src.dataset_generator
python -m src.ml_model
```

---

## Résultats attendus (Version 1)

### Objectifs de performance

| Métrique | Objectif V1 | Explication |
|----------|-------------|-------------|
| **R²** | > 0.85 | Qualité de l'ajustement |
| **MAE** | < 5 m/s | Erreur moyenne acceptable |
| **Temps prédiction** | < 10 ms | 200x plus rapide que l'optimisation |
| **Dataset** | 5000+ scénarios | Suffisant pour généralisation |

### Exemple de résultats

```
MÉTRIQUES DE TEST
────────────────────────────────────
  MAE globale:           3.24 m/s
  RMSE:                  4.87 m/s
  R²:                    0.8912
  Temps prédiction:      6.32 ms

COMPARAISON ML vs OPTIMISATION
────────────────────────────────────
  Erreur moyenne:       3.58 m/s
  Accélération:         316x plus rapide
```

---

## Structure du projet

```
orbital_avoidance_ml/
│
├── src/                          # Code source
│   ├── __init__.py
│   ├── orbital_mechanics.py      # Propagation orbites 2D
│   ├── sensor_model.py           # Capteur optique bruité
│   ├── collision_risk.py         # Évaluation risque collision
│   ├── optimizer.py              # Optimisation manœuvre
│   ├── dataset_generator.py      # Génération dataset ML
│   └── ml_model.py               # Entraînement et prédiction
│
├── data/                         # Datasets générés
│   └── dataset_5000.pkl
│
├── models/                       # Modèles entraînés
│   └── best_model.pkl
│
├── results/                      # Résultats et visualisations
│   ├── prediction_analysis.png
│   ├── comparison_ml_vs_opt.png
│   └── model_comparison.csv
│
├── notebooks/                    # Jupyter notebooks (analyse)
│
├── tests/                        # Tests unitaires
│
├── docs/                         # Documentation détaillée
│
├── main.py                       # Script principal
├── README.md                     # Ce fichier
└── requirements.txt              # Dépendances Python
```

---

## Équations clés

### Dynamique orbitale (2D)

```
d²r/dt² = -μ/r³ · r

où:
  r = [x, y]         : vecteur position [km]
  μ = 398600 km³/s²  : paramètre gravitationnel terrestre
```

### Fonction coût

```
J(Δv) = α·||Δv|| + β·Risk(Δv)

où:
  α = 1              : poids carburant
  β = 10⁴            : poids risque
  Risk = f(P_collision, d_min, v_relative)
```

### Probabilité de collision

```
P_c = Φ((r_combined - d_min) / σ_d)

où:
  Φ                  : fonction de répartition normale
  r_combined = 15 m  : rayon satellite + débris
  σ_d ≈ 0.5 km       : incertitude distance
```

---

## Contexte académique

### Hypothèses simplificatrices (Version 1)

1. **Orbite 2D** : mouvement dans le plan équatorial
2. **Problème à deux corps** : seule attraction terrestre
3. **Débris passif** : pas de manœuvre du débris
4. **Manœuvre impulsionnelle** : Δv instantané
5. **Capteur simple** : bruit gaussien indépendant

### Extensions possibles (V2+)

- [ ] Passage en 3D (inclination, RAAN)
- [ ] Perturbations J2
- [ ] Filtrage de Kalman (fusion multi-mesures)
- [ ] Scénarios multi-débris
- [ ] Reinforcement Learning (décisions multi-étapes)
- [ ] Réseau de neurones profond

---

## Auteurs

Projet développé dans le cadre d'un cours de machine learning appliqué aux systèmes spatiaux.

---

## License

MIT License - Libre d'utilisation pour recherche et éducation.

---

## Références

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*
2. Wiesel, W. E. (2010). *Spaceflight Dynamics*
3. Scikit-learn documentation: https://scikit-learn.org
4. ESA Space Debris Office: https://www.esa.int/space-debris

---

**Date de création** : Mars 2026  
**Version** : 1.0.0  
**Statut** : Production-ready pour démonstration académique ✓