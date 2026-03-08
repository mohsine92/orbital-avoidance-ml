"""
Module de génération de dataset pour l'apprentissage supervisé

Ce module contient les fonctions pour :
- Générer des scénarios aléatoires de collision
- Résoudre l'optimisation pour chaque scénario
- Construire les features et labels
- Sauvegarder/charger les datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

from .orbital_mechanics import (
    create_circular_orbit, create_elliptical_orbit,
    OrbitalState, compute_orbital_period, MU_EARTH, R_EARTH
)
from .sensor_model import OpticalSensor, SensorParameters
from .collision_risk import CollisionRisk
from .optimizer import ManeuverOptimizer


class DatasetGenerator:
    """
    Générateur de dataset pour l'apprentissage supervisé
    """
    
    def __init__(self,
                 altitude_range: Tuple[float, float] = (350, 450),
                 alpha: float = 1.0,
                 beta: float = 1e4,
                 max_delta_v: float = 0.050,
                 sensor_params: Optional[SensorParameters] = None,
                 random_seed: Optional[int] = None):
        """
        Initialise le générateur de dataset
        
        Args:
            altitude_range: Plage d'altitudes [km]
            alpha: Poids carburant pour optimisation
            beta: Poids risque pour optimisation
            max_delta_v: Δv maximal [km/s]
            sensor_params: Paramètres du capteur (défaut si None)
            random_seed: Graine aléatoire
        """
        self.altitude_range = altitude_range
        self.alpha = alpha
        self.beta = beta
        self.max_delta_v = max_delta_v
        
        if sensor_params is None:
            self.sensor_params = SensorParameters(random_seed=random_seed)
        else:
            self.sensor_params = sensor_params
        
        self.sensor = OpticalSensor(self.sensor_params)
        self.optimizer = ManeuverOptimizer(alpha, beta, max_delta_v)
        self.risk_eval = CollisionRisk()
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.rng = np.random.default_rng(random_seed)
    
    def generate_random_scenario(self) -> Tuple[OrbitalState, OrbitalState, float]:
        """
        Génère un scénario aléatoire de collision potentielle
        
        Returns:
            Tuple (state_sat, state_debris, t_horizon)
        """
        # Altitude satellite
        alt_sat = self.rng.uniform(*self.altitude_range)
        
        # Altitude débris (proche du satellite pour avoir collision potentielle)
        # Delta altitude entre -10 km et +10 km
        delta_alt = self.rng.uniform(-10, 10)
        alt_debris = alt_sat + delta_alt
        
        # S'assurer que le débris est au-dessus de l'atmosphère
        alt_debris = max(alt_debris, 150)
        
        # Angles initiaux
        angle_sat = self.rng.uniform(0, 2*np.pi)
        
        # Angle débris : proche du satellite pour intersection possible
        # Delta angle entre -0.5 et +0.5 radians
        delta_angle = self.rng.uniform(-0.5, 0.5)
        angle_debris = angle_sat + delta_angle
        
        # Créer les états orbitaux
        # 80% orbites circulaires, 20% elliptiques
        if self.rng.random() < 0.8:
            state_sat = create_circular_orbit(alt_sat, angle_sat)
        else:
            # Orbite elliptique légèrement excentrique
            e = self.rng.uniform(0.001, 0.05)
            alt_peri = alt_sat * (1 - e)
            alt_apo = alt_sat * (1 + e)
            state_sat = create_elliptical_orbit(alt_peri, alt_apo, angle_sat)
        
        # Débris toujours circulaire (simplification V1)
        state_debris = create_circular_orbit(alt_debris, angle_debris)
        
        # Horizon temporel : une période orbitale du satellite
        t_horizon = compute_orbital_period(state_sat)
        
        return state_sat, state_debris, t_horizon
    
    def extract_features(self,
                         state_sat_obs: OrbitalState,
                         state_debris_obs: OrbitalState) -> np.ndarray:
        """
        Extrait les features d'un scénario
        
        Features (11 dimensions):
        - Position satellite observée (x, y)
        - Vitesse satellite observée (vx, vy)
        - Position débris observée (x, y)
        - Vitesse débris observée (vx, vy)
        - Distance actuelle
        - Vitesse relative
        - Angle d'approche
        
        Args:
            state_sat_obs: État satellite observé
            state_debris_obs: État débris observé
        
        Returns:
            Array de features [11]
        """
        # Positions et vitesses
        pos_sat = state_sat_obs.get_position()
        vel_sat = state_sat_obs.get_velocity()
        pos_debris = state_debris_obs.get_position()
        vel_debris = state_debris_obs.get_velocity()
        
        # Distance actuelle
        d_current = np.linalg.norm(pos_sat - pos_debris)
        
        # Vitesse relative
        v_relative = np.linalg.norm(vel_sat - vel_debris)
        
        # Angle d'approche (angle entre vecteur position relative et vitesse relative)
        r_rel = pos_debris - pos_sat
        v_rel_vec = vel_debris - vel_sat
        
        cos_angle = np.dot(r_rel, v_rel_vec) / (np.linalg.norm(r_rel) * np.linalg.norm(v_rel_vec) + 1e-10)
        angle_approach = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Assembler les features
        features = np.array([
            pos_sat[0], pos_sat[1],
            vel_sat[0], vel_sat[1],
            pos_debris[0], pos_debris[1],
            vel_debris[0], vel_debris[1],
            d_current,
            v_relative,
            angle_approach
        ])
        
        return features
    
    def generate_scenario_data(self, 
                               verbose: bool = False) -> Optional[Dict]:
        """
        Génère un scénario complet avec solution optimale
        
        Returns:
            Dictionnaire contenant:
            - 'features': array de features [11]
            - 'label': manœuvre optimale [2]
            - 'metadata': informations supplémentaires
            
            Retourne None si l'optimisation échoue
        """
        try:
            # Générer scénario
            state_sat_true, state_debris_true, t_horizon = self.generate_random_scenario()
            
            # Simuler observations bruitées
            pos_sat_obs, vel_sat_obs, _ = self.sensor.measure_state(
                state_sat_true.get_position(),
                state_sat_true.get_velocity(),
                force_detection=True
            )
            
            pos_debris_obs, vel_debris_obs, _ = self.sensor.measure_state(
                state_debris_true.get_position(),
                state_debris_true.get_velocity(),
                force_detection=True
            )
            
            # Créer états observés
            state_sat_obs = OrbitalState(
                pos_sat_obs[0], pos_sat_obs[1],
                vel_sat_obs[0], vel_sat_obs[1]
            )
            
            state_debris_obs = OrbitalState(
                pos_debris_obs[0], pos_debris_obs[1],
                vel_debris_obs[0], vel_debris_obs[1]
            )
            
            # Extraire features
            features = self.extract_features(state_sat_obs, state_debris_obs)
            
            # Calculer risque initial
            d_min_init, t_ca_init, v_rel_init = self.risk_eval.find_closest_approach(
                state_sat_obs, state_debris_obs, t_horizon
            )
            
            # Si pas de risque significatif, manœuvre nulle
            if d_min_init > 10.0:  # Plus de 10 km de séparation
                label = np.array([0.0, 0.0])
                
                metadata = {
                    'd_min_before': d_min_init,
                    'd_min_after': d_min_init,
                    'risk_before': 0.0,
                    'risk_after': 0.0,
                    'delta_v_magnitude': 0.0,
                    'cost': 0.0,
                    'optimization_success': True,
                    't_horizon': t_horizon
                }
            else:
                # Optimiser manœuvre
                result = self.optimizer.optimize_maneuver(
                    state_sat_obs,
                    state_debris_obs,
                    t_horizon=t_horizon,
                    sigma_pos=self.sensor_params.sigma_pos,
                    sigma_vel=self.sensor_params.sigma_vel
                )
                
                if not result.success:
                    if verbose:
                        print(f"Échec optimisation: {result.message}")
                    return None
                
                label = result.delta_v
                
                metadata = {
                    'd_min_before': result.d_min_before,
                    'd_min_after': result.d_min_after,
                    'risk_before': result.risk_before,
                    'risk_after': result.risk_after,
                    'delta_v_magnitude': result.delta_v_magnitude,
                    'cost': result.cost,
                    'optimization_success': result.success,
                    't_horizon': t_horizon
                }
            
            return {
                'features': features,
                'label': label,
                'metadata': metadata
            }
            
        except Exception as e:
            if verbose:
                print(f"Erreur génération scénario: {e}")
            return None
    
    def generate_dataset(self,
                         n_scenarios: int,
                         verbose: bool = True,
                         save_path: Optional[str] = None) -> Dict:
        """
        Génère un dataset complet
        
        Args:
            n_scenarios: Nombre de scénarios à générer
            verbose: Afficher progression
            save_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Dictionnaire contenant:
            - 'X': features [n_scenarios, 11]
            - 'y': labels [n_scenarios, 2]
            - 'metadata': liste de métadonnées
        """
        features_list = []
        labels_list = []
        metadata_list = []
        
        # Boucle avec affichage périodique
        for i in range(n_scenarios):
            if verbose and (i % max(1, n_scenarios // 10) == 0):
                print(f"  Progression: {i}/{n_scenarios} ({i/n_scenarios*100:.0f}%)")
            
            data = self.generate_scenario_data(verbose=False)
            
            if data is not None:
                features_list.append(data['features'])
                labels_list.append(data['label'])
                metadata_list.append(data['metadata'])
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        dataset = {
            'X': X,
            'y': y,
            'metadata': metadata_list,
            'n_scenarios': len(features_list),
            'feature_names': [
                'x_sat', 'y_sat', 'vx_sat', 'vy_sat',
                'x_debris', 'y_debris', 'vx_debris', 'vy_debris',
                'd_current', 'v_relative', 'angle_approach'
            ],
            'label_names': ['delta_vx', 'delta_vy'],
            'parameters': {
                'altitude_range': self.altitude_range,
                'alpha': self.alpha,
                'beta': self.beta,
                'max_delta_v': self.max_delta_v,
                'sigma_pos': self.sensor_params.sigma_pos,
                'sigma_vel': self.sensor_params.sigma_vel
            }
        }
        
        if verbose:
            print(f"\nDataset généré:")
            print(f"  Scénarios réussis: {len(features_list)}/{n_scenarios}")
            print(f"  Shape X: {X.shape}")
            print(f"  Shape y: {y.shape}")
        
        # Sauvegarder si demandé
        if save_path is not None:
            self.save_dataset(dataset, save_path)
            if verbose:
                print(f"  Sauvegardé: {save_path}")
        
        return dataset
    
    @staticmethod
    def save_dataset(dataset: Dict, filepath: str):
        """Sauvegarde le dataset en pickle"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    
    @staticmethod
    def load_dataset(filepath: str) -> Dict:
        """Charge un dataset depuis pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def dataset_to_dataframe(dataset: Dict) -> pd.DataFrame:
        """Convertit le dataset en DataFrame pandas"""
        X = dataset['X']
        y = dataset['y']
        
        # Créer DataFrame features
        df_features = pd.DataFrame(X, columns=dataset['feature_names'])
        
        # Ajouter labels
        df_features['delta_vx'] = y[:, 0]
        df_features['delta_vy'] = y[:, 1]
        df_features['delta_v_magnitude'] = np.linalg.norm(y, axis=1)
        
        # Ajouter métadonnées importantes
        if 'metadata' in dataset:
            df_features['d_min_before'] = [m['d_min_before'] for m in dataset['metadata']]
            df_features['d_min_after'] = [m['d_min_after'] for m in dataset['metadata']]
            df_features['risk_before'] = [m['risk_before'] for m in dataset['metadata']]
            df_features['risk_after'] = [m['risk_after'] for m in dataset['metadata']]
        
        return df_features


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE DE GÉNÉRATION DE DATASET")
    print("=" * 60)
    
    # Créer le générateur
    generator = DatasetGenerator(
        altitude_range=(380, 420),
        alpha=1.0,
        beta=1e4,
        max_delta_v=0.050,
        random_seed=42
    )
    
    print("\nParamètres du générateur:")
    print(f"  Altitudes: {generator.altitude_range} km")
    print(f"  α = {generator.alpha}")
    print(f"  β = {generator.beta}")
    print(f"  Δv_max = {generator.max_delta_v} km/s")
    
    # Générer un scénario de test
    print("\n" + "-" * 60)
    print("Test de génération d'un scénario")
    print("-" * 60)
    
    data = generator.generate_scenario_data(verbose=True)
    
    if data is not None:
        print("\nScénario généré avec succès!")
        print(f"\nFeatures ({len(data['features'])}):")
        for i, name in enumerate(['x_sat', 'y_sat', 'vx_sat', 'vy_sat',
                                   'x_debris', 'y_debris', 'vx_debris', 'vy_debris',
                                   'd_current', 'v_relative', 'angle_approach']):
            print(f"  {name:15s} = {data['features'][i]:.6f}")
        
        print(f"\nLabel (manœuvre optimale):")
        print(f"  Δvₓ = {data['label'][0]*1000:.2f} m/s")
        print(f"  Δvᵧ = {data['label'][1]*1000:.2f} m/s")
        print(f"  ||Δv|| = {np.linalg.norm(data['label'])*1000:.2f} m/s")
        
        print(f"\nMétadonnées:")
        for key, value in data['metadata'].items():
            if isinstance(value, float):
                print(f"  {key:25s} = {value:.6e}")
            else:
                print(f"  {key:25s} = {value}")
    
    # Générer un petit dataset
    print("\n" + "-" * 60)
    print("Génération d'un dataset de test (50 scénarios)")
    print("-" * 60)
    
    dataset = generator.generate_dataset(
        n_scenarios=50,
        verbose=True,
        save_path='/home/claude/orbital_avoidance_ml/data/test_dataset.pkl'
    )
    
    # Statistiques du dataset
    print("\n" + "-" * 60)
    print("Statistiques du dataset")
    print("-" * 60)
    
    y = dataset['y']
    delta_v_magnitudes = np.linalg.norm(y, axis=1)
    
    print(f"\nMagnitudes Δv [m/s]:")
    print(f"  Min: {delta_v_magnitudes.min()*1000:.2f}")
    print(f"  Max: {delta_v_magnitudes.max()*1000:.2f}")
    print(f"  Moyenne: {delta_v_magnitudes.mean()*1000:.2f}")
    print(f"  Médiane: {np.median(delta_v_magnitudes)*1000:.2f}")
    print(f"  Std: {delta_v_magnitudes.std()*1000:.2f}")
    
    print(f"\n% de manœuvres nulles: {(delta_v_magnitudes < 1e-6).sum() / len(delta_v_magnitudes) * 100:.1f}%")
    
    # Convertir en DataFrame
    df = DatasetGenerator.dataset_to_dataframe(dataset)
    print(f"\nDataFrame créé: {df.shape}")
    print(df.describe())
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)