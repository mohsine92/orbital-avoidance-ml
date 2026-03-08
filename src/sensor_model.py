"""
Module de simulation de capteur optique pour détection de débris spatiaux

Ce module contient les fonctions pour :
- Simuler un capteur optique avec bruit gaussien
- Modéliser la probabilité de détection en fonction de la distance
- Ajouter des incertitudes réalistes sur position et vitesse
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SensorParameters:
    """
    Paramètres du capteur optique
    
    Attributs:
        sigma_pos (float): Écart-type du bruit sur la position [km]
        sigma_vel (float): Écart-type du bruit sur la vitesse [km/s]
        max_range (float): Portée maximale de détection [km]
        max_detection_prob (float): Probabilité maximale de détection (à distance nulle)
        random_seed (Optional[int]): Graine aléatoire pour reproductibilité
    """
    sigma_pos: float = 0.5      # 500 m d'incertitude sur position
    sigma_vel: float = 0.005    # 5 m/s d'incertitude sur vitesse
    max_range: float = 1000.0   # Portée max 1000 km
    max_detection_prob: float = 0.95  # 95% de probabilité max
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialise le générateur aléatoire si une graine est fournie"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


class OpticalSensor:
    """
    Classe représentant un capteur optique de débris spatiaux
    """
    
    def __init__(self, params: Optional[SensorParameters] = None):
        """
        Initialise le capteur optique
        
        Args:
            params: Paramètres du capteur (utilise valeurs par défaut si None)
        """
        self.params = params if params is not None else SensorParameters()
        self.rng = np.random.default_rng(self.params.random_seed)
    
    def measure_position(self, true_position: np.ndarray) -> np.ndarray:
        """
        Mesure bruitée de la position
        
        position_obs = position_true + ε_pos
        où ε_pos ~ N(0, σ_pos² I)
        
        Args:
            true_position: Position réelle [x, y] [km]
        
        Returns:
            Position observée [x_obs, y_obs] [km]
        """
        noise = self.rng.normal(0, self.params.sigma_pos, size=2)
        return true_position + noise
    
    def measure_velocity(self, true_velocity: np.ndarray) -> np.ndarray:
        """
        Mesure bruitée de la vitesse
        
        velocity_obs = velocity_true + ε_vel
        où ε_vel ~ N(0, σ_vel² I)
        
        Args:
            true_velocity: Vitesse réelle [vx, vy] [km/s]
        
        Returns:
            Vitesse observée [vx_obs, vy_obs] [km/s]
        """
        noise = self.rng.normal(0, self.params.sigma_vel, size=2)
        return true_velocity + noise
    
    def compute_detection_probability(self, distance: float) -> float:
        """
        Calcule la probabilité de détection en fonction de la distance
        
        P_detect = P_max * exp(-d² / (2 * d_max²))
        
        Args:
            distance: Distance entre satellite et débris [km]
        
        Returns:
            Probabilité de détection [0, 1]
        """
        exponent = -(distance**2) / (2 * self.params.max_range**2)
        prob = self.params.max_detection_prob * np.exp(exponent)
        return prob
    
    def is_detected(self, distance: float) -> bool:
        """
        Détermine si le débris est détecté (tirage aléatoire)
        
        Args:
            distance: Distance entre satellite et débris [km]
        
        Returns:
            True si détecté, False sinon
        """
        prob = self.compute_detection_probability(distance)
        return self.rng.random() < prob
    
    def measure_state(self, 
                      true_position: np.ndarray, 
                      true_velocity: np.ndarray,
                      distance: Optional[float] = None,
                      force_detection: bool = False) -> Tuple[Optional[np.ndarray], 
                                                                Optional[np.ndarray], 
                                                                bool]:
        """
        Mesure complète de l'état (position + vitesse) avec simulation de détection
        
        Args:
            true_position: Position réelle [x, y] [km]
            true_velocity: Vitesse réelle [vx, vy] [km/s]
            distance: Distance au débris [km] (optionnel)
            force_detection: Force la détection même si probabilité faible
        
        Returns:
            Tuple (position_obs, velocity_obs, detected)
            - Si detected = False: (None, None, False)
            - Si detected = True: (position_obs, velocity_obs, True)
        """
        # Vérifier si le débris est détecté
        detected = force_detection
        if not force_detection and distance is not None:
            detected = self.is_detected(distance)
        
        if not detected:
            return None, None, False
        
        # Mesurer position et vitesse avec bruit
        pos_obs = self.measure_position(true_position)
        vel_obs = self.measure_velocity(true_velocity)
        
        return pos_obs, vel_obs, True
    
    def get_measurement_uncertainty(self) -> Tuple[float, float]:
        """
        Retourne les incertitudes de mesure
        
        Returns:
            Tuple (sigma_pos, sigma_vel) [km, km/s]
        """
        return self.params.sigma_pos, self.params.sigma_vel
    
    def simulate_multiple_measurements(self,
                                       true_position: np.ndarray,
                                       true_velocity: np.ndarray,
                                       n_measurements: int = 10) -> dict:
        """
        Simule plusieurs mesures pour analyse statistique
        
        Args:
            true_position: Position réelle [x, y] [km]
            true_velocity: Vitesse réelle [vx, vy] [km/s]
            n_measurements: Nombre de mesures à simuler
        
        Returns:
            Dictionnaire contenant:
                - 'positions': array de positions mesurées [n, 2]
                - 'velocities': array de vitesses mesurées [n, 2]
                - 'mean_position': position moyenne
                - 'std_position': écart-type position
                - 'mean_velocity': vitesse moyenne
                - 'std_velocity': écart-type vitesse
        """
        positions = []
        velocities = []
        
        for _ in range(n_measurements):
            pos_obs = self.measure_position(true_position)
            vel_obs = self.measure_velocity(true_velocity)
            positions.append(pos_obs)
            velocities.append(vel_obs)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'mean_position': np.mean(positions, axis=0),
            'std_position': np.std(positions, axis=0),
            'mean_velocity': np.mean(velocities, axis=0),
            'std_velocity': np.std(velocities, axis=0)
        }


def add_measurement_noise(position: np.ndarray, 
                          velocity: np.ndarray,
                          sigma_pos: float = 0.5,
                          sigma_vel: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fonction utilitaire pour ajouter du bruit à des mesures
    
    Args:
        position: Position vraie [x, y] [km]
        velocity: Vitesse vraie [vx, vy] [km/s]
        sigma_pos: Écart-type du bruit position [km]
        sigma_vel: Écart-type du bruit vitesse [km/s]
    
    Returns:
        Tuple (position_noisy, velocity_noisy)
    """
    pos_noise = np.random.normal(0, sigma_pos, size=2)
    vel_noise = np.random.normal(0, sigma_vel, size=2)
    
    return position + pos_noise, velocity + vel_noise


def estimate_position_uncertainty(measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estime l'incertitude à partir de multiples mesures
    
    Args:
        measurements: Array de mesures [n_measurements, 2]
    
    Returns:
        Tuple (mean, std) - moyenne et écart-type
    """
    mean = np.mean(measurements, axis=0)
    std = np.std(measurements, axis=0)
    return mean, std


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE DE CAPTEUR OPTIQUE")
    print("=" * 60)
    
    # Position et vitesse vraies d'un débris
    true_pos = np.array([7000.0, 1000.0])  # km
    true_vel = np.array([0.5, 7.5])        # km/s
    
    print("\nÉtat réel du débris:")
    print(f"  Position: {true_pos} km")
    print(f"  Vitesse:  {true_vel} km/s")
    
    # Créer un capteur avec paramètres par défaut
    sensor = OpticalSensor()
    
    print(f"\nParamètres du capteur:")
    print(f"  σ_pos = {sensor.params.sigma_pos} km")
    print(f"  σ_vel = {sensor.params.sigma_vel} km/s")
    print(f"  Portée max = {sensor.params.max_range} km")
    print(f"  P_detect max = {sensor.params.max_detection_prob}")
    
    # Test de probabilité de détection
    print("\n" + "-" * 60)
    print("Test de probabilité de détection")
    print("-" * 60)
    
    distances = [100, 500, 1000, 1500, 2000]
    for d in distances:
        prob = sensor.compute_detection_probability(d)
        print(f"  Distance = {d:4d} km → P_detect = {prob:.4f}")
    
    # Test de mesures multiples
    print("\n" + "-" * 60)
    print("Simulation de 1000 mesures")
    print("-" * 60)
    
    results = sensor.simulate_multiple_measurements(true_pos, true_vel, n_measurements=1000)
    
    print(f"\nPosition:")
    print(f"  Vraie:    {true_pos}")
    print(f"  Moyenne:  {results['mean_position']}")
    print(f"  Écart-type: {results['std_position']}")
    print(f"  Erreur:   {np.linalg.norm(results['mean_position'] - true_pos):.4f} km")
    
    print(f"\nVitesse:")
    print(f"  Vraie:    {true_vel}")
    print(f"  Moyenne:  {results['mean_velocity']}")
    print(f"  Écart-type: {results['std_velocity']}")
    print(f"  Erreur:   {np.linalg.norm(results['mean_velocity'] - true_vel):.6f} km/s")
    
    # Test de détection avec différentes distances
    print("\n" + "-" * 60)
    print("Test de détection (100 tentatives par distance)")
    print("-" * 60)
    
    for d in [500, 1000, 1500]:
        detections = 0
        n_trials = 100
        
        for _ in range(n_trials):
            if sensor.is_detected(d):
                detections += 1
        
        prob_theoretical = sensor.compute_detection_probability(d)
        prob_empirical = detections / n_trials
        
        print(f"  Distance {d} km:")
        print(f"    P_théorique = {prob_theoretical:.3f}")
        print(f"    P_empirique = {prob_empirical:.3f}")
    
    # Visualisation de la distribution des mesures
    print("\n" + "-" * 60)
    print("Génération de visualisation...")
    print("-" * 60)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position X
    axes[0, 0].hist(results['positions'][:, 0], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(true_pos[0], color='r', linestyle='--', linewidth=2, label='Vraie valeur')
    axes[0, 0].axvline(results['mean_position'][0], color='g', linestyle='--', linewidth=2, label='Moyenne')
    axes[0, 0].set_xlabel('Position X [km]')
    axes[0, 0].set_ylabel('Densité de probabilité')
    axes[0, 0].set_title(f'Distribution Position X (σ = {results["std_position"][0]:.3f} km)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position Y
    axes[0, 1].hist(results['positions'][:, 1], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(true_pos[1], color='r', linestyle='--', linewidth=2, label='Vraie valeur')
    axes[0, 1].axvline(results['mean_position'][1], color='g', linestyle='--', linewidth=2, label='Moyenne')
    axes[0, 1].set_xlabel('Position Y [km]')
    axes[0, 1].set_ylabel('Densité de probabilité')
    axes[0, 1].set_title(f'Distribution Position Y (σ = {results["std_position"][1]:.3f} km)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vitesse X
    axes[1, 0].hist(results['velocities'][:, 0], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(true_vel[0], color='r', linestyle='--', linewidth=2, label='Vraie valeur')
    axes[1, 0].axvline(results['mean_velocity'][0], color='g', linestyle='--', linewidth=2, label='Moyenne')
    axes[1, 0].set_xlabel('Vitesse X [km/s]')
    axes[1, 0].set_ylabel('Densité de probabilité')
    axes[1, 0].set_title(f'Distribution Vitesse X (σ = {results["std_velocity"][0]:.6f} km/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vitesse Y
    axes[1, 1].hist(results['velocities'][:, 1], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(true_vel[1], color='r', linestyle='--', linewidth=2, label='Vraie valeur')
    axes[1, 1].axvline(results['mean_velocity'][1], color='g', linestyle='--', linewidth=2, label='Moyenne')
    axes[1, 1].set_xlabel('Vitesse Y [km/s]')
    axes[1, 1].set_ylabel('Densité de probabilité')
    axes[1, 1].set_title(f'Distribution Vitesse Y (σ = {results["std_velocity"][1]:.6f} km/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/orbital_avoidance_ml/results/test_sensor.png', dpi=150)
    print("Figure sauvegardée: results/test_sensor.png")
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)