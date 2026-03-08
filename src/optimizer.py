"""
Module d'optimisation de manœuvres d'évitement orbital

Ce module contient les fonctions pour :
- Définir une fonction coût combinant carburant et risque
- Optimiser la manœuvre Δv
- Analyser les tradeoffs α/β
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict, Callable
from dataclasses import dataclass

from .orbital_mechanics import (
    OrbitalState, apply_maneuver, propagate_orbit, 
    compute_orbital_period, MU_EARTH
)
from .collision_risk import CollisionRisk


@dataclass
class ManeuverOptimizationResult:
    """
    Résultat de l'optimisation d'une manœuvre
    
    Attributs:
        delta_v (np.ndarray): Manœuvre optimale [dvx, dvy] [km/s]
        delta_v_magnitude (float): Magnitude du Δv [km/s]
        cost (float): Coût total J
        risk_before (float): Risque avant manœuvre
        risk_after (float): Risque après manœuvre
        d_min_before (float): Distance minimale avant [km]
        d_min_after (float): Distance minimale après [km]
        success (bool): Succès de l'optimisation
        message (str): Message de l'optimiseur
        n_iterations (int): Nombre d'itérations
    """
    delta_v: np.ndarray
    delta_v_magnitude: float
    cost: float
    risk_before: float
    risk_after: float
    d_min_before: float
    d_min_after: float
    success: bool
    message: str
    n_iterations: int


class ManeuverOptimizer:
    """
    Classe pour optimiser les manœuvres d'évitement orbital
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1e4,
                 max_delta_v: float = 0.050,
                 mu: float = MU_EARTH):
        """
        Initialise l'optimiseur de manœuvres
        
        Args:
            alpha: Poids du coût carburant (coût par km/s)
            beta: Poids du risque de collision
            max_delta_v: Magnitude maximale du Δv [km/s]
            mu: Paramètre gravitationnel [km³/s²]
        """
        self.alpha = alpha
        self.beta = beta
        self.max_delta_v = max_delta_v
        self.mu = mu
        self.risk_evaluator = CollisionRisk(mu=mu)
        
        # Compteur d'évaluations
        self.n_evaluations = 0
    
    def evaluate_maneuver(self,
                          delta_v: np.ndarray,
                          state_sat: OrbitalState,
                          state_debris: OrbitalState,
                          t_horizon: float,
                          sigma_pos: float = 0.5,
                          sigma_vel: float = 0.005) -> Tuple[float, Dict]:
        """
        Évalue le coût d'une manœuvre donnée
        
        J = α * ||Δv|| + β * Risk
        
        Args:
            delta_v: Vecteur de manœuvre [dvx, dvy] [km/s]
            state_sat: État initial du satellite
            state_debris: État du débris
            t_horizon: Horizon temporel [s]
            sigma_pos: Incertitude position [km]
            sigma_vel: Incertitude vitesse [km/s]
        
        Returns:
            Tuple (cost, info_dict)
            - cost: Coût total J
            - info_dict: Informations détaillées (risk, d_min, etc.)
        """
        self.n_evaluations += 1
        
        # Appliquer la manœuvre
        state_sat_new = apply_maneuver(state_sat, delta_v)
        
        # Propager et trouver l'approche la plus rapprochée
        d_min, t_ca, v_rel = self.risk_evaluator.find_closest_approach(
            state_sat_new, state_debris, t_horizon
        )
        
        # Estimer l'incertitude (version simplifiée)
        # Pour la V1, on utilise une approximation linéaire
        sigma_d = sigma_pos
        
        # Calculer le risque
        risk = self.risk_evaluator.compute_risk_score(
            d_min, sigma_d, t_ca, v_rel
        )
        
        # Magnitude du Δv
        dv_magnitude = np.linalg.norm(delta_v)
        
        # Coût total
        cost = self.alpha * dv_magnitude + self.beta * risk
        
        # Informations détaillées
        info = {
            'risk': risk,
            'd_min': d_min,
            't_ca': t_ca,
            'v_rel': v_rel,
            'sigma_d': sigma_d,
            'dv_magnitude': dv_magnitude,
            'fuel_cost': self.alpha * dv_magnitude,
            'risk_cost': self.beta * risk
        }
        
        return cost, info
    
    def optimize_maneuver(self,
                          state_sat: OrbitalState,
                          state_debris: OrbitalState,
                          t_horizon: Optional[float] = None,
                          method: str = 'SLSQP',
                          initial_guess: Optional[np.ndarray] = None,
                          sigma_pos: float = 0.5,
                          sigma_vel: float = 0.005) -> ManeuverOptimizationResult:
        """
        Optimise la manœuvre d'évitement
        
        Args:
            state_sat: État initial du satellite
            state_debris: État du débris
            t_horizon: Horizon temporel [s] (une période orbitale si None)
            method: Méthode d'optimisation ('SLSQP', 'L-BFGS-B', 'DE')
            initial_guess: Estimation initiale [dvx, dvy] (zéro si None)
            sigma_pos: Incertitude position [km]
            sigma_vel: Incertitude vitesse [km/s]
        
        Returns:
            Résultat d'optimisation
        """
        # Horizon temporel par défaut : une période orbitale
        if t_horizon is None:
            t_horizon = compute_orbital_period(state_sat)
        
        # Estimation initiale
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0])
        
        # Fonction objectif pour l'optimiseur
        def objective(dv):
            cost, _ = self.evaluate_maneuver(
                dv, state_sat, state_debris, t_horizon, sigma_pos, sigma_vel
            )
            return cost
        
        # Contraintes : ||Δv|| ≤ max_delta_v
        bounds = [(-self.max_delta_v, self.max_delta_v),
                  (-self.max_delta_v, self.max_delta_v)]
        
        # Réinitialiser le compteur
        self.n_evaluations = 0
        
        # Calculer le risque avant manœuvre
        _, info_before = self.evaluate_maneuver(
            np.array([0.0, 0.0]), state_sat, state_debris, t_horizon
        )
        
        # Optimisation
        if method == 'DE':
            # Differential Evolution (global)
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=100,
                atol=1e-6,
                seed=42
            )
        else:
            # Méthodes locales (SLSQP, L-BFGS-B, etc.)
            result = minimize(
                objective,
                x0=initial_guess,
                method=method,
                bounds=bounds,
                options={'maxiter': 100}
            )
        
        # Évaluer la solution optimale
        delta_v_opt = result.x
        cost_opt, info_after = self.evaluate_maneuver(
            delta_v_opt, state_sat, state_debris, t_horizon
        )
        
        # Construire le résultat
        return ManeuverOptimizationResult(
            delta_v=delta_v_opt,
            delta_v_magnitude=np.linalg.norm(delta_v_opt),
            cost=cost_opt,
            risk_before=info_before['risk'],
            risk_after=info_after['risk'],
            d_min_before=info_before['d_min'],
            d_min_after=info_after['d_min'],
            success=result.success,
            message=result.message if hasattr(result, 'message') else 'Success',
            n_iterations=self.n_evaluations
        )
    
    def sensitivity_analysis(self,
                             state_sat: OrbitalState,
                             state_debris: OrbitalState,
                             alpha_values: np.ndarray,
                             beta_values: np.ndarray,
                             t_horizon: Optional[float] = None) -> Dict:
        """
        Analyse de sensibilité aux poids α et β
        
        Args:
            state_sat: État satellite
            state_debris: État débris
            alpha_values: Valeurs de α à tester
            beta_values: Valeurs de β à tester
            t_horizon: Horizon temporel [s]
        
        Returns:
            Dictionnaire avec résultats pour chaque combinaison (α, β)
        """
        results = {
            'alpha_values': alpha_values,
            'beta_values': beta_values,
            'delta_v_magnitudes': [],
            'risks_after': [],
            'd_mins_after': []
        }
        
        for alpha in alpha_values:
            for beta in beta_values:
                # Créer optimiseur avec ces poids
                optimizer = ManeuverOptimizer(
                    alpha=alpha,
                    beta=beta,
                    max_delta_v=self.max_delta_v,
                    mu=self.mu
                )
                
                # Optimiser
                opt_result = optimizer.optimize_maneuver(
                    state_sat, state_debris, t_horizon
                )
                
                # Stocker résultats
                results['delta_v_magnitudes'].append(opt_result.delta_v_magnitude)
                results['risks_after'].append(opt_result.risk_after)
                results['d_mins_after'].append(opt_result.d_min_after)
        
        # Convertir en arrays 2D
        n_alpha = len(alpha_values)
        n_beta = len(beta_values)
        
        results['delta_v_magnitudes'] = np.array(results['delta_v_magnitudes']).reshape(n_alpha, n_beta)
        results['risks_after'] = np.array(results['risks_after']).reshape(n_alpha, n_beta)
        results['d_mins_after'] = np.array(results['d_mins_after']).reshape(n_alpha, n_beta)
        
        return results


def analyze_tradeoff_curve(optimizer: ManeuverOptimizer,
                            state_sat: OrbitalState,
                            state_debris: OrbitalState,
                            t_horizon: float,
                            n_points: int = 20) -> Dict:
    """
    Génère la courbe de Pareto fuel vs risk
    
    Args:
        optimizer: Optimiseur configuré
        state_sat: État satellite
        state_debris: État débris
        t_horizon: Horizon temporel [s]
        n_points: Nombre de points sur la courbe
    
    Returns:
        Dictionnaire avec les données de la courbe de Pareto
    """
    # Varier le ratio β/α
    beta_over_alpha = np.logspace(2, 6, n_points)
    
    delta_vs = []
    risks = []
    d_mins = []
    
    for ratio in beta_over_alpha:
        opt = ManeuverOptimizer(
            alpha=1.0,
            beta=ratio,
            max_delta_v=optimizer.max_delta_v,
            mu=optimizer.mu
        )
        
        result = opt.optimize_maneuver(state_sat, state_debris, t_horizon)
        
        delta_vs.append(result.delta_v_magnitude)
        risks.append(result.risk_after)
        d_mins.append(result.d_min_after)
    
    return {
        'beta_over_alpha': beta_over_alpha,
        'delta_v_magnitudes': np.array(delta_vs),
        'risks': np.array(risks),
        'd_mins': np.array(d_mins)
    }


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE D'OPTIMISATION")
    print("=" * 60)
    
    from .orbital_mechanics import create_circular_orbit
    
    # Créer un scénario de collision
    state_sat = create_circular_orbit(altitude=400, angle=0)
    state_debris = create_circular_orbit(altitude=400.5, angle=0.01)
    
    print("\nScénario de test:")
    print(f"  Satellite: altitude 400 km")
    print(f"  Débris: altitude 400.5 km, angle décalé")
    
    # Distance initiale
    d_init = np.sqrt((state_sat.x - state_debris.x)**2 + 
                     (state_sat.y - state_debris.y)**2)
    print(f"  Distance initiale: {d_init:.2f} km")
    
    # Créer l'optimiseur
    optimizer = ManeuverOptimizer(alpha=1.0, beta=1e4, max_delta_v=0.050)
    
    print(f"\nParamètres d'optimisation:")
    print(f"  α = {optimizer.alpha}")
    print(f"  β = {optimizer.beta}")
    print(f"  Δv_max = {optimizer.max_delta_v} km/s")
    
    # Optimiser la manœuvre
    print("\n" + "-" * 60)
    print("Optimisation de la manœuvre...")
    print("-" * 60)
    
    result = optimizer.optimize_maneuver(state_sat, state_debris, method='SLSQP')
    
    print(f"\nRésultat:")
    print(f"  Succès: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Itérations: {result.n_iterations}")
    
    print(f"\nManœuvre optimale:")
    print(f"  Δvₓ = {result.delta_v[0]*1000:.2f} m/s")
    print(f"  Δvᵧ = {result.delta_v[1]*1000:.2f} m/s")
    print(f"  ||Δv|| = {result.delta_v_magnitude*1000:.2f} m/s")
    
    print(f"\nAvant manœuvre:")
    print(f"  Risque: {result.risk_before:.2e}")
    print(f"  d_min: {result.d_min_before*1000:.2f} m")
    
    print(f"\nAprès manœuvre:")
    print(f"  Risque: {result.risk_after:.2e}")
    print(f"  d_min: {result.d_min_after*1000:.2f} m")
    print(f"  Réduction du risque: {(1 - result.risk_after/result.risk_before)*100:.2f}%")
    print(f"  Augmentation de d_min: {(result.d_min_after - result.d_min_before)*1000:.2f} m")
    
    print(f"\nCoût total: {result.cost:.2e}")
    
    # Test avec méthode globale
    print("\n" + "-" * 60)
    print("Test avec Differential Evolution (global)...")
    print("-" * 60)
    
    result_de = optimizer.optimize_maneuver(state_sat, state_debris, method='DE')
    
    print(f"\nManœuvre optimale (DE):")
    print(f"  ||Δv|| = {result_de.delta_v_magnitude*1000:.2f} m/s")
    print(f"  Risque après: {result_de.risk_after:.2e}")
    print(f"  Coût: {result_de.cost:.2e}")
    
    # Comparaison
    print(f"\nComparaison SLSQP vs DE:")
    print(f"  Différence Δv: {abs(result.delta_v_magnitude - result_de.delta_v_magnitude)*1000:.3f} m/s")
    print(f"  Différence coût: {abs(result.cost - result_de.cost):.2e}")
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)