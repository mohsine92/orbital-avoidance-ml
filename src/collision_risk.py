"""
Module d'évaluation du risque de collision entre satellite et débris

Ce module contient les fonctions pour :
- Calculer la distance minimale d'approche (Closest Approach)
- Estimer le temps avant collision
- Évaluer la probabilité de collision sous incertitude
- Calculer un score de risque
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Callable
from .orbital_mechanics import OrbitalState, propagate_orbit, MU_EARTH


# Constantes
R_SATELLITE = 0.010  # km (10 m - rayon typique satellite LEO)
R_DEBRIS = 0.005     # km (5 m - rayon typique débris)
R_COMBINED = R_SATELLITE + R_DEBRIS  # 15 m = 0.015 km


class CollisionRisk:
    """
    Classe pour évaluer le risque de collision
    """
    
    def __init__(self, 
                 r_combined: float = R_COMBINED,
                 mu: float = MU_EARTH):
        """
        Initialise l'évaluateur de risque
        
        Args:
            r_combined: Rayon combiné satellite + débris [km]
            mu: Paramètre gravitationnel [km³/s²]
        """
        self.r_combined = r_combined
        self.mu = mu
    
    def compute_distance_at_time(self,
                                 t: float,
                                 state_sat: OrbitalState,
                                 state_debris: OrbitalState,
                                 sol_sat = None,
                                 sol_debris = None) -> float:
        """
        Calcule la distance entre satellite et débris à un instant donné
        
        Args:
            t: Temps [s]
            state_sat: État initial du satellite
            state_debris: État initial du débris
            sol_sat: Solution de propagation du satellite (optionnel)
            sol_debris: Solution de propagation du débris (optionnel)
        
        Returns:
            Distance [km]
        """
        if sol_sat is not None:
            X_sat = sol_sat.sol(t)
        else:
            result = propagate_orbit(state_sat, (0, t))
            X_sat = result['X'][:, -1]
        
        if sol_debris is not None:
            X_debris = sol_debris.sol(t)
        else:
            result = propagate_orbit(state_debris, (0, t))
            X_debris = result['X'][:, -1]
        
        dx = X_sat[0] - X_debris[0]
        dy = X_sat[1] - X_debris[1]
        
        return np.sqrt(dx**2 + dy**2)
    
    def find_closest_approach(self,
                              state_sat: OrbitalState,
                              state_debris: OrbitalState,
                              t_horizon: float,
                              n_samples: int = 100) -> Tuple[float, float, float]:
        """
        Trouve le temps et la distance de l'approche la plus rapprochée
        
        Méthode:
        1. Propager les deux orbites sur l'horizon temporel
        2. Échantillonner la distance à intervalles réguliers
        3. Affiner avec optimisation locale
        
        Args:
            state_sat: État initial du satellite
            state_debris: État initial du débris
            t_horizon: Horizon temporel de recherche [s]
            n_samples: Nombre d'échantillons pour recherche initiale
        
        Returns:
            Tuple (d_min, t_ca, v_relative)
            - d_min: Distance minimale [km]
            - t_ca: Temps de l'approche rapprochée [s]
            - v_relative: Vitesse relative au moment de l'approche [km/s]
        """
        # Propager les deux orbites
        result_sat = propagate_orbit(state_sat, (0, t_horizon), dense_output=True)
        result_debris = propagate_orbit(state_debris, (0, t_horizon), dense_output=True)
        
        # Fonction de distance à optimiser
        def distance_func(t):
            X_sat = result_sat['sol'].sol(t)
            X_debris = result_debris['sol'].sol(t)
            dx = X_sat[0] - X_debris[0]
            dy = X_sat[1] - X_debris[1]
            return np.sqrt(dx**2 + dy**2)
        
        # Recherche du minimum sur un échantillonnage grossier
        t_samples = np.linspace(0, t_horizon, n_samples)
        distances = np.array([distance_func(t) for t in t_samples])
        idx_min = np.argmin(distances)
        t_guess = t_samples[idx_min]
        
        # Affiner avec optimisation locale
        result = minimize_scalar(
            distance_func,
            bounds=(max(0, t_guess - t_horizon/n_samples), 
                    min(t_horizon, t_guess + t_horizon/n_samples)),
            method='bounded'
        )
        
        t_ca = result.x
        d_min = result.fun
        
        # Calculer la vitesse relative au moment de l'approche
        X_sat_ca = result_sat['sol'].sol(t_ca)
        X_debris_ca = result_debris['sol'].sol(t_ca)
        
        v_rel_x = X_sat_ca[2] - X_debris_ca[2]
        v_rel_y = X_sat_ca[3] - X_debris_ca[3]
        v_relative = np.sqrt(v_rel_x**2 + v_rel_y**2)
        
        return d_min, t_ca, v_relative
    
    def compute_collision_probability(self,
                                      d_min: float,
                                      sigma_d: float) -> float:
        """
        Calcule la probabilité de collision basée sur un modèle gaussien
        
        On modélise la distance minimale comme une variable aléatoire normale:
        d_min ~ N(μ_d, σ_d²)
        
        Probabilité de collision:
        P_c = P(d_min < r_combined) = Φ((r_combined - μ_d) / σ_d)
        
        où Φ est la fonction de répartition de la loi normale standard
        
        Args:
            d_min: Distance minimale moyenne [km]
            sigma_d: Incertitude sur la distance minimale [km]
        
        Returns:
            Probabilité de collision [0, 1]
        """
        if sigma_d <= 0:
            # Cas déterministe
            return 1.0 if d_min < self.r_combined else 0.0
        
        # Calcul avec la fonction de répartition normale
        z = (self.r_combined - d_min) / sigma_d
        p_collision = norm.cdf(z)
        
        # S'assurer que la probabilité est dans [0, 1]
        p_collision = np.clip(p_collision, 0.0, 1.0)
        
        return p_collision
    
    def estimate_distance_uncertainty(self,
                                      state_sat: OrbitalState,
                                      state_debris: OrbitalState,
                                      sigma_pos_sat: float,
                                      sigma_vel_sat: float,
                                      sigma_pos_debris: float,
                                      sigma_vel_debris: float,
                                      t_ca: float,
                                      n_monte_carlo: int = 100) -> float:
        """
        Estime l'incertitude sur la distance minimale par Monte Carlo
        
        Args:
            state_sat: État satellite
            state_debris: État débris
            sigma_pos_sat: Incertitude position satellite [km]
            sigma_vel_sat: Incertitude vitesse satellite [km/s]
            sigma_pos_debris: Incertitude position débris [km]
            sigma_vel_debris: Incertitude vitesse débris [km/s]
            t_ca: Temps de l'approche rapprochée [s]
            n_monte_carlo: Nombre de simulations Monte Carlo
        
        Returns:
            Écart-type de la distance à l'approche rapprochée [km]
        """
        distances_ca = []
        
        for _ in range(n_monte_carlo):
            # Perturber l'état du satellite
            X_sat_perturbed = state_sat.to_array()
            X_sat_perturbed[:2] += np.random.normal(0, sigma_pos_sat, 2)
            X_sat_perturbed[2:] += np.random.normal(0, sigma_vel_sat, 2)
            state_sat_perturbed = OrbitalState.from_array(X_sat_perturbed)
            
            # Perturber l'état du débris
            X_debris_perturbed = state_debris.to_array()
            X_debris_perturbed[:2] += np.random.normal(0, sigma_pos_debris, 2)
            X_debris_perturbed[2:] += np.random.normal(0, sigma_vel_debris, 2)
            state_debris_perturbed = OrbitalState.from_array(X_debris_perturbed)
            
            # Propager jusqu'au temps de l'approche
            result_sat = propagate_orbit(state_sat_perturbed, (0, t_ca))
            result_debris = propagate_orbit(state_debris_perturbed, (0, t_ca))
            
            X_sat_ca = result_sat['X'][:, -1]
            X_debris_ca = result_debris['X'][:, -1]
            
            # Distance à l'approche
            d_ca = np.sqrt((X_sat_ca[0] - X_debris_ca[0])**2 + 
                          (X_sat_ca[1] - X_debris_ca[1])**2)
            distances_ca.append(d_ca)
        
        return np.std(distances_ca)
    
    def compute_risk_score(self,
                           d_min: float,
                           sigma_d: float,
                           t_ca: float,
                           v_relative: float,
                           weight_prob: float = 1e6,
                           weight_severity: float = 1e3) -> float:
        """
        Calcule un score de risque composite
        
        Risk = w_prob * P_collision + w_sev * Severity
        
        où Severity dépend de:
        - Vitesse relative (énergie cinétique de l'impact)
        - Proximité temporelle de l'événement
        
        Args:
            d_min: Distance minimale [km]
            sigma_d: Incertitude distance [km]
            t_ca: Temps avant approche [s]
            v_relative: Vitesse relative [km/s]
            weight_prob: Poids de la probabilité de collision
            weight_severity: Poids de la sévérité
        
        Returns:
            Score de risque (plus élevé = plus dangereux)
        """
        # Probabilité de collision
        p_collision = self.compute_collision_probability(d_min, sigma_d)
        
        # Sévérité (proportionnelle à l'énergie cinétique)
        # E_k ∝ v²
        severity = v_relative**2
        
        # Facteur d'urgence (plus l'approche est proche, plus c'est urgent)
        # On utilise une décroissance exponentielle
        urgency_factor = np.exp(-t_ca / 3600)  # Décroissance avec τ = 1 heure
        
        # Score total
        risk = weight_prob * p_collision + weight_severity * severity * urgency_factor
        
        return risk
    
    def classify_risk_level(self, risk_score: float) -> str:
        """
        Classifie le niveau de risque
        
        Args:
            risk_score: Score de risque
        
        Returns:
            Niveau de risque: 'CRITIQUE', 'ÉLEVÉ', 'MODÉRÉ', 'FAIBLE', 'NÉGLIGEABLE'
        """
        if risk_score > 1e5:
            return 'CRITIQUE'
        elif risk_score > 1e4:
            return 'ÉLEVÉ'
        elif risk_score > 1e3:
            return 'MODÉRÉ'
        elif risk_score > 1e2:
            return 'FAIBLE'
        else:
            return 'NÉGLIGEABLE'


def quick_collision_check(state_sat: OrbitalState,
                          state_debris: OrbitalState,
                          t_horizon: float,
                          r_collision: float = R_COMBINED) -> bool:
    """
    Vérification rapide de collision potentielle
    
    Args:
        state_sat: État satellite
        state_debris: État débris
        t_horizon: Horizon temporel [s]
        r_collision: Rayon de collision [km]
    
    Returns:
        True si collision potentielle, False sinon
    """
    risk_eval = CollisionRisk(r_combined=r_collision)
    d_min, _, _ = risk_eval.find_closest_approach(state_sat, state_debris, t_horizon)
    
    return d_min < r_collision


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE D'ÉVALUATION DU RISQUE")
    print("=" * 60)
    
    from .orbital_mechanics import create_circular_orbit, compute_orbital_period
    
    # Créer un satellite sur orbite circulaire
    state_sat = create_circular_orbit(altitude=400, angle=0)
    T_sat = compute_orbital_period(state_sat)
    
    print("\nSatellite:")
    print(f"  {state_sat}")
    print(f"  Période: {T_sat/60:.2f} min")
    
    # Créer un débris sur une trajectoire de collision
    # Débris sur orbite similaire mais légèrement différente
    state_debris = create_circular_orbit(altitude=405, angle=np.pi/4)
    
    print("\nDébris:")
    print(f"  {state_debris}")
    
    # Distance initiale
    d_initial = np.sqrt((state_sat.x - state_debris.x)**2 + 
                        (state_sat.y - state_debris.y)**2)
    print(f"\nDistance initiale: {d_initial:.2f} km")
    
    # Évaluation du risque
    print("\n" + "-" * 60)
    print("Recherche de l'approche la plus rapprochée")
    print("-" * 60)
    
    risk_eval = CollisionRisk()
    t_horizon = T_sat  # Chercher sur une orbite complète
    
    d_min, t_ca, v_rel = risk_eval.find_closest_approach(
        state_sat, state_debris, t_horizon
    )
    
    print(f"\nRésultats:")
    print(f"  Distance minimale: {d_min:.4f} km = {d_min*1000:.1f} m")
    print(f"  Temps d'approche: {t_ca:.2f} s = {t_ca/60:.2f} min")
    print(f"  Vitesse relative: {v_rel:.4f} km/s = {v_rel*1000:.1f} m/s")
    
    # Évaluation avec incertitude
    print("\n" + "-" * 60)
    print("Évaluation du risque avec incertitude")
    print("-" * 60)
    
    sigma_pos = 0.5   # km
    sigma_vel = 0.005  # km/s
    
    print(f"\nIncertitudes:")
    print(f"  σ_pos = {sigma_pos} km")
    print(f"  σ_vel = {sigma_vel} km/s")
    
    # Estimation Monte Carlo
    print("\nEstimation Monte Carlo de σ_d...")
    sigma_d = risk_eval.estimate_distance_uncertainty(
        state_sat, state_debris,
        sigma_pos, sigma_vel,
        sigma_pos, sigma_vel,
        t_ca,
        n_monte_carlo=50
    )
    
    print(f"  σ_d ≈ {sigma_d:.4f} km = {sigma_d*1000:.1f} m")
    
    # Probabilité de collision
    p_collision = risk_eval.compute_collision_probability(d_min, sigma_d)
    print(f"\nProbabilité de collision: {p_collision:.6f} ({p_collision*100:.4f}%)")
    
    # Score de risque
    risk_score = risk_eval.compute_risk_score(d_min, sigma_d, t_ca, v_rel)
    risk_level = risk_eval.classify_risk_level(risk_score)
    
    print(f"\nScore de risque: {risk_score:.2e}")
    print(f"Niveau: {risk_level}")
    
    # Test avec collision plus probable
    print("\n" + "=" * 60)
    print("TEST AVEC COLLISION PLUS PROBABLE")
    print("=" * 60)
    
    # Débris plus proche
    state_debris_close = create_circular_orbit(altitude=400.05, angle=0.001)
    
    d_min2, t_ca2, v_rel2 = risk_eval.find_closest_approach(
        state_sat, state_debris_close, t_horizon
    )
    
    print(f"\nDistance minimale: {d_min2:.6f} km = {d_min2*1000:.2f} m")
    print(f"Temps d'approche: {t_ca2:.2f} s = {t_ca2/60:.2f} min")
    
    sigma_d2 = 0.5  # km (simplifié)
    p_collision2 = risk_eval.compute_collision_probability(d_min2, sigma_d2)
    risk_score2 = risk_eval.compute_risk_score(d_min2, sigma_d2, t_ca2, v_rel2)
    risk_level2 = risk_eval.classify_risk_level(risk_score2)
    
    print(f"\nProbabilité de collision: {p_collision2:.6f} ({p_collision2*100:.4f}%)")
    print(f"Score de risque: {risk_score2:.2e}")
    print(f"Niveau: {risk_level2}")
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)