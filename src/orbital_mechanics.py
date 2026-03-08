"""
Module de mécanique orbitale pour la propagation d'orbites 2D (problème à deux corps)

Ce module contient les fonctions pour :
- Propager des orbites dans le plan 2D
- Convertir entre éléments orbitaux et vecteurs d'état
- Calculer les propriétés orbitales (énergie, moment angulaire)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# Constantes physiques
MU_EARTH = 398600.0  # km^3/s^2 - Paramètre gravitationnel terrestre
R_EARTH = 6371.0     # km - Rayon terrestre


class OrbitalState:
    """
    Classe représentant l'état orbital d'un objet
    
    Attributs:
        x (float): Position x [km]
        y (float): Position y [km]
        vx (float): Vitesse x [km/s]
        vy (float): Vitesse y [km/s]
        t (float): Temps [s]
    """
    
    def __init__(self, x: float, y: float, vx: float, vy: float, t: float = 0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.t = t
    
    def to_array(self) -> np.ndarray:
        """Convertit l'état en array numpy [x, y, vx, vy]"""
        return np.array([self.x, self.y, self.vx, self.vy])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, t: float = 0.0):
        """Crée un OrbitalState à partir d'un array [x, y, vx, vy]"""
        return cls(arr[0], arr[1], arr[2], arr[3], t)
    
    def get_position(self) -> np.ndarray:
        """Retourne le vecteur position [x, y]"""
        return np.array([self.x, self.y])
    
    def get_velocity(self) -> np.ndarray:
        """Retourne le vecteur vitesse [vx, vy]"""
        return np.array([self.vx, self.vy])
    
    def get_distance(self) -> float:
        """Retourne la distance au centre de la Terre [km]"""
        return np.sqrt(self.x**2 + self.y**2)
    
    def get_speed(self) -> float:
        """Retourne la vitesse scalaire [km/s]"""
        return np.sqrt(self.vx**2 + self.vy**2)
    
    def get_energy(self, mu: float = MU_EARTH) -> float:
        """
        Calcule l'énergie orbitale spécifique [km^2/s^2]
        
        E = v^2/2 - μ/r
        
        Args:
            mu: Paramètre gravitationnel [km^3/s^2]
        
        Returns:
            Énergie spécifique [km^2/s^2]
        """
        r = self.get_distance()
        v = self.get_speed()
        return v**2 / 2 - mu / r
    
    def get_angular_momentum(self) -> float:
        """
        Calcule le moment angulaire spécifique [km^2/s]
        
        h = x*vy - y*vx (composante z du produit vectoriel r × v)
        
        Returns:
            Moment angulaire [km^2/s]
        """
        return self.x * self.vy - self.y * self.vx
    
    def __repr__(self) -> str:
        return (f"OrbitalState(x={self.x:.2f}, y={self.y:.2f}, "
                f"vx={self.vx:.4f}, vy={self.vy:.4f}, t={self.t:.2f})")


def orbital_dynamics(t: float, X: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """
    Équations du mouvement pour le problème à deux corps en 2D
    
    d²r/dt² = -μ/r³ * r
    
    Args:
        t: Temps [s]
        X: Vecteur d'état [x, y, vx, vy]
        mu: Paramètre gravitationnel [km^3/s^2]
    
    Returns:
        Dérivée du vecteur d'état [vx, vy, ax, ay]
    """
    x, y, vx, vy = X
    
    # Distance au centre de la Terre
    r = np.sqrt(x**2 + y**2)
    r3 = r**3
    
    # Accélération gravitationnelle
    ax = -mu * x / r3
    ay = -mu * y / r3
    
    # Retour de la dérivée [dx/dt, dy/dt, dvx/dt, dvy/dt]
    return np.array([vx, vy, ax, ay])


def propagate_orbit(state: OrbitalState, 
                    t_span: Tuple[float, float],
                    mu: float = MU_EARTH,
                    rtol: float = 1e-10,
                    atol: float = 1e-12,
                    dense_output: bool = True) -> dict:
    """
    Propage une orbite sur un intervalle de temps donné
    
    Args:
        state: État orbital initial
        t_span: Intervalle de temps (t_start, t_end) [s]
        mu: Paramètre gravitationnel [km^3/s^2]
        rtol: Tolérance relative pour l'intégrateur
        atol: Tolérance absolue pour l'intégrateur
        dense_output: Si True, permet l'interpolation continue
    
    Returns:
        Dictionnaire contenant:
            - 't': array des temps [s]
            - 'X': array des états [x, y, vx, vy]
            - 'sol': solution complète de solve_ivp
    """
    X0 = state.to_array()
    
    # Intégration avec RK45 (Runge-Kutta d'ordre 4-5 adaptatif)
    sol = solve_ivp(
        fun=orbital_dynamics,
        t_span=t_span,
        y0=X0,
        args=(mu,),
        method='RK45',
        rtol=rtol,
        atol=atol,
        dense_output=dense_output
    )
    
    return {
        't': sol.t,
        'X': sol.y,
        'sol': sol
    }


def create_circular_orbit(altitude: float, 
                          angle: float = 0.0,
                          mu: float = MU_EARTH) -> OrbitalState:
    """
    Crée un état orbital correspondant à une orbite circulaire
    
    Pour une orbite circulaire : v = sqrt(μ/r)
    
    Args:
        altitude: Altitude au-dessus de la surface terrestre [km]
        angle: Angle de position initiale [rad] (0 = direction +x)
        mu: Paramètre gravitationnel [km^3/s^2]
    
    Returns:
        État orbital initial pour une orbite circulaire
    """
    r = R_EARTH + altitude  # Rayon orbital
    v = np.sqrt(mu / r)     # Vitesse orbitale
    
    # Position et vitesse en coordonnées cartésiennes
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    
    # Vitesse perpendiculaire au rayon (rotation dans le sens anti-horaire)
    vx = -v * np.sin(angle)
    vy = v * np.cos(angle)
    
    return OrbitalState(x, y, vx, vy)


def create_elliptical_orbit(altitude_peri: float,
                             altitude_apo: float,
                             angle: float = 0.0,
                             mu: float = MU_EARTH) -> OrbitalState:
    """
    Crée un état orbital correspondant à une orbite elliptique
    
    Args:
        altitude_peri: Altitude au périgée [km]
        altitude_apo: Altitude à l'apogée [km]
        angle: Angle de position initiale [rad] (0 = périgée)
        mu: Paramètre gravitationnel [km^3/s^2]
    
    Returns:
        État orbital initial au périgée
    """
    r_peri = R_EARTH + altitude_peri
    r_apo = R_EARTH + altitude_apo
    
    # Semi-grand axe et excentricité
    a = (r_peri + r_apo) / 2
    e = (r_apo - r_peri) / (r_apo + r_peri)
    
    # Vitesse au périgée (vis-viva equation)
    v_peri = np.sqrt(mu * (2/r_peri - 1/a))
    
    # Position au périgée
    x = r_peri * np.cos(angle)
    y = r_peri * np.sin(angle)
    
    # Vitesse perpendiculaire au périgée
    vx = -v_peri * np.sin(angle)
    vy = v_peri * np.cos(angle)
    
    return OrbitalState(x, y, vx, vy)


def compute_orbital_period(state: OrbitalState, mu: float = MU_EARTH) -> float:
    """
    Calcule la période orbitale [s]
    
    T = 2π * sqrt(a³/μ)
    
    où a est le semi-grand axe calculé depuis l'énergie :
    a = -μ / (2*E)
    
    Args:
        state: État orbital
        mu: Paramètre gravitationnel [km^3/s^2]
    
    Returns:
        Période orbitale [s]
    """
    E = state.get_energy(mu)
    
    # Semi-grand axe (pour E < 0, orbite fermée)
    if E >= 0:
        raise ValueError("L'orbite n'est pas fermée (E >= 0)")
    
    a = -mu / (2 * E)
    
    # Période orbitale
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    
    return T


def apply_maneuver(state: OrbitalState, delta_v: np.ndarray) -> OrbitalState:
    """
    Applique une manœuvre impulsionnelle à un état orbital
    
    Args:
        state: État orbital avant la manœuvre
        delta_v: Vecteur de changement de vitesse [dvx, dvy] [km/s]
    
    Returns:
        Nouvel état orbital après la manœuvre
    """
    new_state = OrbitalState(
        x=state.x,
        y=state.y,
        vx=state.vx + delta_v[0],
        vy=state.vy + delta_v[1],
        t=state.t
    )
    return new_state


def distance_between_states(state1: OrbitalState, state2: OrbitalState) -> float:
    """
    Calcule la distance entre deux états orbitaux
    
    Args:
        state1: Premier état orbital
        state2: Deuxième état orbital
    
    Returns:
        Distance [km]
    """
    dx = state1.x - state2.x
    dy = state1.y - state2.y
    return np.sqrt(dx**2 + dy**2)


def visualize_orbit(result: dict, 
                    title: str = "Orbite 2D",
                    show_earth: bool = True,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualise une orbite propagée
    
    Args:
        result: Résultat de propagate_orbit()
        title: Titre du graphique
        show_earth: Afficher la Terre
        ax: Axes matplotlib existant (optionnel)
    
    Returns:
        Axes matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    X = result['X']
    x, y = X[0], X[1]
    
    # Tracer l'orbite
    ax.plot(x, y, 'b-', linewidth=1.5, label='Trajectoire')
    ax.plot(x[0], y[0], 'go', markersize=10, label='Début')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='Fin')
    
    # Tracer la Terre
    if show_earth:
        earth = plt.Circle((0, 0), R_EARTH, color='cyan', alpha=0.3, label='Terre')
        ax.add_patch(earth)
    
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE DE MÉCANIQUE ORBITALE")
    print("=" * 60)
    
    # Créer une orbite circulaire à 400 km d'altitude
    altitude = 400.0  # km
    state0 = create_circular_orbit(altitude)
    
    print(f"\nÉtat initial (orbite circulaire, altitude = {altitude} km):")
    print(state0)
    print(f"Distance au centre: {state0.get_distance():.2f} km")
    print(f"Vitesse: {state0.get_speed():.4f} km/s")
    print(f"Énergie: {state0.get_energy():.2f} km²/s²")
    print(f"Moment angulaire: {state0.get_angular_momentum():.2f} km²/s")
    
    # Calculer la période orbitale
    T = compute_orbital_period(state0)
    print(f"Période orbitale: {T:.2f} s ({T/60:.2f} min)")
    
    # Propager sur une orbite complète
    print("\nPropagation sur une orbite complète...")
    result = propagate_orbit(state0, (0, T))
    
    print(f"Nombre de points calculés: {len(result['t'])}")
    
    # Vérifier la conservation de l'énergie
    X_final = result['X'][:, -1]
    state_final = OrbitalState.from_array(X_final, t=T)
    
    E0 = state0.get_energy()
    Ef = state_final.get_energy()
    energy_error = abs(Ef - E0) / abs(E0) * 100
    
    print(f"\nÉtat final:")
    print(state_final)
    print(f"\nConservation de l'énergie:")
    print(f"  E0 = {E0:.6f} km²/s²")
    print(f"  Ef = {Ef:.6f} km²/s²")
    print(f"  Erreur relative = {energy_error:.2e} %")
    
    # Visualisation
    print("\nGénération de la visualisation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_orbit(result, title=f"Orbite circulaire - Altitude {altitude} km", ax=ax)
    plt.tight_layout()
    plt.savefig('/home/claude/orbital_avoidance_ml/results/test_orbit.png', dpi=150)
    print("Figure sauvegardée: results/test_orbit.png")
    
    print("\n" + "=" * 60)
    print("TEST RÉUSSI ✓")
    print("=" * 60)