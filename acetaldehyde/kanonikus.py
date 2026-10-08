from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt


# SI constants, exact where applicable.
KB = 1.380649e-23              # J K^-1
H = 6.62607015e-34             # J s
C_CM_S = 2.99792458e10         # cm s^-1
EH = 4.3597447222060e-18       # J
AMU = 1.66053906660e-27        # kg
NA = 6.02214076e23             # mol^-1
CM1_PER_MHZ = 1.0 / 29979.2458


@dataclass(frozen=True)
class Species:
    name: str
    energy_hartree: float
    mass_amu: float
    freqs_cm: tuple[float, ...]
    rot_constants_cm: tuple[float, ...] = ()
    symmetry_number: int = 1
    electronic_degeneracy: int = 1
    zpe_hartree: float | None = None

    def positive_freqs(self, cutoff_cm: float = 0.0) -> tuple[float, ...]:
        return tuple(nu for nu in self.freqs_cm if nu > cutoff_cm)

    def zpe_j(self, freq_cutoff_cm: float = 0.0) -> float:
        if self.zpe_hartree is not None:
            return self.zpe_hartree * EH
        return 0.5 * H * C_CM_S * sum(self.positive_freqs(freq_cutoff_cm))

    def q_vib(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        q = 1.0
        for nu_cm in self.positive_freqs(freq_cutoff_cm):
            x = H * C_CM_S * nu_cm / (KB * temperature_k)
            q *= 1.0 / (1.0 - exp(-x))
        return q

    def q_rot(self, temperature_k: float) -> float:
        if len(self.rot_constants_cm) == 0:
            return 1.0
        if len(self.rot_constants_cm) == 1:
            theta = H * C_CM_S * self.rot_constants_cm[0] / KB
            return temperature_k / (self.symmetry_number * theta)
        if len(self.rot_constants_cm) == 3:
            theta_a, theta_b, theta_c = (
                H * C_CM_S * b_cm / KB for b_cm in self.rot_constants_cm
            )
            return (
                sqrt(pi)
                * temperature_k**1.5
                / (self.symmetry_number * sqrt(theta_a * theta_b * theta_c))
            )
        raise ValueError(
            f"{self.name}: use 0, 1, or 3 rotational constants in cm^-1."
        )

    def q_int(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        return (
            self.electronic_degeneracy
            * self.q_rot(temperature_k)
            * self.q_vib(temperature_k, freq_cutoff_cm)
        )


def rotational_constants_mhz_to_cm(values_mhz: tuple[float, ...]) -> tuple[float, ...]:
    """Convert CFOUR-style rotational constants in MHz to cm^-1."""
    return tuple(value * CM1_PER_MHZ for value in values_mhz)


def reduced_mass_kg(a: Species, b: Species) -> float:
    mu_amu = a.mass_amu * b.mass_amu / (a.mass_amu + b.mass_amu)
    return mu_amu * AMU


def zpe_corrected_barrier_j(
    reactant_a: Species,
    reactant_b: Species,
    transition_state: Species,
    freq_cutoff_cm: float = 0.0,
) -> float:
    delta_e_elec = (
        transition_state.energy_hartree
        - reactant_a.energy_hartree
        - reactant_b.energy_hartree
    ) * EH
    delta_zpe = (
        transition_state.zpe_j(freq_cutoff_cm)
        - reactant_a.zpe_j(freq_cutoff_cm)
        - reactant_b.zpe_j(freq_cutoff_cm)
    )
    return delta_e_elec + delta_zpe


def canonical_tst_bimolecular_rate(
        reactant_a: Species,
        reactant_b: Species,
        transition_state: Species,
        temperature_k: float,
        reaction_path_degeneracy: int = 1,
        freq_cutoff_cm: float = 0.0,
) -> dict[str, float]:
    """Canonical harmonic TST rate without tunneling, for A + B -> TS.

    Uses the bimolecular expression:
        k(T) = (kB T / h) * (h^2 / (2 pi mu kB T))^(3/2)
               * q_int(TS)/(q_int(A) q_int(B)) * exp(-DeltaE0/kB T)

    The returned SI rate is in m^3 molecule^-1 s^-1.
    """
    delta_e0 = zpe_corrected_barrier_j(
        reactant_a, reactant_b, transition_state, freq_cutoff_cm
    )
    mu = reduced_mass_kg(reactant_a, reactant_b)
    translational_factor = (H * H / (2.0 * pi * mu * KB * temperature_k)) ** 1.5
    q_ratio = transition_state.q_int(temperature_k, freq_cutoff_cm) / (
            reactant_a.q_int(temperature_k, freq_cutoff_cm)
            * reactant_b.q_int(temperature_k, freq_cutoff_cm)
    )
    k_m3_molecule_s = (
            reaction_path_degeneracy
            * (KB * temperature_k / H)
            * translational_factor
            * q_ratio
            * exp(-delta_e0 / (KB * temperature_k))
    )
    return {
        "k_m3_molecule_s": k_m3_molecule_s,
        "k_cm3_molecule_s": k_m3_molecule_s * 1.0e6,
        "k_L_mol_s": k_m3_molecule_s * 1000.0 * NA,
        "delta_E0_kJ_mol": delta_e0 * NA / 1000.0,
        "q_int_ratio": q_ratio,
        "translational_factor_m3": translational_factor,
    }

def main() -> None:
    temperature_k = 298.15

    # Fill these from CFOUR. The numbers below are placeholders.
    # - energy_hartree: electronic energy at the optimized geometry.
    # - freqs_cm: harmonic frequencies; for the TS include only the stable
    #   positive modes, or leave the imaginary mode as negative and it will be
    #   ignored by the default positive-frequency filter.
    # - rot_constants_cm: rotational constants in cm^-1. If CFOUR gives MHz,
    #   wrap them with rotational_constants_mhz_to_cm((A, B, C)).
    ch3cho = Species(
        name="CH3CHO",
        energy_hartree=-153.429498589980540,  # TODO
        mass_amu=44.026215,
        freqs_cm=(0.0339,
0.0123,
0.4762 ,
0.5457  ,
1.0632   ,
166.3427,
499.2581 ,
771.3471,
896.1367,
1120.6587 ,
1125.5271 ,
1370.982,
1453.3390 ,
1465.6330 ,
1802.8522,
2907.9137 ,
3045.5309,
3126.2628 ,
3172.4558),  # TODO: 15 positive frequencies for nonlinear C2H4O
        rot_constants_cm=(0.2983800487 ,            0.3328657081 ,            1.8612519936),  # TODO: 3 constants for nonlinear CH3CHO
        symmetry_number=1,
        electronic_degeneracy=1,
)
    h_atom = Species(
        name="H",
        energy_hartree=-0.49982, # TODO
        mass_amu=1.00782503223,
        freqs_cm=(),
        rot_constants_cm=(),
        symmetry_number=1,
        electronic_degeneracy=2
    )
    ts = Species(
        name="TS",
        energy_hartree=-153.918403186890885,  # TODO
        mass_amu=ch3cho.mass_amu + h_atom.mass_amu,
        freqs_cm=(-1487.0405,135.0482,265.6897,
300.2467,
491.6734,
878.3455,
892.2855,
1053.6440,
1165.0740,
1275.6397,
1353.1128,
1394.2242,
1450.8119,
1458.4285,
1885.4714,
3051.0869,
3141.5794,
171.6758),  # TODO: 17 stable TS modes; omit the imaginary mode or keep it negative
        rot_constants_cm=(0.2605435343,             0.3283089104,             1.0177594421),  # TODO: 3 constants for the TS
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    if (
        ch3cho.energy_hartree == 0.0
        or h_atom.energy_hartree == 0.0
        or ts.energy_hartree == 0.0
        or not ch3cho.freqs_cm
        or not ts.freqs_cm
        or len(ch3cho.rot_constants_cm) != 3
        or len(ts.rot_constants_cm) != 3
    ):
        raise SystemExit("Edit the TODO values in main() before running.")

    result = canonical_tst_bimolecular_rate(
        reactant_a=ch3cho,
        reactant_b=h_atom,
        transition_state=ts,
        temperature_k=temperature_k,
        reaction_path_degeneracy=1,
    )

    print(f"T = {temperature_k:.2f} K")
    print(f"Delta E0 = {result['delta_E0_kJ_mol']:.6f} kJ mol^-1")
    print(f"q_int ratio = {result['q_int_ratio']:.6e}")
    print(f"k = {result['k_m3_molecule_s']:.6e} m^3 molecule^-1 s^-1")
    print(f"k = {result['k_cm3_molecule_s']:.6e} cm^3 molecule^-1 s^-1")
    print(f"k = {result['k_L_mol_s']:.6e} L mol^-1 s^-1")


if __name__ == "__main__":
    main()