from __future__ import annotations

from dataclasses import dataclass
from math import cos, cosh, exp, pi, sqrt
import matplotlib.pyplot as plt
from scipy.integrate import quad

# SI constants, exact where applicable.
KB = 1.380649e-23  # J K^-1
H = 6.62607015e-34  # J s
HBAR = H / (2.0 * pi)  # J s
C_CM_S = 2.99792458e10  # cm s^-1
EH = 4.3597447222060e-18  # J
AMU = 1.66053906660e-27  # kg
NA = 6.02214076e23  # mol^-1
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
                    * temperature_k ** 1.5
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


def eckart_transmission_probability(
        energy_j: float, a_j: float, b_j: float, l_m: float, eff_mass_kg: float
) -> float:
    """Kiszámítja a P(E) áthaladási valószínűséget az Eckart-potenciálra."""
    if energy_j <= 0.0:
        return 0.0

    # Dimensionless parameters (Eq. 10, 11, 12)
    alpha = sqrt(2.0 * eff_mass_kg * energy_j) / HBAR * (l_m / (2.0 * pi))

    # Ha E < A, az E - A negatív, de az Eckart P(E) zárt alakja kezeli
    diff_e_a = energy_j - a_j
    if diff_e_a >= 0.0:
        beta = sqrt(2.0 * eff_mass_kg * diff_e_a) / HBAR * (l_m / (2.0 * pi))
    else:
        # Complex beta -> cosh term under analytical continuation
        beta = sqrt(2.0 * eff_mass_kg * abs(diff_e_a)) / HBAR * (l_m / (2.0 * pi))

    radicand = (8.0 * eff_mass_kg * b_j * l_m ** 2) / (H ** 2) - 1.0

    if radicand >= 0.0:
        delta = 0.5 * sqrt(radicand)
        cos_2pi_delta = cos(2.0 * pi * delta)
    else:
        gamma = 0.5 * sqrt(abs(radicand))
        cos_2pi_delta = cosh(2.0 * pi * gamma)

    if diff_e_a >= 0.0:
        num = cosh(2.0 * pi * (alpha - beta)) + cos_2pi_delta
        den = cosh(2.0 * pi * (alpha + beta)) + cos_2pi_delta
    else:
        # Below the product potential level
        num = cosh(2.0 * pi * alpha) + cos_2pi_delta
        den = cosh(2.0 * pi * alpha) + cos_2pi_delta  # Safe real extension

    p_e = 1.0 - (num / den)
    return max(0.0, min(1.0, float(p_e)))


def calculate_eckart_kappa(
        delta_ef_j: float,
        delta_er_j: float,
        imag_freq_cm: float,
        eff_mass_kg: float,
        temperature_k: float,
) -> float:
    """Kiszámítja az Eckart-alagút korrekciós tényezőt kappa(T)."""
    # 1. Eckart paraméterek feltérképezése (Eq. 13 és felette)
    A_j = delta_ef_j - delta_er_j
    B_j = (sqrt(max(0.0, delta_ef_j)) + sqrt(max(0.0, delta_er_j))) ** 2

    nu_imag_hz = abs(imag_freq_cm) * C_CM_S
    inv_l_factor = (1.0 / (2.0 * eff_mass_kg)) * (
                1.0 / sqrt(max(1e-30, delta_ef_j)) + 1.0 / sqrt(max(1e-30, delta_er_j)))
    L_m = (1.0 / nu_imag_hz) * (inv_l_factor ** -0.5)

    # 2. Integrálás a Boltzmann-eloszlásra (Eq. 8)
    beta_kt = KB * temperature_k

    def integrand(E: float) -> float:
        p_e = eckart_transmission_probability(E, A_j, B_j, L_m, eff_mass_kg)
        return p_e * exp(-E / beta_kt)

    # Integrálási határok: 0-tól a gát feletti régióig (pl. 10 * gátmagasság)
    upper_limit = max(delta_ef_j * 5.0, 20.0 * beta_kt)
    integral_val, _ = quad(integrand, 0.0, upper_limit, limit=200)

    kappa = (1.0 / beta_kt) * exp(delta_ef_j / beta_kt) * integral_val
    return max(1.0, kappa)


def canonical_tst_bimolecular_rate(
        reactant_a: Species,
        reactant_b: Species,
        transition_state: Species,
        product_a: Species,
        product_b: Species,
        temperature_k: float,
        imag_freq_cm: float,
        reaction_path_degeneracy: int = 1,
        freq_cutoff_cm: float = 0.0,
) -> dict[str, float]:
    # Előrehaladó gát (Delta E0)
    delta_ef = zpe_corrected_barrier_j(reactant_a, reactant_b, transition_state, freq_cutoff_cm)

    # Visszafelé irányuló gát (Delta Er)
    delta_er = zpe_corrected_barrier_j(product_a, product_b, transition_state, freq_cutoff_cm)

    # Transzlációs tényező és állapotösszegek
    mu = reduced_mass_kg(reactant_a, reactant_b)
    translational_factor = (H * H / (2.0 * pi * mu * KB * temperature_k)) ** 1.5
    q_ratio = transition_state.q_int(temperature_k, freq_cutoff_cm) / (
            reactant_a.q_int(temperature_k, freq_cutoff_cm)
            * reactant_b.q_int(temperature_k, freq_cutoff_cm)
    )

    # Klasszikus TST sebességi állandó (alagút nélkül)
    k_tst = (
            reaction_path_degeneracy
            * (KB * temperature_k / H)
            * translational_factor
            * q_ratio
            * exp(-delta_ef / (KB * temperature_k))
    )

    # Eckart-alagút korrekció (kappa)
    eff_mass_kg = 1.00782503223 * AMU  # H-atom alagutazási tömege
    kappa_eckart = calculate_eckart_kappa(
        delta_ef_j=delta_ef,
        delta_er_j=delta_er,
        imag_freq_cm=imag_freq_cm,
        eff_mass_kg=eff_mass_kg,
        temperature_k=temperature_k,
    )

    k_corrected = k_tst * kappa_eckart

    return {
        "k_tst_cm3": k_tst * 1.0e6,
        "k_corrected_cm3": k_corrected * 1.0e6,
        "kappa_eckart": kappa_eckart,
        "delta_Ef_kJ_mol": delta_ef * NA / 1000.0,
        "delta_Er_kJ_mol": delta_er * NA / 1000.0,
    }


def main() -> None:
    # 1. REAKTÁNSOK
    ch3cho = Species(
        name="CH3CHO",
        energy_hartree=-153.429498589980540,
        mass_amu=44.026215,
        freqs_cm=(
            166.3427, 499.2581, 771.3471, 896.1367, 1120.6587,
            1125.5271, 1370.982, 1453.3390, 1465.6330, 1802.8522,
            2907.9137, 3045.5309, 3126.2628, 3172.4558
        ),
        rot_constants_cm=(0.2983800487, 0.3328657081, 1.8612519936),
        symmetry_number=1,
        electronic_degeneracy=1,
    )

    h_atom = Species(
        name="H",
        energy_hartree=-0.49982,
        mass_amu=1.00782503223,
        freqs_cm=(),
        rot_constants_cm=(),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    # 2. ÁTMENETI ÁLLAPOT (TS)
    ts = Species(
        name="TS",
        energy_hartree=-153.9000, #-153.918403186890885,
        mass_amu=ch3cho.mass_amu + h_atom.mass_amu,
        freqs_cm=(
            135.0482, 171.6758, 265.6897, 300.2467, 491.6734,
            878.3455, 892.2855, 1053.6440, 1165.0740, 1275.6397,
            1353.1128, 1394.2242, 1450.8119, 1458.4285, 1885.4714,
            3051.0869, 3141.5794
        ),
        rot_constants_cm=(0.2605435343, 0.3283089104, 1.0177594421),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    # 3. TERMÉKEK (Megadott adataid alapján)
    h2 = Species(
        name="H2",
        energy_hartree=-1.128746115957980,
        mass_amu=2.01565007,
        freqs_cm=(4583.9090,),  # Csak a pozitív rezgési frekvencia
        rot_constants_cm=(59.7986200445,),  # 1 db rotációs konstans (lineáris)
        symmetry_number=2,
        electronic_degeneracy=1,
    )

    ch3co = Species(
        name="CH3CO",
        energy_hartree=-152.784900719202085,
        mass_amu=43.0183897,
        freqs_cm=(
            457.0776, 847.6872, 879.7312, 1039.0968, 1336.8913,
            1445.1669, 1469.0428, 1893.7103, 3051.3571, 3146.0186, 3185.0905
        ),  # A -142.4490 (a belső rotáció/imaginárius módus) kihagyva
        rot_constants_cm=(0.3099207198, 0.3286687762, 2.6476526610),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    imag_freq_cm = 1487.0405  # TS képzetes frekvenciájának abszolút értéke

    # Számítás 200 K - 1000 K között
    temperatures = [float(T) for T in range(200, 1010, 20)]
    k_tst_list = []
    k_corr_list = []
    kappa_list = []

    for T in temperatures:
        res = canonical_tst_bimolecular_rate(
            reactant_a=ch3cho,
            reactant_b=h_atom,
            transition_state=ts,
            product_a=ch3co,
            product_b=h2,
            temperature_k=T,
            imag_freq_cm=imag_freq_cm,
        )
        k_tst_list.append(res["k_tst_cm3"])
        k_corr_list.append(res["k_corrected_cm3"])
        kappa_list.append(res["kappa_eckart"])

    # Eredmények kiírása 298.15 K-re
    res_298 = canonical_tst_bimolecular_rate(
        ch3cho, h_atom, ts, ch3co, h2, 298.15, imag_freq_cm
    )
    print("=== EREDMÉNYEK (T = 298.15 K) ===")
    print(f"Előre gát (Delta Ef): {res_298['delta_Ef_kJ_mol']:.2f} kJ/mol")
    print(f"Vissza gát (Delta Er): {res_298['delta_Er_kJ_mol']:.2f} kJ/mol")
    print(f"Eckart kappa(298.15 K): {res_298['kappa_eckart']:.3f}")
    print(f"k (Klasszikus TST) : {res_298['k_tst_cm3']:.6e} cm^3 molecule^-1 s^-1")
    print(f"k (Eckart-korrigált): {res_298['k_corrected_cm3']:.6e} cm^3 molecule^-1 s^-1")

    # Grafikonok kirajzolása
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Grafikon: Sebességi állandók (TST vs. Eckart)
    ax1.plot(temperatures, k_tst_list, 'r--', label='Klasszikus TST')
    ax1.plot(temperatures, k_corr_list, 'b-', label='Eckart-korrigált TST')
    ax1.set_yscale('log')
    ax1.set_xlabel('Hőmérséklet (T / K)')
    ax1.set_ylabel(r'$k(T)$  ($\mathrm{cm^3\ molecule^{-1}\ s^{-1}}$)')
    ax1.set_title('Sebességi állandók a hőmérséklet függvényében')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # 2. Grafikon: Eckart kappa tényező T függvényében
    ax2.plot(temperatures, kappa_list, 'g-', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Hőmérséklet (T / K)')
    ax2.set_ylabel(r'Eckart alagút-tényező ($\kappa_{\mathrm{Eckart}}$)')
    ax2.set_title('Alagúteffektus mértéke (Kappa) a hőmérséklettel')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()