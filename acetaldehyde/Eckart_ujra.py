from __future__ import annotations

import csv
from dataclasses import dataclass
from math import cos, cosh, exp, isfinite, pi, sqrt
from pathlib import Path

# ---------------------------------------------------------------------------
# ATOMI EGYSÉGEK (Atomic Units, a.u.) ÉS FIZIKAI KONSTANSOK
#
# Ebben a modulban az Eckart-alagútkoefficiens és a TST reakciósebességi
# együttható számítása alapvetően atomi egységekben (a.u.) történik:
#   - Energia: Hartree (Eh)
#   - Tömeg: elektron tömeg (me)
#   - Hosszúság: Bohr (a0)
#   - Idő: atomic time unit (tau_au = hbar / Eh)
#   - Hatás: hbar = 1 a.u., h = 2 * pi a.u.
# ---------------------------------------------------------------------------

# Atomi egységek és alapkontansok
KB_AU = 3.166811429e-6  # Boltzmann-állandó Hartree / K-ben (kB)
HBAR_AU = 1.0  # Redukált Planck-állandó atomi egységben
H_AU = 2.0 * pi  # Planck-állandó atomi egységben (h = 2 * pi * hbar)
ME_AU = 1.0  # Elektron tömege atomi egységben (me)
AMU_TO_ME = 1822.888486209  # 1 amu hány elektron-tömeg (me)
TIME_AU_S = 2.4188843265857e-17  # 1 atomi időegység másodpercben (s)
BOHR_CM = 5.29177210903e-9  # 1 Bohr cm-ben (a0)

# SI és hagyományos egységek / konverziós tényezők
KB_SI = 1.380649e-23  # J K^-1
H_SI = 6.62607015e-34  # J s
C_CM_S = 2.99792458e10  # cm s^-1
EH_J = 4.3597447222060e-18  # J / Hartree
AMU_KG = 1.66053906660e-27  # kg
NA = 6.02214076e23  # mol^-1

# 1 cm^-1 hullámszámú rezgési kvantum energiája Hartree-ban (h*c*nu / Eh)
HC_PER_CM_HARTREE = H_SI * C_CM_S / EH_J

# Sebességi együttható átváltási tényezője: (a0^3 / tau_au) -> cm^3 molecule^-1 s^-1
AU_RATE_TO_CM3_S = (BOHR_CM ** 3) / TIME_AU_S


@dataclass(frozen=True)
class Species:
    """Kémiai részecskék (reaktánsok, termékek, átmeneti állapot) adatstruktúrája."""
    name: str
    energy_hartree: float  # Elektronikus energia (Hartree, Eh)
    mass_amu: float  # Tömeg atomic mass unitban (amu)
    freqs_cm: tuple[float, ...]  # Rezgési frekvenciák cm^-1-ben
    rot_constants_cm: tuple[float, ...] = ()  # Rotációs állandók cm^-1-ben (0, 1 vagy 3)
    symmetry_number: int = 1  # Rotációs szimmetriaszám (sigma)
    electronic_degeneracy: int = 1  # Elektronikus degeneráció (ge)
    zpe_hartree: float | None = None  # Explicit ZPE (ha nincs megadva, frekvenciákból számoljuk)

    def positive_freqs(self, cutoff_cm: float = 0.0) -> tuple[float, ...]:
        """Csak a stabil (pozitív) rezgési módokat tartja meg.

        Az átmeneti állapot (TS) képzetes (imaginárius) frekvenciáját kiszűri a ZPE számításból.
        """
        return tuple(nu for nu in self.freqs_cm if nu > cutoff_cm)

    def zpe_eh(self, freq_cutoff_cm: float = 0.0) -> float:
        """Nullponti energia (Zero-Point Energy, ZPE) Hartree-ban (Eh).

        TST.pdf Eq. (4): ZPE = sum_i (1/2 * h * nu_i)
        """
        if self.zpe_hartree is not None:
            return self.zpe_hartree
        return 0.5 * HC_PER_CM_HARTREE * sum(self.positive_freqs(freq_cutoff_cm))

    def q_vib(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        """Harmonikus oszcillátor rezgési állapotösszeg (TST.pdf Eq. 2 & 3).

        q_vib = prod_i [ 1 / (1 - exp(-h * nu_i / (kB * T))) ]
        """
        q = 1.0
        kt_eh = KB_AU * temperature_k
        for nu_cm in self.positive_freqs(freq_cutoff_cm):
            x = HC_PER_CM_HARTREE * nu_cm / kt_eh
            q *= 1.0 / (1.0 - exp(-x))
        return q

    def q_rot(self, temperature_k: float) -> float:
        """Merev rotor rotációs állapotösszeg.

        - Atom / gömbi esetben (0 rotációs állandó): q_rot = 1.0
        - Lineáris molekula (1 rotációs állandó): q_rot = T / (sigma * theta_rot)
        - Nem-lineáris molekula (3 rotációs állandó): q_rot = sqrt(pi) * T^1.5 / (sigma * sqrt(theta_A * theta_B * theta_C))
        """
        if len(self.rot_constants_cm) == 0:
            return 1.0
        if len(self.rot_constants_cm) == 1:
            theta = HC_PER_CM_HARTREE * self.rot_constants_cm[0] / KB_AU
            return temperature_k / (self.symmetry_number * theta)
        if len(self.rot_constants_cm) == 3:
            theta_a, theta_b, theta_c = (
                HC_PER_CM_HARTREE * b_cm / KB_AU
                for b_cm in self.rot_constants_cm
            )
            return (
                    sqrt(pi)
                    * (temperature_k ** 1.5)
                    / (self.symmetry_number * sqrt(theta_a * theta_b * theta_c))
            )
        raise ValueError(
            f"{self.name}: 0, 1 vagy 3 rotációs állandót adj meg cm^-1-ben."
        )

    def q_int(self, temperature_k: float, freq_cutoff_cm: float = 0.0) -> float:
        """Belső állapotösszeg (TST.pdf Eq. 19): q_int = ge * q_rot * q_vib."""
        return (
                self.electronic_degeneracy
                * self.q_rot(temperature_k)
                * self.q_vib(temperature_k, freq_cutoff_cm)
        )


def amu_to_electron_mass(mass_amu: float) -> float:
    """Tömeg átváltása amu-ból elektron-tömegbe (a.u. mass, me = 1)."""
    return mass_amu * AMU_TO_ME


def reduced_mass_amu(a: Species, b: Species) -> float:
    """Redukált tömeg kiszámítása amu-ban: mu = (m_A * m_B) / (m_A + m_B)."""
    return (a.mass_amu * b.mass_amu) / (a.mass_amu + b.mass_amu)


def zpe_corrected_barrier_eh(
        reactant_a: Species,
        reactant_b: Species,
        transition_state: Species,
        freq_cutoff_cm: float = 0.0,
) -> float:
    """ZPE-vel korrigált gátmagasság Hartree-ban (TST.pdf Eq. 4: Delta E_0).

    Delta E_0 = Delta E_e + ZPE(TS) - ZPE(R1) - ZPE(R2)
    """
    delta_e_elec = (
            transition_state.energy_hartree
            - reactant_a.energy_hartree
            - reactant_b.energy_hartree
    )
    delta_zpe = (
            transition_state.zpe_eh(freq_cutoff_cm)
            - reactant_a.zpe_eh(freq_cutoff_cm)
            - reactant_b.zpe_eh(freq_cutoff_cm)
    )
    return delta_e_elec + delta_zpe


def wavenumber_cm_to_hz(nu_cm: float) -> float:
    """Hullámszám (cm^-1) átváltása ciklometriás frekvenciába (Hz = s^-1): nu = nu_cm * c."""
    return abs(nu_cm) * C_CM_S


def hz_to_atomic_frequency(nu_hz: float) -> float:
    """Frekvencia átváltása Hz-ből atomi frekvenciába (1 / tau_au)."""
    return nu_hz * TIME_AU_S


def eckart_width_bohr(
        delta_ef_eh: float,
        delta_er_eh: float,
        imag_freq_cm: float,
        eff_mass_me: float,
) -> float:
    """Eckart-gát karakterisztikus szélessége (L) Bohr-ban (TST.pdf Eq. 13).

    A nyeregponti görbület illesztése az imaginárius frekvenciához:
    L = (1 / nu_au) * sqrt(2 / m_eff) / (1/sqrt(Delta E_f) + 1/sqrt(Delta E_r))
    """
    if delta_ef_eh <= 0.0 or delta_er_eh <= 0.0:
        raise ValueError("Az Eckart-korrekcióhoz mindkét gátmagasság legyen pozitív.")

    nu_hz = wavenumber_cm_to_hz(imag_freq_cm)
    nu_au = hz_to_atomic_frequency(nu_hz)

    reciprocal_sum = 1.0 / sqrt(delta_ef_eh) + 1.0 / sqrt(delta_er_eh)
    return (1.0 / nu_au) * sqrt(2.0 / eff_mass_me) / reciprocal_sum


def eckart_transmission_probability_au(
        energy_eh: float,
        a_eh: float,
        b_eh: float,
        width_bohr: float,
        eff_mass_me: float,
) -> float:
    """Áthaladási valószínűség P(E) atomi egységekben az aszimmetrikus Eckart-gátra (TST.pdf Eq. 9-12).

    Paraméterek:
      alpha = sqrt(2 * m * E) * (L / (2 * pi * hbar))  [hbar = 1 a.u.]
      beta  = sqrt(2 * m * (E - A)) * (L / (2 * pi * hbar))
      delta = 1/2 * sqrt( (8 * m * B * L^2) / h^2 - 1 )  [h = 2 * pi a.u.]
    """
    if energy_eh <= 0.0:
        return 0.0

    # Aszimmetria miatti küszöb: E - A alatt a termékoldalon klasszikusan tiltott
    diff_e_a = energy_eh - a_eh
    if diff_e_a <= 0.0:
        return 0.0

    alpha = sqrt(2.0 * eff_mass_me * energy_eh) * (width_bohr / (2.0 * pi))
    beta = sqrt(2.0 * eff_mass_me * diff_e_a) * (width_bohr / (2.0 * pi))

    # delta paraméter kiszámítása (TST.pdf Eq. 12)
    radicand = (8.0 * eff_mass_me * b_eh * width_bohr ** 2) / (H_AU ** 2) - 1.0
    if radicand >= 0.0:
        delta = 0.5 * sqrt(radicand)
        cos_2pi_delta = cos(2.0 * pi * delta)
    else:
        # Ha a gyök alatti kifejezés negatív, delta képzetes (delta = i * gamma)
        # cos(2*pi*i*gamma) = cosh(2*pi*gamma)
        gamma = 0.5 * sqrt(abs(radicand))
        cos_2pi_delta = cosh(2.0 * pi * gamma)

    try:
        numerator = cosh(2.0 * pi * (alpha - beta)) + cos_2pi_delta
        denominator = cosh(2.0 * pi * (alpha + beta)) + cos_2pi_delta
        p_e = 1.0 - numerator / denominator
    except OverflowError:
        # Nagy energiáknál a cosh tagok túlcsordulhatnak.
        # Aszimptotikus közelítés: P(E) ~ 1 - exp(-4 * pi * beta)
        p_e = 1.0 - exp(-4.0 * pi * beta)

    if not isfinite(p_e):
        return 0.0
    return max(0.0, min(1.0, float(p_e)))


def simpson_integral(func, lower: float, upper: float, intervals: int = 6000) -> float:
    """Simpson-módszer szerinti numerikus integrálás."""
    if intervals % 2:
        intervals += 1
    step = (upper - lower) / intervals
    total = func(lower) + func(upper)
    for i in range(1, intervals):
        coefficient = 4.0 if i % 2 else 2.0
        total += coefficient * func(lower + i * step)
    return total * step / 3.0


def calculate_eckart_kappa_au(
        delta_ef_eh: float,
        delta_er_eh: float,
        imag_freq_cm: float,
        eff_mass_amu: float,
        temperature_k: float,
) -> float:
    """Eckart alagút-korrekciós tényező kappa(T) kiszámítása atomi egységekben (TST.pdf Eq. 8).

    kappa(T) = (1 / (kB * T)) * exp(Delta E_0 / (kB * T)) * integral_0^inf P(E) * exp(-E / (kB * T)) dE
    """
    eff_mass_me = amu_to_electron_mass(eff_mass_amu)

    # Eckart-potenciál paraméterei (TST.pdf Sec. 3.2.2)
    a_eh = delta_ef_eh - delta_er_eh
    b_eh = (sqrt(delta_ef_eh) + sqrt(delta_er_eh)) ** 2
    width_bohr = eckart_width_bohr(
        delta_ef_eh=delta_ef_eh,
        delta_er_eh=delta_er_eh,
        imag_freq_cm=imag_freq_cm,
        eff_mass_me=eff_mass_me,
    )

    kt_eh = KB_AU * temperature_k

    def integrand(energy_eh: float) -> float:
        p_e = eckart_transmission_probability_au(
            energy_eh=energy_eh,
            a_eh=a_eh,
            b_eh=b_eh,
            width_bohr=width_bohr,
            eff_mass_me=eff_mass_me,
        )
        # Exponenciális tagon belüli összevonás a numerikus stabilitásért:
        # P(E) * exp((Delta E_f - E) / (kB * T)) / (kB * T)
        exponent = (delta_ef_eh - energy_eh) / kt_eh
        if exponent > 700.0:  # Overflow elkerülése
            return 0.0
        return p_e * exp(exponent) / kt_eh

    # Felső integrálási határ meghatározása (ahol a Boltzmann-farok elenyésző)
    upper_limit_eh = max(delta_ef_eh * 5.0, 20.0 * kt_eh)

    kappa = simpson_integral(integrand, 0.0, upper_limit_eh)
    return max(1.0, float(kappa))


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
    """Kanonikus TST reakciósebességi együttható számítása atomi egységekben (TST.pdf Eq. 19).

    Atomi egységekben a sebességi együttható (k_TST_au):
      k_TST_au = L * (kB_au * T / (2 * pi)) * ( (2 * pi) / (mu_au * kB_au * T) )^(3/2) * (q_TS / (q_A * q_B)) * exp(-Delta E_0 / (kB_au * T))

    A kapott k_TST_au értéket konvertáljuk cm^3 molecule^-1 s^-1 egységbe.
    """
    # Gátmagasságok ZPE-vel korrigálva (Hartree)
    delta_ef_eh = zpe_corrected_barrier_eh(
        reactant_a, reactant_b, transition_state, freq_cutoff_cm
    )
    delta_er_eh = zpe_corrected_barrier_eh(
        product_a, product_b, transition_state, freq_cutoff_cm
    )

    # Redukált tömeg atomi egységben (elektron tömeg, me)
    mu_amu = reduced_mass_amu(reactant_a, reactant_b)
    mu_au = amu_to_electron_mass(mu_amu)

    kt_eh = KB_AU * temperature_k

    # 3D transzlációs tényező atomi egységekben (TST.pdf Eq. 17 & 19):
    # (h^2 / (2 * pi * mu * kB * T))^(3/2) atomi egységben [h = 2*pi]:
    # ( (2 * pi) / (mu_au * kt_eh) )^(3/2)  [egysége: Bohr^3]
    trans_factor_au = (2.0 * pi / (mu_au * kt_eh)) ** 1.5

    # Belső állapotösszegek aránya (q_int^TS / (q_int_A * q_int_B))
    q_ratio = transition_state.q_int(temperature_k, freq_cutoff_cm) / (
            reactant_a.q_int(temperature_k, freq_cutoff_cm)
            * reactant_b.q_int(temperature_k, freq_cutoff_cm)
    )

    # Klasszikus Eyring/TST sebességi együttható atomi egységekben (a0^3 / tau_au)
    # Eyring pre-faktor: (kB * T) / h = kt_eh / (2 * pi)
    k_tst_au = (
            reaction_path_degeneracy
            * (kt_eh / (2.0 * pi))
            * trans_factor_au
            * q_ratio
            * exp(-delta_ef_eh / kt_eh)
    )

    # Eckart alagút-korrekció kiszámítása (hatásos tömeg H-atom átadásnál ~ 1.0078 amu)
    kappa_eckart = calculate_eckart_kappa_au(
        delta_ef_eh=delta_ef_eh,
        delta_er_eh=delta_er_eh,
        imag_freq_cm=imag_freq_cm,
        eff_mass_amu=1.00782503223,
        temperature_k=temperature_k,
    )

    # Sebességi együtthatók konvertálása cm^3 molecule^-1 s^-1 egységre
    k_tst_cm3 = k_tst_au * AU_RATE_TO_CM3_S
    k_corrected_cm3 = k_tst_cm3 * kappa_eckart

    return {
        "k_tst_cm3": k_tst_cm3,
        "k_corrected_cm3": k_corrected_cm3,
        "kappa_eckart": kappa_eckart,
        "delta_Ef_kJ_mol": delta_ef_eh * EH_J * NA / 1000.0,
        "delta_Er_kJ_mol": delta_er_eh * EH_J * NA / 1000.0,
        "delta_Ef_Eh": delta_ef_eh,
        "delta_Er_Eh": delta_er_eh,
    }


def write_results_csv(
        temperatures: list[float],
        k_tst_list: list[float],
        k_corr_list: list[float],
        kappa_list: list[float],
        csv_path: Path,
) -> None:
    """Az eredmények mentése CSV táblázatba."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "T_K",
                "k_tst_cm3_molecule_s",
                "k_eckart_cm3_molecule_s",
                "kappa_eckart",
            ]
        )
        writer.writerows(zip(temperatures, k_tst_list, k_corr_list, kappa_list))


def save_plot_if_matplotlib_available(
        temperatures: list[float],
        k_tst_list: list[float],
        k_corr_list: list[float],
        kappa_list: list[float],
        figure_path: Path,
) -> bool:
    """Grafikon mentése matplotlib megléte esetén."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(temperatures, k_tst_list, "r--", label="Klasszikus TST")
    ax1.plot(temperatures, k_corr_list, "b-", label="Eckart-korrigált TST")
    ax1.set_yscale("log")
    ax1.set_xlabel("Hőmérséklet (T / K)")
    ax1.set_ylabel(r"$k(T)$  ($\mathrm{cm^3\ molecule^{-1}\ s^{-1}}$)")
    ax1.set_title("Sebességi együtthatók a hőmérséklet függvényében")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(temperatures, kappa_list, "g-", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Hőmérséklet (T / K)")
    ax2.set_ylabel(r"Eckart alagúttényező ($\kappa_{\mathrm{Eckart}}$)")
    ax2.set_title("Alagúteffektus mértéke a hőmérséklettel")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(figure_path, dpi=200)
    return True


def main():
    # 1. Reaktánsok definíciója
    ch3cho = Species(
        name="CH3CHO",
        energy_hartree=-153.429498589980540,
        mass_amu=44.026215,
        freqs_cm=(
            166.3427,
            499.2581,
            771.3471,
            896.1367,
            1120.6587,
            1125.5271,
            1370.982,
            1453.3390,
            1465.6330,
            1802.8522,
            2907.9137,
            3045.5309,
            3126.2628,
            3172.4558,
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

    # 2. Átmeneti állapot (TS)
    ts = Species(
        name="TS",
        energy_hartree=-153.9000,
        mass_amu=ch3cho.mass_amu + h_atom.mass_amu,
        freqs_cm=(
            135.0482,
            171.6758,
            265.6897,
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
        ),
        rot_constants_cm=(0.2605435343, 0.3283089104, 1.0177594421),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    # 3. Termékek (a fordított gátmagassághoz Delta Er)
    h2 = Species(
        name="H2",
        energy_hartree=-1.128746115957980,
        mass_amu=2.01565007,
        freqs_cm=(4583.9090,),
        rot_constants_cm=(59.7986200445,),
        symmetry_number=2,
        electronic_degeneracy=1,
    )

    ch3co = Species(
        name="CH3CO",
        energy_hartree=-152.784900719202085,
        mass_amu=43.0183897,
        freqs_cm=(
            457.0776,
            847.6872,
            879.7312,
            1039.0968,
            1336.8913,
            1445.1669,
            1469.0428,
            1893.7103,
            3051.3571,
            3146.0186,
            3185.0905,
        ),
        rot_constants_cm=(0.3099207198, 0.3286687762, 2.6476526610),
        symmetry_number=1,
        electronic_degeneracy=2,
    )

    # TS imaginárius frekvencia abszolút értéke cm^-1-ben
    imag_freq_cm = 1487.0405

    temperatures = [float(T) for T in range(200, 1050, 50)]  # Hőmérsékleti tartomány
    k_tst_list = []
    k_corr_list = []
    kappa_list = []

    for temperature_k in temperatures:
        res = canonical_tst_bimolecular_rate(
            reactant_a=ch3cho,
            reactant_b=h_atom,
            transition_state=ts,
            product_a=ch3co,
            product_b=h2,
            temperature_k=temperature_k,
            imag_freq_cm=imag_freq_cm,
        )
        k_tst_list.append(res["k_tst_cm3"])
        k_corr_list.append(res["k_corrected_cm3"])
        kappa_list.append(res["kappa_eckart"])

    res_298 = canonical_tst_bimolecular_rate(
        ch3cho, h_atom, ts, ch3co, h2, 298.15, imag_freq_cm
    )
    print("=== EREDMÉNYEK (T = 298.15 K) ===")
    print(f"Forward gát, Delta Ef: {res_298['delta_Ef_kJ_mol']:.2f} kJ/mol")
    print(f"Reverse gát, Delta Er: {res_298['delta_Er_kJ_mol']:.2f} kJ/mol")
    print(f"Delta Ef: {res_298['delta_Ef_Eh']:.8f} Eh")
    print(f"Delta Er: {res_298['delta_Er_Eh']:.8f} Eh")
    print(f"Eckart kappa(298.15 K): {res_298['kappa_eckart']:.6g}")
    print(f"k, klasszikus TST: {res_298['k_tst_cm3']:.6e} cm^3 molecule^-1 s^-1")
    print(
        f"k, Eckart-korrigált: {res_298['k_corrected_cm3']:.6e} "
        "cm^3 molecule^-1 s^-1"
    )

    output_dir = Path(__file__).parent
    csv_path = output_dir / "eckart_tst_atomic_units.csv"
    write_results_csv(temperatures, k_tst_list, k_corr_list, kappa_list, csv_path)
    print(f"Táblázat mentve ide: {csv_path}")

    figure_path = output_dir / "eckart_tst_atomic_units.png"
    if save_plot_if_matplotlib_available(
            temperatures, k_tst_list, k_corr_list, kappa_list, figure_path
    ):
        print(f"Grafikon mentve ide: {figure_path}")
    else:
        print("Grafikon nem készült: nincs telepítve matplotlib ebben a Python-környezetben.")


if __name__ == "__main__":
    main()
