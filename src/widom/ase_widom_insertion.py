"""
The code in this file is derived from
https://github.com/hspark1212/DAC-SIM
MIT-licensed
"""

from io import BytesIO

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from pydantic import BaseModel
from tqdm_loggable.auto import tqdm

from .utils import check_accessibility, create_supercell_if_needed, sample_gas_positions


class WidomInsertionResults(BaseModel):
    henry_coefficient: float  # [mol/kg/Pa]
    henry_coefficient_std: float  # [mol/kg/Pa]
    averaged_interaction_energy: float  # [eV]
    averaged_interaction_energy_std: float  # [eV]
    heat_of_adsorption: float  # [kJ/mol]
    heat_of_adsorption_std: float  # [kJ/mol]
    atomic_density: float  # [kg/m³]
    total_energies: list[float]  # [eV]
    energy_gas: float  # [eV]
    energy_structure: float  # [eV]
    interaction_energies: list[float]  # [eV]
    is_accessible: list[bool]  # Guests non-overlapping with framework atoms
    is_valid: list[bool]  # Guests with valid interaction energy (> -1.25 eV)
    gas_positions: list[list[list[float]]]  # Positions of the inserted gas molecules
    optimized_structure_cif: str  # CIF string of the optimized structure


def ase_widom_insertion(
    calculator: Calculator,
    structure: Atoms,
    gas: Atoms,
    num_insertions: int,
    cutoff_distance: float,
    cutoff_to_com: bool,
    min_interplanar_distance: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Widom insertion algorithm to calculate interaction energies.

    This function performs Widom insertion of gas molecules into a structure to calculate
    interaction energies, which can be used for computing the Henry coefficient and heat
    of adsorption.

    Args:
        calculator: Calculator object from ASE to calculate energies
        structure: Atoms object representing the framework structure
        gas: Atoms object representing the gas molecule to insert
        num_insertions: Number of random insertions to perform
        cutoff_distance: Minimum allowed distance between framework atoms and gas molecule, in angstroms
        min_interplanar_distance: Minimum interplanar distance before constructing a supercell, in angstroms
        random_seed: Seed for random number generator to ensure reproducibility

    Returns:
        Array of total energies (in eV) for each insertion attempt
        Array of booleans indicating whether the insertion was successful (non-overlapping with framework atoms)
        Array of positions of the inserted gas molecules
    """
    structure = structure.copy()
    gas = gas.copy()

    # Create supercell if needed
    structure = create_supercell_if_needed(structure, min_interplanar_distance)

    # Set up random number generator
    rng = np.random.default_rng(random_seed)
    # Sample gas positions and orientations
    gas_positions = sample_gas_positions(structure, gas, num_insertions, rng)

    # Check accessibility of insertions
    framework_coords = structure.get_positions()
    lattice_matrix = np.array(structure.cell)  # 3x3 matrix
    is_accessible = check_accessibility(
        gas_positions, framework_coords, lattice_matrix, cutoff_distance, cutoff_to_com
    )

    print(f"Number of accessible positions: {np.sum(is_accessible)} out of {num_insertions}")

    # Prepare arrays for results
    energies = np.zeros(num_insertions)  # [eV], total energy

    # Set inaccessible positions to high energy
    energies[~is_accessible] = 1e10

    # Batch evaluate energies for accessible positions only
    accessible_indices = np.where(is_accessible)[0]
    structure_with_gas_original = structure + gas
    for i in tqdm(accessible_indices):
        structure_with_gas = structure_with_gas_original.copy()
        structure_with_gas.arrays["positions"][-len(gas) :] = gas_positions[i]
        structure_with_gas.wrap()  # wrap atoms to unit cell
        structure_with_gas.calc = calculator
        energies[i] = structure_with_gas.get_potential_energy()  # [eV]

    return energies, is_accessible, gas_positions  # [eV]


def analyze_widom_insertions(
    energies: np.ndarray,
    is_accessible: np.ndarray,
    gas_positions: np.ndarray,
    energy_structure: float,
    energy_gas: float,
    temperature: float,
    structure: Atoms,
    energies_are_interaction: bool,
    random_seed: int,
) -> WidomInsertionResults:
    if energies_are_interaction:
        interaction_energies = energies  # [eV]
    else:
        interaction_energies = energies - energy_structure - energy_gas  # [eV]

    num_samples = len(interaction_energies)

    # If the model gives too low interaction energy, we treat it as invalid.
    is_valid = interaction_energies > -1.25
    # Set invalid energies to a large value
    interaction_energies_valid = np.where(is_valid, interaction_energies, 1e10)  # [eV]

    # Calculate ensemble averages properties
    # units._e [J/eV], units._k [J/K], units._k / units._e # [eV/K]
    boltzmann_factor = np.exp(-interaction_energies_valid / (temperature * units._k / units._e))

    # KH = <exp(-E/RT)> / (R * T)
    atomic_density = calculate_atomic_density(structure)  # [kg / m^3]
    kh = (
        boltzmann_factor.mean()
        / (units._k * units._Nav)  # R = [J / mol K] = [Pa m^3 / mol K]
        / temperature  # T = [K] -> [mol/ m^3 Pa]
        / atomic_density  #  = [kg / m^3] -> [mol / kg Pa]
    )  # [mol/kg Pa]

    # Estimate std dev of estimator. Is a mean, so we take the std dev of terms/sqrt(N)
    kh_std = (
        boltzmann_factor.std()
        / (units._k * units._Nav)
        / temperature
        / atomic_density
        / np.sqrt(num_samples)
    )  # [mol/kg Pa]

    # U = < E * exp(-E/RT) > / <exp(-E/RT)> # [eV]
    # If we only hit large interaction energies, we don't get a good estimator due to numerics.
    # Correct for this by shifting the interaction energies.
    interaction_energies_shift = interaction_energies_valid - interaction_energies_valid.min()
    boltzmann_factor_shift = np.exp(
        -interaction_energies_shift / (temperature * units._k / units._e)
    )
    u = (interaction_energies_valid * boltzmann_factor_shift).sum() / boltzmann_factor_shift.sum()

    u_std = bootstrap_ratio_std(
        interaction_energies_valid * boltzmann_factor_shift,
        boltzmann_factor_shift,
        n_bootstrap=100,
        random_seed=random_seed,
    )  # [eV]

    # Qst = U - RT # [kJ/mol]
    qst = (u * units._e - units._k * temperature) * units._Nav * 1e-3

    # Std dev simple linear transformation of u_std
    qst_std = u_std * units._e * units._Nav * 1e-3

    cif_writer = BytesIO()
    structure.write(cif_writer, format="cif")
    cif_writer.seek(0)

    results = WidomInsertionResults(
        henry_coefficient=float(kh),
        henry_coefficient_std=float(kh_std),
        averaged_interaction_energy=float(u),
        averaged_interaction_energy_std=float(u_std),
        heat_of_adsorption=float(qst),
        heat_of_adsorption_std=float(qst_std),
        atomic_density=float(atomic_density),
        total_energies=energies.tolist(),
        energy_gas=float(energy_gas),
        energy_structure=float(energy_structure),
        interaction_energies=interaction_energies.tolist(),
        is_accessible=is_accessible.tolist(),
        is_valid=is_valid.tolist(),
        gas_positions=gas_positions.tolist(),
        optimized_structure_cif=cif_writer.read().decode("ascii"),
    )
    return results


def calculate_atomic_density(atoms: Atoms) -> float:
    """
    Calculate atomic density of the atoms.

    Args:
        atoms: The Atoms object to operate on.

    Returns:
        Atomic density of the atoms in kg/m³.
    """
    volume = atoms.get_volume() * 1e-30  # Convert Å³ to m³
    total_mass = np.sum(atoms.get_masses()) * units._amu  # Convert amu to kg # type: ignore
    return total_mass / volume


def bootstrap_ratio_std(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
) -> float:
    """Calculate standard deviation of ratio estimator using bootstrap.

    Args:
        numerator: Array of numerator values
        denominator: Array of denominator values
        n_bootstrap: Number of bootstrap samples

    Returns:
        float: Standard deviation of the ratio estimator
    """
    n = len(numerator)
    ratios = np.zeros(n_bootstrap)

    rng = np.random.default_rng(random_seed)

    for i in range(n_bootstrap):
        # Generate bootstrap sample indices (with replacement)
        indices = rng.choice(n, size=n, replace=True)

        # Calculate ratio for this bootstrap sample
        num_sample = numerator[indices]
        denom_sample = denominator[indices]
        ratios[i] = num_sample.mean() / denom_sample.mean()

    # Return standard deviation of bootstrap ratios
    return float(np.std(ratios))
