import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from tqdm_loggable.auto import tqdm

from widom.utils import check_accessibility, create_supercell_if_needed, sample_gas_positions


def sample_compute_energies(
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
        calculator: Calculator object from ASE to calculate energies.
        structure: Atoms object representing the framework structure.
        gas: Atoms object representing the gas molecule to insert.
        num_insertions: Number of random insertions to perform.
        cutoff_distance: Minimum allowed distance between framework atoms and gas molecule, in angstroms.
        cutoff_to_com: Whether to use the center of mass for distance calculations.
        min_interplanar_distance: Minimum interplanar distance before constructing a supercell, in angstroms.
        random_seed: Seed for random number generator to ensure reproducibility.

    Returns:
        A tuple containing:
            - Array of total energies (in eV) for each insertion attempt.
            - Array of booleans indicating whether the insertion was successful (non-overlapping with framework atoms).
            - Array of positions of the inserted gas molecules.
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
