from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator

from .analyze import (
    WidomInsertionResults,
    analyze_widom_insertions,
)
from .sample_compute_energies import sample_compute_energies
from .utils import optimize_atoms


def run_widom_insertion(
    calculator: Calculator,
    structure: Atoms,
    gas: str,
    temperature: float,
    model_outputs_interaction_energy: bool,
    num_insertions: int = 10000,
    optimize_structures: bool = False,
    cutoff_distance: float = 1.00,
    cutoff_to_com: bool = False,  # Whether to use center of mass for cutoff
    min_interplanar_distance: float = 6.0,
    random_seed: int = 0,
    min_interaction_energy: float = -1.25,
) -> WidomInsertionResults:
    """Run Widom insertion simulation to calculate Henry coefficient and heat of adsorption.

    The code in this function is derived from
    https://github.com/hspark1212/DAC-SIM
    MIT-licensed

    Args:
        calculator: ASE calculator for energy calculations.
        structure: ASE Atoms object representing the framework structure.
        gas: Name of the gas molecule to insert (e.g., 'H2', 'CO2').
        temperature: Temperature in Kelvin for the simulation.
        model_outputs_interaction_energy: Whether the calculator outputs interaction energies directly.
        num_insertions: Number of random insertion attempts.
        optimize_structures: Whether to optimize the structure and gas molecule before insertion.
        cutoff_distance: Minimum allowed distance between framework and gas atoms in angstroms.
        cutoff_to_com: Whether to use center of mass for distance calculations.
        min_interplanar_distance: Minimum interplanar distance before constructing a supercell, in angstroms.
        random_seed: Seed for random number generator to ensure reproducibility.
        min_interaction_energy: Minimum valid interaction energy for the gas molecule, in eV.

    Returns:
        Results containing Henry coefficient, heat of adsorption, and other computed properties.
    """
    gas_atoms = molecule(gas)

    # Optionally optimize structures
    if optimize_structures:
        print("Optimizing structure...")
        optimized_structure = optimize_atoms(
            calculator=calculator,
            atoms=structure,
        )
        if optimized_structure is None:
            raise ValueError("Structure optimization failed.")

        print("Optimizing gas molecule...")
        optimized_gas = optimize_atoms(
            calculator=calculator,
            atoms=gas_atoms,
            cell_relax=False,
        )
        if optimized_gas is None:
            raise ValueError("Gas molecule optimization failed.")
    else:
        optimized_structure = structure
        optimized_gas = gas_atoms

    # Run Widom insertion
    print(f"Running Widom insertion with {num_insertions} insertions...")
    energies, is_accessible, gas_positions = sample_compute_energies(
        calculator=calculator,
        structure=optimized_structure,
        gas=optimized_gas,
        num_insertions=num_insertions,
        cutoff_distance=cutoff_distance,
        cutoff_to_com=cutoff_to_com,
        min_interplanar_distance=min_interplanar_distance,
        random_seed=random_seed,
    )

    energy_structure = calculator.get_potential_energy(optimized_structure)
    energy_gas = calculator.get_potential_energy(optimized_gas)
    print(f"Energy of structure: {energy_structure} eV")
    print(f"Energy of gas: {energy_gas} eV")

    assert energy_structure is not None, "Energy of the structure must be computed."
    assert energy_gas is not None, "Energy of the gas molecule must be computed."

    # Analyze results
    print("Analyzing results...")
    results = analyze_widom_insertions(
        energies=energies,
        is_accessible=is_accessible,
        gas_positions=gas_positions,
        temperature=temperature,
        structure=optimized_structure,
        energy_structure=energy_structure,
        energy_gas=energy_gas,
        energies_are_interaction=model_outputs_interaction_energy,
        random_seed=random_seed,
        min_interaction_energy=min_interaction_energy,
    )

    print("Results:")
    print(results)
    return results
