from pathlib import Path

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.io import read

from .ase_widom_insertion import (
    WidomInsertionResults,
    analyze_widom_insertions,
    ase_widom_insertion,
)
from .utils import optimize_atoms


def run_widom_insertion(
    calculator: Calculator,
    structure: Path,
    gas: str,
    temperature: float,
    model_outputs_interaction_energy: bool,
    num_insertions: int = 10000,
    optimize_structures: bool = False,
    cutoff_distance: float = 1.00,
    cutoff_to_com: bool = False,  # Whether to use center of mass for cutoff
    min_interplanar_distance: float = 6.0,
    random_seed: int = 0,
) -> WidomInsertionResults:
    # Download files from remote storage

    # Load structure
    structure_atoms = read(structure)
    assert isinstance(structure_atoms, Atoms), "Structure must be an ASE Atoms object."
    gas_atoms = molecule(gas)

    # Optionally optimize structures
    if optimize_structures:
        print("Optimizing structure...")
        optimized_structure = optimize_atoms(
            calculator=calculator,
            atoms=structure_atoms,
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
        optimized_structure = structure_atoms
        optimized_gas = gas_atoms

    # Run Widom insertion
    print(f"Running Widom insertion with {num_insertions} insertions...")
    energies, is_accessible, gas_positions = ase_widom_insertion(
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

    # Analyze results
    print("Analyzing results...")
    results = analyze_widom_insertions(
        energies=energies,
        is_accessible=is_accessible,
        gas_positions=gas_positions,
        temperature=temperature,
        structure=optimized_structure,
        energy_structure=energy_structure,  # type: ignore
        energy_gas=energy_gas,  # type: ignore
        energies_are_interaction=model_outputs_interaction_energy,
        random_seed=random_seed,
    )

    print("Results:")
    print(results)
    return results
