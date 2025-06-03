import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from widom.utils import check_accessibility, sample_gas_positions


@pytest.fixture
def rng():
    """Fixed random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_framework():
    """Simple cubic framework for testing."""
    return bulk("Cu", cubic=True, a=5.0)


@pytest.fixture
def simple_gas():
    """Simple diatomic gas molecule."""
    gas = Atoms("N2", positions=[[0, 0, 0], [1.1, 0, 0]])
    return gas


@pytest.fixture
def single_atom_gas():
    """Single atom gas for simpler testing."""
    return Atoms("Ar", positions=[[0, 0, 0]])


class TestSampleGasPositions:
    """Tests for the sample_gas_positions function."""

    def test_output_shape(self, simple_framework, simple_gas, rng):
        """Test that output has correct shape."""
        num_insertions = 10
        positions = sample_gas_positions(simple_framework, simple_gas, num_insertions, rng)

        expected_shape = (num_insertions, len(simple_gas), 3)
        assert positions.shape == expected_shape

    def test_insertions(self, simple_framework: Atoms, simple_gas: Atoms, rng: np.random.Generator):
        """Test single gas insertion."""
        num_insertions = 10_000
        positions = sample_gas_positions(simple_framework, simple_gas, num_insertions, rng)

        assert positions.shape == (num_insertions, len(simple_gas), 3)
        # unit_positions = np.linalg.pinv(simple_framework.cell.array).T @ positions.reshape(-1, 3).T
        unit_positions = simple_framework.cell.scaled_positions(positions.reshape(-1, 3))
        assert np.all(unit_positions >= 0)
        assert np.all(unit_positions <= 1)

    def test_reproducibility(self, simple_framework, simple_gas):
        """Test that same seed produces same results."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        positions1 = sample_gas_positions(simple_framework, simple_gas, 5, rng1)
        positions2 = sample_gas_positions(simple_framework, simple_gas, 5, rng2)

        np.testing.assert_array_almost_equal(positions1, positions2)

    def test_gas_molecule_integrity(self, simple_framework, simple_gas, rng):
        """Test that gas molecule bond lengths are preserved."""
        original_distance = np.linalg.norm(simple_gas.get_distances(0, 1, mic=False))

        positions = sample_gas_positions(simple_framework, simple_gas, 10, rng)

        from ase.geometry import get_distances

        for i in range(positions.shape[0]):
            gas_distance = get_distances(
                positions[i], cell=simple_framework.cell, pbc=simple_framework.pbc
            )[1][0, 1]
            np.testing.assert_almost_equal(gas_distance, original_distance, decimal=10)


class TestCheckAccessibility:
    """Tests for the check_accessibility function."""

    def test_no_overlaps(self, simple_framework):
        """Test case where gas molecules don't overlap with framework."""
        # Place gas molecules far from framework atoms (considering PBC)
        # Framework has Cu at [0,0,0] so place gas near center of cell [2.5, 2.5, 2.5]
        gas_positions = np.array(
            [[[2.5, 2.5, 2.5], [2.6, 2.6, 2.6]], [[2.0, 2.0, 2.0], [2.1, 2.1, 2.1]]]
        )  # 2 insertions, 2 atoms each
        framework_coords = simple_framework.get_positions()
        lattice_matrix = simple_framework.cell.array

        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert accessible.shape == (2,)
        assert np.all(accessible)  # All should be accessible

    def test_all_overlaps(self, simple_framework):
        """Test case where all gas molecules overlap with framework."""
        # Place gas molecules at framework atom positions
        framework_coords = simple_framework.get_positions()
        gas_positions = framework_coords[:2].reshape(2, 1, 3)  # 2 insertions, 1 atom each
        gas_positions = np.concatenate(
            [gas_positions, gas_positions + 5, gas_positions + 0.1], axis=0
        )  # Test wrapping, and slightly offset
        lattice_matrix = simple_framework.cell.array

        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert len(accessible) == len(gas_positions)
        assert not np.any(accessible)  # None should be accessible

    def test_mixed_accessibility(self, simple_framework):
        """Test case with both accessible and inaccessible positions."""
        framework_coords = simple_framework.get_positions()
        lattice_matrix = simple_framework.cell.array

        # First insertion overlaps, second doesn't
        gas_positions = np.array(
            [
                [framework_coords[0], [2.5, 2.5, 2.5]],  # Overlaps with first framework atom
                [[2.5, 2.5, 2.5], [2.6, 2.6, 2.6]],  # Far from any framework atom (center of cell)
            ]
        )

        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert accessible.shape == (2,)
        assert not accessible[0]  # First should be inaccessible
        assert accessible[1]  # Second should be accessible

    def test_cutoff_distance_effect(self, simple_framework):
        """Test that cutoff distance affects accessibility."""
        framework_coords = simple_framework.get_positions()
        lattice_matrix = simple_framework.cell.array

        # Place gas at moderate distance from framework
        gas_positions = np.array([[[framework_coords[0] + np.array([1.5, 0, 0])]]])

        # With small cutoff, should be accessible
        accessible_small = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 1.0, cutoff_to_com=False
        )
        assert accessible_small[0]

        # With large cutoff, should be inaccessible
        accessible_large = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )
        assert not accessible_large[0]

    def test_multi_atom_gas(self, simple_framework):
        """Test accessibility checking with multi-atom gas molecules."""
        framework_coords = simple_framework.get_positions()
        lattice_matrix = simple_framework.cell.array

        # Two-atom gas molecule, one atom overlaps
        gas_positions = np.array(
            [
                [framework_coords[0], [2.5, 2.5, 2.5]]  # First atom overlaps, second doesn't
            ]
        )

        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert not accessible[0]  # Should be inaccessible due to overlap of first atom

    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        gas_positions = np.empty((0, 1, 3))
        framework_coords = np.array([[0, 0, 0]])
        lattice_matrix = np.eye(3) * 5

        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert accessible.shape == (0,)

    def test_periodic_boundary_conditions(self):
        """Test that periodic boundary conditions are handled correctly."""
        # Create a small cell where gas near one edge overlaps with framework near opposite edge
        lattice_matrix = np.eye(3) * 3.0
        framework_coords = np.array([[0.1, 0.1, 0.1]])  # Near origin
        gas_positions = np.array([[[2.9, 2.9, 2.9]]])  # Near opposite corner

        # Due to PBC, these should be close and overlap
        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        assert not accessible[0]  # Should detect overlap across PBC


class TestIntegration:
    """Integration tests combining both functions."""

    def test_widom_insertion_workflow(self, simple_framework, single_atom_gas, rng):
        """Test a complete Widom insertion workflow."""
        num_insertions = 100

        # Sample gas positions
        gas_positions = sample_gas_positions(simple_framework, single_atom_gas, num_insertions, rng)

        # Check accessibility
        framework_coords = simple_framework.get_positions()
        lattice_matrix = simple_framework.cell.array
        accessible = check_accessibility(
            gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
        )

        # Should have some accessible and some inaccessible positions
        assert 0 < np.sum(accessible) < num_insertions
        assert accessible.shape == (num_insertions,)

    def test_different_framework_sizes(self, single_atom_gas, rng):
        """Test with different framework cell sizes."""
        for cell_size in [3.0, 5.0, 10.0]:
            framework = bulk("Cu", cubic=True, a=cell_size)

            gas_positions = sample_gas_positions(framework, single_atom_gas, 10, rng)
            framework_coords = framework.get_positions()
            lattice_matrix = framework.cell.array

            accessible = check_accessibility(
                gas_positions, framework_coords, lattice_matrix, 2.0, cutoff_to_com=False
            )

            # Larger cells should generally have more accessible positions
            assert accessible.shape == (10,)
            if cell_size >= 10.0:
                assert np.sum(accessible) > 5  # Most positions should be accessible in large cell
