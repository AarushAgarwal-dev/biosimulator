"""The fast CPM paths must compute exactly what the reference per-pixel/per-cell code computes.

run_mcs used to call _attempt_pixel_copy once per attempt (random module + numpy scalar
indexing) and _update_all_cells/_update_fields did a full-lattice mask per cell. The PDAC
preset took >300 s on the deployment and timed out. The fast paths are pinned here against
the reference implementations, which are kept for exactly this purpose.
"""
import random
import unittest

import numpy as np

import abm_blueprints
from abm_engine import build_cpm_from_blueprint


def _model(name, mcs=30, seed=3):
    random.seed(seed)
    np.random.seed(seed)
    cpm = build_cpm_from_blueprint(abm_blueprints.get_abm_preset(name))
    for _ in range(mcs):          # let cells grow, move, and fields develop
        cpm.run_mcs()
    return cpm


class FastKernelEquivalenceTests(unittest.TestCase):
    def test_delta_h_matches_the_reference_hamiltonian(self):
        # wound_healing has chemotaxis, pdac_tme several types and fields, cell_sorting adhesion only.
        for name in ("cell_sorting", "wound_healing", "pdac_tme"):
            with self.subTest(preset=name):
                cpm = _model(name)
                delta_h, _ = cpm._make_fast_kernel()
                rng = np.random.default_rng(0)
                checked = 0
                for _ in range(20000):
                    x = int(rng.integers(cpm.width))
                    y = int(rng.integers(cpm.height))
                    dx, dy = cpm._neighbor_offsets[int(rng.integers(len(cpm._neighbor_offsets)))]
                    s = int(cpm.lattice[x, y])
                    t = int(cpm.lattice[(x + dx) % cpm.width, (y + dy) % cpm.height])
                    if s == t:
                        continue
                    self.assertAlmostEqual(delta_h(x, y, s, t), cpm._compute_delta_H(x, y, s, t), places=9)
                    checked += 1
                self.assertGreater(checked, 500)

    def test_cell_statistics_match_per_cell_argwhere(self):
        cpm = _model("tumor_growth")
        cpm._update_all_cells()
        for cid, cell in cpm.cells.items():
            if not cell.alive:
                continue
            coords = np.argwhere(cpm.lattice == cid)
            self.assertEqual(cell.volume, len(coords))
            self.assertAlmostEqual(cell.center_x, float(np.mean(coords[:, 0])), places=12)
            self.assertAlmostEqual(cell.center_y, float(np.mean(coords[:, 1])), places=12)

    def test_field_update_matches_the_per_cell_loop(self):
        cpm = _model("pdac_tme")
        reference = {name: f.grid.copy() for name, f in cpm.fields.items()}
        # Reference: the original per-cell loop, then the same diffusion step.
        for name, grid in reference.items():
            for cid, cell in cpm.cells.items():
                if not cell.alive:
                    continue
                ct = cell.cell_type
                if name in ct.secretion_rates:
                    grid[cpm.lattice == cid] += ct.secretion_rates[name]
                if name in ct.uptake_rates:
                    mask = cpm.lattice == cid
                    grid[mask] = np.maximum(0, grid[mask] - ct.uptake_rates[name])
        cpm._update_fields()
        for name, f in cpm.fields.items():
            saved = f.grid.copy()
            f.grid = reference[name]
            f.diffuse_and_decay()
            np.testing.assert_allclose(saved, f.grid, rtol=0, atol=1e-12, err_msg=name)

    def test_a_seeded_run_repeats_exactly(self):
        first = _model("wound_healing", mcs=15, seed=11)
        second = _model("wound_healing", mcs=15, seed=11)
        np.testing.assert_array_equal(first.lattice, second.lattice)
        third = _model("wound_healing", mcs=15, seed=12)
        self.assertFalse(np.array_equal(first.lattice, third.lattice))

    def test_field_rounding_only_changes_transport_precision(self):
        random.seed(5)
        np.random.seed(5)
        cpm = build_cpm_from_blueprint(abm_blueprints.get_abm_preset("tumor_growth"))
        result = cpm.simulate(num_mcs=6, save_every=3, field_decimals=4)
        frame = np.asarray(result["fields"]["nutrient"][-1])
        np.testing.assert_allclose(frame, np.round(cpm.fields["nutrient"].grid, 4), atol=0)


if __name__ == "__main__":
    unittest.main()
