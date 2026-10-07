"""pyCHARMM CGenFF rigid-monomer interaction energy for the H2O–MeOH dimer."""

from __future__ import annotations

import pytest
import numpy as np

from karml.analysis.dimer_molecules import make_oriented_scan_geometries


@pytest.mark.pycharmm
def test_pycharmm_cgenff_dimer_regression():
    try:
        import pycharmm
        import pycharmm.settings as settings
        import pycharmm.generate as gen
        import pycharmm.coor as coor
        import pycharmm.energy as energy
        import pycharmm.psf as psf
        import pycharmm.read as read
        import pandas as pd
        from karml.interfaces.pycharmmInterface.import_pycharmm import (
            CGENFF_PRM,
            CGENFF_RTF,
            reset_block,
            pycharmm_quiet,
        )
    except ImportError:
        pytest.skip("pyCHARMM not installed or importable")

    from karml.data.cgenff_dataset import load_reference, reorder_to_cgenff_template

    pycharmm_quiet()
    reset_block()
    settings.set_bomb_level(-2)

    read.rtf(CGENFF_RTF)
    read.prm(CGENFF_PRM)

    # 4.0 A COM separation, near the CGenFF minimum for this fixed orientation.
    # NOT the 2.8 A of h2o_meoh_dimer_atoms: that sits far up the repulsive wall
    # (E_int ~ +5100 kcal/mol) and only became usable via a minimisation that
    # descended 2-3 A into whichever local minimum the optimiser happened to
    # find -- which differed by build (macOS reached -3.79, pc-studix -8.43).
    dimer = list(make_oriented_scan_geometries("TIP3", "MEOH", [4.0], [0.0]))[0].atoms
    z_all = np.asarray(dimer.get_atomic_numbers())
    pos_all = np.asarray(dimer.get_positions())

    # coor.set_positions fills the PSF positionally, so the geometry must be in
    # CGenFF RESI order first. ASE's CH3OH puts the hydroxyl H fourth; MEOH wants
    # it third (HG1). Feeding the raw order stretches an O-H bond to ~2 A and
    # costs ~5x10^5 kcal/mol -- silently, because nothing checks.
    ref = load_reference()
    _, pos_a = reorder_to_cgenff_template(ref, z_all[:3], pos_all[:3], "TIP3")
    _, pos_b = reorder_to_cgenff_template(ref, z_all[3:], pos_all[3:], "MEOH")

    def total_energy(segments, positions):
        """Full-system CGenFF energy for a freshly built PSF."""
        psf.delete_atoms()
        for seg_name, resi in segments:
            read.sequence_string(resi)
            gen.new_segment(seg_name)
        coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))
        pycharmm.lingo.charmm_script("ENER")
        return float(energy.get_total())

    # Three independent systems. `ENER sele segid A end` does NOT restrict the
    # energy on this libcharmm build -- energy.get_total() returns the same
    # full-system total after every such call, so the previous
    # `e_d - e_m1 - e_m2` silently evaluated to -e_d. INTE is equally inert here.
    e_dimer = total_energy([("A", "TIP3"), ("B", "MEOH")], np.vstack([pos_a, pos_b]))
    e_mon_a = total_energy([("A", "TIP3")], pos_a)
    e_mon_b = total_energy([("B", "MEOH")], pos_b)

    e_int_kcal = e_dimer - e_mon_a - e_mon_b

    # Rigid-monomer CGenFF interaction energy at 4.0 A COM separation. With no
    # minimisation this is deterministic: macOS and pc-studix both give
    # -3.2625 kcal/mol (dimer 7.1602, TIP3 0.1208, MEOH 10.3020) to 4 decimals.
    # The band is left at 0.15 to tolerate force-field/parser revisions rather
    # than because the number moves between builds.
    assert e_int_kcal == pytest.approx(-3.26, abs=0.15)
