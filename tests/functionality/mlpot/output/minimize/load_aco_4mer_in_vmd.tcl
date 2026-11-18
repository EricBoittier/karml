# VMD: topology written BEFORE MLpot (bonds intact).
# Atoms: 40 — must match trajectory frame count.
mol new {${HOME}/karml/tests/functionality/mlpot/output/minimize/cluster_for_vmd_aco_4mer.psf}
mol addfile {${HOME}/karml/tests/functionality/mlpot/output/minimize/mini_full_mlpot_aco_4mer.dcd} waitfor all
animate goto 0
display update
