# VMD: topology written BEFORE MLpot (bonds intact).
# Atoms: 100 — must match trajectory frame count.
mol new {${HOME}/karml/tests/functionality/mlpot/output/dynamics/cluster_for_vmd_aco_10mer.psf}
mol addfile {${HOME}/karml/tests/functionality/mlpot/output/dynamics/nve_aco_10mer.dcd} waitfor all
animate goto 0
display update
