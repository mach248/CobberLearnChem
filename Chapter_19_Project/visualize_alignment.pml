# PyMOL script to visualize structural alignment
load exp_structure.pdb, experimental
load af_structure_aligned.pdb, alphafold
hide everything
show cartoon
color blue, experimental
color red, alphafold
set cartoon_transparency, 0.5, experimental
zoom
ray 1200, 1000
png alignment.png
