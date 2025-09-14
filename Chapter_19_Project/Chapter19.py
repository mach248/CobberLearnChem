from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import numpy as np
import os
import math

# Initialize the parser
parser = PDBParser(QUIET=True)

# Path to your PDB file
pdb_file = "your_protein.pdb"  # Replace with your actual file path

# Parse the PDB file
structure_id = os.path.basename(pdb_file).split('.')[0]
structure = parser.get_structure(structure_id, pdb_file)

# Get the first model
model = structure[0]

# Define amino acid codes
aa_codes = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# Simple secondary structure assignment based on phi/psi angles
def assign_ss(phi, psi):
    if phi is None or psi is None:
        return '-'  # Coil for termini

    # Alpha helix
    if -145 <= phi <= -35 and -70 <= psi <= 50:
        return 'H'
    # Beta sheet
    elif -180 <= phi <= -40 and 90 <= psi <= 180:
        return 'E'
    # Everything else is coil
    else:
        return '-'


print("Secondary Structure Analysis:")
print("-" * 50)
print("Chain\tResidue\tAA\tSS")
print("-" * 50)

# Analyze each chain
for chain in model:
    residues = list(chain.get_residues())

    # Calculate phi/psi angles and assign secondary structure
    for i in range(len(residues)):
        res = residues[i]
        res_name = res.get_resname()
        res_id = res.get_id()[1]

        # Skip non-standard residues
        if res_name not in aa_codes:
            continue

        # Get phi angle (needs previous residue)
        phi = None
        if i > 0:
            prev_res = residues[i - 1]
            if 'C' in prev_res and 'N' in res and 'CA' in res and 'C' in res:
                phi = calc_dihedral(
                    prev_res['C'].get_vector(),
                    res['N'].get_vector(),
                    res['CA'].get_vector(),
                    res['C'].get_vector()
                )
                phi = np.degrees(phi)

        # Get psi angle (needs next residue)
        psi = None
        if i < len(residues) - 1:
            next_res = residues[i + 1]
            if 'N' in res and 'CA' in res and 'C' in res and 'N' in next_res:
                psi = calc_dihedral(
                    res['N'].get_vector(),
                    res['CA'].get_vector(),
                    res['C'].get_vector(),
                    next_res['N'].get_vector()
                )
                psi = np.degrees(psi)

        # Assign secondary structure
        ss = assign_ss(phi, psi)

        # Print result
        aa_code = aa_codes.get(res_name, 'X')
        print(f"{chain.id}\t{res_id}\t{aa_code}\t{ss}")