from rdkit import Chem
from rdkit.Chem import Descriptors

# Define the SMILES strings for some example molecules
compounds = {
    'theobromine': 'CN1C=NC2=C1C(=O)NC(=O)N2C',
    'ethanol': 'CCO',
    'acetic_acid': 'CC(=O)O'
}

# Loop through each compound and calculate properties
for name, smiles in compounds.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_weight = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)

        print(f"\n{name.capitalize()}:")
        print(f"  Molecular Weight: {mol_weight:.2f} g/mol")
        print(f"  Rotatable Bonds: {rot_bonds}")
        print(f"  Topological Polar Surface Area (TPSA): {tpsa:.2f} Å²")
    else:
        print(f"\nInvalid SMILES for {name}.")

