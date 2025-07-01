import pubchempy as pcp

# Fetch compound data for 'caffeine' using its name
compounds = pcp.get_compounds('caffeine', 'name')

# Check if we got any results
if compounds:
    caffeine = compounds[0]

    # Print some basic properties
    print(f"Name: {caffeine.iupac_name}")
    print(f"Molecular Formula: {caffeine.molecular_formula}")
    print(f"Molecular Weight: {caffeine.molecular_weight} g/mol")
    print(f"Canonical SMILES: {caffeine.canonical_smiles}")
    print(f"InChIKey: {caffeine.inchikey}")
else:
    print("No data found for caffeine.")






