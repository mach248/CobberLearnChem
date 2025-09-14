from Bio.PDB import PDBParser, Superimposer, PDBIO
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import subprocess
import sys


def analyze_pdb_structure(pdb_file, structure_name="Structure"):
    """Analyze basic structural elements in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Basic statistics
    residue_count = 0
    atom_count = 0
    chain_ids = []

    for model in structure:
        for chain in model:
            chain_ids.append(chain.id)
            for residue in chain:
                residue_count += 1
                for atom in residue:
                    atom_count += 1

    # Print basic information
    print(f"\n{structure_name} Analysis:")
    print(f"PDB file: {pdb_file}")
    print(f"Chains: {', '.join(chain_ids)}")
    print(f"Total residues: {residue_count}")
    print(f"Total atoms: {atom_count}")

    # Return structure for further analysis
    return structure, chain_ids


def compare_structures(exp_structure, af_structure, exp_chain_ids, af_chain_ids, exp_file, af_file):
    """Compare experimental structure with AlphaFold prediction."""
    print("\nComparing Experimental and AlphaFold Structures:")

    # Select the first chain from each structure for comparison
    exp_chain_id = exp_chain_ids[0] if exp_chain_ids else None
    af_chain_id = af_chain_ids[0] if af_chain_ids else None

    if not exp_chain_id or not af_chain_id:
        print("Error: Could not find chains for comparison")
        return

    # Get CA atoms from both structures for the selected chains
    exp_atoms = []
    af_atoms = []
    exp_residues = []
    af_residues = []

    for residue in exp_structure[0][exp_chain_id]:
        if 'CA' in residue:
            exp_atoms.append(residue['CA'])
            exp_residues.append(residue)

    for residue in af_structure[0][af_chain_id]:
        if 'CA' in residue:
            af_atoms.append(residue['CA'])
            af_residues.append(residue)

    # Use the shorter list length
    min_length = min(len(exp_atoms), len(af_atoms))
    if min_length < 3:
        print("Error: Not enough CA atoms for comparison")
        return

    print(f"Comparing {min_length} CA atoms")
    exp_atoms = exp_atoms[:min_length]
    af_atoms = af_atoms[:min_length]
    exp_residues = exp_residues[:min_length]
    af_residues = af_residues[:min_length]

    # Superimpose structures
    super_imposer = Superimposer()
    super_imposer.set_atoms(exp_atoms, af_atoms)

    # Calculate RMSD
    rmsd = super_imposer.rms

    print(f"RMSD between structures: {rmsd:.2f} Å")
    print("RMSD (Root Mean Square Deviation) measures the average distance between atoms.")
    print("Lower values indicate better structural agreement:")
    print("  < 1.0 Å: Excellent match")
    print("  1.0-3.0 Å: Good match")
    print("  3.0-5.0 Å: Moderate match")
    print("  > 5.0 Å: Poor match")

    # Calculate per-residue distances
    distances = []
    residue_names = []

    for i in range(min_length):
        distance = np.linalg.norm(exp_atoms[i].get_coord() - af_atoms[i].get_coord())
        distances.append(distance)
        residue_names.append(f"{exp_residues[i].get_resname()}{exp_residues[i].get_id()[1]}")

    # Plot distances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(distances)), distances, color=plt.cm.viridis(np.array(distances) / max(distances)))
    plt.xlabel('Residue Position')
    plt.ylabel('Distance (Å)')
    plt.title('Per-residue Distance Between Experimental and AlphaFold Structures')
    plt.axhline(y=1.5, color='g', linestyle='--', alpha=0.5, label='Excellent match (< 1.5 Å)')
    plt.axhline(y=3.0, color='y', linestyle='--', alpha=0.5, label='Good match (< 3.0 Å)')
    plt.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Poor match (> 5.0 Å)')

    # Add residue labels for the largest differences
    threshold = max(3.0, np.percentile(distances, 75))  # Label residues with distances above 3Å or top 25%
    for i, distance in enumerate(distances):
        if distance > threshold:
            plt.text(i, distance + 0.2, residue_names[i], ha='center', rotation=90, fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.savefig("structure_comparison.png")
    print("Structure comparison plot saved as 'structure_comparison.png'")

    # Create a superimposed PDB file for visualization
    # Apply the rotation/translation to the AlphaFold structure
    super_imposer.apply(af_structure.get_atoms())

    # Save the superimposed structures to a single file
    io = PDBIO()

    # First, save the experimental structure
    io.set_structure(exp_structure)
    io.save("exp_structure.pdb")

    # Then save the transformed AlphaFold structure
    io.set_structure(af_structure)
    io.save("af_structure_aligned.pdb")

    # Create a PyMOL script to visualize the superposition
    with open("visualize_alignment.pml", "w") as f:
        f.write("# PyMOL script to visualize structural alignment\n")
        f.write(f"load exp_structure.pdb, experimental\n")
        f.write(f"load af_structure_aligned.pdb, alphafold\n")
        f.write("hide everything\n")
        f.write("show cartoon\n")
        f.write("color blue, experimental\n")
        f.write("color red, alphafold\n")
        f.write("set cartoon_transparency, 0.5, experimental\n")
        f.write("zoom\n")
        f.write("ray 1200, 1000\n")
        f.write("png alignment.png\n")

    print("\nCreated files for visualization:")
    print("1. exp_structure.pdb - The experimental structure")
    print("2. af_structure_aligned.pdb - The aligned AlphaFold prediction")
    print("3. visualize_alignment.pml - A PyMOL script to visualize the alignment")

    # Try to run PyMOL if available
    try_run_pymol = input("\nWould you like to try running PyMOL to visualize the alignment? (y/n): ")
    if try_run_pymol.lower() == 'y':
        try:
            # Try different PyMOL executable names
            pymol_commands = [
                ["pymol", "-r", "visualize_alignment.pml"],
                ["PyMOL", "-r", "visualize_alignment.pml"],
                ["pymol.exe", "-r", "visualize_alignment.pml"]
            ]

            success = False
            for cmd in pymol_commands:
                try:
                    print(f"Trying to run PyMOL with command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    success = True
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

            if not success:
                print("\nCould not run PyMOL automatically.")
                print("To visualize the alignment:")
                print("1. Open PyMOL")
                print("2. In PyMOL, use File > Run Script... and select 'visualize_alignment.pml'")
        except Exception as e:
            print(f"\nError running PyMOL: {e}")
            print("To visualize the alignment:")
            print("1. Open PyMOL")
            print("2. In PyMOL, use File > Run Script... and select 'visualize_alignment.pml'")
    else:
        print("\nTo visualize the alignment:")
        print("1. Open PyMOL")
        print("2. In PyMOL, use File > Run Script... and select 'visualize_alignment.pml'")

    return rmsd, distances


def analyze_alphafold_confidence(pdb_file):
    """Analyze the pLDDT confidence scores from an AlphaFold PDB file."""
    residues = []
    scores = []
    residue_names = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # Alpha carbon atoms
                residue_num = int(line[22:26].strip())
                residue_name = line[17:20].strip()
                b_factor = float(line[60:66].strip())  # pLDDT score

                residues.append(residue_num)
                residue_names.append(f"{residue_name}{residue_num}")
                scores.append(b_factor)

    if not scores:
        print("No pLDDT scores found in the AlphaFold PDB file.")
        return None, None

    # Check if these look like pLDDT scores (typically 0-100)
    if max(scores) > 100 or min(scores) < 0:
        print("Warning: B-factor values don't appear to be pLDDT scores (not in 0-100 range).")
        print(f"Range: {min(scores)} to {max(scores)}")
        use_scores = input("Continue with B-factor analysis anyway? (y/n): ")
        if use_scores.lower() != 'y':
            return None, None

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Create a colormap based on confidence levels
    colors = []
    for score in scores:
        if score > 90:
            colors.append('green')  # Very high confidence
        elif score > 70:
            colors.append('blue')  # Confident
        elif score > 50:
            colors.append('orange')  # Low confidence
        else:
            colors.append('red')  # Very low confidence

    plt.bar(range(len(scores)), scores, color=colors)
    plt.xlabel('Residue Position')
    plt.ylabel('pLDDT Score')
    plt.title('AlphaFold Prediction Confidence by Residue (pLDDT)')
    plt.axhline(y=90, color='g', linestyle='-', alpha=0.3, label='Very high (pLDDT > 90)')
    plt.axhline(y=70, color='b', linestyle='-', alpha=0.3, label='Confident (70 < pLDDT < 90)')
    plt.axhline(y=50, color='orange', linestyle='-', alpha=0.3, label='Low (50 < pLDDT < 70)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Very low (pLDDT < 50)')

    # Add residue labels for the lowest confidence regions
    threshold = min(50, np.percentile(scores, 25))  # Label residues with pLDDT below 50 or bottom 25%
    for i, score in enumerate(scores):
        if score < threshold:
            plt.text(i, score + 2, residue_names[i], ha='center', rotation=90, fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.savefig("alphafold_confidence.png")
    print("AlphaFold confidence plot saved as 'alphafold_confidence.png'")

    # Print summary statistics
    print("\npLDDT Score Summary:")
    print(f"Average pLDDT: {np.mean(scores):.2f}")
    print(f"Minimum pLDDT: {min(scores):.2f}")
    print(f"Maximum pLDDT: {max(scores):.2f}")

    # Categorize regions by confidence
    very_high = [s for s in scores if s > 90]
    high = [s for s in scores if 70 < s <= 90]
    low = [s for s in scores if 50 < s <= 70]
    very_low = [s for s in scores if s <= 50]

    print(f"\nVery high confidence regions: {len(very_high)} residues ({len(very_high) / len(scores) * 100:.1f}%)")
    print(f"High confidence regions: {len(high)} residues ({len(high) / len(scores) * 100:.1f}%)")
    print(f"Low confidence regions: {len(low)} residues ({len(low) / len(scores) * 100:.1f}%)")
    print(f"Very low confidence regions: {len(very_low)} residues ({len(very_low) / len(scores) * 100:.1f}%)")

    return residues, scores


# Create a root window but hide it
root = tk.Tk()
root.withdraw()

# Open file dialog to select experimental PDB file
print("Please select the EXPERIMENTAL PDB file (e.g., 1TRZ.pdb)...")
exp_pdb_file = filedialog.askopenfilename(
    title="Select Experimental PDB File",
    filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
)

if not exp_pdb_file:  # If user cancels the dialog
    print("No experimental file selected. Exiting.")
    exit()

# Open file dialog to select AlphaFold prediction PDB file
print("\nPlease select the ALPHAFOLD PREDICTION PDB file from the extracted ColabFold zip...")
af_pdb_file = filedialog.askopenfilename(
    title="Select AlphaFold PDB File",
    filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
)

if not af_pdb_file:  # If user cancels the dialog
    print("No AlphaFold file selected. Exiting.")
    exit()

# Check if files exist
if not os.path.exists(exp_pdb_file) or not os.path.exists(af_pdb_file):
    print("Error: One or both files not found!")
    exit()

# Analyze the experimental structure
exp_structure, exp_chain_ids = analyze_pdb_structure(exp_pdb_file, "Experimental Structure")

# Analyze the AlphaFold structure
af_structure, af_chain_ids = analyze_pdb_structure(af_pdb_file, "AlphaFold Prediction")

# Analyze AlphaFold confidence scores (pLDDT)
print("\nAnalyzing AlphaFold confidence scores (pLDDT)...")
residues, scores = analyze_alphafold_confidence(af_pdb_file)


def create_nglview_visualization(exp_pdb, af_pdb_aligned):
    """Create an HTML file with an interactive 3D visualization using NGLView."""
    try:
        import nglview as nv

        # Create HTML with both structures
        html_content = """
        <html>
        <head>
            <title>Structure Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { display: flex; flex-direction: column; }
                .viewer { width: 800px; height: 600px; margin-bottom: 20px; }
                .description { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Experimental vs. AlphaFold Structure Comparison</h1>
            <div class="description">
                <p>This visualization shows the superposition of the experimental structure (blue) 
                and the AlphaFold prediction (red).</p>
                <p>Use your mouse to rotate, zoom, and pan the structure.</p>
            </div>
            <div class="container">
                <div class="viewer" id="viewport"></div>
            </div>
            <script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>
            <script>
                document.addEventListener("DOMContentLoaded", function() {
                    var stage = new NGL.Stage("viewport");

                    Promise.all([
                        stage.loadFile("exp_structure.pdb"),
                        stage.loadFile("af_structure_aligned.pdb")
                    ]).then(function(components) {
                        // Set representations
                        components[0].addRepresentation("cartoon", {color: "blue"});
                        components[1].addRepresentation("cartoon", {color: "red"});

                        // Center view
                        stage.autoView();
                    });
                });
            </script>
        </body>
        </html>
        """

        with open("structure_comparison.html", "w") as f:
            f.write(html_content)

        print("\nCreated 'structure_comparison.html' for interactive visualization.")
        print("Open this file in a web browser to view the aligned structures.")

    except ImportError:
        print("\nNGLView not installed. Skipping HTML visualization creation.")
        print("To install NGLView: pip install nglview")

# Compare 3D structures
compare_structures(exp_structure, af_structure, exp_chain_ids, af_chain_ids, exp_pdb_file, af_pdb_file)

# Add this after creating the PyMOL script in the compare_structures function
create_nglview_visualization("exp_structure.pdb", "af_structure_aligned.pdb")

# Show all plots
plt.show()