"""
Configuration parameters for solubility data exploration.
"""

# Data paths
INPUT_PATH = "../../Project_Input/"
SOLUBILITY_DATA = INPUT_PATH + "solubility_esol.csv"
OUTPUT_PATH = "./"
REPORT_PATH = OUTPUT_PATH + "data_exploration.html"

# Visualization settings
FIGURE_SIZE = (10, 6)
HIST_BINS = 30
CORRELATION_CMAP = "coolwarm"
OUTLIER_THRESHOLD = 1.5  # For IQR-based outlier detection

# Feature extraction settings
MOLECULE_FEATURES = [
    'MolWt',        # Molecular weight
    'LogP',         # Octanol-water partition coefficient
    'NumHDonors',   # Number of H-bond donors
    'NumHAcceptors',# Number of H-bond acceptors
    'TPSA',         # Topological polar surface area
    'NumRotatableBonds',  # Number of rotatable bonds
    'NumAromaticRings',   # Number of aromatic rings
    'NumHeteroatoms',     # Number of heteroatoms
    'NumHeavyAtoms',      # Number of heavy atoms
    'FractionCSP3',       # Fraction of carbon atoms that are sp3 hybridized
]

# Report settings
REPORT_TITLE = "Solubility Data Exploration"
