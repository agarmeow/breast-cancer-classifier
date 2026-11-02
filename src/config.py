"""Central settings so every module uses the same values."""
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_SELECTED_FEATURES = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

for _d in (PLOTS_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)
