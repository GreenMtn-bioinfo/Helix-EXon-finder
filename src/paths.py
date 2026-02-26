from pathlib import Path
import os 


### This module provides a centralized source of absolute paths for the project 
### (makes changes much easier and path related issues much less likely)


## Get the absolute path to the directory where THIS file lives
PROJECT_ROOT = Path(__file__).resolve().parent.parent


## Important sub-directories relative to root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEMO_SEQS_DIR  = PROJECT_ROOT / "demo_sequences"
PROFILES_DIR = PROJECT_ROOT / "profiles"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
TEMP_DIR = PROJECT_ROOT / "temp"
REFERENCE_DIR = DATA_DIR / "reference_genome"
NORM_DIR = DATA_DIR / 'normalization_params'


## Create these directories (useful in case GitHub drops an empty directory or it gets deleted by accident)
make_dirs = [PROFILES_DIR,
             PREDICTIONS_DIR,
             TEMP_DIR,
             REFERENCE_DIR]
for dir in make_dirs:
    dir.mkdir(parents=True, exist_ok=True)


## Define specific file names and sub-sub-directories used in the core modules

# Used by get_demo_seqs.py
HELD_OUT = DATA_DIR / "held_out_regions_GRCh38_p14.txt"
FIG_11_DEMO = DEMO_SEQS_DIR / "Figure11_demo_regions"
EXAMPLE_DIST = DEMO_SEQS_DIR / "example_length_distribution"

# Used by hex_finder.py
NORM_DATA_PATHS = [NORM_DIR / 'z_norm_training_means.npy', 
                   NORM_DIR / 'z_norm_training_sdevs.npy', 
                   NORM_DIR / 'min_training_post_z.npy', 
                   NORM_DIR / 'max_training_post_z.npy']
EXON_LENGTHS_DIST = DATA_DIR / 'human_exon_length_distribution_Mokry_et_al_2010.csv'

# Used by profile_generator_inference.py
SKIPPED_LOG_NAME = "skipped_sequences_log.txt"
TRINUCLEO = DATA_DIR / 'trinucleo_Sharma_et_al_2025_params.csv'
TETRANUCLEO = DATA_DIR / 'tetranucleo_Sharma_et_al_2025_params.csv'

# Used by visualize_predictions.py
VISUAL_PREDICTIONS = PREDICTIONS_DIR / "visual_report.html"
FAVICON = PROJECT_ROOT / "dependencies" / "hex_finder_simple_icon.png"


## Define utility functions for handling paths

def shorten_path(path,
                 levels: int = 2) -> str:
    """
    Facilitates displaying only the deepest N levels of the path (useful for console prints)
    """
    
    # Ensure the path is a Path object
    if not isinstance(path, Path):
        path = Path(path)
    
    # Get the last 'levels' components of the path
    return os.path.sep.join(path.parts[-levels:])