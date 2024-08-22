import os
import sys
import logging
import warnings

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep.compute.compute_dpo_score import main as compute_dpo_scores_main

warnings.filterwarnings("ignore")

def run_compute_dpo_scores():
    # Define the configuration path
    config_path = "../configs/config.yaml"
    
    # Setup logging (console only)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # Log the configuration path
    logger.info(f"Using configuration file at: {config_path}")
    
    # Set the configuration path in the environment, so the main function can access it
    os.environ['CONFIG_PATH'] = config_path

    # Run the main function from the src script
    compute_dpo_scores_main()

if __name__ == "__main__":
    run_compute_dpo_scores()
