import os
import warnings

def prepare_environment(root_dir):
    # make sure models dir exists
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # clean up console logs
    warnings.simplefilter("ignore", FutureWarning)
