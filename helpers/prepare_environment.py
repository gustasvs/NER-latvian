import os
import warnings

def prepare_environment(root_dir):

    # models dir
    os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)

    # cache dir
    os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)


    # clean up console logs
    warnings.simplefilter("ignore", FutureWarning)
