import torch

# training settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 4

# sample settings
MAX_SAMPLE_LENGTH = 128

# generic
MAX_SAMPLES_TO_USE = -1 # all samples
# MAX_SAMPLES_TO_USE = 1200