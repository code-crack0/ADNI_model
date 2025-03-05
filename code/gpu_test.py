import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the device name
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")

    # Check if the GPU is GTX 1650
    if "GTX 1650" in device_name:
        print("The GPU being used is GTX 1650.")
        device = torch.device("cuda:0")
    else:
        print("The GPU being used is not GTX 1650. Using CPU instead.")
        device = torch.device("cpu")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

print(f"Device set to: {device}")