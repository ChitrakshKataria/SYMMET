import torch
import torch_directml


from models.baseline_mlp import BaselineMLP


### Device agnostic code (GPU is not required on a dataset this small)
def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available(): ## NVIDIA
        return torch.device("cuda")
    if torch.backends.mps.is_available(): ## Apple Silicon
        return torch.device("mps")
    try:
        import torch_directml ## AMD
        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")

device = get_device(force_cpu=True)

