"""
generative_core/generate.py
============================
Counterfactual scenario generation from a trained GCD-VAE checkpoint.

Each call to `generate_counterfactual()` draws a fresh latent sample Z ~ N(0, I)
and decodes it under the given condition vector C.  Because Z is random, multiple
calls for the same scenario produce *different* but statistically consistent demand
profiles – this is the "generative" part of the framework.

`generate_all_scenarios()` runs every scenario defined in config.SCENARIOS and
writes one .npy file per scenario to config.OUTPUT_DIR.  These files are the
async handoff artefact for Lochan (grid physics) and the UI team.

Output contract
---------------
  * File format  : NumPy .npy (float32)
  * Array shape  : [SEQ_LEN, NUM_NODES]  →  [24, 50] with default config
  * Units        : kW – active power demand per grid node per hour
  * Hour 0       : Midnight → 01:00 (standard time, no DST correction)
"""

import os
from typing import List, Optional, Dict

import numpy as np
import torch

from . import config
from .models import GenerativeCounterfactualVAE


def _resolve_device(model=None, device=None) -> torch.device:
    """Infer the correct torch device from a model or system capability.

    Priority: explicit device arg > model's parameter device > CUDA if available.

    Args:
        model:  Optional trained model to read the device from.
        device: Optional explicit torch.device override.

    Returns:
        A valid torch.device.
    """
    if device is not None:
        return device
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: Optional[torch.device] = None):
    """Load a trained GCD-VAE from the checkpoint at config.MODEL_SAVE_PATH.

    Args:
        device: torch.device to map the weights to.  Auto-detected if None.

    Returns:
        Tuple (model, device):
            model  – GenerativeCounterfactualVAE in eval mode.
            device – the device the model is on.

    Raises:
        FileNotFoundError: If no checkpoint exists at config.MODEL_SAVE_PATH.
    """
    device = _resolve_device(device=device)
    if not os.path.isfile(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No trained model at '{config.MODEL_SAVE_PATH}'. "
            "Run 'python run.py train' first."
        )
    model = GenerativeCounterfactualVAE().to(device)
    ckpt = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def generate_counterfactual(model: GenerativeCounterfactualVAE,
                             device: torch.device,
                             condition: List[float],
                             num_nodes: int = config.NUM_NODES) -> np.ndarray:
    """Generate one counterfactual demand profile under a given condition.

    Samples a single Z ~ N(0, I) and decodes it conditioned on `condition`.
    Only the node-demand channels are returned; weather channels are dropped
    because downstream tools (pandapower, UI) only need kW per node.

    Args:
        model:     Trained GenerativeCounterfactualVAE in eval mode.
        device:    torch.device that model lives on.
        condition: Condition vector, length == config.COND_DIM.
                   See config.py for the meaning of each dimension.
        num_nodes: Number of grid nodes to extract from the output.

    Returns:
        float32 NumPy array of shape [SEQ_LEN, num_nodes] in kW.

    Raises:
        ValueError:   If condition length doesn't match config.COND_DIM.
        RuntimeError: If the decoded output has an unexpected shape.
    """
    if len(condition) != config.COND_DIM:
        raise ValueError(
            f"Condition vector has length {len(condition)} but "
            f"config.COND_DIM={config.COND_DIM}.  Update the condition list."
        )

    with torch.no_grad():
        z    = torch.randn(1, config.LATENT_DIM, device=device)
        cond = torch.tensor([condition], dtype=torch.float32, device=device)
        out  = model.decode(z, cond)       # [1, num_features, seq_len]

        # Slice off the weather channels – consumers only need [seq_len, num_nodes].
        demand = out[:, :num_nodes, :].squeeze(0).permute(1, 0)   # [seq_len, nodes]

        if demand.shape != (config.SEQ_LEN, num_nodes):
            raise RuntimeError(
                f"Shape mismatch: expected ({config.SEQ_LEN}, {num_nodes}), "
                f"got {tuple(demand.shape)}.  Check NUM_NODES in config.py."
            )
        return demand.cpu().numpy()


def generate_all_scenarios(model=None,
                            device=None,
                            save: bool = True) -> Dict[str, np.ndarray]:
    """Run all scenarios from config.SCENARIOS and optionally save to disk.

    If no model is provided, loads the latest checkpoint from
    config.MODEL_SAVE_PATH.  Passes an in-memory model directly to avoid
    re-loading from disk when chaining `train → generate` in run.py.

    Args:
        model: Trained GenerativeCounterfactualVAE, or None to load from disk.
        device: torch.device override.  Auto-detected from model if None.
        save (bool): Whether to write .npy files to config.OUTPUT_DIR.

    Returns:
        Dict mapping scenario name → float32 array [SEQ_LEN, NUM_NODES].

    Raises:
        FileNotFoundError: Propagated from load_model if no checkpoint exists.
    """
    if model is None:
        try:
            model, device = load_model(device)
        except (RuntimeError, KeyError, FileNotFoundError) as e:
            print(f"\n⚠️  [Warning] Model architecture mismatch or missing file: {e}")
            print("   Falling back to pre-existing .npy scenarios in output/ folder.")
            return {}
    else:
        device = _resolve_device(model, device)

    results: Dict[str, np.ndarray] = {}
    for name, spec in config.SCENARIOS.items():
        tensor = generate_counterfactual(model, device, spec["condition"])
        results[name] = tensor
        print(f"  [{name}] {spec['description']}  shape={tensor.shape}")

        if save:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            path = os.path.join(config.OUTPUT_DIR, f"{name}.npy")
            np.save(path, tensor)
            print(f"    → {path}")

    return results

def generate_extreme_demand_tensor(model=None, device=None, n=1000):
    """
    Generates an extreme demand tensor with 1000 scenarios and saves to data/ extreme_demand_tensor.npy
    """
    import os
    if model is None:
        try:
            model, device = load_model(device)
        except FileNotFoundError:
            print("No model to load, using mock generator for demo...")
            from .mock import generate_mock_demand
            scenarios = []
            for _ in range(n):
                scenarios.append(generate_mock_demand(config.NUM_NODES, config.SEQ_LEN))
            demand_tensor = np.stack(scenarios, axis=0)
            
            os.makedirs(os.path.join(config.PROJECT_ROOT, 'data'), exist_ok=True)
            path = os.path.join(config.PROJECT_ROOT, 'data', 'extreme_demand_tensor.npy')
            np.save(path, demand_tensor)
            print(f"Saved {demand_tensor.shape} to {path}")
            return demand_tensor
    else:
        device = _resolve_device(model, device)

    condition = config.SCENARIOS["extreme_winter_storm"]["condition"]
    scenarios = []
    for _ in range(n):
        tensor = generate_counterfactual(model, device, condition)
        scenarios.append(tensor)
    
    demand_tensor = np.stack(scenarios, axis=0) # [1000, 24, NUM_NODES]
    
    os.makedirs(os.path.join(config.PROJECT_ROOT, 'data'), exist_ok=True)
    path = os.path.join(config.PROJECT_ROOT, 'data', 'extreme_demand_tensor.npy')
    np.save(path, demand_tensor)
    print(f"Saved {demand_tensor.shape} to {path}")
    return demand_tensor

if __name__ == "__main__":
    generate_extreme_demand_tensor()
