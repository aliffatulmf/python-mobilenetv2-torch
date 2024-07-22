import os
from datetime import datetime
from typing import Any, Dict

import torch
import torchvision


def checkpoint(
    root: str,
    name: str,
    model: torch.nn.Module,
    classes: Dict[int, str],
    transform: torchvision.transforms.Compose,
    **kwargs: Dict[str, Any]
) -> None:
    """
    Saves a model checkpoint including metadata, model state, and transformations.

    Args:
        root (str): Directory to save the model to.
        name (str): Filename for the saved model.
        model (torch.nn.Module): Model to save.
        classes (dict): Class index-to-label mapping.
        transform (torchvision.transforms.Compose): Transformations applied to inputs.
        kwargs: Additional metadata for the checkpoint.

    Returns:
        str: Path to the saved model.

    Example:
        >>> checkpoint('runs/train/weight', 'best', model, classes, transform)
        'runs/train/weight/best.pt'
    """
    metadata = {
        'metadata': {
            'datetime': datetime.now().isoformat(),
            'model': model.__class__.__name__,
            'format': '.pt',
        },
        'model_info': {
            'classes': classes,
            'transform': transform,
        },
        **kwargs,
    }

    filename = f'{name}.pt'
    filepath = os.path.join(root, filename)
    torch.save(metadata, filepath)
