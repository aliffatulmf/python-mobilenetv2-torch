import os
from datetime import datetime

import torch
from torch.nn.modules import Module
from torchvision.transforms import v2

from utils.path import SubDir


def _generate_checkpoint(model, classes, transform):
    return {
        'metadata': {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model.__class__.__name__,
            'format': '.pt',
        },
        'model_info': {
            'classes': classes,
            'transform': transform,

        },
        'model_state': model.state_dict(),
    }


def auto_save_model(root: str, name: str, prefix: str, model: Module, classes: dict, transform: v2.Compose):
    """
    Automatically saves the model to a file.

    Args:
        root (str): The root directory where the model will be saved.
        name (str): The name of the model.
        prefix (str): The prefix to be added to the saved file.
        model (Module): The model to be saved.
        num_classes (int): The number of classes in the model.
        transform (v2.Compose): The transformation applied to the model.

    Returns:
        str: The directory where the model is saved.

    Raises:
        FileNotFoundError: If the root directory does not exist.

    Example:
        >>> root = '/path/to/save/directory'
        >>> name = 'my_model'
        >>> prefix = 'v1'
        >>> model = MyModel()
        >>> num_classes = 10
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> auto_save_model(root, name, prefix, model, num_classes, transform)
        '/path/to/save/directory/v1'

    This function saves the model to a file in the specified root directory. The saved file will have the name
    '{name}.pt' and will be stored in a subdirectory of the root directory. The subdirectory name is generated
    using the provided prefix and a unique identifier.

    The model is saved using the PyTorch `torch.save` function. Before saving, a checkpoint is generated using
    the `_generate_checkpoint` function, which includes the model, the number of classes, and the transformation
    applied to the model.

    If the root directory does not exist, a `FileNotFoundError` is raised.

    """
    filename = f'{name}.pt'
    basedir = SubDir(root).next(prefix)
    filepath = os.path.join(basedir, filename)

    torch.save(_generate_checkpoint(model, classes, transform), filepath)
    return basedir
