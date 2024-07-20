import os
from datetime import datetime

from torch import save as torch_save


def _generate_checkpoint(model, classes, transform, **kwargs):
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
        **kwargs
    }


def checkpoint(root, name, model, classes, transform, **kwargs):
    """
    Saves a model checkpoint including metadata, model state, and transformations.

    Args:
        root (str): Directory for saving the model.
        name (str): Filename for the saved model.
        model (torch.nn.Module): Model to save.
        classes (dict): Class index-to-label mapping.
        transform (torchvision.transforms.Compose): Transformations applied to inputs.
        **kwargs: Additional metadata for the checkpoint.

    Returns:
        str: Path to the saved model.
    """
    filename = f'{name}.pt'
    filepath = os.path.join(root, filename)
    torch_save(_generate_checkpoint(model, classes, transform, **kwargs), filepath)
