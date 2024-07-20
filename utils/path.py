import os
import pickle
from pathlib import Path

from PIL import Image

from utils.decorators import run_once

IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')


def scan_images(path, extension=IMAGE_FORMATS, recursive=False):
    """
    Scans for image files in a given directory or checks a single file.

    This function either scans a directory for image files (with extensions
    specified in IMAGE_FORMATS) or checks if a given path is an image file.
    It can perform a non-recursive (default) or recursive scan of directories.

    Args:
        path (str): The directory path or file path to scan for image files.
        extension: (list, optional): A list of image file extensions to scan for. Defaults to IMAGE_FORMATS.
        recursive (bool, optional): Flag to enable recursive directory scanning. Defaults to False.

    Returns:
        list: A list containing the paths to the image files found.
    """
    images = []
    # check if path is a directory and scan for image files
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[-1].lower() in extension:
                    images.append(os.path.join(root, file))
            if not recursive:
                break
    # check if the path itself is an image file
    elif os.path.isfile(path) and os.path.splitext(path)[-1].lower() in extension:
        images.append(path)
    return images


def cache_images(path, transform, recursive=False, verbose=False, **kwargs):
    """
    Caches image files in a given directory or checks a single file.

    This function either caches image files (with extensions specified in
    IMAGE_FORMATS) found in a directory or checks if a given path is an image file.
    It can perform a non-recursive (default) or recursive scan of directories.

    Args:
        path (str): The directory path or file path to cache image files.
        recursive (bool, optional): Flag to enable recursive directory scanning. Defaults to False.

    Returns:
        None
    """
    images = scan_images(path, recursive)
    if verbose:
        if images:
            print(f'Found {len(images)} images.')
        else:
            print('No images found.')
            return

    for i, img in enumerate(images):
        if verbose:
            print(f'Caching image {i + 1}/{len(images)}: {img}')

        im = Image.open(img).convert('RGB')
        if transform:
            im = transform(im).unsqueeze(0)

        filename = os.path.splitext(img)[0] + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(im, f)


class SubDir:
    def __init__(self, root):
        self.root = Path(root)

    def find_id(self, prefix, find_type='max', separator='_'):
        """
        Finds the maximum or minimum numeric ID from directories with a given prefix.

        Args:
            prefix (str): The prefix to match for directory names.
            find_type (str, optional): The type of ID to find. Defaults to 'max'.
                Possible values are 'max' and 'min'.
            separator (str, optional): The separator used in directory names. Defaults to '_'.

        Returns:
            int or None: The maximum or minimum numeric ID found, or None if no matching directories are found.
        """
        ids = []
        for p in self.root.iterdir():
            if p.is_dir() and p.name.startswith(prefix):
                parts = p.name.split(separator)
                if len(parts) > 1 and parts[-1].isdigit():
                    ids.append(int(parts[-1]))
        if not ids:
            return None
        return max(ids) if find_type == 'max' else min(ids)

    def next(self, prefix, separator='_', start_suffix=1, step=1):
        """
        Generate the next available name with a numeric suffix in a sequence.

        Parameters:
            prefix (str): The prefix of the name.
            separator (str): The separator between the prefix and the numeric suffix.
            start_suffix (int): The starting suffix value.
            step (int): The increment step for the suffix.

        Returns:
            str: The next available name with a numeric suffix.

        Raises:
            ValueError: If any of the parameters are of incorrect type or value.

        Examples:
            >>> sub_dir = SubDir('data')
            >>> sub_dir.next('model')
            'data/model_1'
            >>> sub_dir.next('model', start_suffix=1)
            'data/model_2'
        """
        max_suffix = self.find_id(prefix)
        if max_suffix is None:
            suffix = start_suffix
        else:
            suffix = max_suffix + step
        padding = len(str(suffix))

        while True:
            name = f'{prefix}{separator}{suffix:0{padding}d}'
            if not (self.root / name).exists():
                return os.path.join(self.root, name)
            suffix += step
            padding = len(str(suffix))

    def current(self, prefix):
        """
        Finds the current directory with the highest suffix for a given prefix.

        This method searches for the directory with the highest numeric suffix that matches the given prefix.
        It returns the full path to the directory with the highest suffix or None if no matching directories are found.

        Args:
            prefix (str): The prefix of the directory names to search for.

        Returns:
            str: The full path to the directory with the highest suffix for the given prefix, or None if no matching directories are found.

        Examples:
            >>> sub_dir = SubDir('data')
            >>> sub_dir.current('model')
            'data/model_3'
            >>> sub_dir.current('data')
            'data/data_1'
            >>> sub_dir.current('image')
            None
        """
        max_suffix = self.find_id(prefix)
        if max_suffix is None:
            return None
        return str(self.root / f'{prefix}_{max_suffix}')


@run_once
def makedirs(root: str, dirs: set[str] = (), exist_ok=True):
    """
    Create directories within a root directory.

    This function creates directories within a specified root directory.
    It can create multiple directories at once and can skip existing directories.

    Args:
        root (str): The root directory to create the subdirectories in.
        dirs (set): A list of directory names to create.
        exist_ok (bool, optional): Flag to skip existing directories. Defaults to True.

    Returns:
        None
    """
    if not dirs:
        os.makedirs(root, exist_ok=exist_ok)
        return

    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=exist_ok)
