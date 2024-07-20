import os
import pickle
from pathlib import Path

from PIL import Image

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


def cache_images(path, recursive=False, verbose=False, **kwargs):
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

        transform = kwargs.get('transform', None)

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
        Generates the next suffix for a given prefix, using the maximum existing suffix as a starting point.

        This method validates the input parameters and then calculates the next available suffix for the given prefix.
        It ensures that the suffix is numeric and increments it based on the provided step value. If no existing suffixes
        are found that match the prefix, it starts with the provided `start_suffix`. The method continues to increment
        the suffix until an available name is found that does not exist in the directory.

        Args:
            prefix (str): The prefix string.
            separator (str, optional): The separator between the prefix and the suffix. Defaults to '_'.
            start_suffix (int, optional): The starting suffix if no existing suffixes are found. Defaults to 1.
            step (int, optional): The step value for the suffix increment. Defaults to 1.

        Returns:
            str: The next available name with the given prefix and an incremented suffix.

        Raises:
            ValueError: If `prefix` is not a non-empty string, `separator` is not a string, `start_suffix` is not an integer
                        greater than or equal to 1, or `step` is not an integer greater than 0.

        Examples:
            >>> sub_dir = SubDir('data')
            >>> sub_dir.next('model', start_suffix=1)
            'data/model_1'
            >>> sub_dir.next('model', start_suffix=1)
            'data/model_2'
            >>> sub_dir.next('model', start_suffix=1)
            'data/model_3'
        """
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if not isinstance(separator, str):
            raise ValueError("separator must be a string")
        if not isinstance(start_suffix, int) or start_suffix < 1:
            raise ValueError("start_suffix must be an integer greater than or equal to 1")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("step must be an integer greater than 0")

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
