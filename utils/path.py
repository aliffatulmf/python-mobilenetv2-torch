import os
from pathlib import Path
from typing import List, Optional

IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']


def scan_objects(path: str, extension: List[str], recursive: bool = False) -> List[str]:
    """
    Scans for objects in the specified path based on the provided extensions.

    Args:
        path (str): The path where to scan for objects.
        extension (List[str]): A list of file extensions to filter the scanned objects.
        recursive (bool, optional): A boolean indicating whether to scan recursively. Defaults to False.

    Returns:
        list: A list of paths to the scanned objects.
    """
    obj_list = []
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() in extension:
                    filepath = os.path.join(dirpath, filename)
                    obj_list.append(filepath)
            if not recursive:
                break
    elif os.path.isfile(path) and os.path.splitext(path)[-1].lower() in extension:
        obj_list.append(path)
    return obj_list


def scan_images(path: str, recursive: bool = False) -> List[str]:
    """
    Scans images in the specified path based on the provided extensions.

    Args:
        path (str): The path where to scan for images.
        recursive (bool, optional): A boolean indicating whether to scan recursively. Defaults to False.

    Returns:
        list: A list of paths to the scanned images.
    """
    return scan_objects(path, IMAGE_FORMATS, recursive)


class SubDir:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def find_id(
        self,
        prefix: str,
        find_type: str = 'max',
        separator: str = '_'
    ) -> Optional[int]:
        """
        Finds the maximum or minimum numeric ID from directories with a given prefix.

        This method searches for directories with names that start with the given prefix and contain
        a numeric suffix. It returns the maximum or minimum value of those numeric suffixes.

        Args:
            prefix (str): The prefix to match for directory names.
            find_type (str, optional): The type of ID to find. Defaults to 'max'.
                Possible values are 'max' and 'min'. (str)
            separator (str, optional): The separator used in directory names. Defaults to '_'. (str)

        Returns:
            int: The maximum or minimum numeric ID found, or None if no matching directories are found.
                (Union[int, None])
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

    def next(
        self,
        prefix: str,
        separator: str = '_',
        makedir: bool = False,
        start_suffix: int = 1,
        step: int = 1
    ) -> str:
        """
        Generate the next available name with a numeric suffix in a sequence.

        Args:
            prefix (str): The prefix of the name.
            separator (str): The separator between the prefix and the numeric suffix.
            makedir (bool): Flag to create the directory if it does not exist.
            start_suffix (int): The starting suffix value.
            step (int): The increment step for the suffix.

        Returns:
            str: The next available name with a numeric suffix.

        Raises:
            ValueError: If any of the parameters are of incorrect type or value.
        """
        max_suffix = self.find_id(prefix)
        if max_suffix is None:
            suffix = start_suffix
        else:
            suffix = max_suffix + step
        padding = len(str(suffix))

        while True:
            name = f'{prefix}{separator}{suffix:0{padding}d}'
            subdir = Path(self.root / name)
            if not subdir.exists():
                if makedir:
                    subdir.mkdir(parents=True)
                return subdir.as_posix()
            suffix += step
            padding = len(str(suffix))

    def current(self, prefix: str) -> Optional[str]:
        """
        Finds the current directory with the highest suffix for a given prefix.

        This method searches for the directory with the highest numeric suffix that matches the given prefix.
        It returns the full path to the directory with the highest suffix or None if no matching directories are found.

        Args:
            prefix (str): The prefix of the directory names to search for.

        Returns:
            str: The full path to the directory with the highest suffix for the given prefix, or None if no matching directories are found.
        """
        max_suffix = self.find_id(prefix)
        if max_suffix is None:
            return None
        return str(self.root / f'{prefix}_{max_suffix}')


def makedirs(
    root: str,
    dirs: Optional[List[str]] = None,
    skip_exist: bool = True,
) -> List[str]:
    """
    Create directories within a root directory and return the list of created directories.
    This function creates directories within a specified root directory and returns the
    list of directories that were successfully created. It can create multiple directories
    at once and can skip existing directories.

    Args:
        root (str): The root directory to create the subdirectories in.
        dirs (Optional[List[str]]): A list of directory names to create. Defaults to None.
        skip_exist (bool): Flag to skip existing directories. Defaults to True.

    Returns:
        list: The list of directories that were successfully created.
    """
    created_dirs = []
    if not dirs:
        os.makedirs(root, exist_ok=skip_exist)
        created_dirs.append(root)
        return created_dirs
    for d in dirs:
        dir_path = os.path.join(root, d)
        os.makedirs(dir_path, exist_ok=skip_exist)
        created_dirs.append(dir_path)
    return created_dirs
