import os
import pickle

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