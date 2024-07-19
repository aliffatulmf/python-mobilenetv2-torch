import argparse
import os
import sys

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.models import MobileNetV2

from utils.figure import plot_confusion_matrix
from utils.path import scan_images


def main(weights, dataset, images, device):
    if os.path.isfile(weights) is False:
        print(f'Model weights not found: {weights}')
        sys.exit(1)

    device = torch.device(device)
    checkpoint = torch.load(weights, map_location=device)
    classes = checkpoint['classes']
    transform = checkpoint['transform']
    classes_list = list(classes.values())

    model = MobileNetV2(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    true_labels = []
    image_paths = []

    if dataset:
        for root, _, files in os.walk(os.path.join(dataset, 'valid')):
            # extend the true_labels list with the directory name for each file found
            true_labels.extend([os.path.basename(root)] * len(files))

            # extend the image_paths list with the full path for each file found
            image_paths.extend([os.path.join(root, file) for file in files])

            # # find total files with .pkl extension
            # total_pkl_files = len(scan_images(dataset, extension=['.pkl'], recursive=True))
            #
            # # cache images if total_pkl_files is not equal to the number of images found
            # if total_pkl_files != len(image_paths):
            #     cache_images(dataset, recursive=True)
    else:
        for path in images:
            image_paths.extend(scan_images(path))
            # dummy labels for images
            true_labels.extend([0] * len(image_paths))
            #
            # total_pkl_files = len(scan_images(path, extension=['.pkl'], recursive=True))
            # if total_pkl_files != len(image_paths):
            #     cache_images(path, recursive=True)
            #
            # image_paths = scan_images(path, extension=['.pkl'], recursive=True)
            #
            # print(f'image path: {len(image_paths)}, true labels: {len(true_labels)}')

    if len(true_labels) != len(image_paths):
        raise ValueError('Error: true_labels and image_paths are not the same length.')

    if not true_labels or not image_paths:
        raise ValueError('Error: No files found in the dataset directory.')

    scores = []
    all_pred = []
    all_label = []

    print('''Label              Score              Image\n''')
    for image_path, true_label in zip(image_paths, true_labels):
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

            scores.append((image_path, conf.item()))
            if dataset:
                all_pred.append(classes[pred.item()])
                all_label.append(true_label)

            print(f'''{classes[pred.item()]}              {conf.item():.4f}              {image_path}''')

    if dataset:
        cm = confusion_matrix(all_label, all_pred, labels=classes_list)
        plot_confusion_matrix(cm, classes_list, 'confusion_matrix.png')
        print()
        print(f'Confusion matrix saved to confusion_matrix.png')


def args():
    parser = argparse.ArgumentParser(description='CNN for Image Classification')
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--dataset', type=str, help='dataset path')
    parser.add_argument('--images', nargs='+', help='image path')
    parser.add_argument('--device', type=str, default='cpu', help='device to use (cpu or cuda)')
    return parser.parse_args()


if __name__ == '__main__':
    opts = args()

    if opts.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available.')
        exit(1)

    main(opts.weights, opts.dataset, opts.images, opts.device)
