import argparse
import os

import torch
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.table import Table
from torchvision.models import MobileNetV2

table = Table(title='Detection', show_header=True)
table.add_column('images')
table.add_column('confidence', justify='center')
table.add_column('predicted_label')

console = Console()


def main(weights, images, device):
    checkpoint = torch.load(weights, map_location=device)
    classes = checkpoint['classes']
    transform = checkpoint['transform']

    model = MobileNetV2(num_classes=len(classes))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with Live(table, console=console):
        processed_images = []
        for img in images:
            if os.path.isdir(img):
                files = []
                for file in os.listdir(img):
                    if os.path.isfile(os.path.join(img, file)):
                        files.append(file)
                for file in files:
                    processed_images.append(os.path.join(img, file))
            else:
                processed_images.append(img)

        for image in processed_images:
            img = Image.open(image).convert('RGB')
            img = transform(img).unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                output = model(img)
                prob = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

                conf, pred = conf.item(), pred.item()
                table.add_row(image, f'{conf:.4f}', classes[pred])


def args():
    parser = argparse.ArgumentParser(description='CNN for Image Classification')
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--images', nargs='+', required=True, help='image path')
    parser.add_argument('--device', type=str, default='cpu', help='device to use (cpu or cuda)')
    return parser.parse_args()


if __name__ == '__main__':
    opts = args()

    if opts.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available.')
        exit(1)

    main(opts.weights, opts.images, opts.device)
