import argparse
import os
import pathlib
import numpy as np
import sklearn.metrics as metrics
import torch

from datetime import datetime
from textwrap import dedent
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNetV2
from torchvision.transforms import v2


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, transform, classes, device='cpu'):
        """
        Represents a Trainer object that is responsible for training a neural network model.

        Args:
            model (torch.nn.Module): the neural network model
            criterion (torch.nn.Module): the loss function
            optimizer (torch.optim.Optimizer): the optimization algorithm
            scheduler (torch.optim.lr_scheduler._LRScheduler): the learning rate scheduler
            transform (Callable[[Any], Any]): the data transformation
            classes (List[str]): list of all classes
            device (str, optional): the device to run the model on (default is 'cpu')

        Returns:
            None
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.transform = transform
        self.classes = classes
        self.device = device

        self.model_writer = ModelWriter()

    def _train_epoch(self, tensors):
        """
        Trains the model for one epoch using the given tensors.

        Args:
            tensors (Iterable): An iterable of tuples containing images and labels.

        Returns:
            None
        """
        self.model.train()
        for images, labels in tensors:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(images)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

    def evaluate(self, data):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for images, labels in data:
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)
                loss = self.criterion(pred, labels)
                total_loss += loss.item()

                _, pred_labels = torch.max(pred, 1)
                total_correct += (pred_labels == labels).float().sum().item()
                total_samples += labels.size(0)

        loss_avg = total_loss / total_samples
        acc = metrics.accuracy_score(labels.cpu(), pred_labels.cpu())
        prec = metrics.precision_score(labels.cpu(), pred_labels.cpu(), average='macro', zero_division=0)
        recall = metrics.recall_score(labels.cpu(), pred_labels.cpu(), average='macro', zero_division=0)
        return [loss_avg, acc, prec, recall]

    def train(self, data, epochs=10):
        top_score = 0.0

        if epochs < 1:
            raise ValueError('Epochs must be greater than 0')

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            self._train_epoch(data['train'])
            eval_res = self.evaluate(data['valid'])

            print_epoch(*eval_res)

            last_model = self.model_writer.save('last', self.model,
                                                transform=self.transform,
                                                classes=self.classes)
            if epoch == epochs - 1:
                print(f'Saved model to {last_model}')

            score = metrics_score(eval_res)
            if top_score < score:
                top_score = score
                best_model = self.model_writer.save('best', self.model,
                                                    transform=self.transform,
                                                    classes=self.classes)

                if epoch == epochs - 1:
                    print(f'Saved model to {best_model}')

    def _device(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)
        else:
            raise ValueError('Data must be a tensor')


class ModelWriter:
    def __init__(self, root='weights'):
        """
        Initializes the ModelWriter object.

        Args:
            root (str): The root directory path. Defaults to `weights`.

        Raises:
            ValueError: If the root path is not a directory.
        """
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.root.is_dir():
            raise ValueError('Root path must be a directory')

        self.filetype = '.pt'
        self.latest_id = self._latest_runs(add=1)

    def _latest_runs(self, start=0, add=1):
        max_id = start + add
        with os.scandir(self.root.absolute()) as it:
            for run in it:
                if run.is_dir() and run.name.startswith('runs_'):
                    try:
                        num = int(run.name.split('_')[1])
                        max_id = max(max_id, num)
                    except ValueError:
                        continue
        return max_id

    def _create_run(self, run_id):
        run_path = self.root / f'runs_{run_id}'
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def exists(self, run_id):
        run_path = self.root / f'runs_{run_id}'
        return run_path.is_dir()

    def save(self, label, model, **kwargs):
        run_path = self._create_run(self.latest_id)
        filepath = str(run_path / f'{label}{self.filetype}')

        try:
            state_dict = model.state_dict()
            metadata = {
                'training_date': datetime.now().isoformat(),
                'label': label,
            }

            torch.save({**metadata, **kwargs, 'model_state': state_dict}, filepath)
        except Exception:
            raise ValueError('Error saving model')
        return filepath


def metrics_score(metric, weights=None):
    """
    Calculates the score based on the given metric and weights.

    Args:
        metric (list): A list of values representing the metric.
        weights (list, optional): A list of weights corresponding to the metric values. Defaults to None.

    Returns:
        float: The calculated score.

    Raises:
        ValueError: If the lengths of metric and weights are not the same.
    """
    if weights is not None and len(metric) != len(weights):
        raise ValueError('Lengths of metric and weights must be the same')

    if weights is None:
        weights = [-1.0, 1.0, 1.0, 1.0]

    metric_np = np.array(metric)
    weights = np.array(weights)

    score = np.dot(metric_np, weights)
    score = (score + np.dot(metric_np, weights)) / 2

    return score


def print_epoch(loss, acc, prec, recall):
    """
    Print the loss, accuracy, precision, and recall for an epoch.

    Args:
        loss (float): The loss value for the epoch.
        acc (float): The accuracy value for the epoch.
        prec (float): The precision value for the epoch.
        recall (float): The recall value for the epoch.

    Returns:
        None

    This function prints the loss, accuracy, precision, and recall for an epoch
    in a formatted table. The values are formatted to four decimal places.
    """
    log_fmt = dedent('''
    \t     Loss           Accuracy           Precision           Recall
    \t   {:.4f}             {:.4f}              {:.4f}           {:.4f}
    '''.format(loss, acc, prec, recall))

    print(log_fmt)


def get_paths(root):
    required_dirs = ['train', 'test', 'valid']
    paths = {r: os.path.join(root, r) for r in required_dirs}

    missing_dirs = [r for r in required_dirs if not os.path.exists(paths[r])]
    if missing_dirs:
        raise ValueError(f'Missing required subdirectories: {", ".join(missing_dirs)}')

    return paths


def size_transform(size):
    if isinstance(size, list) and len(size) < 2:
        return size[0], size[0]
    else:
        return size[:2]


def args():
    parser = argparse.ArgumentParser(description='CNN for Image Classification')

    # dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to the dataset directory. should contain subdirectories for each class'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs='+',
        default=(256, 256),
        help='image size to resize to, specified as either a single integer or two integers (height, width)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size for training'
    )

    # training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='number of epochs to train the model'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=('cuda', 'cpu'),
        default='cpu',
        help='device to use for training and evaluation. choices are "cuda" for GPU or "cpu"'
    )

    # optimizer and scheduler parameters
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='momentum factor for the optimizer'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='weight decay (L2 penalty) for the optimizer'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=('min', 'max'),
        default='min',
        help='in min mode, lr will be reduced when the quantity monitored has stopped decreasing. in max mode, lr will be reduced when the quantity monitored has stopped increasing'
    )
    parser.add_argument(
        '--factor',
        type=float,
        default=0.1,
        help='factor by which the learning rate will be reduced. new_lr = lr * factor'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='number of epochs with no improvement after which learning rate will be reduced'
    )

    return parser


def main():
    opts = args().parse_args()
    paths = get_paths(opts.dataset)

    transform = v2.Compose([
        v2.Resize(size_transform(opts.size)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
    ])

    train_ds = ImageFolder(paths['train'], transform)
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True)

    valid_ds = ImageFolder(paths['valid'], transform)
    valid_dl = DataLoader(valid_ds, batch_size=opts.batch_size, shuffle=False)

    classes = {v: k for k, v in train_ds.class_to_idx.items()}

    model = MobileNetV2(train_ds.classes.__len__())
    criterion = CrossEntropyLoss()
    optimizer = RMSprop(
        model.parameters(),
        lr=opts.lr,
        weight_decay=opts.weight_decay,
        momentum=opts.momentum
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=opts.mode,
        factor=opts.factor,
        patience=opts.patience,
    )

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        transform,
        classes,
        device=opts.device
    )
    trainer.train({'train': train_dl, 'valid': valid_dl}, opts.epochs)


if __name__ == '__main__':
    main()
