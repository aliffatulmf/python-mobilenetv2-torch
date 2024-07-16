# Python MobileNetV2

PyTorch MobileNetV2

## Installation
```bash
pip install -r requirements.txt
```

## Use

### Training

```bash
python train.py --dataset dataset --size 320 --device cuda --epochs 10
```

### Inference

```bash
python detect.py --weights weights/run_1/best.pt --device cpu --images images_path flower.jpg car.jpg
```