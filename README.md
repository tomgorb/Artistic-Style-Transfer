Artistic Style Transfer
---

- [COCO dataset](https://cocodataset.org/#download)

- [TensorFlow 1.x models](https://drive.google.com/file/d/1G-J0KlSp9Z4QJR98T_aAbSruW5uHG42z/view?usp=sharing).

- convert content images to *jpg* before doing stylization (*png* have **4** channels).

- pyTorch
  - requirements: torch==1.3.1+cpu
  - usage: ```python neural_style.py eval --content-image 01.jpg --model legion.pth --output-image 01-legion.png --cuda 0```

- TensorFlow
  - requirements: tensorflow==1.15.0, imageio==2.9.0
  - usage: ```python stylize_image.py --content results/01.jpg --network-path models/dora-marr-network --output-path results/01-dora-marr.jpg```
