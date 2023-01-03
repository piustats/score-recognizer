## A machine learning-based approach to extracting PIU scores from images

![schema](https://github.com/piustats/score-recognizer/blob/main/imgs/schema.jpg?raw=true)

#### Installation
1. Clone this repository,
2. Install all necessary packages from `requirements.txt`.
3. Install [tesseract](https://github.com/tesseract-ocr/tesseract) through your package manager. We used `tesseract 5.2.0` for development and deploy.

#### Pretrained models
This code uses two models to transcribe scores from images:

1. A Faster R-CNN pytorch model for object detection ([download pretained model](https://drive.google.com/file/d/1t7euKqEFbaEb8DzXt1ZcQk0nxwDMRS5C/view?usp=sharing))
2. A tesseract model for OCR ([download pretrained model](https://drive.google.com/file/d/1r1yUIHYWUoi0pxdm1HG5KDzmkj0qhJm0/view?usp=sharing))

#### Usage
##### Comand line syntax
```
usage: main.py predict [-h] [--disable-logs] [--enable-debug-images] [--device DEVICE] cnn_model tesseract_model image certainty

positional arguments:
  cnn_model             Path to trained Faster RCNN model
  tesseract_model       Path to tesseract model
  image                 Path to image (for one prediction)
  certainty             Certainty

options:
  -h, --help            show this help message and exit
  --disable-logs        Disable logs. Only prediction is shown (default: False)
  --enable-debug-images
                        Show images of object detection and OCR (default: False)
  --device DEVICE, -d DEVICE
                        Choose computing device (default: cpu)

```
##### Example

For prediction, just typen on your shell:
```sh
$ python3 main.py predict <pytorch_model_path> <tesseract_model_path> <image_path>\
 0.7 -d 'cpu' --disable-logs
```
where `<pytorch_model_path>` is the path to the pretrained Faster R-CNN model, `<tesseract_model_path>` is the folder path to the tesseract model (after extraction), and `<image_path>` is the path to the image.
