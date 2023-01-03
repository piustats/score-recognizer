import argparse
import json
import os
import sys

import torch
from ai.coach import coach
from dataset.coco_detection import get_dataset
from ai.model import Model
from ocr.ocr import recognize
from visualizer import objects
import visualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='A Faster RCNN and tesseract-based utility to transcribe PIU scores from images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_subparsers(title='mode', help='Select the mode', dest='mode', required=True)

    train_parser = mode.add_parser('train', help='Learn the parameters of a the Object Detector',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--batch-size', '-b', type=int, help='Sets the batch size for mini-batching training.',
                              default=4)
    train_parser.add_argument('--epochs', '-e', type=int, help='Number of epochs for training',
                              default=10)
    train_parser.add_argument('--device', '-d', type=str, help='Choose computing device', default='cpu')
    train_parser.add_argument('--output-model', '-o', type=str, help='Output model', default=None)
    train_parser.add_argument('train_json', type=str, help='JSON of training dataset')
    train_parser.add_argument('train_images', type=str, help='Folder of training images')

    train_parser.add_argument('test_json', type=str, help='JSON of test dataset')
    train_parser.add_argument('test_images', type=str, help='Folder of test images')

    test_parser = mode.add_parser('predict', help='Predict given a pretrained model',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    test_parser.add_argument('cnn_model', type=str, help='Path to trained Faster RCNN model')
    test_parser.add_argument('tesseract_model', type=str, help='Path to tesseract model')
    test_parser.add_argument('image', type=str, help='Path to image (for one prediction)')
    test_parser.add_argument('certainty', type=float, help='Certainty', default=None)
    # test_parser.add_argument('--output-json', '-o', type=str,
    #                          help='Select path to save json with predicted scores', default=None)

    test_parser.add_argument('--disable-logs', action='store_true',
                             help='Disable logs. Only prediction is shown', default=False)
    test_parser.add_argument('--enable-debug-images', action='store_true',
                             help='Show images of object detection and OCR', default=False)
    test_parser.add_argument('--device', '-d', type=str, help='Choose computing device', default='cpu')
    return vars(parser.parse_args())


def train(train_json, train_images, test_json, test_images, output_model, **kwargs):
    train_dataset = get_dataset(train_images, train_json)
    # draw(train_dataset, train_dataset.coco.anns)
    model = Model(len(train_dataset.coco.cats))

    test_dataset = get_dataset(test_images, test_json, training_transforms=False)
    c = coach(model)
    c.fit(train_dataset=train_dataset, test_dataset=test_dataset, **kwargs)
    if output_model != None:
        o = {
            'model': model.to('cpu'),
            'cats': train_dataset.coco.cats
        }
        torch.save(o, output_model)


def predict(image, device, certainty, cnn_model, tesseract_model, disable_logs, enable_debug_images, **kwargs):
    if disable_logs:
        sys.stdout = open(os.devnull, 'w')

    visualizer.objects.__DEBUG__ = enable_debug_images

    if device == 'cpu':
        model = torch.load(cnn_model, map_location=torch.device('cpu'))
    else:
        model = torch.load(cnn_model)
    m, c = model['model'], model['cats']
    m.to(device)
    m.eval()
    scores = recognize(image, certainty, m, tesseract_model, c, device)
    sys.stdout = sys.__stdout__
    print(json.dumps(scores, indent=4))

def predict_image(model, cats, image, certainty, output_image, device):
    objects.object_detection_api(image, model, cats, certainty, device=device, output_image=output_image)


def main(mode, **kwargs):
    if mode == "train":
        train(**kwargs)
    elif mode == "predict":
        predict(**kwargs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    main(**args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
