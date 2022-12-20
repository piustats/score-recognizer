import argparse
import json
from pathlib import Path

import torch

from ai.coach import coach
from dataset.coco_detection import get_dataset
from ai.model import Model
from visualizer import objects


def parse_args():
    parser = argparse.ArgumentParser(description='A Faster RCNN for PIU scores.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_subparsers(title='mode', help='Select the mode', dest='mode', required=True)

    train_parser = mode.add_parser('train', help='Learn the parameters',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--batch-size', '-b', type=int, help='Sets the batch size for mini-batching training.',
                              default=4)
    train_parser.add_argument('--epochs', '-e', type=int, help='Number of epochs for training',
                              default=10)
    train_parser.add_argument('--device', '-d', type=str, help='Choose compute device', default='cpu')
    train_parser.add_argument('--output-model', '-o', type=str, help='Output model', default=None)
    train_parser.add_argument('train_json', type=str, help='JSON of training dataset')
    train_parser.add_argument('train_images', type=str, help='Folder of training images')

    train_parser.add_argument('test_json', type=str, help='JSON of test dataset')
    train_parser.add_argument('test_images', type=str, help='Folder of test images')

    test_parser = mode.add_parser('test', help='Predict given a pretrained model',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    test_parser.add_argument('model', type=str, help='Path to pretrained model')
    test_parser.add_argument('image', type=str, help='Path to image')
    test_parser.add_argument('certainty', type=float, help='Certainty', default=None)
    test_parser.add_argument('--output-image', '-o', type=str, help='Select path to save image', default=None)

    return vars(parser.parse_args())


def train(train_json, train_images, test_json, test_images, output_model, **kwargs):
    train_dataset = get_dataset(train_images, train_json)
    model = Model(len(train_dataset.coco.cats))

    test_dataset = get_dataset(test_images, test_json, training_transforms=False)
    c = coach(model)
    c.fit(train_dataset=train_dataset, test_dataset=test_dataset, **kwargs)
    if output_model != None:
        o = {
            'model': model,
            'cats': train_dataset.coco.cats
        }
        torch.save(o, output_model)


def test(model, image,  certainty, output_image):
    lp = torch.load(model)
    m, c = lp['model'], lp['cats']

    m.eval()
    m.to('cpu')
    objects.object_detection_api(image, m, c, certainty, output_image=output_image)


def main(mode, **kwargs):
    if mode == "train":
        train(**kwargs)
    elif mode == "test":
        test(**kwargs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    main(**args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
