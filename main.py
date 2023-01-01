import argparse
import json
import os
from pathlib import Path

import torch

from ai.coach import coach
from dataset.coco_detection import get_dataset
from ai.model import Model
from ocr.ocr import recognize
from visualizer import objects
from visualizer.draw_dataset import draw


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

    test_parser = mode.add_parser('predict', help='Predict given a pretrained model',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    test_parser.add_argument('model', type=str, help='Path to pretrained model')
    test_parser.add_argument('-i', '--image', type=str, help='Path to image (for one prediction)', default=None)
    test_parser.add_argument('-f', '--image-folder', type=str, help='Path to folder with images', default=None)
    test_parser.add_argument('certainty', type=float, help='Certainty', default=None)
    test_parser.add_argument('--output-image', '-o', type=str,
                             help='Select path to save image (if -i) or folder (if -f)', default=None)
    test_parser.add_argument('--device', '-d', type=str, help='Choose compute device', default='cpu')
    test_parser.add_argument('prediction_type', type=str, choices=['ocr', 'objects'],
                             help='Choose to predict ocr or objects')

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


def predict(prediction_type, **kwargs):
    if prediction_type == 'objects':
        predict_objects(**kwargs)
    elif prediction_type == 'ocr':
        predict_ocr(**kwargs)


def predict_ocr(image, device, certainty, image_folder, output_image, model, **kwargs):
    model = torch.load(model)
    m, c = model['model'], model['cats']
    m.to(device)
    m.eval()
    recognize(image, certainty, m, c, device)


def predict_objects(image, device, image_folder, output_image, model, **kwargs):
    model = torch.load(model)
    m, c = model['model'], model['cats']
    m.to(device)
    m.eval()
    if image != None:
        predict_image(model=m, cats=c, image=image, output_image=output_image, device=device, **kwargs)
    elif image_folder != None:
        filelist = os.listdir(image_folder)
        for i, file in enumerate(filelist):  # filelist[:] makes a copy of filelist.
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                print(f'predicting "{file}"')
                predict_image(model=m, cats=c, image=os.path.join(image_folder, file),
                              output_image=os.path.join(output_image, f"{i}.png"), device=device, **kwargs)


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
