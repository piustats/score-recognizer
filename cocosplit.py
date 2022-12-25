# Code kindly provided by:
# https://github.com/akarazniewicz/cocosplit

import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def create_splits(annotations, having_annotations, multi_class, split, train, test):
    with open(annotations, 'rt', encoding='UTF-8') as ann:
        coco = json.load(ann)
        info = coco['info']
        licenses = ''
        images = coco['images']
        ann = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), ann)

        if having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        if multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), ann)

            # bottle neck 1
            # remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories = funcy.lremove(lambda i: annotation_categories.count(i) <= 1, annotation_categories)

            ann = funcy.lremove(lambda i: i['category_id'] not in annotation_categories, ann)

            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([ann]).T,
                                                                          np.array([annotation_categories]).T,
                                                                          test_size=1 - split)

            save_coco(train, info, licenses, filter_images(images, X_train.reshape(-1)),
                      X_train.reshape(-1).tolist(), categories)
            save_coco(test, info, licenses, filter_images(images, X_test.reshape(-1)), X_test.reshape(-1).tolist(),
                      categories)

            print("Saved {} entries in {} and {} in {}".format(len(X_train), train, len(X_test), test))

        else:

            X_train, X_test = train_test_split(images, train_size=split)

            anns_train = filter_annotations(ann, X_train)
            anns_test = filter_annotations(ann, X_test)

            save_coco(train, info, licenses, X_train, anns_train, categories)
            save_coco(test, info, licenses, X_test, anns_test, categories)

            print("Saved {} entries in {} and {} in {}".format(len(anns_train), train, len(anns_test), test))
