import json
from os.path import join
import os
from typing import Dict


class dataset(object):

    def __init__(self, folder_path: str):
        json_path = join(folder_path, 'result.json')

        with open(json_path, 'r') as fp:
            self.json = json.load(fp)

        self.folder_path = folder_path
        self.class_count = len(self.json['categories'])


    def _get_annotations_from_image_id(self, id: int):
        return [d for d in self.json['annotations'] if d['image_id'] == id]

    def _create_samples(self):
        samples = []
        for img_dict in self.json['images']:
            id = img_dict['id']
            image_path = join(join(self.folder_path, 'images'), os.path.basename(img_dict['file_name']))
            annotations = self._get_annotations_from_image_id(id)

