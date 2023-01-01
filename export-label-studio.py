import argparse
import json
import requests, zipfile, io
from os import path
from cocosplit import create_splits


def parse_args():
    parser = argparse.ArgumentParser(description='Automatically export data from label-studio.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--label-studio-url', '-u', type=str,
                        help='Sets the label studio url to use (without trailing slash).',
                        default='https://label.piustats.com')
    parser.add_argument('--label-studio-project-id', '-p', type=str, help='Sets the label studio project id.',
                        default='4', dest="project_id")
    parser.add_argument('--label-studio-access-token', '-t', type=str, help='Sets the label studio API Access Token.',
                        dest="auth_token", required=True)
    parser.add_argument('--train-test-split', '-s', type=float, help='Sets the train/test split. Choose a float between [0, 1]',
                        dest="split", required=True, default=0.8)
    return vars(parser.parse_args())


def main(label_studio_url, project_id, auth_token, split):
    print(f"Exporting data from label-studio project @ {label_studio_url}\n\tUsing project id = {project_id}")
    headers = {"Authorization": "Token %s" % auth_token}
    r = requests.get("%s/api/projects/%s/export?exportType=COCO" % (label_studio_url, project_id), headers=headers,
                     stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./data")
    result_file_path = './data/result.json'
    with open(result_file_path, "r") as jsonFile:
        data = json.load(jsonFile)
    for image in data['images']:
        image['file_name'] = str.split(image['file_name'], '/')[-1]
        # convert to png

    with open(result_file_path, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

    # create splits
    create_splits(annotations='./data/result.json', having_annotations=False, multi_class=True, split=split,
                  train='./data/train.json', test='./data/test.json')


if __name__ == '__main__':
    args = parse_args()
    main(**args)
