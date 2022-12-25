import argparse
import json
import requests, zipfile, io

def parse_args():
    parser = argparse.ArgumentParser(description='Automatically export data from label-studio.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--label-studio-url', '-u', type=str, help='Sets the label studio url to use (without trailing slash).',
                              default='https://label.piustats.com')
    parser.add_argument('--label-studio-project-id', '-p', type=str, help='Sets the label studio project id.',
        default='4', dest="project_id")
    parser.add_argument('--label-studio-access-token', '-t', type=str, help='Sets the label studio API Access Token.', dest="auth_token", required=True)
    return vars(parser.parse_args())

def main(label_studio_url, project_id, auth_token):
    print("Exporting data from label-studio project @ %s\n\tUsing project id = %s", label_studio_url, project_id)
    headers = {"Authorization": "Token %s" % auth_token}
    r = requests.get("%s/api/projects/%s/export?exportType=COCO" % (label_studio_url, project_id), headers=headers, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./data/train")
    result_file_path = './data/train/result.json'
    with open(result_file_path, "r") as jsonFile:
        data = json.load(jsonFile)
    for image in data['images']:
        image['file_name'] = str.split(image['file_name'], '/')[-1]
    with open(result_file_path, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(**args)