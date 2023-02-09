import argparse
import io
import json
import os
import sys
import zipfile

import requests
from check_dataset import check_dataset

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(project_path)


def get_image_data(image_ids, url, mode):
    url = url + "/request?collection=" + mode
    payload = json.dumps({"id": image_ids})
    headers = {}
    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.raise_for_status():
            raise response.reason
        return response.content
    except requests.exceptions.RequestException as err:
        raise Exception(err)


def get_dataset(image_ids, dataset_path, url, mode):

    if mode not in ["train", "test"]:
        raise Exception("mode must be in train or test")

    tmp_file_name = "tmp.zip"
    working_file_path = os.path.join(project_path + dataset_path, tmp_file_name)
    res = get_image_data(image_ids, url, mode)
    with open(working_file_path, "wb") as f:
        f.write(io.BytesIO(res).getbuffer())

    with zipfile.ZipFile(working_file_path) as f:
        f.extractall(os.path.join(project_path + dataset_path))

    os.remove(working_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/test")
    parser.add_argument("--url", type=str, default="https://kyc-system.mynetgear.com")
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()
    request_file_ids = check_dataset(dataset_path=args.dataset_path, url=args.url, mode=args.mode)
    if request_file_ids:
        get_dataset(request_file_ids, dataset_path=args.dataset_path, url=args.url, mode=args.mode)
