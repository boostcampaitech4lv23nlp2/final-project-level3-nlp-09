import argparse
import base64
import hashlib
import json
import os
import sys

import requests
from tqdm import tqdm

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(project_path)


def print_check_results(lack_files, over_files, crash_files, print_num=10):
    print(">> list for lack files in local")
    print(f"total num: {len(lack_files)}")
    print(lack_files[:print_num], end="\n\n")
    print(">> list for over files in local")
    print(f"total num: {len(over_files)}")
    print(over_files[:print_num], end="\n\n")
    print(">> list for crash files in local")
    print(f"total num: {len(crash_files)}")
    print(crash_files[:print_num], end="\n\n")


def get_image_data(url, mode):
    url = url + "/item?collection=" + mode
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    res_items = json.loads(response.json()["item"]["body"])
    return res_items


def check_dataset(dataset_path="/data/train", url="http://kyc-system.mynetgear.com", mode="train"):

    if mode not in ["train", "test"]:
        raise Exception("mode must be in train or test")

    file_list = os.listdir(project_path + dataset_path)

    crash_files = []

    remote_digest_to_filename_dict = {}
    remote_filename_to_digest_dict = {}
    remote_filename_to_id_dict = {}

    local_digest_to_filename_dict = {}
    local_filename_to_digest_dict = {}

    print(">>> get image data to food api...")

    res_items = get_image_data(url, mode)

    for item in res_items:
        remote_digest_to_filename_dict[item["SHA256"]] = item["file_name"]
        remote_filename_to_digest_dict[item["file_name"]] = item["SHA256"]
        remote_filename_to_id_dict[item["file_name"]] = item["id"]

    print(">>> check digest for food image...")

    for file_name in tqdm(file_list):
        image_path = os.path.join(project_path + dataset_path, file_name)
        with open(image_path, "rb") as f:
            image_hash_digest = hashlib.sha256(base64.b64encode(f.read())).hexdigest()
            local_digest_to_filename_dict[image_hash_digest] = image_path
            local_filename_to_digest_dict[image_path] = image_hash_digest
            if (
                image_path in remote_filename_to_digest_dict
                and remote_filename_to_digest_dict[image_path] != image_hash_digest
            ):
                crash_files.append(image_path)

    lack_digests = list(remote_digest_to_filename_dict.keys() - local_digest_to_filename_dict.keys())
    over_digests = list(local_digest_to_filename_dict.keys() - remote_digest_to_filename_dict.keys())

    lack_files = [remote_digest_to_filename_dict[digest] for digest in lack_digests]
    over_files = [local_digest_to_filename_dict[digest] for digest in over_digests]

    print_check_results(lack_files, over_files, crash_files)

    request_files = lack_files + crash_files
    request_file_ids = [remote_filename_to_id_dict[item] for item in request_files]

    return request_file_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/train")
    parser.add_argument("--url", type=str, default="http://kyc-system.mynetgear.com")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    check_dataset(dataset_path=args.dataset_path, url=args.url, mode=args.mode)
