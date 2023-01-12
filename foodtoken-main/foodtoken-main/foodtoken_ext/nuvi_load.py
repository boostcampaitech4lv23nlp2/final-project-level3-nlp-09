from nuviAPI.s3 import S3API
import json


#TODO
def download_dict():
    api = S3API()
    dictionary = api.get_json('tokenizer-dict', 'tokens_by_length.json')
    with open('tokens_by_length.json', 'w') as f:
        json.dump(dictionary, f)
