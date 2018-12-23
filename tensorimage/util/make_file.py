import json


def make_json_file(path, data):
    with open(path, 'a') as f:
        json.dump(data, f)
