import json
import argparse
from util import str2bool
import os


def make_question(args):
    all_entries = []

    f = open(args.csv_path)
    rdr = csv.reader(f)
    feature_names = None
    label_names = None
    for idx, line in enumerate(rdr):
        if idx == 0:
            feature_names = line[2:-6]
            label_names = line[-6:]
        else:
            prompt = ""

            for feature_idx, feature in enumerate(feature_names):
                prompt += "{}: {}\n".format(feature, line[feature_idx + 2])

            for label_idx, label in enumerate(line[-6:]):
                json_entry = {
                    "id": idx,
                    "image": os.path.join(args.img_path,"{}.jpeg".format(line[1])),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Predict {}\n<image>\n{}".format(label_names[label_idx], prompt)
                        },
                        {
                            "from": "gpt",
                            "value": "{}".format(float(label))
                        },
                    ]
                }
                all_entries.append(json_entry)

    with open(f'./kaggle/test_{args.file_name}.jsonl', 'w') as json_file:
        json.dump(all_entries, json_file, indent=2)


def make_train(args):
    all_entries = []

    f = open(args.csv_path)
    rdr = csv.reader(f)
    feature_names = None
    label_names = None
    for idx, line in enumerate(rdr):
        if idx == 0:
            feature_names = line[2:-6]
            label_names = [line[-1]]
        else:
            prompt = ""

            for feature_idx, feature in enumerate(feature_names):
                prompt += "F{}: {} \n ".format(feature_idx, line[feature_idx + 2])

            for label_idx, label in enumerate([line[-1]]):
                json_entry = {
                    "id": idx,
                    "image": os.path.join(args.img_path,"{}.jpeg".format(line[1])),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nPredict {} \n {} ".format(label_names[label_idx], prompt)
                        },
                        {
                            "from": "gpt",
                            "value": "{}".format(format(float(label), '.4f'))
                        },
                    ]
                }
                all_entries.append(json_entry)

    with open(f'./kaggle/train_{args.file_name}.jsonl', 'w') as json_file:
        json.dump(all_entries, json_file, indent=2)

import csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--file_name", type=str, default='x3112_75')
    parser.add_argument("--csv_path", type=str, default='/home/kaggle/train/X3112_mean_75.csv')
    parser.add_argument("--img_path", type=str, default='/home/kaggle/train/train_images')
    args = parser.parse_args()

    if args.train:
        make_train(args)
    else:
        make_question(args)


