import os
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import wget

from fastestimator.dataset.csv_dataset import CSVDataset
from fastestimator.util.wget_util import bar_custom

class_dict = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


def _bbox_info(x, path):
    bboxes = []
    root = ET.parse(os.path.join(path, x)).getroot()
    for item in root.findall('./object'):
        bbox = []
        for child in item:
            if child.tag == "name":
                class_id = class_dict[child.text]
            elif child.tag == "bndbox":
                bbox = [
                    float(child.find("xmin").text),
                    float(child.find("ymax").text),
                    float(child.find("xmax").text),
                    float(child.find("ymin").text)
                ]
        bbox.append(class_id)
        bboxes.append(bbox)

    return bboxes


def _create_csv(image_list_path: str, root_dir: str, csv_path: str) -> None:
    """A helper function to create and save csv files.
    Args:
        image_list_path: Path to list of image id's.
        root_dir: The path of the downloaded data.
        csv_path: Path to save the csv file.
    """

    annotation_path = os.path.join(root_dir, "VOCdevkit/VOC2012/Annotations")
    image_path = os.path.join(root_dir, "VOCdevkit/VOC2012/JPEGImages")

    img_list = [line.rstrip('\n') for line in open(image_list_path)]

    df = pd.DataFrame(img_list, columns=["image"])
    df["bbox"] = df['image'].apply(lambda x: _bbox_info(x + ".xml", path=annotation_path))
    df['image'] = df['image'].apply(lambda x: os.path.join(image_path, x + ".png"))
    df.to_csv(csv_path, index=False)
    return None


def load_data(root_dir: Optional[str] = None) -> Tuple[CSVDataset, CSVDataset]:

    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'PASCAL_VOC')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'PASCAL_VOC')
    os.makedirs(root_dir, exist_ok=True)

    train_csv_path = os.path.join(root_dir, 'train.csv')
    val_csv_path = os.path.join(root_dir, 'val.csv')

    compressed_path = os.path.join(root_dir, 'VOCtrainval_11-May-2012.tar')
    extracted_path = os.path.join(root_dir, 'VOCdevkit')

    print(root_dir)

    if not os.path.exists(extracted_path):
        # download
        if not os.path.exists(compressed_path):
            print("Downloading data to {}".format(root_dir))
            wget.download('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
                          root_dir,
                          bar=bar_custom)

        # extract
        print("\nExtracting files ...")
        with tarfile.open(compressed_path) as tar_file:
            tar_file.extractall(root_dir)

    train_path = os.path.join(root_dir, "VOCdevkit/VOC2012/ImageSets/Main/train.txt")
    val_path = os.path.join(root_dir, "VOCdevkit/VOC2012/ImageSets/Main/val.txt")

    _create_csv(train_path, root_dir, train_csv_path)
    _create_csv(val_path, root_dir, val_csv_path)

    return CSVDataset(train_csv_path), CSVDataset(val_csv_path)
