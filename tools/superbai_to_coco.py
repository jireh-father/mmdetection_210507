import argparse

import json
import os
import glob
import pandas as pd
import sys
from pycocotools import mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--meta_dir', type=str, default=None)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--except_files', type=str, default=None)
    args = parser.parse_args()
    return args


def get_except_img_files(except_files):
    except_img_files = []
    for except_file in except_files:
        df = pd.read_csv(except_file)
        except_img_files += list(df.file.values)
    return {f: True for f in except_img_files}


def get_coco_template():
    template = {
        "info": {'year': 2021, 'version': '1.0',
                 'description': 'VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)',
                 'contributor': '', 'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/',
                 'date_created': 'Wed May 05 2021 14:18:44 GMT+0900 (한국 표준시)'}
        ,
        "licenses":
            [{'id': 0, 'name': 'Unknown License', 'url': ''}],
        "images": [
            # {
            #     "id": 1,
            #     "width": 400,
            #     "height": 400,
            #     'file_name': '1.jpg'
            # }
        ],
        "annotations": [
            # {
            #     "segmentation": [[2, 442, 996, 442, 996, 597, 2, 597]],
            #     "area": 11111,
            #     "bbox": [2, 4, 2, 4],
            #     "iscrowd": 0,
            #     "id": 21,
            #     "image_id": 4,
            #     "category_id": 1
            # }
        ],
        "categories": [{'supercategory': 'foot', 'id': 1, 'name': 'foot'},
                       {'supercategory': 'triangle', 'id': 2, 'name': 'left_triangle'},
                       {'supercategory': 'triangle', 'id': 3, 'name': 'right_triangle'}],
    }

    return template


def make_anno(polygon, width, height):
    seg = []
    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0
    for vertex in polygon:
        if vertex["x"] < min_x:
            min_x = vertex["x"]
        if vertex["x"] > max_x:
            max_x = vertex["x"]
        if vertex["y"] < min_y:
            min_y = vertex["y"]
        if vertex["y"] > max_y:
            max_y = vertex["y"]
        seg.append(vertex["x"])
        seg.append(vertex["y"])

    rles = mask.frPyObjects([seg], height, width)
    rle = mask.merge(rles)
    # bbox = mask.toBbox(rle)
    area = int(mask.area(rle))
    annotation = {
        "segmentation": [seg],
        "area": float(area),
        "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
        "iscrowd": 0,
    }

    return annotation


def main():
    args = parse_args()

    except_files_dict = get_except_img_files(args.except_files.split(","))

    meta_files = glob.glob(os.path.join(args.meta_dir, "*.json"))

    anno_id = 1
    im_id = 1
    annotations = []
    images = []
    for i, meta_file in enumerate(meta_files):
        if i % 10 == 0:
            print(i, len(meta_files))
        meta = json.load(open(meta_file))
        im_fname = meta["data_key"]
        if im_fname in except_files_dict:
            print(im_fname, "except image")
            continue
        label_fname = meta["label_id"] + ".json"
        width = meta["image_info"]["width"]
        height = meta["image_info"]["height"]

        label_path = os.path.join(args.label_dir, label_fname)
        anno = json.load(open(label_path))

        for ob in anno["result"]["objects"]:
            if ob["class"] == "Foot":
                new_anno = make_anno(ob["shape"]["polygon"], width, height)
                new_anno["category_id"] = 1
            elif ob["class"] == "Triangle":
                if ob["properties"][0]["value"] == "Left":
                    new_anno = make_anno(ob["shape"]["polygon"], width, height)
                    new_anno["category_id"] = 2
                elif ob["properties"][0]["value"] == "Right":
                    new_anno = make_anno(ob["shape"]["polygon"], width, height)
                    new_anno["category_id"] = 3
                else:
                    raise Exception("unknown property value of triangle", ob["properties"]["value"])
            else:
                raise Exception("unknown class", ob["class"])
            new_anno["id"] = anno_id
            anno_id += 1
            new_anno["image_id"] = im_id
            annotations.append(new_anno)
        im_data = {
            "id": im_id,
            "width": width,
            "height": height,
            'file_name': im_fname
        }
        images.append(im_data)
        im_id += 1

    coco = get_coco_template()
    coco["annotations"] = annotations
    coco["images"] = images
    json.dump(coco, open(args.output_path, "w+"))

    print("done")


if __name__ == '__main__':
    main()
