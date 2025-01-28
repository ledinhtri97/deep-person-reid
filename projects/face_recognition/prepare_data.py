import os
import glob
import re
import random
import string
import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
# import cv2                
import shutil
""" bounding_box_train
query
bounding_box_test """

""" 
python3 ../../tools/visualize_actmap.py \
    --root /workspace/dataset/product/rp_reid_Market/ \
    -d market1501 \
    -m osnet_x1_0 \
    --weights new_log0909/resnet50_market1501_softmax/model/model.pth.tar-8 \
    --save-dir log/resnet50_market1501_softmax --height 224 --width 224
"""

np.random.seed(123)

def collect_image(input_folder):
    IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm" 
    im_files = []
    try:
        f = []  # image files
        for p in input_folder if isinstance(input_folder, list) else [input_folder]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob("*.*"))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"{input_folder}{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        # assert im_files, f"No images found in {input_folder}"
    except Exception as e:
        raise FileNotFoundError(f"{input_folder}Error loading data\n") from e
    return im_files

def copy_file(source, destination):
    try:
        shutil.copy2(source, destination)
        # print("File copied successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e} {source}")

def convert_and_split_dataset(input_folder, output_dir, prefix_name, train_ratio=1, cam_id_cb=None, limit=100):
    box_train = f"{output_dir}/market1501/bounding_box_train"
    box_test = f"{output_dir}/market1501/bounding_box_test"
    box_query = f"{output_dir}/market1501/query"
    os.makedirs(box_train, exist_ok=True)
    os.makedirs(box_test, exist_ok=True)
    os.makedirs(box_query, exist_ok=True)
    
    pid_file = f"{output_dir}/pid.txt"
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            start_pid = int(f.read())
    else:
        start_pid = 0
                
    folders = [folder for folder in glob.glob(os.path.join(input_folder, "*")) if os.path.isdir(folder)]
    random.shuffle(folders)
    
    num_train = int(len(folders) * train_ratio)
    num_test = len(folders) - num_train
    
    train_data_folder = folders[:num_train]
    test_data_folder = folders[num_train:]
    
    def gen_img_id(prefix):
        return prefix + "".join(random.choices(string.ascii_letters + string.digits, k=8))
    
    progress_bar = tqdm(total=num_train, unit=f" {box_train} Rendering..")
    id_object = start_pid
    for id_img, train_folder in enumerate(train_data_folder):
        progress_bar.update(1)
        sub_train_images = collect_image(train_folder)[:limit]
        id_object += 1
        for img_path in sub_train_images:
            cam_id = np.random.randint(1, 7)
            img_id = gen_img_id(prefix_name)
            new_name_image = f"{id_object:08d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            copy_file(img_path, f"{box_train}/{new_name_image}")
            # os.system(f"cp -f {img_path} {box_train}/{new_name_image}")
    progress_bar = tqdm(total=num_test, unit=f" {box_test} Rendering..")
    
    for id_img, test_folder in enumerate(test_data_folder):
        progress_bar.update(1)
        sub_test_images = collect_image(test_folder)[:limit]
        random.shuffle(sub_test_images)
        num_test_image = int(len(sub_test_images) * 0.7)
        num_query = len(sub_test_images) - num_test_image
        
        sub_sub_test_images = sub_test_images[:num_test_image]
        sub_sub_query_images = sub_test_images[num_test_image:]
        id_object += 1
        for img_path in sub_sub_test_images:
            if cam_id_cb is None:
                cam_id = np.random.randint(2, 7)
            else:
                cam_id = cam_id_cb(img_path)
            img_id = gen_img_id(prefix_name)
            new_name_image = f"{id_object:08d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            # os.system(f"cp -f {img_path} {box_test}/{new_name_image}")
            copy_file(img_path, f"{box_test}/{new_name_image}")
        for img_path in sub_sub_query_images:
            if cam_id_cb is None:
                cam_id = 1
            else:
                cam_id = cam_id_cb(img_path)
            img_id = gen_img_id(prefix_name)
            new_name_image = f"{id_object:08d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            # os.system(f"cp -f {img_path} {box_query}/{new_name_image}")
            copy_file(img_path, f"{box_query}/{new_name_image}")

    print("=========================\n\n")
    print(f"[DataReport] {prefix_name} [PIDs] Number of train: {num_train}, Number of test: {num_test}")
    with open(pid_file, "w") as f:
        f.write(str(id_object+1))
        
    return id_object + 1

def group_flat_images_to_folder(input_folder, output_folder):
    images_list = collect_image(input_folder)
    limit_frame = 100
    count_frame_pid = {}
    random.shuffle(images_list)
    progress_bar = tqdm(total=len(images_list), unit=f" {input_folder} Rendering..")
    for img_path in images_list:
        progress_bar.update(1)
        # inhouse_whatson_dataset_2024_101624/0001/0001_0001.jpg
        img_name = os.path.basename(img_path).split(".")[0]
        split_data = img_path.split("/")
        person_id = split_data[-2]
        if person_id not in count_frame_pid:
            count_frame_pid[person_id] = 0
        if count_frame_pid[person_id] > limit_frame:
            continue
        count_frame_pid[person_id] += 1
        person_folder = f"{output_folder}/{person_id}"
        if not os.path.exists(person_folder):
            os.makedirs(person_folder, exist_ok=True)
        copy_file(img_path, f"{person_folder}/{img_name}.jpg")

if __name__ == "__main__":

    root_folder = "/home/trild/Devs/driplym-project/driply-ai-cores/build/face_test/"
    combine_folder = f"{root_folder}/facerec_combine"
    
    # process real celebq data
    raw_celebahq_dir = f"{root_folder}/test_data"
    # celebahq_dir = f"{root_folder}/1preprocessed_celebahq"
    # group_flat_images_to_folder(raw_celebahq_dir, celebahq_dir)
    start_pid = convert_and_split_dataset(raw_celebahq_dir, combine_folder, "tinyface", train_ratio=0.8)
    
    