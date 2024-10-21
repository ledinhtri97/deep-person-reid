import os
import glob
import re
import random
import string
import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2                
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
        assert im_files, f"No images found in {input_folder}"
    except Exception as e:
        raise FileNotFoundError(f"{input_folder}Error loading data\n") from e
    return im_files

def collect_data_rp(input_dir, output_dir):
    
    list_images = collect_image(input_dir)
    progress_bar = tqdm(total=len(list_images), unit=f" {input_dir} Rendering..")
    for image_file in list_images:
        progress_bar.update(1)
        if "camera2" in image_file:
            continue
        img_cv = cv2.imread(image_file)
        image_name = image_file.split("/")[-1]
        prod_id = image_name.split("-")[0]
        try:
            int(prod_id)
        except:
            prod_id = image_name.split("_")[0]
            
        prod_folder = f"{output_dir}/{prod_id}"
        os.makedirs(prod_folder, exist_ok=True)
        label_file = Path(image_file.replace("/images/", "/labels/"))
        label_file = str(label_file.with_suffix(".txt"))
        lbdata = []
        if os.path.exists(label_file):
            img_height, img_width = img_cv.shape[:2]
            with open(label_file, "r") as f:
                lbdata = f.readlines()
            for lb in lbdata:
                lb = lb.strip().split(" ")
                x_c = float(lb[1]) * img_width
                y_c = float(lb[2]) * img_height
                o_w = float(lb[3]) * img_width
                o_h = float(lb[4]) * img_height
                
                xmin = int(x_c - o_w / 2)
                ymin = int(y_c - o_h / 2)
                xmax = int(x_c + o_w / 2)
                ymax = int(y_c + o_h / 2)

                img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                img_crop = img_cv[ymin:ymax, xmin:xmax]
                cv2.imwrite(f"{prod_folder}/{img_id}.jpg", img_crop)
                
        # print(f"{label_file} done -> {len(lbdata)}")
        # break


def copy_file(source, destination):
    try:
        shutil.copy2(source, destination)
        # print("File copied successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e} {source}")

def convert_and_split_dataset(input_folder, output_dir):
    box_train = f"{output_dir}/market1501/bounding_box_train"
    box_test = f"{output_dir}/market1501/bounding_box_test"
    box_query = f"{output_dir}/market1501/query"
    os.makedirs(box_train, exist_ok=True)
    os.makedirs(box_test, exist_ok=True)
    os.makedirs(box_query, exist_ok=True)
    
    folders = [folder for folder in glob.glob(os.path.join(input_folder, "*")) if os.path.isdir(folder)]
    random.shuffle(folders)
    
    num_train = int(len(folders) * 0.9)
    num_test = len(folders) - num_train
    
    train_data_folder = folders[:num_train]
    test_data_folder = folders[num_train:]
    
    progress_bar = tqdm(total=num_train, unit=f" {box_train} Rendering..")
    id_object = 1
    for id_img, train_folder in enumerate(train_data_folder):
        progress_bar.update(1)
        sub_train_images = collect_image(train_folder)
        id_object += 1
        for img_path in sub_train_images:
            cam_id = np.random.randint(1, 7)
            img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            new_name_image = f"{id_object:04d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            copy_file(img_path, f"{box_train}/{new_name_image}")
            # os.system(f"cp -f {img_path} {box_train}/{new_name_image}")

    progress_bar = tqdm(total=num_test, unit=f" {box_test} Rendering..")
    
    for id_img, train_folder in enumerate(test_data_folder):
        progress_bar.update(1)
        sub_test_images = glob.glob(f"{train_folder}/*.jpg")
        random.shuffle(sub_test_images)
        num_test_image = int(len(sub_test_images) * 0.7)
        num_query = len(sub_test_images) - num_test_image
        
        sub_sub_test_images = sub_test_images[:num_test_image]
        sub_sub_query_images = sub_test_images[num_test_image:]
        id_object += 1
        for img_path in sub_sub_test_images:
            cam_id = np.random.randint(2, 7)
            img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            new_name_image = f"{id_object:04d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            # os.system(f"cp -f {img_path} {box_test}/{new_name_image}")
            copy_file(img_path, f"{box_test}/{new_name_image}")
        for img_path in sub_sub_query_images:
            cam_id = 1
            img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            new_name_image = f"{id_object:04d}_c{cam_id}_{id_img:04d}{img_id}.jpg"
            # os.system(f"cp -f {img_path} {box_query}/{new_name_image}")
            copy_file(img_path, f"{box_query}/{new_name_image}")
            
if __name__ == "__main__":
    convert_and_split_dataset(
        "/workspace/dataset/product/rp_reid/",
        "/workspace/dataset/product/rp_reid_Market")
    
    # collect_data_rp("/workspace/dataset/product/retail_product_checkout/images/train2019/", 
    #                 "/workspace/dataset/product/rp_reid/")
            
        
    