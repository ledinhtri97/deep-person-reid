import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import glob
from tqdm import tqdm

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

def remove_duplicate_images(data_folder_path):
    image_list = collect_image(data_folder_path)
    
    num_images = len(image_list)
    pd = tqdm(total=num_images, unit=f" {data_folder_path} Rendering..")
    # Remove duplicate images
    for i in range(len(image_list)):
        pd.update(1)
        image_i_path = image_list[i]
        name_i = image_i_path.split("/")[-1]
        for j in range(i+1, len(image_list)):
            name_j_path = image_list[j]
            name_j = name_j_path.split("/")[-1]
            if name_i == name_j:
                if os.path.exists(name_j_path):
                    os.remove(name_j_path)
                    print(f"Removed: {name_j_path}")
        
        # """ glob by name_i in for folder """
        # dup_images = glob.glob(data_folder_path + f"/**/{name_i}", recursive=True)
        # for dup_image in dup_images:
        #     if dup_image != image_path:
        #         print(f"Removed: {dup_image}")
        #         # os.remove(dup_image)

root_folder = "/home/trild/Downloads/face_groups/"
list_sub_folder = os.listdir(root_folder)
for sub_folder in list_sub_folder:
    # print(sub_folder)
    # remove_duplicate_images(f"{root_folder}/{sub_folder}")
    
    # list_sub2_folder = os.listdir(f"{root_folder}/{sub_folder}")
    # for sub2_folder in list_sub2_folder:
    #     list_image = collect_image(f"{root_folder}/{sub_folder}/{sub2_folder}")
    #     num_image = len(list_image)
    #     if num_image < 5:
    #         print(f"Remove {sub2_folder} with {num_image} images")
    #         shutil.rmtree(f"{root_folder}/{sub_folder}/{sub2_folder}")
    # #     # print(f"id {sub2_folder} num images: {len(list_image)}")
        
    pass