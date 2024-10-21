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

def copy_file(source, destination):
    try:
        shutil.copy2(source, destination)
        # print("File copied successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e} {source}")

def convert_and_split_dataset(input_folder, output_dir, prefix_name, start_pid, train_ratio=1, cam_id_cb=None, limit=100):
    box_train = f"{output_dir}/market1501/bounding_box_train"
    box_test = f"{output_dir}/market1501/bounding_box_test"
    box_query = f"{output_dir}/market1501/query"
    os.makedirs(box_train, exist_ok=True)
    os.makedirs(box_test, exist_ok=True)
    os.makedirs(box_query, exist_ok=True)
    
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
    return id_object + 1

def group_PKU_images_to_folder(input_folder, output_folder):
    images_list = collect_image(input_folder)
    random.shuffle(images_list)
    progress_bar = tqdm(total=len(images_list), unit=f" {input_folder} Rendering..")
    for img_path in images_list:
        progress_bar.update(1)
        img_name = os.path.basename(img_path).split(".")[0]
        split_data = img_name.split("_")
        person_id = split_data[0]
        person_folder = f"{output_folder}/{person_id}"
        if not os.path.exists(person_folder):
            os.makedirs(person_folder, exist_ok=True)
        copy_file(img_path, f"{person_folder}/{img_name}.jpg")

def group_RPI_images_to_folder(input_folder, output_folder):
    images_list = collect_image(input_folder)
    progress_bar = tqdm(total=len(images_list), unit=f" {input_folder} Rendering..")
    for img_path in images_list:
        progress_bar.update(1)
        # RPIfield_v0/Data/Cam_9/67/Cam_9f_190889.png
        img_name = os.path.basename(img_path).split(".")[0]
        split_data = img_path.split("/")
        person_id = split_data[-2]
        person_folder = f"{output_folder}/{person_id}"
        if not os.path.exists(person_folder):
            os.makedirs(person_folder, exist_ok=True)
        copy_file(img_path, f"{person_folder}/{img_name}.jpg")

def group_mars_images_to_folder(input_folder, output_folder):
    limit_frame = 100
    images_list = collect_image(input_folder)
    random.shuffle(images_list)
    progress_bar = tqdm(total=len(images_list), unit=f" {input_folder} Rendering..")
    count_frame_pid = {}
    for img_path in images_list:
        progress_bar.update(1)
        # mars-motion-analysis-and-reidentification-set/bbox_train/0001/0001_C1_000001.jpg
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
        
def group_pep_images_to_folder(input_folder, output_folder):
    sub_scen_folder = [folder for folder in glob.glob(os.path.join(input_folder, "*")) if os.path.isdir(folder)]
    map_new_pids = {}
    for sub_folder in sub_scen_folder:
        images_list = collect_image(sub_folder)
        random.shuffle(images_list)
        progress_bar = tqdm(total=len(images_list), unit=f" {sub_folder} Rendering..")
        folder_name = os.path.basename(sub_folder)
        for img_path in images_list:
            progress_bar.update(1)
            # pep_256x128/scen3/view4/181/10213.jpg
            img_name = os.path.basename(img_path).split(".")[0]
            split_data = img_path.split("/")
            person_id = split_data[-2]
            pid_local = f"{folder_name}_{person_id}"
            if pid_local not in map_new_pids:
                map_new_pids[pid_local] = len(map_new_pids)
            new_pid = map_new_pids[pid_local]
            camera_id = split_data[-3].split("view")[-1]
            person_folder = f"{output_folder}/{new_pid}"
            if not os.path.exists(person_folder):
                os.makedirs(person_folder, exist_ok=True)
            copy_file(img_path, f"{person_folder}/{camera_id}_{pid_local}_{img_name}.jpg")

def group_wson_images_to_folder(input_folder, output_folder):
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

    root_folder = "/workspace/dataset/pose/reid/"
    combine_folder = "/workspace/dataset/pose/reid_combine"
    
    start_pid = 0
    # process PKU-Reid-Dataset data
    """ convert pku data to to subfolder """
    raw_pku_dir = f"{root_folder}/PKU-Reid-Dataset"
    pku_dir = f"{root_folder}/1preprocessed_PKU-Reid-Dataset"
    group_PKU_images_to_folder(raw_pku_dir, pku_dir)
    start_pid = convert_and_split_dataset(pku_dir, combine_folder, "PKU", start_pid, train_ratio=0.9, cam_id_cb=None)
    
    # process RPIfield_v0 data
    """ convert rpifield data to to subfolder """
    raw_rpi_dir = f"{root_folder}/RPIfield_v0/Data"
    rpi_dir = f"{root_folder}/1preprocessed_RPIfield_v0"
    group_RPI_images_to_folder(raw_rpi_dir, rpi_dir)
    def cam_id_cb_rpi(img_path):
        image_name = img_path.split("/")[-1].split(".")[0]
        cam_id = int(image_name.split("_")[1].split("f")[0])
        return cam_id
    start_pid = convert_and_split_dataset(rpi_dir, combine_folder, "RPI", start_pid, train_ratio=0.9, cam_id_cb=cam_id_cb_rpi)
     
    # [SKIP] process last data skip last because the data with same person (actor) but have different clothes
    # raw_last_dir = f"{root_folder}/last"
    
    # [CHECKING] process mars-motion-analysis-and-reidentification-set limit images for each fodler to 200frames/person
    raw_mars_dir = f"{root_folder}/mars-motion-analysis-and-reidentification-set"
    mars_dir = f"{root_folder}/1preprocessed_mars-motion-analysis-and-reidentification-set"
    group_mars_images_to_folder(f"{raw_mars_dir}/bbox_train", f"{mars_dir}/bbox_train")
    group_mars_images_to_folder(f"{raw_mars_dir}/bbox_test", f"{mars_dir}/bbox_test")
    def cam_id_cb_mars(img_path):
        # 0003C5T0010F046.jpg
        image_name = img_path.split("/")[-1].split(".")[0]
        cam_id = int(image_name.split("T")[0].split("C")[-1])
        return cam_id
    start_pid = convert_and_split_dataset(f"{mars_dir}/bbox_train", combine_folder, "mars", start_pid, train_ratio=1, cam_id_cb=cam_id_cb_mars)
    start_pid = convert_and_split_dataset(f"{mars_dir}/bbox_test", combine_folder, "mars", start_pid, train_ratio=0, cam_id_cb=cam_id_cb_mars)
    
    # process pep_256x128
    raw_pep_dir = f"{root_folder}/pep_256x128"
    pep_dir = f"{root_folder}/1preprocessed_pep_256x128"
    group_pep_images_to_folder(raw_pep_dir, pep_dir)
    def cam_id_cb_pep(img_path):
        # scen3/view4/181/10213.jpg
        image_name = img_path.split("/")[-1].split(".")[0]
        cam_id = int(image_name.split("_")[0])
        return cam_id
    start_pid = convert_and_split_dataset(pep_dir, combine_folder, "pep", start_pid, train_ratio=0.9, cam_id_cb=cam_id_cb_pep)
    
    # process inhouse_whatson_dataset_2024_101624
    raw_wson_dir = f"{root_folder}/inhouse_whatson_dataset_2024_101624"
    wson_dir = f"{root_folder}/1preprocessed_inhouse_whatson_dataset_2024_101624"
    group_wson_images_to_folder(raw_wson_dir, wson_dir)
    start_pid = convert_and_split_dataset(wson_dir, combine_folder, "wson", start_pid, train_ratio=1)
    start_pid = convert_and_split_dataset(wson_dir, combine_folder, "wson", start_pid, train_ratio=0)

            
        
    