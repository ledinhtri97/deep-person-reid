#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_osnet_ain_x1_0_arcface_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_osnet_ain_x1_0_triplet_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_r50_softmax_256x128_amsgrad.yaml --root /home/trild/dataset/facerec/facerec_combine/

# python3 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    # --root /home/trild/dataset/facerec/facerec_combine/

# python3 ../../tools/export.py --weights resutls/resnet50_market1501_triplet1/model/model.pth.tar-50 --dynamic --include onnx --name resnet50_fc512
# python3 main.py --config-file im_osnet_ain_x1_0_arcface_unpg_256x128_amsgrad_cosine.yaml --root /home/trild/Devs/driplym-project/driply-ai-cores/build/face_test/facerec_combine/