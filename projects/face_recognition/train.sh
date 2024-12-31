#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --master_port=7777 --nproc_per_node=1 main.py --config-file im_osnet_ain_x1_0_arcface_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/

# python3 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    # --root /home/trild/dataset/facerec/facerec_combine/

# python3 ../../tools/export.py --weights resutls/osnet_ain_x1_0_market1501_softmax/241021/model/model.pth.tar-18 --dynamic --include onnx
