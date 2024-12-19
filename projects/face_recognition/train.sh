#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml --root /home/trild/dataset/facerec/facerec_combine/

# python3 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    # --root /home/trild/dataset/facerec/facerec_combine/
