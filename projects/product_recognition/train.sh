#!/bin/bash

# python3 main.py --config-file im_r50_softmax_256x256_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_osnet_x1_0_softmax_256x256_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_deit_small_patch16_softmax_224_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_osnet_x1_0_softmax_256x128_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
python3 main.py --config-file im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_osnet_ibn_x1_0_softmax_256x128_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_swin_base_patch4_window7_224_amsgrad_cosine.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 main.py --config-file im_mamba_s_softmax_224_amsgrad.yaml --root /workspace/dataset/product/rp_reid_Market/
# python3 ../../tools/export.py --weights new_log0508/osnet_ain_x1_0_market1501_softmax/model/model.pth.tar-44 --dynamic --include onnx
