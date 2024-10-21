# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 libglib2.0-0

#RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.0.2 mamba-ssm==1.0.1 timm==0.9.0 einops

# Set working directory and copy application code
WORKDIR /workspace
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt