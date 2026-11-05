#!/bin/bash
set -Eeuo pipefail
S=/scratch/Ananya_Kulkarni/slam; cd $S/DROID-SLAM
export PIP_CACHE_DIR=$S/.pipcache TMPDIR=$S/.tmp CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
mkdir -p $PIP_CACHE_DIR $TMPDIR
[ -x $S/droid_env/bin/python ] || ~/miniconda3/envs/visia/bin/python -m venv $S/droid_env
source $S/droid_env/bin/activate
pip install -q -U pip "setuptools<70" wheel
pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q tqdm evo scipy open3d gdown tensorboard opencv-python-headless pyyaml "numpy<2"
pip install -q thirdparty/lietorch
pip install -q thirdparty/pytorch_scatter
pip install -q -e .
[ -f droid.pth ] || gdown -q 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O droid.pth
ls -la droid.pth
python -c "import torch,lietorch,droid_backends,torch_scatter,evo,cv2;print('IMPORT_OK',torch.__version__,torch.cuda.device_count())"
echo INSTALL_DONE
