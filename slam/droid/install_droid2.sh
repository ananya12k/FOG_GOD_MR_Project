#!/bin/bash
# Continues install_droid.sh after lietorch failed: torch is not visible inside pip build isolation.
set -Eeuo pipefail
S=/scratch/Ananya_Kulkarni/slam; cd $S/DROID-SLAM
export PIP_CACHE_DIR=$S/.pipcache TMPDIR=$S/.tmp CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONNOUSERSITE=1
source $S/droid_env/bin/activate
pip install -q --no-build-isolation thirdparty/lietorch
pip install -q --no-build-isolation thirdparty/pytorch_scatter
pip install -q --no-build-isolation -e .
[ -f droid.pth ] || gdown -q 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O droid.pth
ls -la droid.pth
python -c "import torch,lietorch,droid_backends,torch_scatter,evo,cv2;print(\"IMPORT_OK\",torch.__version__,torch.cuda.device_count())"
echo INSTALL_DONE
