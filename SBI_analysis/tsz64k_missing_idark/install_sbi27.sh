#!/bin/bash
# Python 3.11 env for the tSZ 64k SBI on CentOS 7 (glibc 2.17): compiled packages from conda-forge builds that support
# glibc 2.17, then the pure-Python sbi 0.27.0 (+ nflows, zuko) with pip, no dependency resolution by pip.
set -euo pipefail
cd $HOME/envs
echo "start $(date -Is)"
[ -x bin/micromamba ] || curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=$HOME/envs/mamba
unset LD_LIBRARY_PATH PYTHONPATH
rm -rf $HOME/envs/sbi27
./bin/micromamba create -y -p $HOME/envs/sbi27 -c conda-forge python=3.11 "numpy=1.26" scipy matplotlib scikit-learn \
  "pytorch=*=cpu*" tensorboard pillow tqdm joblib skorch tabulate pip
PY=$HOME/envs/sbi27/bin/python
$PY -m pip install --no-cache-dir --no-deps sbi==0.27.0 nflows==0.14 zuko==1.6.0
$PY -c "import numpy, scipy, torch, sbi, sklearn, matplotlib, zuko, nflows; print(numpy.__version__, scipy.__version__, torch.__version__, sbi.__version__, sklearn.__version__, matplotlib.__version__)"
echo "all done $(date -Is)"
