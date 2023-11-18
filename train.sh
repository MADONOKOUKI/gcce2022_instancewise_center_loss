cd /fs1/groups1/gaa50073/madono/bmvc2021/em_classifier
source ~/.bashrc
module load python/3.6/3.6.12
python3 -m venv pkd
source pkd/bin/activate
module load python/3.6/3.6.12
module load nccl/2.2/2.2.13-1
# module load python/3.6/3.6.12
module load cuda/10.1/10.1.243
module load 7.6/7.6.4
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# pip install tqdm
# pip install torch torchvision
# pip install hydra
# pip install hydra-core --upgrade
# pip install logging
# pip install pytorch-lightning
# pip install omegaconf 
# pip install mlflow
# pip install scipy
pip install logzero
pip install git+https://github.com/ildoonet/pytorch-randaugment 
python train.py 

