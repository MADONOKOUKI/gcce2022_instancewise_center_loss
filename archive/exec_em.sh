cd /fs1/groups1/gaa50073/madono/bmvc2021/em_classifier
source ~/.bashrc
module load python/3.6/3.6.5
python3 -m venv pkd
source pkd/bin/activate
module load python/3.6/3.6.5
module load nccl/2.2/2.2.13-1
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
module load 7.6/7.6.4
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
pip install tqdm
pip install torch torchvision
python main_em.py > results/main_em.txt
#python main.py > results/main.txt
