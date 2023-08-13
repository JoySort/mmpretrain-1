conda create -n mmpretrain python=3.8 -y
conda activate mmpretrain
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install openmim
conda install -c anaconda scikit-learn
pip install wandb
pip install future tensorboard
pip install sklearn funcy argparse scikit-multilearn
mim install -e .