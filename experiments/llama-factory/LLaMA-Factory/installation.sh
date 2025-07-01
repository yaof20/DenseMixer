conda create -n ste python=3.12
conda activate ste
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install transformers==4.51.3
pip install deepspeed==0.16.7
pip install wandb
pip install densemixer==0.1.0.post2
densemixer setup
