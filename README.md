## DDPOPlace
Code for DDPOPlace: Reinforcement-Learned Diffusion Placement with Intermediate Rewards

This project based on https://github.com/vint-1/chipdiffusion

## How to train

'''CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/$program \
    method=ddpo task=$task mode@_global_=ddpo \
    from_checkpoint=large-v2.ckpt 
'''
large-v2.ckpt can get from (https://github.com/vint-1/chipdiffusion#pre-trained-models) https://drive.google.com/drive/folders/16b8RkVwMqcrlV_55JKwgprv-DevZOX8v?usp=sharing

## How to get Synthesized dataset
'''
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v1

PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=5000 num_val_samples=2500
'''

## How to Run
'''
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py task=v2.61 method=eval from_checkpoint=DDPOPlaceModel.ckpt legalizer@_global_=none guidance@_global_=none num_output_samples=128
'''

## DDPOPlace model
For convenience, we provide the training checkpoint for the DDPOPlace model at <a href="https://drive.google.com/file/d/11JQo5AEueToZ-zEb_gPcipWxSgrh9wBh/view?usp=sharing">this link</a>.


