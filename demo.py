# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import pickle as pkl

# import inference code
sys.path.append('notebook')
from inference import Inference, load_image, load_single_mask

# load model
tag = 'hf'
config_path = f'checkpoints/{tag}/pipeline.yaml'
inference = Inference(
    config_path,
    compile=False,
    stage1_only=True,
)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image('notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png')
mask = load_single_mask('notebook/images/shutterstock_stylish_kidsroom_1640806567', index=14)

# run model
output = inference(image, mask, seed=42)
print('Inference done!')
print('Inference output keys:', output.keys())
# save output
print('Saving output to output_demo.pkl')
pkl.dump(output, open('output_demo.pkl', 'wb'))
