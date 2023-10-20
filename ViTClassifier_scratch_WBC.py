import torch
import numpy as np

import os
import sys

from models.ViTClassifier import ViTClassifier

from load_dataset import DataloaderGenerator
from torch_run_utils import run_iter

args = sys.argv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_lr = 0.01
batch_size = 64
weight_decay = 0.03
warmup_steps = 10000
warmup_epochs = np.ceil(warmup_steps / batch_size)

num_epochs = 90

input_image_shape = (224, 224)
input_channels = 3

patch_shape = (16, 16)

latent_dims = 768
num_classes = 10
num_head = 12
num_layer = 12

dataset = "wbc100"
if len(args1) > 1:
    dataset = args[1]

test_dataset = "wbc100"

train_dataloader = DataloaderGenerator.dataloader(dataset, batch_size, input_image_shape=input_image_shape, subfolder="train")
test_dataloader = DataloaderGenerator.dataloader(test_dataset, batch_size, input_image_shape=input_image_shape, subfolder="val")


model_tested = "ViTClassifier"
training_type = "scratch" # pretrained
optimizer_type = "AdamW"

model_name = f'{model_tested}_p{patch_shape[0]}_b_b{batch_size}_{dataset}_{training_type}_lr{str(base_lr)[2:]}_decay{str(weight_decay)[2:]}_warmup{warmup_steps}_{optimizer_type}'

model_base_datapath = f"./model_weights/"

# Sinusoidal decay from 1 to 0. But original ViT use linear: https://github.com/google-research/vision_transformer/issues/2
lr_func = lambda epoch: min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (np.cos(epoch / num_epochs * np.pi) + 1))

loaded_loss = None

print(os.getcwd())
print(model_base_datapath + f"{model_name}.pt")

checkpoint_exists = os.path.isfile(model_base_datapath + f"{model_name}.pt")

if checkpoint_exists:
    wbc100_scratch_classifier = torch.load(model_base_datapath + f"{model_name}.pt")
    print("Loaded Model!")

    loaded_loss = torch.load(model_base_datapath + f"{model_name}.pickle")
    print("Loaded Losses!")

else:
    print("Initializing new model!")
    wbc100_scratch_classifier = ViTClassifier(embedding_dim=latent_dims, num_classes=num_classes,
                                            img_size=input_image_shape,
                                            patch_size=patch_shape, in_channels=3,
                                            num_head=num_head, num_layer=num_layer)
    
wbc100_scratch_optim = torch.optim.AdamW(wbc100_scratch_classifier.parameters(), lr=base_lr * batch_size / 256, betas=(0.9, 0.999), weight_decay=weight_decay)
if checkpoint_exists:
    optim_state_dict = torch.load(model_base_datapath + f"{model_name}_optimizer.pt")
    wbc100_scratch_optim.load_state_dict(state_dict=optim_state_dict)
    print("Loaded Optimizer!")

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(wbc100_scratch_optim, lr_lambda=lr_func, verbose=True)
if checkpoint_exists:
    lr_state_dict = torch.load(model_base_datapath + f"{model_name}_lrscheduler.pt")
    lr_scheduler.load_state_dict(state_dict=lr_state_dict)
    print("Loaded LR Scheduler!")

loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())


run_iter(wbc100_scratch_classifier, num_epochs, loss_fn, train_dataloader, test_dataloader, None, acc_fn, wbc100_scratch_optim, lr_func, lr_scheduler, device, model_name=model_name, loaded_loss=loaded_loss)
