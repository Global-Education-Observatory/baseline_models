from torchvision import transforms, models
from datetime import datetime
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import argparse
import random
import torch
import time
import copy
import os

from dataloader import *
from utils import *




if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--folder_name', type = str, required = True)
    # args = parser.parse_args()

    now = datetime.now()    
    fname = 'run_' + now.strftime("%m-%d-%Y_%H-%M-%S")
    username = "hmbaier"
    base_dir = f"/sciclone/home/{username}/geo/"

    records_dir = os.path.join(base_dir, "models", fname)
    os.mkdir(records_dir)
    os.mkdir(os.path.join(records_dir, "models"))        

    image_names = ["/sciclone/geograd/Heather/explain/phl_clips/" + i for i in os.listdir("/sciclone/geograd/Heather/explain/phl_clips/")]
    image_names = [i for i in image_names if "ipynb" not in i]
    
    print(image_names[0:5])

    data = Dataloader(image_names, os.path.join(base_dir, "ph_hpc.json"), records_dir, batch_size = 32)

    device = "cpu"
    model_ft = models.resnet50(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)
    criterion = nn.L1Loss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data, device, os.path.join(records_dir, "models"), num_epochs = 50)
