import torch
from torch.autograd import _tensor_or_tensors_to_tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils_test import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_predictions_as_imgs_test,
)
import os
import sys

# Hyperparameters etc.
LEARNING_RATE = 1e-4
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 1200 # Resize value
IMAGE_WIDTH = 1920 # Resize value
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR =sys.argv[1]

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop =  tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data =  data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transform = A.Compose(
        [
             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
             A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transform = A.Compose(
        [
             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
             A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    _model = UNET(in_channels=3, out_channels=1).cuda()
    model = nn.DataParallel(_model).to(DEVICE)
    # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_loader = get_loaders(
        TEST_IMG_DIR,
        test_transform,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("../../ckpts/unet_best.pth.tar"), model)


    scaler = torch.cuda.amp.GradScaler()
    save_predictions_as_imgs_test(
            test_loader, model, 
            folder=sys.argv[2], 
            device=DEVICE
        )
    

if __name__ == "__main__":
    if not os.path.isdir(sys.argv[2]) :
        os.mkdir(sys.argv[2])
    main()

