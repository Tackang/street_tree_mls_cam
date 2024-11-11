from logging.config import valid_ident
import torch
from torch.autograd import _tensor_or_tensors_to_tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import wandb
import os

wandb.init(project="UNET_TK", entity="didxorkd")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 300
NUM_WORKERS = 2
IMAGE_HEIGHT = 1200  # Resize value
IMAGE_WIDTH = 1920 # Resize value
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/esail3/Tackang/98AimmoData/train_images/"
TRAIN_MASK_DIR = "/esail3/Tackang/98AimmoData/train_masks/"
VAL_IMG_DIR = "/esail3/Tackang/98AimmoData/val_images/"
VAL_MASK_DIR = "/esail3/Tackang/98AimmoData/val_masks/"

wandb.config = {
  "learning_rate": LEARNING_RATE,
  "epochs": NUM_EPOCHS,
  "batch_size": BATCH_SIZE
}

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop =  tqdm(loader)
    loss_avg = []
    train_dice_score=0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data =  data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        predictions_dice=torch.sigmoid(predictions)    
        train_dice_score += (2*(predictions_dice*targets).sum()) / (
                (predictions_dice + targets).sum() + 1e-8
            )
        loss_avg.append(loss)
       
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    loss_avg=sum(loss_avg)/len(loss_avg)
   
    return loss_avg, train_dice_score/len(loader)
      

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.RandomCrop(width=450 , height=900),
            A.CLAHE(p = 0.1),
            A.RandomBrightnessContrast(p = 0.1),
            # A.RandomBrightness(p = 0.1),
            # A.VerticalFlip(p=0.1),
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
            A.CLAHE(p = 0.1),
            A.RandomBrightnessContrast(p = 0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    _model = UNET(in_channels=3, out_channels=1).cuda()
    model = nn.DataParallel(_model).to(DEVICE)
    wandb.watch(model)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    
    saveChecker = []
    for epoch in range(NUM_EPOCHS):
        print(f"current epoch : {epoch+1}")

        loss_avg, train_dice_score = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # check accuracy
        val_loss_avg, validation_dice_score = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
        
        saveChecker.append(val_loss_avg)
        if val_loss_avg <= min(saveChecker):
            save_checkpoint(checkpoint)
            
        wandb.log({"Epoch": epoch+1,
        "Train loss" : loss_avg,
        "Validation loss" : val_loss_avg,
        "Train Dice Score" : train_dice_score,
        "Validation Dice Score" : validation_dice_score
        })      

        del loss_avg
        del val_loss_avg
        del train_dice_score
        del validation_dice_score

        # print some examples to folder
        if epoch % 5 == 0:
            save_predictions_as_imgs(
             val_loader, model, folder="saved_images/", device=DEVICE
            )


if __name__ == "__main__":
    main()

