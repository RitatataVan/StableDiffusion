#!pip install open_clip_torch==2.19.0
#!pip install timm==0.8.6.dev0
#!pip install sentence_transformers

IMG_PATH = '/root/SDIP/data/image211k/'#211k image path
CSV_PATH = '/root/SDIP/data/prompts/sd2_meta_211k_filter.csv'#211k csv path

IMG_PATH_540kP1 = '/root/SDIP/data/image540kp1/'#540kp1 image path
CSV_PATH_540kP1 = '/root/SDIP/data/prompts/sd2_meta_540k_part1.csv'#540kp1 csv path
IMG_PATH_540kP2 = '/root/SDIP/data/images540kp2/'#540kp2 image path
CSV_PATH_540kP2 = '/root/SDIP/data/prompts/sd2_meta_540k_part2.csv'#540kp2 csv path

IMG_PATH_660kP1 = '/root/SDIP/data/images660kp1/'#660kp1 image path
CSV_PATH_660kP1 = '/root/SDIP/data/prompts/sd2_meta_cc660k_part1.csv'#660kp1 csv path
IMG_PATH_660kP2 = '/root/SDIP/data/images660kp2/'#660kp2 image path
CSV_PATH_660kP2 = '/root/SDIP/data/prompts/sd2_meta_cc660k_part2.csv'#660kp2 csv path

IMG_PATH_Pszemraj = '/root/SDIP/data/imagesPszemraj/'#Pszemraj image path
CSV_PATH_Pszemraj = '/root/SDIP/data/prompts/sd2_meta_pszemraj.csv'#Pszemraj csv path

IMG_PATH_Hard = '/root/SDIP/data/imagesHard/'#Hard image path
CSV_PATH_Hard = '/root/SDIP/data/prompts/sd2_meta_hard_filter.csv'#Hard csv path

IMG_PATH_laincocop1 = '/root/SDIP/data/imagesLaioncocop1/'#Hard image path
CSV_PATH_laioncocop1 = '/root/SDIP/data/prompts/sd2_meta_laioncoco_p1.csv'#Hard csv path

IMG_PATH_laincocop2 = '/root/SDIP/data/imagesLaioncocop2/'#Hard image path
CSV_PATH_laioncocop2 = '/root/SDIP/data/prompts/sd2_meta_laioncoco_p2.csv'#Hard csv path

BS = 32
NEPOCH=25
run_name = f'train_convnext_xxlarge_littleboat_on3000k_size512'
FOLD = 0
#MODEL = 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
DEBUG = 0
if DEBUG:
    BS=64
from sklearn.model_selection import KFold
import pandas as pd
df = pd.read_csv(CSV_PATH)#.head(1000)
if DEBUG:
    df = df.head(100)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
for num, (train_index, val_index) in enumerate(kfold.split(df)):
    df.loc[val_index, 'fold'] = int(num)
df['fold'] = df['fold'].astype(int)
df = df[['prompt','image_name', 'fold']]
df['filepath'] = IMG_PATH+df['image_name']

trn_df = df[df['fold']!=FOLD].reset_index(drop=True)
val_df = df[df['fold']==FOLD].reset_index(drop=True)

##540k Dataset
df_540kp1 = pd.read_csv(CSV_PATH_540kP1)#.head(1000)
if DEBUG:
    df_540kp1 = df_540kp1.head(100)
df_540kp1['filepath'] = IMG_PATH_540kP1 + df_540kp1['image_name']
df_540kp1['fold'] = -1

df_540kp2 = pd.read_csv(CSV_PATH_540kP2)#.head(1000)
if DEBUG:
    df_540kp2 = df_540kp2.head(100)
df_540kp2['filepath'] = IMG_PATH_540kP2 + df_540kp2['image_name']
df_540kp2['fold'] = -1

##660k Dataset
#trn_df = pd.concat([trn_df, df_540k, df_540k2], ignore_index=True)
df_660kp1 = pd.read_csv(CSV_PATH_660kP1)#.head(1000)
if DEBUG:
    df_660kp1 = df_660kp1.head(100)
df_660kp1['filepath'] = IMG_PATH_660kP1+df_660kp1['image_name']
#df_660kp1['fold'] = -1

kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
for num, (train_index, val_index) in enumerate(kfold.split(df_660kp1)):
    df_660kp1.loc[val_index, 'fold'] = int(num)

trn_660kp1 = df_660kp1[df_660kp1['fold']!=FOLD].reset_index(drop=True)
val_660kp1 = df_660kp1[df_660kp1['fold']==FOLD].reset_index(drop=True)

df_660kp2 = pd.read_csv(CSV_PATH_660kP2)#.head(1000)
if DEBUG:
    df_660kp2 = df_660kp2.head(100)
df_660kp2['filepath'] = IMG_PATH_660kP2+df_660kp2['image_name']
df_660kp2['fold'] = -1

df_psze = pd.read_csv(CSV_PATH_Pszemraj)#.head(1000)
if DEBUG:
    df_psze = df_psze.head(100)
df_psze['filepath'] = IMG_PATH_Pszemraj+df_psze['image_name']
df_psze['fold'] = -1

df_hard = pd.read_csv(CSV_PATH_Hard)#.head(1000)
if DEBUG:
    df_hard = df_hard.head(100)
df_hard['filepath'] = IMG_PATH_Hard+df_hard['image_name']
df_hard['fold'] = -1

df_laionp1 = pd.read_csv(CSV_PATH_laioncocop1)#.head(1000)
if DEBUG:
    df_laionp1 = df_laionp1.head(100)
df_laionp1['filepath'] = IMG_PATH_laincocop1+df_laionp1['image_name']
df_laionp1['fold'] = -1

df_laionp2 = pd.read_csv(CSV_PATH_laioncocop2)#.head(1000)
if DEBUG:
    df_laionp2 = df_laionp2.head(100)
df_laionp2['filepath'] = IMG_PATH_laincocop2+df_laionp2['image_name']
df_laionp2['fold'] = -1

trn_df = pd.concat([trn_df, df_540kp1, df_540kp2, trn_660kp1, df_660kp2, df_psze, df_hard, val_660kp1, df_laionp1], ignore_index=True)
print(trn_df.shape)

import sys
#sys.path.append('../input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_vector = model.encode(list(trn_df["prompt"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=True)
val_vector = model.encode(list(val_df["prompt"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=True)
train_vector = train_vector.cpu()
val_vector = val_vector.cpu()

#val_vector_660kp1 = model.encode(list(val_660kp1["prompt"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=True)
#val_vector_660kp1 = val_vector_660kp1.cpu()

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from glob import glob
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoProcessor
import open_clip
from torchvision import transforms
def _convert_to_rgb(image):
    return image.convert('RGB')
#clip_processor = AutoProcessor.from_pretrained(MODEL)
clip_model, _, _ = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')#"layers": 32
clip_processor = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
BATCHSIZE=BS
SAVE_OPT_CKP = False
SAVE_MODEL_CKP = True
UNFREEZE_START = 24 # set it to lower number when significantly more samples are included.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.cuda.empty_cache()


import os
from copy import deepcopy
import random
import time
class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output


def get_train_test_split():
    """add your image paths and embedding labels here"""
    train_images, train_labels, test_images, test_labels = list(trn_df['filepath']), train_vector, list(val_df['filepath']), val_vector
    return train_images, train_labels, test_images, test_labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = clip_model
        self.vision = clip.visual
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)
        return self.fc(out)


def load_pretrained_model():
    model = Net()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to(device)


class IMGDataset:
    def __init__(self, image_paths, targets, clip_processor=clip_processor):
        self.images = image_paths
        self.labels = targets
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(image)
        target = self.labels[item]
        return image, target


if __name__ == "__main__":
    """main training"""
    Path(f"../{run_name}").mkdir(exist_ok=True)

    
    BestEpoch=0
    BestSim = 0
    train_images, train_targets, test_images, test_targets = get_train_test_split()
    #test_images2, test_targets2 = list(val_660kp1['filepath']), val_vector_660kp1
    print(f"test size: {len(test_images)}, train size: {len(train_images)}")

    nn_model = load_pretrained_model()
    #nn_model = torch.compile(nn_model)
    optimizer = optim.AdamW(nn_model.parameters(), lr=5e-5, weight_decay=1e-4, amsgrad=False)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(NEPOCH*0.5), T_mult=1, eta_min=1e-7, last_epoch=-1)
    optimizer.zero_grad()
    test_dataloader = DataLoader(dataset=IMGDataset(test_images, test_targets),
                                 batch_size=BATCHSIZE, shuffle=False, num_workers=8)
    #test_dataloader2 = DataLoader(dataset=IMGDataset(test_images2, test_targets2),
    #                             batch_size=BATCHSIZE, shuffle=False, num_workers=8)
    train_dataloader = DataLoader(dataset=IMGDataset(train_images, train_targets),
                                 batch_size=BATCHSIZE, shuffle=True, num_workers=8)
    #ema_model = ModelEma(nn_model, 0.9999)
    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch_data in enumerate(tqdm(train_dataloader)):
            batch_images, batch_targets = batch_data
            batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
            #batch_images['pixel_values'] = batch_images['pixel_values'][:, 0]
            pred = nn_model(batch_images)
            cosine_loss = cosine_similarity_loss(pred, batch_targets)
            loss = cosine_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += -cosine_loss.item()
            
            #ema_model.update(nn_model)
            
        epoch_loss /= len(train_dataloader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")
        scheduler.step()
        
        #valid
        epoch_loss = 0
        with torch.no_grad():
            for batch_images, batch_targets in tqdm(test_dataloader):
                batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                #batch_images['pixel_values'] = batch_images['pixel_values'][:, 0]
                
                pred = nn_model(batch_images)
                loss = -cosine_similarity_loss(pred, batch_targets)
                epoch_loss += loss.item()
            epoch_loss /= len(test_dataloader)
        print(f"epoch: {epoch}, 211k test loss: {epoch_loss}")
        
        #epoch_loss2 = 0
        #with torch.no_grad():
        #    for batch_images, batch_targets in tqdm(test_dataloader2):
        #        batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                #batch_images['pixel_values'] = batch_images['pixel_values'][:, 0]
                
        #        pred = nn_model(batch_images['pixel_values'][:, 0])
        #        loss = -cosine_similarity_loss(pred, batch_targets)
        #        epoch_loss2 += loss.item()
        #    epoch_loss2 /= len(test_dataloader2)
        #print(f"epoch: {epoch}, 660k test loss: {epoch_loss2}")
        
        all_loss = epoch_loss#(epoch_loss+epoch_loss2)/2
        if all_loss > BestSim:
            BestSim = all_loss
            BestEpoch = epoch
            print(f"save best model at {BestSim} with epoch {BestEpoch}")
            if SAVE_MODEL_CKP:
                torch.save(nn_model.state_dict(), f"/root/SDIP/save/{run_name}_f{FOLD}_cv{epoch_loss:.4f}_ep{epoch}_unfree{UNFREEZE_START}.pt")


        if epoch - 1 >= BestEpoch:
            print(f"early stop at {epoch+1} with best epoch {BestEpoch} and test similarity {BestSim}.")
            break
        if DEBUG or epoch==4:
            break
