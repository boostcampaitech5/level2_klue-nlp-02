import os
import argparse
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from models.models import Model, TAPTModel
from utils import utils, data_controller
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from shutil import copyfile
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """---Setting---"""
    # config 파일 불러오기
    config_path = "results/05-16_16:35:49_기범_ainize-klue-bert-base-re-2/config.yaml" # 검증할 config 파일
    with open(config_path) as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
        
    # seed 설정
    pl.seed_everything(CFG['seed'])

    """---Train---"""
    # 데이터 로더와 모델 가져오기
    tokenizer = AutoTokenizer.from_pretrained(CFG['train']['model_name'])
    CFG['train']['special_tokens_list'] = utils.get_add_special_tokens()
    tokenizer.add_special_tokens({
        'additional_special_tokens': CFG['train']['special_tokens_list']
    })
    
    dataloader = data_controller.Dataloader(tokenizer, CFG)
    dataloader.setup()
    
    if CFG['train']['TAPT']:
        # Pretrain on combined train and test data (TAPT)
        if not os.path.exists("./tapt_model"):
            LM = AutoModelForMaskedLM.from_pretrained(CFG['train']['model_name'])
            tapt_tokenizer = AutoTokenizer.from_pretrained(CFG['train']['model_name'])
            tapt_dataloader = data_controller.TAPTDataloader(tapt_tokenizer)  # You'll need to implement this
            tapt_model = TAPTModel(LM)  # You'll need to implement twhis

            tapt_logger = WandbLogger(name="TAPT", project="TAPT")
            
            checkpoint_callback = ModelCheckpoint(
                        monitor='val_loss',
                        dirpath='./tapt_model',
                        filename='best_model-{epoch:02d}-{val_loss:.2f}',
                        save_top_k=1,
                        mode='min',
                    )
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           mode='min',
                                           verbose=True)

            tapt_trainer = pl.Trainer(accelerator='gpu',
                                      max_epochs=100,
                                      logger = tapt_logger,
                                      callbacks = [early_stopping, checkpoint_callback])
            tapt_trainer.fit(tapt_model, tapt_dataloader)
            tapt_model.LM.save_pretrained("./tapt_model")
            # Fine-tune on actual training data
        LM = AutoModelForSequenceClassification.from_pretrained("./tapt_model", num_labels=30)
            
    else:    
        LM = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=CFG['train']['model_name'], num_labels=30,
        output_hidden_states=True, output_attentions=True)
    
    # LM.resize_token_embeddings(len(tokenizer))
    # model = Model(LM, tokenizer, CFG)
    ckpt_path = "results/05-16_16:35:49_기범_ainize-klue-bert-base-re-2/checkpoints/epoch=4-val_micro_f1_Score=94.2629.ckpt"
    model = Model.load_from_checkpoint(ckpt_path)
    device = torch.device("cuda")
    
    """---evaluation---"""
    num2label = data_controller.load_num2label()

    predictions = []
    true_y = []
    for batch in dataloader.val_dataloader():
        preds, probs = model.confusion_matrix_inference(batch[0])
        predictions.extend(preds)
        true_y.extend(batch[1])
    
    print(len(predictions))
    print(len(true_y))

    cm = confusion_matrix(true_y, predictions)

    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('dataset/test.png')
