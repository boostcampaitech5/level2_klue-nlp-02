import os
from tqdm.auto import tqdm
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import wandb

from models.models import ADVSModel
from utils import utils, data_controller
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """---Setting---"""
    # config 파일 불러오기
    wandb_logger = WandbLogger(name="Adversarial_Validation", project="Adversarial_Validation_Project")
    wandb_logger = WandbLogger()

    """---Train---"""
    # 데이터 로더와 모델 가져오기
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    special_tokens_list = utils.get_add_special_tokens()
    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens_list
    })
    dataloader = data_controller.ADVSDataloader(tokenizer)
    
    LM = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='klue/roberta-large', num_labels=2)
    LM.resize_token_embeddings(len(tokenizer))
    
    
    model = ADVSModel(LM)
    # check point
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 save_top_k=1,
                                 save_last=False,
                                 verbose=True,
                                 dirpath="./ADVS/checkpoints",
                                 filename="best-checkpoint",
                                 mode='min')
    callbacks = [checkpoint]
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   mode='min',
                                   verbose=True)
    callbacks.append(early_stopping)

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs= 100,
                         default_root_dir='./ADVS',
                         log_every_n_steps=1,
                         val_check_interval=0.5,           # 1 epoch 당 valid loss 2번 체크: 학습여부 빠르게 체크
                         logger=wandb_logger,
                         callbacks=callbacks)

    # Start training
    trainer.fit(model = model, datamodule = dataloader)

    predictions = trainer.predict(model=model, datamodule=dataloader)
    probs = []
    for prediction in predictions:
        probs.extend([prob[1] for prob in prediction])
    
    train_df = pd.read_csv('./dataset/new_train.csv')
    
    # Convert probs to a DataFrame
    probs_df = pd.DataFrame(probs, columns=['prob_1'])

    # Concatenate the original DataFrame with the probabilities DataFrame
    train_df = pd.concat([train_df.reset_index(drop=True), probs_df.reset_index(drop=True)], axis=1)

    # Save the new DataFrame to a csv file
    train_df.to_csv('./ADVS/new_train_ADVS.csv', index=False)