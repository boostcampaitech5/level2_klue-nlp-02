import os
from tqdm.auto import tqdm
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import wandb

from models.models import Model
from utils import utils, data_controller
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """---Setting---"""
    # config 파일 불러오기
    with open('./use_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    # 실험 결과 파일 생성 및 폴더명 가져오기
    folder_name, save_path = utils.get_folder_name(CFG)
    # seed 설정
    pl.seed_everything(CFG['seed'])
    # wandb 설정
    wandb_logger = wandb.init(
        name=folder_name, project="level2", entity=CFG['wandb']['id'], dir=save_path)
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(CFG)

    """---Train---"""
    # 데이터 로더와 모델 가져오기
    tokenizer = AutoTokenizer.from_pretrained(CFG['train']['model_name'])
    special_tokens_list = utils.get_add_special_tokens()
    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens_list
    })
    
    dataloader = data_controller.Dataloader(tokenizer, CFG)
    LM = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=CFG['train']['model_name'], num_labels=30, output_hidden_states=CFG['train']['LSTM']['Do'])
    LM.resize_token_embeddings(len(tokenizer))
    model = Model(LM, tokenizer, CFG)
    # check point
    checkpoint = ModelCheckpoint(monitor='val_micro_f1_Score',
                                 save_top_k=CFG['train']['save_top_k'],
                                 save_last=False,
                                 save_weights_only=True,
                                 verbose=True,
                                 dirpath=f"{save_path}/checkpoints",
                                 filename="{epoch}-{val_micro_f1_Score:.4f}",
                                 mode='max')
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint,lr_monitor]
    # Earlystopping
    if CFG['option']['early_stop']:
        early_stopping = EarlyStopping(
            monitor='val_micro_f1_Score', patience=CFG['train']['patience'], mode='max', verbose=True)
        callbacks.append(early_stopping)
    # fit
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=CFG['train']['epoch'],
                         default_root_dir=save_path,
                         log_every_n_steps=1,
                         val_check_interval=0.5,           # 1 epoch 당 valid loss 2번 체크: 학습여부 빠르게 체크
                         logger=wandb_logger,
                         callbacks=callbacks)

    trainer.fit(model=model, datamodule=dataloader)

    """---Inference---"""
    def inference_model(model, dataloader):
        predictions = trainer.predict(model=model, datamodule=dataloader)

        num2label = data_controller.load_num2label()
        pred_label, probs = [], []
        for prediction in predictions:
            for pred in prediction[0]:
                pred_label.append(num2label[pred])
            for prob in prediction[1]:
                probs.append(list(map(float, prob)))

        return pred_label, probs

    pred_label, probs = inference_model(model, dataloader)

    """---save---"""
    # save submit
    submit = pd.read_csv('./code/prediction/sample_submission.csv')

    utils.save_csv(submit, pred_label, probs, save_path, folder_name)

    for ckpt_name in tqdm(os.listdir(f"{save_path}/checkpoints"), desc="inferencing_ckpt"):
        print("Now...  "+ ckpt_name)
        checkpoint = torch.load(f"{save_path}/checkpoints/{ckpt_name}")
        model.load_state_dict(checkpoint['state_dict'])

        pred_label, probs = inference_model(model, dataloader)
        utils.save_csv(submit, pred_label, probs, save_path, folder_name, ckpt_name.split('=')[-1][:7])
