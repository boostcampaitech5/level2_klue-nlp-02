import yaml
import torch
import transformers
import pandas as pd
import pytorch_lightning as pl
import wandb

from models.models import Model
from utils import utils, data_controller
from transformers import AutoTokenizer, AutoModel
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
    wandb.init(name=folder_name, project="level2", entity=CFG['wandb']['id'], dir=save_path)
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(CFG)

    """---Train---"""
    # 데이터 로더와 모델 가져오기
    tokenizer = AutoTokenizer.from_pretrained(CFG['train']['model_name'])
    dataloader = data_controller.Dataloader(tokenizer, CFG)
    LM = AutoModel.from_pretrained(pretrained_model_name_or_path=CFG['train']['model_name'], num_labels=30)
    model = Model(LM, CFG)
    # check point
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                save_top_k=3,
                                save_last=True,
                                save_weights_only=True,
                                verbose=False,
                                filename='{epoch}-{val_loss:.4f}',
                                mode='min')
    # Earlystopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=CFG['train']['patience'], mode='min')
    # fit
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=CFG['train']['epoch'],
                         default_root_dir=save_path,
                         log_every_n_steps=1,
                         logger = wandb_logger,
                         callbacks = [checkpoint, early_stopping])
            
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    """---Inference---"""
    preds_y, probs_y = trainer.predict(model=model, datamodule=dataloader)

    """---save---"""
    # write yaml
    # with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
    #     yaml.dump(CFG, f)
    # save mode
    # torch.save(model, f'{save_path}/{folder_name}_model.pt')
    # save submit
    submit = pd.read_csv('./code/prediction/sample_submission.csv')
    submit['pred_label'] = preds_y
    submit['probs'] = probs_y
    submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)