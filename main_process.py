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
    wandb_logger = WandbLogger(save_dir=save_path) # 체크포인트 저장을 위한 경로 지정
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
        pretrained_model_name_or_path=CFG['train']['model_name'], num_labels=30)
    LM.resize_token_embeddings(len(tokenizer))
    model = Model(LM, CFG)
    # check point
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 save_top_k=3,
                                 save_last=True,
                                 save_weights_only=True,
                                 verbose=False,
                                 filename='{epoch}-{val_loss:.4f}',
                                 mode='min')
    callbacks = [checkpoint]
    # Earlystopping
    if CFG['option']['early_stop']:
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=CFG['train']['patience'], mode='min', verbose=True)
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
    predictions = trainer.predict(model=model, datamodule=dataloader)
    
    num2label = data_controller.load_num2label()
    pred_label, probs = [], []
    for prediction in predictions:
        for pred in prediction[0]:
            pred_label.append(num2label[pred])
        for prob in prediction[1]:
            probs.append(list(map(float, prob)))

    """---save---"""
    # write yaml
    # with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
    #     yaml.dump(CFG, f)
    # save mode
    # torch.save(model, f'{save_path}/{folder_name}_model.pt')
    # save submit
    submit = pd.read_csv('./code/prediction/sample_submission.csv')
    submit['pred_label'] = pred_label
    submit['probs'] = probs
    submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)