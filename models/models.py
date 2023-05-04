import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

# import sys
# sys.path.append("/opt/ml/utils")
from utils import metrics

class Model(pl.LightningModule):
    def __init__(self, LM, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG
        
        # 사용할 모델을 호출
        self.LM = LM # Language Model
        self.loss_func = eval("torch.nn." + self.CFG['train']['lossF'])()
        self.optim = eval("torch.optim." + self.CFG['train']['optim'])

    def forward(self, input_ids, attention_mask, token_type_ids): 
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1] # [CLS] token에 대한 출력

        return pooled_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(outputs, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(outputs, y)
        self.log("val_loss", loss)

        metric = metrics.compute_metrics(outputs, y)
        self.log('val_micro_f1_Score', metric['micro f1 score'])
        self.log('val_AUPRC', metric['auprc'])
        self.log('val_acc', metric['accuracy'])

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )

        metric = metrics.compute_metrics(outputs, y)
        self.log('test_micro_f1_Score', metric['micro f1 score'])
        self.log('test_AUPRC', metric['auprc'])
        self.log('test_acc', metric['accuracy'])

    def predict_step(self, batch, batch_idx):
        x = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        logits = outputs[0]
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        return preds, probs

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR'])
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda epoch: 0.95 ** epoch,
        #     last_epoch=-1,
        #     verbose=False)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.7,
            verbose=True)

        lr_scheduler = {
        'scheduler': scheduler,
        'name': 'LR_schedule'
        }

        return [optimizer]#, [lr_scheduler]