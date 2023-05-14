import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import metrics


def Adaptive_Threshold_loss(logits, labels):
    """
    Note:       ALTOP - Adaptive Threshold loss를 계산합니다.
                TH label(기준)을 설정하여 Positive label logit은 더 높게, negative label logits은 더 낮도록 패널티를 줍니다.
                참고 링크: https://arxiv.org/pdf/2010.11304.pdf

    Arguments:
    logits: (batch, num_classes)
    labels: (batch, )
    
    Return:
    loss:   (batch, ), torch.tensor([scalar])
    """
    
    # 1. Threshold 라벨 설정: TH label = no_relation -> no_relation 보다는 logit 값이 높도록 설정
    labels = F.one_hot(labels, num_classes=logits.size(-1))
    threshold_label = torch.zeros_like(labels, dtype=torch.float).to(labels)   # (batch, num_classes)
    threshold_label[:, 0] = 1                   # TH label = no_relation
    labels[:, 0] = 0

    positive_mask = labels + threshold_label    # (batch, num_classes) -> 원래 label 과 같다.
    negative_mask = 1 - labels                  # (batch, num_classes) -> 정답 label이 아닌 다른 모든 label이 1로 설정된다.
    
    # 2. loss = l1 + l2
    logit1 = logits - (1 - positive_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
    
    logit2 = logits - (1 - negative_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * threshold_label).sum(1)
    
    # 3. Sum
    loss = loss1 + loss2
    loss = loss.mean()
    
    return loss


class Model(pl.LightningModule):
    def __init__(self, LM, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM  # Language Model
        self.loss_func = eval("torch.nn." + self.CFG['train']['lossF'])()
        self.optim = eval("torch.optim." + self.CFG['train']['optim'])

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        return outputs['logits']

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(logits, y) if not self.CFG['train']['Adaptive_Threshold_loss'] \
        else Adaptive_Threshold_loss(logits, y)
        
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)

        metric = metrics.compute_metrics(
            F.softmax(logits, dim=-1), y)
        self.log('val_micro_f1_Score', metric['micro f1 score'])
        self.log('val_AUPRC', metric['auprc'])
        self.log('val_acc', metric['accuracy'])

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        probs = F.softmax(logits, dim=-1)
        preds = np.argmax(probs.cpu().numpy(), axis=-1)

        return preds, probs.tolist()

    def configure_optimizers(self):
        if self.CFG['train']['optim'] == 'SGD':
            optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR'], momentum=0.9)
        else:
            optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR'])
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda epoch: 0.95 ** epoch,
        #     last_epoch=-1,
        #     verbose=False)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer=optimizer,
        #     step_size=10,
        #     gamma=0.7,
        #     verbose=True)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.CFG['train']['LR'] / self.CFG['train']['LR_base'], max_lr=self.CFG['train']['LR'] / self.CFG['train']['LR_max'], step_size_up=self.CFG['train']['LR_step_up'], step_size_down=self.CFG['train']['LR_step_down'], cycle_momentum=False, mode='triangular')

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'LR_schedule'
        }

        return [optimizer], [lr_scheduler]
