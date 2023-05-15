import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle

from utils import metrics


def focal_loss(logits, y, sub_obj_types, types2labelnum, alpha=0.5, gamma=2., penalty_scale=1):
    """
    logits = (batch_size, class_num)의 tensor
    y = (batch_size, )의 tensor
    sub_obj_types = [(subject_type_1, subject_type_2, ...), (object_type_1, object_type_2, ...)]의 튜플을 가지고 있는 list
    types2labelnum = key가 (sub_type, obj_type)이고, value가 가능한 label의 number를 set으로 가지고 있는 dictionary
    penalty_scale = penalty의 scale을 조정할 변수
    """
    
    # sub_obj_types를 batch_size개의 (subject_type, object_type) 튜플을 가지고 있는 list로 바꿔주기 위함
    sub_obj_types = [(sub_type, obj_type) for sub_type, obj_type in zip(*sub_obj_types)]
    
    # y를 one_hot represent함
    y_one_hot = torch.zeros_like(logits)  
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # y_ont_hot = (batch_size, class_num)

    # 원래 Focal Loss 논문에서 제시한 대로 구현한 basic_loss
    ce_loss = torch.nn.CrossEntropyLoss(reduction = 'none')(logits, y) # ce_loss (batch_size, )
    pt = torch.exp(-ce_loss)
    basic_loss = torch.mean(alpha * ((1-pt) ** gamma) * ce_loss)  # tensor([scalar])

    # Penalty 계산을 위한 binary cross entropy loss 계산 
    penalty_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_one_hot, reduction='none')  # penalty_loss = (batch_size, class_num)

    # 특정 (sub_type, obj_type) 조합에 부적합한 label에 기반하여 binary mask 구현
    mask = torch.zeros_like(logits)  # mask = (batch_size, class_num)
    for i, types in enumerate(sub_obj_types):
        valid_labels = types2labelnum[types]
        invalid_labels = set(range(logits.shape[1])) - valid_labels
        mask[i, list(invalid_labels)] = 1

    # mask와 penalty_loss를 element-wise multiply. 이후 총합이 될 loss에 더해준다.
    penalty_term = mask * penalty_loss
    loss = basic_loss + penalty_scale * penalty_term.mean()

    return loss


class Model(pl.LightningModule):
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM  # Language Model
        self.tokenizer = tokenizer  # Tokenizer
        self.sep_id = tokenizer.sep_token_id  # Separator token ID
        
        self.loss_func = eval("torch.nn." + self.CFG['train']['lossF'])()
        self.optim = eval("torch.optim." + self.CFG['train']['optim'])
        self.types2labelnum = load_types2labelnum() if self.CFG['train']['focal_loss'] else None
        self.lstm = torch.nn.LSTM(input_size = self.LM.config.hidden_size,
                                  hidden_size = self.LM.config.hidden_size,
                                  num_layers = 4,
                                  dropout = 0.1,
                                  batch_first = True,
                                  bidirectional = True) if self.CFG['train']['LSTM']['Do'] else None
        self.fc = torch.nn.Linear(self.LM.config.hidden_size * 2, 30) if self.CFG['train']['LSTM']['Do'] else None

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        if not self.CFG['train']['LSTM']['Do']:
            return outputs['logits']
        
        else:
            """
            batch 내의 각 sentence에 대해 LSTM을 수행.
            범위는 가장 앞 [CLS] 부터 처음으로 나오는 [SEP]까지만 수행하는 truncate와 문장 전체에 수행하는 not truncate가 있음
            """
            if not self.CFG['train']['LSTM']['truncate']:
                lstm_outputs, (hidden, cell) = self.lstm(outputs['hidden_states'][-1])
            else:
                # Find the position of the [SEP] token
                sep_positions = []
                for seq in input_ids:
                    sep_positions.append((seq == self.sep_id).nonzero(as_tuple=True)[0][0])

                # Prepare the sequences for LSTM
                lstm_input = []
                for i, pos in enumerate(sep_positions):
                    lstm_input.append(outputs["hidden_states"][-1][i, :pos])
                lstm_input = torch.nn.utils.rnn.pad_sequence(lstm_input, batch_first=True)

                # Pass the sequences through LSTM
                lstm_outputs, (hidden, cell) = self.lstm(lstm_input)

            # Concatenate the forward and backward hidden states
            output = torch.cat((hidden[0], hidden[1]), dim=1)

            # Pass the output through a fully connected layer
            logits = self.fc(output)

            return logits
            

    def training_step(self, batch, batch_idx):
        x, y, sub_obj_types = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(logits, y) if not self.CFG['train']['focal_loss'] \
        else focal_loss(logits, y, sub_obj_types, self.types2labelnum, self.CFG['train']['focal_loss_scale'])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
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
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.CFG['train']['LR'] / self.CFG['train']['LR_base'], max_lr=self.CFG['train']['LR'] / self.CFG['train']['LR_max'], step_size_up=self.CFG['train']['LR_step_up'], step_size_down=self.CFG['train']['LR_step_down'], cycle_momentum=False, mode='triangular')
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=3)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'LR_schedule'
        }

        return [optimizer], [lr_scheduler]
    

class ADVSModel(pl.LightningModule):
    def __init__(self, LM):
        super().__init__()
        self.save_hyperparameters()
        # 사용할 모델을 호출
        self.LM = LM  # Language Model
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW

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
        loss = self.loss_func(logits, y)

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

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        probs = F.softmax(logits, dim=-1)

        return probs.tolist()

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=0.00001)

        return [optimizer]
    
    
class TAPTModel(pl.LightningModule):
    def __init__(self, LM):
        super().__init__()
        self.LM = LM

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']  # Labels for MLM are the same as input_ids
        )
        loss = outputs['loss']  # Loss is calculated by the model
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        loss = outputs['loss']
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.000001)
        return [optimizer]

    
    
def load_types2labelnum():
    with open('./code/dict_types_to_labelnum.pkl', 'rb') as f:
        types2labelnum = pickle.load(f)
    
    return types2labelnum


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            step += 1
            if step < warmup_steps + 1:
                return float(step) / float(max(1.0, warmup_steps + 1))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)