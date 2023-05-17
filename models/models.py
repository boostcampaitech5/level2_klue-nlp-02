import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle

from utils import metrics
from utils.data_controller import load_types2labelnum
from . import train

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
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM                            # Language Model
        self.tokenizer = tokenizer              # Tokenizer
        self.sep_id = tokenizer.sep_token_id    # Separator token ID
        self.ent_id = tokenizer.convert_tokens_to_ids('[ENT]')
        
        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing = self.CFG['train']['lossF']['smooth_scale'])
        self.optim = eval("torch.optim." + self.CFG['train']['optim'])
        
        # LSTM layer 초기화
        self.types2labelnum = load_types2labelnum() if self.CFG['train']['lossF']['name'] == 'focal_loss' else None
        self.lstm = torch.nn.LSTM(input_size = self.LM.config.hidden_size,
                                  hidden_size = self.LM.config.hidden_size,
                                  num_layers = 4,
                                  dropout = 0.1,
                                  batch_first = True,
                                  bidirectional = True) if self.CFG['train']['LSTM']['Do'] else None
        self.fc = torch.nn.Linear(self.LM.config.hidden_size * 2, 30) if self.CFG['train']['LSTM']['Do'] else None
        
        # entity type 분류 레이어 추가 - multi task 를 위한 세팅
        if self.CFG['select_DC'] is not None and "add_entity_tokens_base" in self.CFG['select_DC'] and self.CFG['train']['type_classify']:
            self.type_classify_init()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        if self.CFG['train']['LSTM']['Do']:
            """
            batch 내의 각 sentence에 대해 LSTM을 수행.
            범위는 가장 앞 [CLS] 부터 처음으로 나오는 [SEP]까지만 수행하는 truncate와 문장 전체에 수행하는 not truncate가 있음
            """
            if not self.CFG['train']['LSTM']['truncate']:
                lstm_outputs, (hidden, cell) = self.lstm(outputs['hidden_states'][-1])  # outputs['hidden_states'][-1] == (batch_size * seq_len * hidden_dim)
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
            outputs['logits'] = self.fc(output)
        
        return outputs

    def training_step(self, batch, batch_idx):
        x, y, sub_obj_types = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(outputs['logits'], y) if self.CFG['train']['lossF']['name'] != 'focal_loss' else\
            Adaptive_Threshold_loss(outputs['logits'], y) if self.CFG['train']['lossF']['name'] == 'Adaptive_Threshold_loss' else\
            focal_loss(outputs['logits'], y, sub_obj_types, self.types2labelnum, self.CFG['train']['lossF']['focal_loss_scale'])

        # multi-task learning: type classify 학습
        if self.CFG['select_DC'] is not None and "add_entity_tokens_base" in self.CFG['select_DC'] and self.CFG['train']['type_classify']:
            preds = self.type_classify_forward(x, outputs['hidden_states'])
            loss += self.type_classify_cal_loss(preds, sub_obj_types, alpha=0.5)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if not self.CFG['train']['no_valid']:
            x, y, _ = batch
            outputs = self(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                token_type_ids=x['token_type_ids']
            )
            loss = self.loss_func(outputs['logits'], y)
            self.log("val_loss", loss)

            metric = metrics.compute_metrics(
                F.softmax(outputs['logits'], dim=-1), y)
            self.log('val_micro_f1_Score', metric['micro f1 score'])
            self.log('val_AUPRC', metric['auprc'])
            self.log('val_acc', metric['accuracy'])

            return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        probs = F.softmax(outputs['logits'], dim=-1)
        preds = np.argmax(probs.cpu().numpy(), axis=-1)

        return preds, probs.tolist()

    def configure_optimizers(self):
        if self.CFG['train']['optim'] == 'SGD':
            optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR']['lr'], momentum=0.9,weight_decay=1e-7)
        else:
            optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR']['lr'])

        if self.CFG['train']['LR']['name'] == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda epoch: 0.95 ** epoch,
                last_epoch=-1,
                verbose=False)
        elif self.CFG['train']['LR']['name'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=5,
                gamma=0.3,
                verbose=True)
        elif self.CFG['train']['LR']['name'] == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.CFG['train']['LR']['lr'] / self.CFG['train']['LR']['base'], 
                                                          max_lr=self.CFG['train']['LR']['lr'] / self.CFG['train']['LR']['max'], 
                                                          step_size_up=self.CFG['train']['LR']['step_up'], 
                                                          step_size_down=self.CFG['train']['LR']['step_down'], cycle_momentum=False, mode='exp_range')
        elif self.CFG['train']['LR']['name'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5,verbose=True)
        elif self.CFG['train']['LR']['name'] == 'WarmupConstantLR':
            scheduler = train.WarmupConstantLR(optimizer, warmup_steps=self.CFG['train']['LR']['warmupconstantLR_step'])
        elif self.CFG['train']['LR']['name'] == 'WarmupDecayLR':
            scheduler = train.WarmupDecayLR(optimizer, warmup_steps=self.CFG['train']['LR']['warmupdecayLR_warmup'], total_steps=self.CFG['train']['LR']['warmupdecayLR_total'])

        
        lr_scheduler = {
            'scheduler': scheduler,
            'interval' : self.CFG['train']['LR']['interval'],
            'name': self.CFG['train']['LR']['name']
        }

        return [optimizer], [lr_scheduler]

    def type_classify_init(self):
        """
        Note:   문장 내 head [ENT], tail [ENT]를 가지고 type classify를 진행하는 multi task learning을 수행합니다.
                type classify를 진행하기 위한 initialize를 진행합니다.
        """
        
        self.type_list = ['PER', 'NOH', 'ORG', 'LOC', 'POH', 'DUR', 'PNT', 'TIM', 'MNY', 'DAT']
        self.type_dict = {key:value for value, key in enumerate(self.type_list)}
        
        self.dropout = torch.nn.Dropout(0.1)
        self.pooler_layer = torch.nn.Linear(768, len(self.type_list))
        
        self.type_classify_loss = torch.nn.CrossEntropyLoss()
        # self.type_classify_optim = eval("torch.optim." + self.CFG['train']['optim']) 
        
    def type_classify_forward(self, x, hidden_state):
        """
        Note:   tokenizing된 문장과 model의 output을 가지고 head [ENT], tail [ENT]의 type을 예측합니다.

        Arguments:
        x: Dict({'input_ids', 'attention_mask', 'token_type_ids})
        hidden_state: (batch, num_seq, hidden_state)
            
        Return:
        entity_outs: Tuple((batch, num_types), (batch, num_types))
        """
        
        # 1. Batch 각각에 대해 ENT token의 위치를 [first ent idx, second ent idx] 형태로 저장.
        ent_indices = []
        for i in range(len(x['input_ids'])):
            ent_indices_batch = torch.where(x['input_ids'][i] == self.ent_id)[0]
            # 1-1. 입력문장이 길 경우 ENT가 2개 마킹되지 않고 truncated 되는 경우에 대한 예외처리
            if ent_indices_batch.size(-1) == 2:
                ent_indices.append(ent_indices_batch)
            else:
                ent_indices.append(torch.tensor([0, 0]).to('cuda:0'))
        ent_indices = torch.stack(ent_indices, axis=0)                      # (batch, 2)
    
        # 2. ENT 토큰의 hidden_state를 추출하기 위한 작업
        ent_head_h, ent_tail_h = [], []
        for i in range(len(x['input_ids'])):                                # 각 batch에 대하여
            ent_head_h.append(hidden_state[-1][i, ent_indices[i, 0], :])    # head entity token idx에 해당하는 hidden_state
            ent_tail_h.append(hidden_state[-1][i, ent_indices[i, 1], :])
        entity_head_hidden = torch.stack(ent_head_h, axis=0)       
        entity_tail_hidden = torch.stack(ent_tail_h, axis=0)                    # (batch, hidden_state=768)

        # 3. ENT 토큰의 hidden_state -> PER, ORG, ... 등등의 type 으로 분류
        entity_head_out = self.pooler_layer(self.dropout(entity_head_hidden))
        entity_tail_out = self.pooler_layer(self.dropout(entity_tail_hidden))   # (batch, num_types)
        
        return (entity_head_out, entity_tail_out)
        
    def type_classify_cal_loss(self, preds, _type, alpha=0.8):
        """
        Note:   head [ENT], tail [ENT]로부터 분류한 각각의 entity type 에 대한 loss를 계산합니다.

        Arguments:
        preds: Tuple(head, tail), head와 tail이 예측한 preds 값. 각각 (batch, num_types)의 형태를 갖습니다.
        _type: Tuple(head type, tail type), 각각 (batch, )의 형태를 갖습니다.
            
        Return:
        loss: head와 tail 각각의 type 분류에 대한 CrossEntropyLoss 값
        """
        
        preds_head = preds[0]       # (batch, num_types)
        preds_tail = preds[1]
        
        # entity type 분류 모델 학습을 위한 type 데이터 생성: string -> number(using dict)
        type_list = ['PER', 'NOH', 'ORG', 'LOC', 'POH', 'DUR', 'PNT', 'TIM', 'MNY', 'DAT']
        type_dict = {key:value for value, key in enumerate(type_list)}
        
        target_head = torch.tensor([type_dict[i] for i in _type[0]]).type(torch.LongTensor).to('cuda:0')
        target_tail = torch.tensor([type_dict[i] for i in _type[1]]).type(torch.LongTensor).to('cuda:0')
        loss_head = self.type_classify_loss(preds_head, target_head)
        loss_tail = self.type_classify_loss(preds_tail, target_tail)
        
        type_loss = (loss_head + loss_tail)*alpha
        self.log('type_class_loss', type_loss)
        
        return type_loss

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