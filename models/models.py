import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import metrics


class Model(pl.LightningModule):
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM                    # Language Model
        self.tokenizer = tokenizer      # Tokenizer
        self.loss_func = eval("torch.nn." + self.CFG['train']['lossF'])()
        self.optim = eval("torch.optim." + self.CFG['train']['optim'])
        
        # entity type 분류 레이어 추가 - multi task 를 위한 세팅
        if "add_entity_tokens_base" in self.CFG['select_DC'] and self.CFG['train']['type_classify']:
            self.type_classify_init()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        return {'logits': outputs['logits'],
                'hidden_states': outputs['hidden_states'],
                'attentions': outputs['attentions']
        }

    def training_step(self, batch, batch_idx):
        x, y, _type = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(logits['logits'], y)
        
        if "add_entity_tokens_base" in self.CFG['select_DC'] and self.CFG['train']['type_classify']:
            preds = self.type_classify_forward(x, logits['hidden_states'])
            loss += self.type_classify_cal_loss(preds, _type, alpha=0.5)
            
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _type = batch
        logits = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        loss = self.loss_func(logits['logits'], y)
        self.log("val_loss", loss)

        metric = metrics.compute_metrics(
            F.softmax(logits['logits'], dim=-1), y)
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
        probs = F.softmax(logits['logits'], dim=-1)
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

    def type_classify_init(self):
        """
        Note:   문장 내 head [ENT], tail [ENT]를 가지고 type classify를 진행하는 multi task learning을 수행합니다.
                type classify를 진행하기 위한 initialize를 진행합니다.
        """
        
        self.type_list = ['PER', 'NOH', 'ORG', 'LOC', 'POH', 'DUR', 'PNT', 'TIM', 'MNY', 'DAT']
        self.type_dict = {key:value for value, key in enumerate(self.type_list)}
        
        self.dropout = torch.nn.Dropout(0.1)
        self.pooler_layer = torch.nn.Linear(768, len(self.type_list))
        
        self.type_classify_loss = eval("torch.nn." + self.CFG['train']['lossF'])()
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
        
        ENT_TOKEN_ID = self.tokenizer.convert_tokens_to_ids('[ENT]')
        
        # 1. Batch 각각에 대해 ENT token의 위치를 [first ent idx, second ent idx] 형태로 저장.
        ent_indices = []
        for i in range(len(x['input_ids'])):
            ent_indices_batch = torch.where(x['input_ids'][i] == ENT_TOKEN_ID)[0]
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
        
    def type_classify_cal_loss(self, preds, _type, alpha=0.5):
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
        
        loss_head = self.type_classify_loss(preds_head, _type[0])
        loss_tail = self.type_classify_loss(preds_tail, _type[1])
        
        type_loss = (loss_head + loss_tail)*alpha
        self.log('type_class_loss', type_loss)
        
        return type_loss