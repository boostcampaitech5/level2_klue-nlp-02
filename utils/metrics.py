import numpy as np
import torch
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def micro_f1_score(preds, labels):
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")  # print : 0   / no relation은 계산에서 제외한다.
    label_indices = list(range(len(label_list)))     # print : [0,1, ..., 29]
    label_indices.remove(no_relation_label_idx)     # print : [1, ..., 29]
    f1 = f1_score(labels, preds, average="micro", labels=label_indices)
    f1 *= 100.0    # 확률로 바꿔주기 위한 100 곱셈

    return f1


def auprc(probs, labels):
    """ KLUE-RE AUPRC
      args:
        probs:  (batch, num_labels)
        labels: (batch,)
        
      return:
        float number
    """
    labels = np.eye(30)[labels] # label에 대한 one-hot encoding으로 변환 [0, 0, 1, 0, ... 0]. 리턴사이즈=(batch, num_label) -> prob과 사이즈 맞추기 위함. classification으로 각 라벨에 대한 에러 계산을 위한 것
    score = np.zeros((30,))     # label 각각에 대한 점수 기록을 위한 빈 numpy
    
    for c in range(30):         # x축 recall, y축 precision 에서 면적 계산=auprc. 각 라벨별(c)로 auprc를 계산하고 평균낸다.
        targets_c = labels.take([c], axis=1).ravel()    # axis=1, c 번째 column을 가져와서 1차원 array로 만듦(ravel). 사이즈 = (batch, )
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)   # 배치 전체에 대한 계산 metric
        score[c] = sklearn.metrics.auc(recall, precision)                                   # 라벨 c 에 대해서 batch 전체를 가지고 auc 계산.
    
    return np.average(score) * 100.0  # 배치 전체에 대해 계산했던 걸 다시 전체 평균


def compute_metrics(probs, preds, y):
  """ validation을 위한 metrics function """
  y = y.cpu()

  # calculate accuracy using sklearn's function
  f1 = micro_f1_score(preds, y)
  # aup = auprc(probs, y)
  acc = accuracy_score(y, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : 0,
      'accuracy': acc,
  }
