import numpy as np

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

def auprc(preds, labels):
    pass




def compute_metrics(outputs, y):
  """ validation을 위한 metrics function """
  outputs = outputs.cpu()
  y = y.cpu()
  labels = y
  preds = outputs.argmax(-1)
  probs = outputs

  # calculate accuracy using sklearn's function
  f1 = micro_f1_score(preds, labels)
  auprc = auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }