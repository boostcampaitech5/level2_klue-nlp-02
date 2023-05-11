import pickle
import torch
import pandas as pd
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        inputs = {key: val[idx].clone().detach()
                  for key, val in self.inputs.items()}

        if self.targets:
            targets = torch.tensor(self.targets[idx])

            return inputs, targets
        else:
            return inputs

    def __len__(self):
        return len(self.inputs['input_ids'])


class Dataloader(pl.LightningDataModule):
    """
    원본 데이터를 불러와 전처리 후 Dataloader 만들어 Dataset에 넘겨 최종적으로 사용할 데이터셋 만들기
    """

    def __init__(self, tokenizer, CFG):
        super(Dataloader, self).__init__()
        self.CFG = CFG
        self.tokenizer = tokenizer
        train_x, train_y, predict_x = load_data()
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                          stratify=train_y,
                                                          test_size=self.CFG['train']['test_size'],
                                                          shuffle=self.CFG['train']['shuffle'],
                                                          random_state=self.CFG['seed'])
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.predict_x = predict_x
        self.label2num = load_label2num()

        self.train_dataset = None
        self.val_dataset = None  # val == test
        self.test_dataset = None
        self.predict_dataset = None

    def tokenizing(self, x):
        """ 토크나이징 함수

        Note:   두 entity를 [SEP]토큰으로 이어붙인 리스트와
                원래 x['sentence'] 리스트를 토크나이저에 인자로 집어넣습니다.
                inputs는 따라서 input_ids, attention_mask, token_type_ids가 각각 포함된 배열형태로 구성됩니다.

        Arguments:
        x: pd.DataFrame

        Returns:
        inputs: Dict({'input_ids', 'token_type_ids', 'attention_mask'}), 각 tensor(num_data, max_length)
        """
        concat_entity = []
        for sub_ent, obj_ent in zip(x['subject_entity'], x['object_entity']):
            concat_entity.append(obj_ent + " [SEP] " + sub_ent)

        inputs = self.tokenizer(
            concat_entity,
            list(x['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.CFG['train']['token_max_len'],
            add_special_tokens=True,
        )

        return inputs

    def preprocessing(self, x, y=[], train=False):
        DC = DataCleaning(self.CFG['select_DC'])
        DA = DataAugmentation(self.CFG['select_DA'])
        
        if type(y) != list:
            x = pd.concat([x, y], axis = 1)
            
        x = DC.process(x)
        # 데이터 증강
        if train:
            x = DA.process(x)
            y = x['label']

        # 텍스트 데이터 토큰화
        inputs = self.tokenizing(x)
        targets = [self.label2num[label] for label in y]

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            breakpoint()
            train_inputs, train_targets = self.preprocessing(
                self.train_x, self.train_y, train=True)
            self.train_dataset = Dataset(train_inputs, train_targets)
            
            val_inputs, val_targets = self.preprocessing(self.val_x, self.val_y)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가 데이터 호출
            predict_inputs, predict_targets = self.preprocessing(
                self.predict_x)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=self.CFG['train']['shuffle'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.CFG['train']['batch_size'])

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.CFG['train']['batch_size'])


class DataCleaning():
    """
    config select DC에 명시된 Data Cleaning 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list

    def process(self, df):
        if self.select_list:
            for method_name in self.select_list:
                method = eval("self." + method_name)
                df = method(df)

        return df

    """
    data cleaning 코드
    """

    def entity_parsing(self, df):
        """ entity에서 word, start_idx, end_idx, type 분리하기
        Note: <데이터 예시>
            subject_entity : {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}
            object_entity : {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}
            
        Arguments:
        df: Cleaning을 수행하고자 하는 DataFrame
        
        Return:
        df: Cleaning 작업이 완료된 DataFrame
        """
        for type_entity in ['subject', 'object']:
            column = f"{type_entity}_entity"

            word_list, type_list = [], []
            start_idx_list, end_idx_list = [], []

            for i in range(len(df)):
                dictionary = eval(df.iloc[i][column])

                word_list.append(dictionary['word'])
                start_idx_list.append(dictionary['start_idx'])
                end_idx_list.append(dictionary['end_idx'])
                type_list.append(dictionary['type'])

            df[column] = word_list
            for key in ['start_idx', 'end_idx', 'type']:
                df[f"{type_entity}_{key}"] = eval(f"{key}_list")
        
        return df
    
    def normalize_class(self, df):
        """ 특정 label에 대해 들어올 수 없는 entity_type이 기재되어있는 경우,
        해당 type을 올바른 entity_type으로 바꾸어 주고자 함.
        
        Note: 어떤 label에 대해 어떤 entity_type이 허용되는지는 CJW_vis.ipynb 참고바람

        Arguments:
        df: (entity_parsing이 끝난) normalize_class을 수행하고자 하는 DataFrame

        Return:
        df: normalize_class 작업이 완료된 DataFrame
        """

        if 'labels' not in df.columns:
            return df
        allowed_obj_for_class = \
        {'org:founded_by' : {'allowed' : set(['PER', 'ORG']), 'change_into' : 'ORG'},
        'org:member_of' : {'allowed' : set(['ORG', 'LOC', 'POH']), 'change_into' : 'ORG'},
        'org:members' : {'allowed' : set(['ORG', 'LOC', 'POH']), 'change_into' : 'ORG'},
        'org:place_of_headquarters' : {'allowed' : set(['ORG', 'LOC', 'POH']), 'change_into' : 'LOC'},
        'org:political/religious_affiliation' : {'allowed' : set(['ORG', 'LOC', 'POH']), 'change_into' : 'ORG'},
        'org:product' : {'allowed' : set(['ORG', 'LOC', 'POH', 'PER']), 'change_into' : 'POH'},
        'org:top_members/employees' : {'allowed' : set(['ORG', 'LOC', 'POH', 'PER']), 'change_into' : 'PER'},
        'per:alternate_names' : {'allowed' : set(['ORG','POH', 'PER']), 'change_into' : 'PER'},
        'per:children' : {'allowed' : set(['PER']), 'change_into' : 'PER'},
        'per:colleagues' : {'allowed' : set(['PER', 'POH', 'ORG']), 'change_into' : 'PER'},
        'per:date_of_birth' : {'allowed' : set(['DAT']), 'change_into' : 'DAT'},
        'per:date_of_death' : {'allowed' : set(['DAT']), 'change_into' : 'DAT'},
        'per:employee_of' : {'allowed' : set(['PER', 'POH', 'ORG', 'LOC']), 'change_into' : 'ORG'},
        'per:origin' : {'allowed' : set(['LOC', 'POH', 'DAT', 'ORG']), 'change_into' : 'ORG'},
        'per:other_family' : {'allowed' : set(['PER', 'POH']), 'change_into' : 'POH'},
        'per:parents' : {'allowed' : set(['PER']), 'change_into' : 'PER'},
        'per:place_of_birth' : {'allowed' : set(['LOC']), 'change_into' : 'LOC'},
        'per:place_of_death' : {'allowed' : set(['LOC']), 'change_into' : 'LOC'},
        'per:place_of_residence' : {'allowed' : set(['LOC', 'ORG', 'POH', 'DAT']), 'change_into' : 'LOC'},
        'per:product' : {'allowed' : set(['POH']), 'change_into' : 'POH'},
        'per:religion' : {'allowed' : set(['POH', 'ORG']), 'change_into' : 'ORG'},
        'per:siblings' : {'allowed' : set(['PER']), 'change_into' : 'PER'},
        'per:spouse' : {'allowed' : set(['PER']), 'change_into' : 'PER'},
        'per:title' : {'allowed' : set(['PER', 'POH']), 'change_into' : 'POH'}
        }
            
        for label, obj_info in allowed_obj_for_class.items():
            df.loc[(df['label'] == label) & (~df['object_type'].isin(obj_info['allowed'])), 'object_type'] = obj_info['change_into']

        df.loc[df['label'].str.startswith('per'), 'subject_type'] = 'PER'
        df.loc[df['label'].str.startswith('org'), 'subject_type'] = 'ORG'
                
        return df


class DataAugmentation():
    """
    config select DA에 명시된 Data Augmentation 기법을 적용시켜주는 클래스
    """

    def __init__(self, select_list):
        self.select_list = select_list

    def process(self, df):
        if self.select_list:
            aug_df = pd.DataFrame(columns=df.columns)

            for method_name in self.select_list:
                method = eval("self." + method_name)
                aug_df = pd.concat([aug_df, method(df)])

            df = pd.concat([df, aug_df])

        return df
    
    """
    data augmentation 코드
    """
    
    def swap_sentence(self, df):
        """ subject/object swap을 통해 데이터 증강이 가능.
        그러므로 swap이 가능한 문장은 swap을 수행, 데이터를 증강하고자 함
        Note: <swap이 가능한 문장의 classes> {
            colleagues -> colleagues
            other_family -> other_family
            siblings -> siblings
            spouse -> spouse

            alternate_names -> alternate_names
            ------------------------------
            children <-> parents
            members <-> member_of

            top_members/employees <-> employee_of
            }

        Arguments:
        df: (entity_parsing이 끝난) 문장 swap augmentation을 수행하고자 하는 DataFrame

        Return:
        df: augmentation 작업이 완료된 DataFrame
        """
        
        auto_augmentation = set(['org:alternate_names',
                                'per:alternate_names',
                                'per:colleagues',
                                'per:other_family',
                                'per:siblings',
                                'per:spouse'])
        cross_augmentation = {'per:children' : 'per:parents',
                            'per:parents' : 'per:children',
                            'org:members' : 'org:member_of',
                            'org:member_of' : 'org:members',
                            'org:top_members/employees' : 'per:employee_of',
                            'per:employee_of' : 'org:top_members/employees'}
        auto_df = df[df['label'].isin(auto_augmentation)]
        cross_df = df[df['label'].isin(cross_augmentation.keys())]

        auto_df['subject_entity'], auto_df['object_entity'] = auto_df['object_entity'].copy(), auto_df['subject_entity'].copy()

        cross_df['subject_entity'], cross_df['object_entity'] = cross_df['object_entity'].copy(), cross_df['subject_entity'].copy()
        cross_df['label'] = cross_df['label'].map(cross_augmentation)

        return pd.concat([auto_df, cross_df], ignore_index=True)


def load_data():
    """
    학습 데이터와 테스트 데이터 DataFrame 가져오기
    """
    train_df = pd.read_csv('./dataset/new_train.csv')
    train_df.drop(['id', 'source'], axis=1, inplace=True)
    train_x = train_df.drop(['label'], axis=1)
    train_y = train_df['label']
    test_x = pd.read_csv('./dataset/new_test.csv')
    test_x.drop(['id', 'source'], axis=1, inplace=True)
    
    return train_x, train_y, test_x


def load_label2num():
    with open('./code/dict_label_to_num.pkl', 'rb') as f:
        label2num = pickle.load(f)
    
    return label2num


def load_num2label():
    with open('./code/dict_num_to_label.pkl', 'rb') as f:
        num2label = pickle.load(f)
    
    return num2label


if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/train/train.csv')
    test_df = pd.read_csv('./dataset/test/test_data.csv')

    # entity_parsing이 적용된 DataFrame 파일 만들기
    new_train_df = DataCleaning([]).entity_parsing(train_df.copy(deep=True))
    new_test_df = DataCleaning([]).entity_parsing(test_df.copy(deep=True))

    new_train_df.to_csv('./dataset/new_train.csv', index=False)
    new_test_df.to_csv('./dataset/new_test.csv', index=False)