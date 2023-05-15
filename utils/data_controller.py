import re
import pickle
import torch
import pandas as pd
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
# from konlpy.tag import Okt
# from pykospacing import Spacing
# from hanspell import spell_checker
# from pororo import Pororo
# from hangulize import hangulize


class Dataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, inputs, targets=[], sub_obj_types=[], no_valid = False):
        self.inputs = inputs
        self.targets = targets
        self.sub_obj_types = sub_obj_types
        self.no_valid = no_valid

    def __getitem__(self, idx):
        if self.no_valid:
            return None, None, None
        
        inputs = {key: val[idx].clone().detach()
                  for key, val in self.inputs.items()}

        if self.targets:
            targets = torch.tensor(self.targets[idx])
            sub_obj_types = tuple(self.sub_obj_types.iloc[idx])
            
            return inputs, targets, sub_obj_types
        else:
            return inputs
        
    def __len__(self):
        if self.no_valid:
            return 1
        
        return len(self.inputs['input_ids'])
        
class ADVSDataset(Dataset):
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
    
class TAPTDataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        inputs = {key: val[idx].clone().detach()
                  for key, val in self.inputs.items()}
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
        
        train_df, predict_x = load_data()
        self.train_df = train_df
        self.predict_x = predict_x
        self.label2num = load_label2num()

        self.train_dataset = None
        self.val_dataset = None
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
    
    def short_tokenizing(self, x):
        """ 토크나이징 함수

        Note:   원래 x['sentence'] 리스트를 토크나이저에 인자로 집어넣습니다.
                inputs는 따라서 input_ids, attention_mask, token_type_ids가 각각 포함된 배열형태로 구성됩니다.

        Arguments:
        x: pd.DataFrame

        Returns:
        inputs: Dict({'input_ids', 'token_type_ids', 'attention_mask'}), 각 tensor(num_data, max_length)
        """
        inputs = self.tokenizer(
            list(x['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.CFG['train']['token_max_len'],
            add_special_tokens=True,
        )

        return inputs

    def preprocessing(self, x, train=False):
        DC = DataCleaning(self.CFG['select_DC'])
        DA = DataAugmentation(self.CFG['select_DA'])
        # 선택한 토크나이징 방법에 따라 동적으로 작동
        tokenizing_method_name = 'self.short_tokenizing' if self.CFG['option']['short_tokenizing'] else 'self.tokenizing'
        tokenizing_method = eval(tokenizing_method_name)

        if train:
            x = DC.process(x, train=True)
            x = DA.process(x)

            if self.CFG['train']['no_valid']:
                train_x = x.drop(['label'], axis=1)  
                train_y = x['label']
                
                train_inputs = tokenizing_method(train_x)
                train_targets = [self.label2num[label] for label in train_y]
                
                return (train_inputs, train_targets, train_x[['subject_type', 'object_type']]), (None, None, None)
                
                
            if not self.CFG['train']['adverse_valid']:
                train_x = x.drop(['label'], axis=1)
                train_y = x['label']
            
                train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                                stratify=train_y,
                                                                test_size=self.CFG['train']['test_size'],
                                                                shuffle=self.CFG['train']['shuffle'],
                                                                random_state=self.CFG['seed'])
            else:
                # Sort the dataframe by similarity score in descending order
                df = x.sort_values(by='prob_1', ascending=False)

                # Define the validation size
                valid_size = 0.2

                # Initialize an empty dataframe for the validation set
                valid_df = pd.DataFrame()

                # For each unique label...
                for label in df['label'].unique():
                    # Filter the dataframe for the current label
                    df_label = df[df['label'] == label]
                    
                    # Calculate the number of validation instances for the current label
                    num_valid = int(len(df_label) * valid_size)
                    
                    # Add the instances with the highest similarity scores to the validation set
                    valid_df = valid_df.append(df_label.iloc[:num_valid])
                    
                    # Remove these instances from the original dataframe
                    df = df.drop(df_label.iloc[:num_valid].index)

                # What remains in df is the training set
                train_df = df

                # Optionally, reset the indices of the train and validation dataframes
                train_df = train_df.reset_index(drop=True)
                valid_df = valid_df.reset_index(drop=True)
                
                train_x = train_df.drop(['label'], axis=1)
                train_y = train_df['label']
                
                val_x = valid_df.drop(['label'], axis=1)
                val_y = valid_df['label']
            
            train_inputs = tokenizing_method(train_x)
            train_targets = [self.label2num[label] for label in train_y]

            val_inputs = tokenizing_method(val_x)
            val_targets = [self.label2num[label] for label in val_y]      # ([sub_types], [obj_types])

            return (train_inputs, train_targets, train_x[['subject_type', 'object_type']]), (val_inputs, val_targets, val_x[['subject_type', 'object_type']])
        else:
            x = DC.process(x)

            # 텍스트 데이터 토큰화
            test_inputs = tokenizing_method(x)
        
            return test_inputs

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train, val = self.preprocessing(self.train_df, train=True)
            
            self.train_dataset = Dataset(train[0], train[1], train[2])
            self.val_dataset = Dataset(val[0], val[1], val[2], no_valid = self.CFG['train']['no_valid'])
        else:
            # 평가 데이터 호출
            predict_inputs = self.preprocessing(self.predict_x)
            self.predict_dataset = Dataset(predict_inputs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=self.CFG['train']['shuffle'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=self.CFG['train']['shuffle'])

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.CFG['train']['batch_size'])
    
class ADVSDataloader(pl.LightningDataModule):
    """
    Adversarial Validation Set을 위한 데이터 로더
    """

    def __init__(self, tokenizer):
        super(ADVSDataloader, self).__init__()
        self.tokenizer = tokenizer
        
        train_df, test_df = load_data()
        train_df['comes_from'] = 0
        test_df['comes_from'] = 1
        self.train_df = pd.concat([train_df, test_df])
        
        train_only_df, _ = load_data()
        self.predict_df = train_only_df

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def tokenizing(self, x):
        concat_entity = []
        for sub_ent, obj_ent in zip(x['subject_entity'], x['object_entity']):
            concat_entity.append(obj_ent + " [SEP] " + sub_ent)

        inputs = self.tokenizer(
            concat_entity,
            list(x['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=170,
            add_special_tokens=True,
        )

        return inputs

    def preprocessing(self, x, train=False):
        if train:
            train_x = x.drop(['comes_from'], axis=1)
            train_y = x['comes_from']

            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                                stratify=train_y,
                                                                test_size=0.2,
                                                                shuffle=True,
                                                                random_state=42)
            
            train_inputs = self.tokenizing(train_x)
            train_targets = [i for i in train_y]

            val_inputs = self.tokenizing(val_x)
            val_targets = [i for i in val_y]

            return (train_inputs, train_targets), (val_inputs, val_targets)
        else:
            predict_x =  self.tokenizing(x)
            return predict_x

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train, val = self.preprocessing(self.train_df, train=True)
            self.train_dataset = ADVSDataset(train[0], train[1])
            self.val_dataset = ADVSDataset(val[0], val[1])
        else:
            # 평가 데이터 호출
            predict_inputs = self.preprocessing(self.predict_df)
            self.predict_dataset = ADVSDataset(predict_inputs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=64, shuffle=False)
    
class TAPTDataloader(pl.LightningDataModule):
    """
    Adversarial Validation Set을 위한 데이터 로더
    """

    def __init__(self, tokenizer):
        super(TAPTDataloader, self).__init__()
        self.tokenizer = tokenizer
        
        train_df, test_df = load_data()
        self.train_df = train_df
        self.test_df = test_df

        self.train_dataset = None
        self.val_dataset = None

    def tokenizing(self, x):
        inputs = self.tokenizer(
            list(x['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=170,
            add_special_tokens=True,
        )

        return inputs

    def preprocessing(self, x):
        train_x, val_x= train_test_split(x,
                                        test_size=0.3,
                                        shuffle=True,
                                        random_state=42)
        train_x = pd.concat([train_x, self.test_df])
        train_inputs = self.tokenizing(train_x)
        val_inputs = self.tokenizing(val_x)

        return train_inputs, val_inputs

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train, val = self.preprocessing(self.train_df)
            self.train_dataset = TAPTDataset(train)
            self.val_dataset = TAPTDataset(val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, collate_fn = DataCollatorForLanguageModeling(
    tokenizer = self.tokenizer, mlm=True, mlm_probability=0.15))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=True,collate_fn = DataCollatorForLanguageModeling(
    tokenizer = self.tokenizer, mlm=True, mlm_probability=0.15))


class DataCleaning():
    """
    config select DC에 명시된 Data Cleaning 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list
        self.continue_list = ['remove_duplicated']

    def process(self, df, train=False):
        if self.select_list:
            for method_name in self.select_list:
                if not train and method_name in self.continue_list: continue
                method = eval("self." + method_name)
                df = method(df)

        return df

    """
    data cleaning 코드
    """
    def entity_parsing(self, df):
        """ 
        entity에서 word, start_idx, end_idx, type 분리하기
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

        if 100 in df['label']:  # test.csv에는 label이 모두 100이다.
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
                
    def pronounce_japan(self, df):
        """ sentence, subject_entity, object_entity 컬럼에서 일본어 발음을 한글로 변환
        hangulize 라이브러리 사용
        
        Note: <데이터 예시>
        みなみ → 미나미
        
        Arguments:
        df: pronounce_japan을 수행하고자 하는 DataFrame
        
        Return:
        df: pronounce_japan 작업이 완료된 DataFrame
        """
        pattern = re.compile(r'[ぁ-ゔ]+|[ァ-ヴー]+[々〆〤]')
        language = 'jpn'
        for i in range(len(df)):
            if pattern.search(df.loc[i,'sentence']):
                df.loc[i, 'sentence'] = hangulize(df.loc[i, 'sentence'], language)
            if pattern.search(df.loc[i,'subject_entity']):
                df.loc[i, 'subject_entity'] = hangulize(df.loc[i, 'subject_entity'], language)
            if pattern.search(df.loc[i,'object_entity']):
                df.loc[i, 'object_entity'] = hangulize(df.loc[i, 'object_entity'], language)
        return df
    
    def remove_duplicated(self, df):
        """
        sentence, subject_entity, object_entity는 동일하지만 label이 두 개 이상 지정된 경우
        데이터를 직접 봐서 필요한 label 선택하기

        *train dataset에 대해서만 적용
        """
        del_idx = [6749, 8364, 22258, 277, 10202, 4212]
        df.drop(del_idx, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    
    def add_entity_tokens_base(self, df):
        """
        sentence에서 entity 앞뒤로 [ENT] [/ENT] 태그 달아줘 entity임을 명시하기
        """
        # ENT 태크 달아주기
        new_sentence = []
        for _, row in df.iterrows():
            sentence = row["sentence"]

            for check, idx in enumerate(sorted([row['subject_start_idx'], row['subject_end_idx'], row['object_start_idx'], row['object_end_idx']], reverse=True)):
                if check % 2 == 0:
                    sentence = sentence[:idx+1] + " [/ENT] " + sentence[idx+1:]
                else:
                    sentence = sentence[:idx] + "[ENT] " + sentence[idx:]
            
            new_sentence.append(sentence)
        df['sentence'] = new_sentence

        return df
    
    def add_entity_tokens_detail(self, df):
        """
        sentence에서 entity 앞뒤로 [{S|O}:{type}] 태그 달아줘 entity임을 상세하게 명시하기
        """
        # [{S|O}:{type}] 태크 달아주기
        new_sentence = []
        for _, row in df.iterrows():
            sentence = row["sentence"]
            trigger = True if row['object_end_idx'] > row['subject_end_idx'] else False

            for check, idx in enumerate(sorted([row['subject_start_idx'], row['subject_end_idx'], row['object_start_idx'], row['object_end_idx']], reverse=True)):
                if trigger:
                    token = f"O:{row['object_type']}"
                else:
                    token = f"S:{row['subject_type']}"

                if check % 2 == 0:
                    sentence = sentence[:idx+1] + f" [/{token}] " + sentence[idx+1:]
                else:
                    sentence = sentence[:idx] + f"[{token}] " + sentence[idx:]
                    trigger = not trigger
            
            new_sentence.append(sentence)
        df['sentence'] = new_sentence

        return df
    
    def add_only_punct(self, df):
        """
        @ : Subject
        * : Subject_type
        # : Object
        ^ : Object_type
        
        short_tokenizing과 함께 쓸 것을 권장.        
                
        Ex)
        From ->     〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        To ->       〈Something〉는 # ^ 사람 ^ 조지 해리슨 # 이 쓰고 @ * 모임 * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        """
        
        type2korean = {'PER' : '사람',
                       'ORG' : '모임',
                       'DAT' : '날짜',
                       'NOH' : '숫자',
                       'POH' : '명사',
                       'LOC' : '장소'}
        
        new_sentence = []
        for _, row in df.iterrows():
            sentence = row["sentence"]
            sub_type = '* ' + type2korean[row["subject_type"]] + ' * '
            obj_type = '^ ' + type2korean[row["object_type"]] + ' ^ '
            
            trigger = True if row['object_end_idx'] > row['subject_end_idx'] else False

            for check, idx in enumerate(sorted([row['subject_start_idx'], row['subject_end_idx'], row['object_start_idx'], row['object_end_idx']], reverse=True)):
                if trigger:
                    token = " # "
                else:
                    token = " @ "

                if check % 2 == 0:
                    sentence = sentence[:idx+1] + token + sentence[idx+1:]
                else:
                    sentence = sentence[:idx] + (token + (obj_type if trigger else sub_type)) + sentence[idx:]
                    trigger = not trigger
            
            new_sentence.append(sentence)
        df['sentence'] = new_sentence

        return df
    
    def quering_with_punct(self, df):
        """
        add_only_punct 뒤에 사용할 수 있는 버전의 quering           
        From ->       〈Something〉는 # ^ 사람 ^ 조지 해리슨 # 이 쓰고 @ * 모임 * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        To ->       @ * 모임 * 비틀즈 @와 # ^ 사람 ^ 조지 해리슨 #의 관계 [SEP] 〈Something〉는 # ^ 사람 ^ 조지 해리슨 # 이 쓰고 @ * 모임 * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        """
        
        type2korean = {'PER' : '사람',
                       'ORG' : '모임',
                       'DAT' : '날짜',
                       'NOH' : '숫자',
                       'POH' : '명사',
                       'LOC' : '장소'}        
        
        for _, row in df.iterrows():
            subject_entity = row["subject_entity"]
            subject_type = type2korean[row["subject_type"]]
            object_entity = row["object_entity"]
            object_type = type2korean[row["object_type"]]
            
            row["sentence"] = f"@ * {subject_type} * {subject_entity} @와 # ^ {object_type} ^ {object_entity} #의 관계 [SEP] " + row["sentence"]
        
        return df

    def quering_with_detail(self, df):
        """
        add_entity_tokens_detail 뒤에 사용할 수 있는 버전의 quering           
        From ->       〈Something〉는 # ^ PER ^ 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        To ->       @ * ORG * 비틀즈 @와 # ^ PER ^ 조지 해리슨 #의 관계 [SEP] 〈Something〉는 # ^ PER ^ 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        """
        
        # 아직 미구현
        
        return df
    
    def add_others_tokens(self, df):
        """
        sentence에서 일본어와 한자를 [OTH] 토큰으로 바꾸기
        """
        df['sentence'].replace(r'[ぁ-ゔァ-ヴー々〆〤一-龥]+', '[OTH]', regex=True, inplace=True)

        return df

    def stop_words(self, df):
        """
        정적 데이터로 만들어진 불용어 리스트를 기반으로 입력 데이터의 불용어 제거하기
        """
        # 불용어 리스트
        okt = Okt()
        stop_words = set()
        with open('./utils/stop_word.txt', 'r') as f:
            for line in f.readlines():
                stop_words.add(line.strip())

        # 불용어 제거 후 새로운 문장 만들기
        def logic(x):
            new_sentence = []
            for word in okt.morphs(x):
                if word not in stop_words:
                    new_sentence.append(word)
            
            return " ".join(new_sentence)
            
        df['sentence'] = df['sentence'].apply(lambda x: logic(x))

        return df
    
    def spacing(self, df):
        """
        띄어쓰기 문법 교정
        """
        lib = Spacing()

        df['sentence'] = df['sentence'].apply(lambda x: lib(x.replace(" ", "")))

    def entity_mask_base(self, df):
        """
        sentence, subject_entity, object_entity 컬럼에서 subject_entity와 object_entity에 마스킹 처리
        [ENT] subject_entity [/ENT] → [SUB] 마스킹
        [ENT] object_entity [/ENT]→ [OBJ] 마스킹
        
        *** add_entity_tokens_base 함수를 먼저 적용해줘야 함! ***

        Note: <데이터 예시>
        〈Something〉는 [ENT] 조지 해리슨 [/ENT] 이 쓰고 [ENT] 비틀즈 [/ENT] 가 1969년 앨범 《Abbey Road》에 담은 노래다.
            ↓
        〈Something〉는 [OBJ] 이 쓰고 [SUB] 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        Arguments:
        df: add_entity_tokens_base 함수를 적용한 DataFrame
        
        Return:
        df: subject_entity와 object_entity에 마스킹 처리 작업이 완료된 DataFrame
        """
        for idx, row in df.iterrows():
            words=re.findall('\[ENT].+?\[/ENT]',row['sentence'])
            if row['subject_start_idx']<row['object_start_idx']:
                df.loc[idx,'sentence']=row['sentence'].replace(words[0],'[SUB]').replace(words[1],'[OBJ]')
            else:
                df.loc[idx,'sentence']=row['sentence'].replace(words[0],'[OBJ]').replace(words[1],'[SUB]')
        
        df['subject_entity']='[SUB]'
        df['object_entity']='[OBJ]'

        return df

    def entity_mask_detail(self, df):
        """
        sentence, subject_entity, object_entity 컬럼에서 subject_entity와 object_entity에 마스킹 처리
        [S:{type}] subject_entity [/S:{type}] → [S:{type}] 마스킹
        [O:{type}] object_entity [/O:{type}] → [O:{type}] 마스킹
        
        *** add_entity_tokens_detail 함수를 먼저 적용해줘야 함! ***

        Note: <데이터 예시>
        〈Something〉는 [O:PER] 조지 해리슨 [/O:PER] 이 쓰고 [S:ORG] 비틀즈 [/S:ORG] 가 1969년 앨범 《Abbey Road》에 담은 노래다.
            ↓
        〈Something〉는 [O:PER] 이 쓰고 [S:ORG] 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        Arguments:
        df: add_entity_tokens_detail 함수를 적용한 DataFrame
        
        Return:
        df: subject_entity와 object_entity에 마스킹 처리 작업이 완료된 DataFrame
        """
        df['sentence']=df['sentence'].apply(lambda x: re.sub(r'\[S:.+?].+?\[/S:.+?]',re.findall(r'\[S:.+?]',x)[0],x))
        df['sentence']=df['sentence'].apply(lambda x: re.sub(r'\[O:.+?].+?\[/O:.+?]',re.findall(r'\[O:.+?]',x)[0],x))
        df['subject_entity']=df['sentence'].apply(lambda x: re.findall(r'\[S:.+?]',x)[0])
        df['object_entity']=df['sentence'].apply(lambda x: re.findall(r'\[O:.+?]',x)[0])

        return df

    def dc_hanspell(self, df):
        """ 
        Spell check 코드
        sentence, subject_entity, object_entity spell check (띄어쓰기도 해 줌)
        hanspell 라이브러리 사용 : https://github.com/ssut/py-hanspell
        
        Note: <데이터 예시>
        진도군은 진도개를 보기 위해 찾아온 관람객들에게 더욱 흥미롭고 즐거움을 선사하기 위해 ▲팔백리길을 돌아온 백구 생가 토피어리 조형물 ▲어로(犬수영장)수렵장 ▲진도개 애견 캠핑장 등도 운영하고 있다.	
        ↓
        진도군은 진돗개를 보기 위해 찾아온 관람객들에게 더욱 흥미롭고 즐거움을 선사하기 위해 ▲팔백 리 길을 돌아온 백구 생가 토피어리 조형물 ▲어로(犬 수영장) 수렵장 ▲진돗개 애견 캠핑장 등도 운영하고 있다.
            
        Arguments:
        df: Spell check를 수행하고자 하는 DataFrame
        
        Return:
        df: Spell check 작업이 완료된 DataFrame
        """
        for type_entity in ['sentence','subject_entity', 'object_entity']:

            for i in range(len(df)):
                df.loc[ i, type_entity ] = spell_checker.check(df.loc[ i, type_entity ]).checked
        
        return df
    
    def dc_change_per_others(self,df):
        """ 
        문장에서 entity를 제외한 사람 이름을 [PER]로 바꿔주는 코드 (윤리적 이슈)
         sentence를 변환
            Pororo 라이브러리 사용 : https://github.com/kakaobrain/pororo
            issue: torch v 1.6
        
        Note: <데이터 예시>
            sentence: 특히 김동연 전 경제부총리를 비롯한 김두관 국회의원, 안규백 국회의원, 김종민 국회의원, 오제세 국회의원, 최운열 국회의원, 김정우 국회의원, 권칠승 국회의원, 맹성규 국회의원등 더불어민주당 국회의원 8명이 영상 축하 메세지를 보내 눈길을 끌었다.		
            sub_ent: 안규백
            obj_ent: 더불어민주당
            ↓
            sentence: 특히 [PER] 전 경제부총리를 비롯한 [PER] 국회의원, 안규백 국회의원, [PER] 국회의원, [PER] 국회의원, [PER] 국회의원, [PER] 국회의원, [PER] 국회의원, [PER] 국회의원등 더불어민주당 국회의원 8명이 영상 축하 메세지를 보내 눈길을 끌었다.
            
        Arguments:
            df: 원본 DataFrame
        
        Return:
            df: entitiy로 들어가는 이름 제외한 사람 이름을 [PER]로 변환하는 작업이 완료된 DataFrame
        """

        ner = Pororo(task="ner", lang="ko")  # ner 수행해 주는 pororo기능 object화

        def per_change(sen):     
            final_sen = ''
            sen = ner(sen)     # [(특히, O), ( , O), (김동연, PERSON), ( , O), (전, O), ( , O), (경제부총리, CIVILIZATION), ... (끌었다., O)]
            for (w,v) in sen:
                if v =='PERSON':
                    w = '[PER]'     # 김동연 -> [PER]
                else:
                    pass
                final_sen+=w
            return final_sen  # 문장 반환: '특히 [PER] 전 경제부총리를 비롯한 [PER] 국회의원, '
        
        def per_change_sent(x):
            sen_ori = x[0]
            sen_new = []
            sentences = re.split(' ?\[ENT\] ?| ?\[\/ENT\] ?',sen_ori)
            for s in sentences:
                if s != x[1] and s!=x[2]: # 사람 이름인데 sub_ent, obj_ent라면 [PER]로 변환하면 안됨
                    sen_new.append(per_change(s)) # 나머지 사람 이름은 [PER]로 변환
                else:
                    sen_new.append(s)
            return ' '.join(sen_new)
        

        df['sentence']=df[['sentence', 'subject_entity','object_entity']].apply(per_change_sent, axis=1)
        # apply로 데이터 프레임을 직접적으로 처리
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
        auto_df = df[df['label'].isin(auto_augmentation)].copy()
        cross_df = df[df['label'].isin(cross_augmentation.keys())].copy()

        auto_df['subject_entity'], auto_df['object_entity'] = auto_df['object_entity'].copy(), auto_df['subject_entity'].copy()

        cross_df['subject_entity'], cross_df['object_entity'] = cross_df['object_entity'].copy(), cross_df['subject_entity'].copy()
        cross_df['label'] = cross_df['label'].map(cross_augmentation)

        return pd.concat([auto_df, cross_df], ignore_index=True)

    def sub_obj_change_augment(self, df):
        """
        Note:   (sub, obj) 쌍을 같은 라벨, 같은 sub-type, 같은 obj-type 을 가지고 있는 
                다른 문장의 (sub, obj) 를 교환하는 형태로 데이터를 증강시킵니다.
                train.csv에서 상대적으로 적은 obj-type을 가지는 POH, DAT, LOC 만을 증강시킵니다.
                no_relation인 라벨은 증강시키지 않습니다.
        
        Arugments:
        증강하고자 하는 dataframe
                
        Return:
        Note 규칙 기반으로 새롭게 생성된 문장 dataframe
        """
        
        label_list = sorted(list(df['label'].unique()))[1:]         # no_relation 제외
        aug_obj_type = ['POH', 'DAT', 'LOC']
        
        df_copied = df.copy(deep=True)
        copy_element = ['label', 'source', 'subject_start_idx', 
                        'subject_end_idx', 'subject_type', 'object_start_idx', 'object_end_idx', 'object_type']
        aug_dict = {key:[] for idx, key in enumerate(df.columns.tolist())}
        
        # 1. 특정 라벨 선택
        for label in tqdm(label_list, desc="Augmentation by (sub, obj) change"):
            # 2. Object type에서 POH, DAT, LOC 확인
            for obj_type in aug_obj_type:
                temp_df = df_copied[(df_copied['label'] == label) & (df_copied['object_type'] == obj_type)]
                if len(temp_df) == 0:
                    continue
                
                # 3. sub type - obj type 같을 때만 교체해서 증강
                sub_type_list = sorted(list(temp_df['subject_type'].unique()))  # ['PER', 'ORG']
                for sub_type in sub_type_list:
                    # 4. (sub, obj) 쌍을 구하고
                    tuple_list = temp_df[temp_df['subject_type'] == sub_type].apply(lambda x:(x['id'], x['subject_entity'], x['object_entity']), axis=1).tolist()
                    
                    # 5. 거꾸로 서로 교체. -> 데이터 2배 증가
                    for idx, (id, sub_en_target, obj_en_target) in enumerate(tuple_list):
                        sent_target = df_copied.loc[id]['sentence']
                        
                        sub_en_change = tuple_list[len(tuple_list)-idx-1][1]
                        obj_en_chage = tuple_list[len(tuple_list)-idx-1][2]
                        
                        sent_change = sent_target.replace(sub_en_target, sub_en_change).replace(obj_en_target, obj_en_chage)
                        
                        aug_dict['id'].append(id)
                        aug_dict['sentence'].append(sent_change)
                        aug_dict['subject_entity'].append(sub_en_change)
                        aug_dict['object_entity'].append(obj_en_chage)

                        # 5-1. 변경하지 않는 컬럼 성분은 그냥 복붙.
                        for c in copy_element:
                            aug_dict[c].append(df.loc[id][c])
                            
        aug_df = pd.DataFrame(aug_dict)
        
        return aug_df


def load_data():
    """
    학습 데이터와 테스트 데이터 DataFrame 가져오기
    """
    train_df = pd.read_csv('./dataset/new_train.csv')
    train_df.drop(['id', 'source'], axis=1, inplace=True)
    test_x = pd.read_csv('./dataset/new_test.csv')
    test_x.drop(['id', 'source'], axis=1, inplace=True)
    
    return train_df, test_x


def load_label2num():
    with open('./code/dict_label_to_num.pkl', 'rb') as f:
        label2num = pickle.load(f)
    
    return label2num


def load_num2label():
    with open('./code/dict_num_to_label.pkl', 'rb') as f:
        num2label = pickle.load(f)
    
    return num2label


def load_types2labelnum():
    with open('./code/dict_types_to_labelnum.pkl', 'rb') as f:
        types2labelnum = pickle.load(f)
    
    return types2labelnum


if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/train/train.csv')
    test_df = pd.read_csv('./dataset/test/test_data.csv')

    # entity_parsing이 적용된 DataFrame 파일 만들기
    new_train_df = DataCleaning([]).entity_parsing(train_df.copy(deep=True))
    new_test_df = DataCleaning([]).entity_parsing(test_df.copy(deep=True))

    new_train_df.to_csv('./dataset/new_train.csv', index=False)
    new_test_df.to_csv('./dataset/new_test.csv', index=False)
    
    types2labelnum = {('PER', 'PER'): {0, 6, 10, 12, 13, 14, 15, 16, 17, 21},
                      ('PER', 'POH'): {0, 4, 6, 8, 11, 12, 14, 15, 17, 21, 23, 29},
                      ('PER', 'ORG'): {0, 4, 6, 11, 12, 14, 15, 23, 29},
                      ('PER', 'LOC'): {0, 6, 11, 15, 23, 26, 27},
                      ('PER', 'DAT'): {0, 11, 15, 24, 25},
                      ('PER', 'NOH'): {0},
                      ('ORG', 'PER'): {0, 1, 3, 5, 28},
                      ('ORG', 'POH'): {0, 1, 2, 3, 5, 7, 19, 20},
                      ('ORG', 'ORG'): {0, 1, 2, 3, 5, 7, 19, 20, 28},
                      ('ORG', 'LOC'): {0, 1, 2, 3, 5, 7, 19, 20},
                      ('ORG', 'DAT'): {0, 3, 5, 18, 22},
                      ('ORG', 'NOH'): {0, 3, 5, 9}}
    
    with open('./code/dict_types_to_labelnum.pkl', 'wb') as f:
        pickle.dump(types2labelnum, f)