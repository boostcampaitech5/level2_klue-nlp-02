import pickle
import torch
import pandas as pd
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from konlpy.tag import Okt
# from pykospacing import Spacing
# from hanspell import spell_checker
# from pororo import Pororo


# from hangulize import hangulize
import re

class Dataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, inputs, targets=[], sub_obj_types=[]):
        self.inputs = inputs
        self.targets = targets
        self.sub_obj_types = sub_obj_types

    def __getitem__(self, idx):
        inputs = {key: val[idx].clone().detach()
                  for key, val in self.inputs.items()}

        if self.targets:
            targets = torch.tensor(self.targets[idx])
            sub_obj_types = tuple(self.sub_obj_types.iloc[idx])
            
            return inputs, targets, sub_obj_types
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

            train_x = x.drop(['label'], axis=1)
            train_y = x['label']

            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                              stratify=train_y,
                                                              test_size=self.CFG['train']['test_size'],
                                                              shuffle=self.CFG['train']['shuffle'],
                                                              random_state=self.CFG['seed'])
            
            train_inputs = tokenizing_method(train_x)
            train_targets = [self.label2num[label] for label in train_y]

            val_inputs = tokenizing_method(val_x)
            val_targets = [self.label2num[label] for label in val_y]

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
            self.val_dataset = Dataset(val[0], val[1], val[2])
        else:
            # 평가 데이터 호출
            predict_inputs = self.preprocessing(self.predict_x)
            self.predict_dataset = Dataset(predict_inputs)

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

    def entity_mask(self, df):
        """ sentence 컬럼에서 subject_entity와 object_entity에 마스킹 처리(index 기준으로)
        subject_entity → [SUB] 마스킹
        object_entity → [OBJ] 마스킹
        
        Note: <데이터 예시>
        〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
            ↓
        〈Something〉는 [OBJ]이 쓰고 [SUB]가 1969년 앨범 《Abbey Road》에 담은 노래다.
        
        Arguments:
        df: entity_mask를 수행하고자 하는 DataFrame
        
        Return:
        df: entity_mask 작업이 완료된 DataFrame
        """
        for idx, row in df.iterrows():
            if row['subject_start_idx']<row['object_start_idx']:
                df.loc[idx,'sentence']=row['sentence'][:row['subject_start_idx']]+'[SUB]'+row['sentence'][row['subject_end_idx']+1:row['object_start_idx']]+'[OBJ]'+row['sentence'][row['object_end_idx']+1:]
            else:
                df.loc[idx,'sentence']=row['sentence'][:row['object_start_idx']]+'[OBJ]'+row['sentence'][row['object_end_idx']+1:row['subject_start_idx']]+'[SUB]'+row['sentence'][row['subject_end_idx']+1:]
        
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
        To ->       〈Something〉는 # ^ PER ^ 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        """
        
        new_sentence = []
        for _, row in df.iterrows():
            sentence = row["sentence"]
            sub_type = '* ' + row["sub_type"] + ' * '
            obj_type = '^ ' + row["obj_type"] + ' ^ '
            
            trigger = True if row['object_end_idx'] > row['subject_end_idx'] else False

            for check, idx in enumerate(sorted([row['subject_start_idx'], row['subject_end_idx'], row['object_start_idx'], row['object_end_idx']], reverse=True)):
                if trigger:
                    token = " # "
                else:
                    token = " @ "

                if check % 2 == 0:
                    sentence = sentence[:idx+1] + token + sentence[idx+1:]
                else:
                    sentence = sentence[:idx] + obj_type if trigger else sub_type + token  + sentence[idx:]
                    trigger = not trigger
            
            new_sentence.append(sentence)
        df['sentence'] = new_sentence

        return df
    
    def quering_with_punct(self, df):
        """
        add_only_punct 뒤에 사용할 수 있는 버전의 quering           
        From ->       〈Something〉는 # ^ PER ^ 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        To ->       @ * ORG * 비틀즈 @와 # ^ PER ^ 조지 해리슨 #의 관계 [SEP] 〈Something〉는 # ^ PER ^ 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
        """
        
        for _, row in df.iterrows():
            subject_entity = row["subject_entity"]
            subject_type = row["subject_type"]
            object_entity = row["object_entity"]
            object_type = row["object_type"]
            
            row["sentence"] = f"@ * {subject_type} * {subject_entity} @와 # ^ {object_type} ^ {object_entity}의 관계 [SEP] " + row["sentence"]
        
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

    """
    Spell check 코드
    """
    def dc_hanspell(self, df):
        """ sentence, subject_entity, object_entity spell check (띄어쓰기도 해 줌)
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
    
    """
    문장에서 entity를 제외한 사람 이름을 [PER]로 바꿔주는 코드 (윤리적 이슈)
    """

    def dc_change_per_others(self,df):
        """ sentence를 변환
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
                # print(w)
                final_sen+=w
            # print(sen)
            return final_sen  # 문장 반환: '특히 [PER] 전 경제부총리를 비롯한 [PER] 국회의원, '
        
        def per_change_sent(x):
            sen_ori = x[0]
            idx = sorted([0,x[3],x[4]+1,x[5],x[6]+1,len(x[0])])   # index 준비 : subject_start_idx, subject_end_idx,  ,,,
            # print(idx,len(x[0]))
            sentences=[]

            for i in range(len(idx)-1):
                # print(idx[i] , idx[i+1])
                if idx[i]==x[3]:            # subject_start_idx: 사람 이름인데 sub_ent라면 [PER]로 변환하면 안돼서 따로 빼줌
                    sentences.append(x[1])        
                elif idx[i]==x[5]:          # object_start_idx: 마찬가지
                    sentences.append(x[2])        
                else:
                    sentences.append(per_change(sen_ori[ idx[i] : idx[i+1] ]))   # 나머지 사람 이름은 [PER]로 변환
                    # print(sen_ori[ idx[i] : idx[i+1] ])
            # print(sentences)
            return ' '.join(sentences)
        

        df['sentence']=df[['sentence', 'subject_entity','object_entity','subject_start_idx','subject_end_idx','object_start_idx','object_end_idx']].apply(per_change_sent, axis=1)
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


if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/train/train.csv')
    test_df = pd.read_csv('./dataset/test/test_data.csv')

    # entity_parsing이 적용된 DataFrame 파일 만들기
    new_train_df = DataCleaning([]).entity_parsing(train_df.copy(deep=True))
    new_test_df = DataCleaning([]).entity_parsing(test_df.copy(deep=True))

    new_train_df.to_csv('./dataset/new_train.csv', index=False)
    new_test_df.to_csv('./dataset/new_test.csv', index=False)