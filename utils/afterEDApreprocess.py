import pandas as pd


def swap_sentence(df):
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

    auto_augmentation = set('org:alternate_names',
                            'per:alternate_names',
                            'per:colleagues',
                            'per:other_family',
                            'per:siblings',
                            'per:spouse')
    cross_augmentation = {'per:children' : 'per:parents',
                          'per:parents' : 'per:children',
                          'org:members' : 'org:member_of',
                          'org:member_of' : 'org:members',
                          'org:top_members/employees' : 'per:employee_of',
                          'per:employee_of' : 'org:top_members/employees'}
    auto_df = df[df['label'].isin(auto_augmentation)]
    cross_df = df[df['label'].isin(cross_augmentation.keys())]

    auto_df['subject_entity'], auto_df['object_entity'] = auto_df['object_entity'], auto_df['subject_entity']
    cross_df


def normalize_class(df):
    allowed_obj_for_class = \
        {'org:member_of' : set('ORG', 'LOC', 'POH'),
         'org:members' : set('ORG', 'LOC', 'POH')
         }
