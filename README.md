# 문장 내 개체간 관계 추출

- 팀명: 강남 특공대
- 인원: 5명
- 주최: 부스트캠프 ai tech 5기 | level 2 프로젝트
- 프로젝트 기간: 2023년 5월 2일 ~ 2023년 5월 18일
- 주제: KLUE 데이터셋에서 관계 추출(RE; Relation Extraction) 문제 해결
- 데이터셋
  - train data: 32,470개
  - test data: 7,765개
- 평가 방법
  - `Micro F1 Score` `AUPRC`
- 개발 스택: `Git` `Notion` `Ubuntu` `Python` `Pytorch Lightning` `HuggingFace` `Wandb`
- 결과: `Public: 10위`, `Private: 7위`

# 목차
[1. 팀원 소개](#1-팀원-소개) > [2. 프로젝트 구조](#2-프로젝트-구조) > [3. 데이터 전처리 및 모델링](#3-데이터-전처리-및-모델링) > [4. 프로젝트 자체 평가](#4-프로젝트-자체-평가)

# 1. 팀원 소개

<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/gibum1228"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/gibum1228"/></a>
            <br/>
            <a href="https://github.com/gibum1228"><strong>김기범</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/heejinsara"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/heejinsara"/></a>
            <br/>
            <a href="https://github.com/heejinsara"><strong>박희진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/LewisVille-flow"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/LewisVille-flow"/></a>
            <br/>
            <a href="https://github.com/LewisVille-flow"><strong>이주형</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Forbuds"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/Forbuds"/></a>
            <br/>
            <a href="https://github.com/Forbuds"><strong>천소영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/rustic-snob"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/rustic-snob"/></a>
            <br/>
            <a href="https://github.com/rustic-snob"><strong>천재원</strong></a>
            <br />
        </td>
    </tr>
</table>

# 2. 프로젝트 구조

```html
|-- level2_klue-nlp-02/
|   |-- code/ (private)
|   |-- dataset/ (private)
|   |-- results/ (private)
|   |-- data_analysis/
|   |   |-- {name}_pre.ipynb
|   |   `-- {name}_viz.ipynb
|   |-- ensemble/
|   |   `-- ensemble.ipynb
|   |-- models/
|   |   `-- models.py
|   |-- utils/
|   |   |-- data_controller.py
|   |   |-- metrics.py
|   |   `-- utils.py
|   |-- README.md
|   |-- config.yaml
|   |-- confusion_matrix.yaml
|   |-- main_process.py
|   |-- requirements.txt
|   `-- use_config.yaml (private)
`----------------------
```

# 3. 데이터 전처리 및 모델링

> 적용한 전처리 기법 및 모델링 기법이 매우 많은 관계로 코드 내 docstring를 참고

# 4. 프로젝트 자체 평가

## 4.1 잘한 점

- 굵직굵직하게 해야 할 일들과 그에 맞춘 전체적인 날짜를 잘 설정하고, 그에 따라 계획적으로 다같이 프로젝트를 진행할 수 있었음
- 협업 과정에서 Github를 적극적으로 활용함
- Zoom을 적극적으로 활용하여 이슈 발생 시 즉각적인 대응
- 애자일 스크럼 방식을 채택해 팀원 간 진행도를 파악하고 중복 업무를 배제해 효율적으로 프로젝트를 진행할 수 있었음
- EDA를 누군가가 맡아서 하는 것이 아니라, 팀원 모두가 함께 진행한 뒤 각자의 분석 결과를 쇼업함으로써 상호 간의 얻을 수 있는 아이디어가 다양했음
- 서로의 꼼꼼한 코드 리뷰를 통해 구현한 기능을 점검하고 공유하는 문화(with Agile process)
- 일반화된 프로젝트 구조(모듈화) 덕분에 실험 환경 공유가 용이했음
- 주기적인 오프라인 만남을 통한 적극적인 팀 아이디어 공유와 토론
- fp16 precision을 사용하여 마지막에 학습 속도를 비약적으로 빠르게 한 점
- 코드 주석을 명확하게 기입해 다른 사람의 코드를 쉽게 이해하고 읽어 활용할 수 있었음

## 4.2 시도 했으나 잘 되지 않았던 것

- TAPT를 도입하여 성능을 끌어올리고자 했으나, 오히려 하락했음(이미 기학습된 데이터가 KLUE 데이터 셋이라 오버피팅이 일어난 것으로 추측)
- Adversarial Validation을 도입하여 Test와 Valid 사이의 score 차이를 줄이려 했으나, 잘 되지 않았음. Feature에 대한 더 깊은 고민이 필요했던 것으로 추측.
- 몇 가지 Data Augmentation 기법들을 적용했지만, 실제 성능 향상에 기여된 기법들은 극소수에 불과
- 다양한 loss 함수를 설계하였으나 해당 태스크에 적합한 기법을 찾기 어려웠던 점
- 모델의 학습 난이도를 높이기 위한 multi task learning 기법을 다소 단순하게 적용시켜 큰 효과가 없었음.

## 4.3 아쉬웠던 점

- 프로젝트 관련해서 멘토님과 상호작용이 활발하지 못 했던 점 아쉬웠음
- 감정적으로 아무래도 F1 score가 잘 나오지 않다보니, 마음이 급해져서 구현에 구멍을 종종 발생시켰던 점
- Augmentation 기법들을 다양하게 적용해보고 싶었지만 아이디어의 한계에 봉착함
- 다양한 모델을 적용해 보고자 했으나 적용할 수 있는 모델이 한정적이었음
- 리더보드 점수와 실험 모니터링 결과를 비교했을 때 F1은 너무 높게 나오고(과적합) AUPRC는 너무 낮게 나왔는데(과소적합) 이 간극을 해결하지 못 한 게 아쉬움이 남음
- 각종 논문을 토대로 loss 함수 및 모델링을 진행하였으나, 구현에 집중하여 고정된 환경에서 정량적으로 성능 비교를 진행하지 못한 점.
- WandB sweep이나 ray tune과 같은 툴을 사용하여 하이퍼파라미터 튜닝을 좀 더 정량적으로 하지 못했던 점
- fp16을 좀 더 빨리 도입하여, 다양한 실험을 더 많이 진행하지 못한 점
- 같은 조건에서 seed를 바꿔가며 지금 모델의 일반화된 성능을 체크하지 못한 점
- 블렌딩의 꿈을 이루지 못 한 게 아쉬움
- 마지막 날에서야 k-fold가 생각나서 적용 못 함

## 4.4 프로젝트를 통해 배운 점 또는 시사점

- 자나깨나 에러 조심, 완성 코드 다시 보자
    - 오히려 잘 돌아가는 코드가 더 무섭다
- 하이퍼파라미터 튜닝 라이브러리 or 툴 등을 적극 활용해보자
- 3주 라는 시간이 길면 길고 짧으면 짧은데 지치지 않고 꾸준한 열정을 쏟기 위해 팀 분위기가 중요하다는 걸 깨달음
- 체력 관리는 항상 중요하다. 아프거나 컨디션이 안 좋으면 시간 대비 효율이 나오지 않아 오히려 좋지 않다.
- 팀원 간의 코드 설명에 쏟는 시간은 아무리 늘려도 부족하지 않다.
    - 서로의 구현이나 아이디어에 대해 열심히 질문하고, 상세하게 대답하는 것으로 큰 성장을 이룰 수 있음
- 대회 진행 계획을 조금 더 앞당겨서 아이디어 검증의 여유를 둘 수 있게 해야하겠다는 깨달음
- 논문은 이미 검증된 자료다. 논문의 모델 구조와 하이퍼파라미터 세팅을 참고해야겠다는 점.
- AI Stages 제출 횟수인 10회는 적다는 걸 앎
    - 제출 마감 직전에 내지 말기. 여유 시간을 가지고 10분 전에 제출 완료하기.
