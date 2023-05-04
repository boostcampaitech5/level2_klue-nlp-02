# 문장 내 개체간 관계 추출

- 팀명: 강남 특공대
- 주제: 네이버 부스트캠프 ai tech 5기 | level 2 프로젝트
- 프로젝트 기간: 2023년 5월 2일 ~ 2023년 5월 18일
- 데이터셋
  - train data: 32,470개
  - test data: 7,765개
- 평가 방법
  - `Micro F1 Score` `AUPRC`
- 개발 스택: `Git` `Notion` `Ubuntu` `Python` `Pytorch Lightning` `Wandb`
- 결과: 

# 목차
[1. 팀원 소개](#1-팀원-소개) > [2. 프로젝트 구조](#2-프로젝트-구조)

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
|   |-- models/
|   |   `-- models.py
|   |-- utils/
|   |   |-- data_controller.py
|   |   |-- metrics.py
|   |   `-- utils.py
|   |-- README.md
|   |-- config.yaml
|   |-- main_process.py
|   |-- requirements.txt
|   `-- use_config.yaml (private)
`
```