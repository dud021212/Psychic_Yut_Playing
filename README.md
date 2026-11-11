# 초능력 윷놀이 AI

Python으로 자기대국을 학습하고, 학습한 모델과 정적 웹에서 대전하는 프로젝트입니다. 현재 모델은 `checkpoints/model-v3.pt`이며 기본 실행은 **PyTorch CPU·워커 4개**입니다.

- [학습·평가 실행법](SELF_PLAY.md)
- [GitHub 연결과 관리](GITHUB.md)
- [사람 vs AI 웹 대전](WEB_PLAY.md)
- [윷판 경로 명세](윷판%20경로%20명세.md)

## 설치와 검사

프로젝트 폴더에서 실행하세요. 이미 설치했다면 다시 설치할 필요는 없습니다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

## 현재 구현

정책·가치 신경망과 가상 대국 탐색을 이용합니다. 가상 대국에서도 양쪽의 선택에 신경망 정책을 사용하며, 상대의 미공개 능력은 추정합니다. 후속 상황마다 탐색을 중첩하는 완전한 트리 탐색은 아닙니다.

학습 데이터·옵티마이저·진행 상태를 체크포인트에 저장하며, 추가 학습 전후 모델을 보존합니다. 평가는 학습 없이 독립적인 무작위 대국으로 진행합니다. 모델의 실력은 별도 평가로 확인해야 합니다.

`python -m yut_ai`는 합성 데이터 검사 도구입니다. 실제 게임 학습에는 `python -m yut_ai.selfplay`를 사용하세요.
