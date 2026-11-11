# Core ML 참고 (기본 실행에서 사용하지 않음)

현재 학습·평가는 CPU 방식이다. 일반 실행은 [SELF_PLAY.md](SELF_PLAY.md)를 따른다. 이 문서는 코드에서 안내하는 선택형 Core ML 의존성 설치 정보만 보존한다.

별도 Python 3.11 환경이 필요한 경우:

```sh
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install -e '.[coreml]'
```

Core ML을 명시적으로 선택하는 옵션은 `--inference-backend coreml-ne`이다. CPU＋NPU 사용을 허용하며 실제 NPU 실행이나 속도 향상을 보장하지 않는다. 기본 CPU 실행에는 설치할 필요가 없다.
