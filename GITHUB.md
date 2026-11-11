# GitHub 연결과 관리

프로젝트 폴더에서 실행한다. 새 공개 저장소 이름은 `Psychic_Yut_Playing`을 사용한다.

## 최초 연결

GitHub CLI 설치 및 브라우저 로그인:

```sh
brew install gh
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

로그인 계정을 확인하고 로컬 Git 저장소를 만든다.

```sh
gh auth status
git init -b main
git add .
git diff --cached --stat
```

업로드 대상은 코드·문서·윷판 이미지·웹 파일이다. `web/model.json`은 브라우저 대전용 가중치이므로 포함된다. `.venv`, `.venv-coreml`, `checkpoints`, `.env`, 빌드 결과물은 제외한다. 체크포인트는 GitHub에 백업되지 않으므로 별도로 보관한다.

첫 커밋 후 공개 저장소를 생성하고 업로드한다.

```sh
git commit -m "Initial project import"
gh repo create Psychic_Yut_Playing --public --source=. --remote=origin --push
```

동일 이름의 저장소가 이미 있으면 임의로 덮어쓰지 말고 새 이름을 정한다. 위 최초 연결 명령은 한 번만 실행한다. 라이선스는 아직 지정하지 않았다.

## 이후 관리

변경 사항 확인은 `git status`, 차이 확인은 `git diff`를 사용한다. 기능 수정은 별도 브랜치에서 진행하고 테스트 후 커밋·푸시·PR을 만든다. 작업 요청 시 “변경 내용을 검토하고 커밋해줘”, “브랜치를 만들어 수정하고 PR을 만들어줘”처럼 원하는 범위를 지정할 수 있다. 실제 Git 쓰기와 업로드는 해당 세션의 권한 및 GitHub 인증이 필요하다.

GitHub 연동을 연결하면 저장소·이슈·PR 조회에 사용할 수 있다. 터미널의 `gh` 인증과 앱 연동 인증은 별도다. 비밀번호나 토큰은 대화에 붙여 넣지 않는다.

공식 안내: [CLI 로그인](https://cli.github.com/manual/gh_auth_login), [저장소 생성](https://cli.github.com/manual/gh_repo_create).
