# Contributing Guide

## 커밋 메시지 컨벤션

이 프로젝트는 [Conventional Commits](https://www.conventionalcommits.org/) 스타일을 따릅니다.

### 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅, 세미콜론 누락 등 (코드 변경 없음)
- `refactor`: 코드 리팩토링
- `test`: 테스트 코드 추가/수정
- `chore`: 빌드 업무 수정, 패키지 매니저 설정 등
- `perf`: 성능 개선
- `ci`: CI 설정 변경
- `build`: 빌드 시스템 또는 외부 의존성 변경

### Scope

프로젝트의 모듈/디렉토리 이름을 사용합니다:

- `web-client/frontend`: Next.js 프론트엔드
- `web-client/backend`: 백엔드 API
- `web-client/socket`: WebSocket 서버
- `3d-simulator`: 3D 시뮬레이터
- `chatbot`: 챗봇 모듈
- `ML`: 머신러닝 모듈
- `DB`: 데이터베이스 관련
- (비워두기): 루트 레벨 변경사항 (README, .gitignore 등)

### 예시

#### 단일 모듈 변경
```
feat(web-client/frontend): add ThreeViewer component with R3F

- Implement 3D viewer using React Three Fiber
- Add camera preset controls
- Support color and variant toggles
```

#### 버그 수정
```
fix(3d-simulator): resolve websocket connection timeout

Increase timeout duration and add reconnection logic.
```

#### 문서 수정
```
docs: update README with project structure
```

#### 여러 모듈 동시 변경
```
feat: add vehicle comparison feature

- web-client/frontend: implement CompareTable component
- web-client/backend: add comparison API endpoint
- DB: update vehicle schema for comparison data
```

#### 의존성 업데이트
```
chore(web-client): update Next.js to 15.0.0
```

### 주의사항

- **제목은 50자 이내**로 작성
- **제목은 명령형**으로 작성 (예: "add" not "added" or "adds")
- **본문은 72자마다 줄바꿈**
- **본문은 "what"과 "why"를 설명**, "how"는 코드에서 확인 가능
- 여러 모듈을 동시에 변경하는 경우, 본문에 각 모듈별 변경사항을 나열
- Breaking changes가 있는 경우 `BREAKING CHANGE:` 접두사 사용

### Git 템플릿 사용

프로젝트 루트의 `.gitmessage` 파일이 커밋 템플릿으로 설정되어 있습니다.

```bash
# 이미 설정되어 있음
git config commit.template .gitmessage
```

커밋 시 자동으로 템플릿이 열리며, 가이드라인을 참고하여 작성할 수 있습니다.

