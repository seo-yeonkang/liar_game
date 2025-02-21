# liar_game

## 웹 플레이
🎮 [라이어 게임 플레이하기](https://kubig-nlpteam1-liargame.streamlit.app/)

## 로컬 실행

1. 저장소 클론
```bash
Copygit clone https://github.com/your-username/liar-game.git
cd liar-game
```

가상환경 생성 및 활성화

bashCopypython -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

필요한 패키지 설치

bashCopypip install -r requirements.txt

게임 실행

bashCopystreamlit run app.py

OpenAI API 키 입력


게임 실행 후 화면에서 OpenAI API 키를 입력하세요
API 키는 비밀번호 형태로 안전하게 처리됩니다
게임 세션이 종료되면 API 키를 다시 입력해야 합니다


## 소개
이 프로젝트는  '라이어 게임'을 웹 기반으로 구현한 애플리케이션입니다. 플레이어가 AI가 함께 즐길 수 있는 대화형 게임으로, Streamlit을 사용하여 개발되었습니다.

## 주요 기능
사용자와 AI 플레이어 간의 대화형 게임플레이
주제별 단어 시스템
AI 플레이어의 지능적인 응답 생성
라이어를 위한 단어 예측 힌트 시스템
실시간 점수 계산 및 승자 결정

## 기술 스택
Python
Streamlit
OpenAI GPT
BERT (Bidirectional Encoder Representations from Transformers)
Sentence Transformers
PyTorch
