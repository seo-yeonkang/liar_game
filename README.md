# 🃏 liar_game

## 소개
이 프로젝트는 '라이어 게임'을 웹 기반으로 구현한 애플리케이션입니다. 플레이어가 AI와 함께 즐길 수 있는 대화형 게임으로, Streamlit을 사용하여 개발되었습니다.

## 웹 플레이
🎮 [라이어 게임 플레이하기](https://kubig-nlpteam1-liargame.streamlit.app/)

## 로컬 실행

1. 저장소 클론
```bash
git clone https://github.com/seo-yeonkang/liar_game
cd liar_game
```
2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```
4️. OpenAI API 키 설정
- OpenAI API를 사용하려면 환경 변수로 API 키를 설정해야 합니다.
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"
# macOS/Linux (터미널)
export OPENAI_API_KEY="your_api_key_here"
```
5. 게임 실행
```bash
streamlit run app.py
```



## 게임 시작하기
1. 라이어 게임 웹사이트에 접속하거나 로컬에서 실행합니다.
2. OpenAI API 키를 입력합니다 (AI 플레이어와의 상호작용을 위해 필요).
3. 플레이어 수를 선택하고 당신의 이름을 입력합니다.
4. "게임 시작" 버튼을 클릭하여 게임을 시작합니다!

## 기술 스택
- Python
- Streamlit
- OpenAI GPT
- BERT (Bidirectional Encoder Representations from Transformers)
- Sentence Transformers
- PyTorch
