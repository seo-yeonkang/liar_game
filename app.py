import streamlit as st
from player import Player
from liar_game import LiarGame
import random
from ai_utils_bert import compute_secret_embeddings
import time

# Streamlit 페이지 설정
st.set_page_config(page_title="라이어 게임", page_icon="🎭")

# 게임 정보 표시 함수 정의
def display_game_info():
    game = st.session_state.game
    if game and hasattr(game, 'chosen_topic'):
        with st.sidebar:
            st.write("### 게임 정보")
            st.write(f"라운드: {game.current_round}/{game.total_rounds}")
            st.write(f"주제: {game.chosen_topic}")
            human_player = next(p for p in game.players if p.is_human)
            if human_player.is_liar:
                st.write("당신은 라이어입니다!")
            else:
                st.write(f"제시어: {st.session_state.secret_word}")
            
            st.write("\n### 플레이어 점수")
            for player in game.players:
                st.write(f"{player.name}: {player.score}점")

# 예측된 단어들 처리 함수
def process_predicted_words(predicted_dict):
    # tensor 제거하고 단어만 추출
    processed_words = {}
    for word, score in predicted_dict.items():
        # score가 tensor인 경우 float으로 변환
        processed_score = float(score) if hasattr(score, 'item') else score
        processed_words[word] = processed_score
    return dict(sorted(processed_words.items(), key=lambda x: x[1], reverse=True))

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.game = None
    st.session_state.game_phase = 'setup'
    st.session_state.descriptions = {}
    st.session_state.current_player_idx = 0
    st.session_state.secret_word = None
    st.session_state.chosen_topic = None
    st.session_state.players_order = None
    st.session_state.votes = {}
    st.session_state.round_data_initialized = False
    st.session_state.initialized = True
    st.session_state.ai_predicted_words = None
    st.session_state.start_time = None
    st.session_state.show_predicted_words = False  # 새로 추가된 상태 변수

st.title("라이어 게임에 오신 것을 환영합니다!")

# 게임 초기 설정
if st.session_state.game_phase == 'setup':
    total_players = st.number_input("총 플레이어 수를 입력하세요 (최소 3명)", min_value=3, value=3)
    human_name = st.text_input("당신의 이름을 입력하세요")
    
    if st.button("게임 시작") and human_name:
        # 플레이어 생성
        players = [Player(human_name, is_human=True)]
        for i in range(1, total_players):
            players.append(Player(f"AI_{i+1}"))
        
        # 게임 인스턴스 생성
        st.session_state.game = LiarGame(players)
        st.session_state.game_phase = 'role_reveal'
        st.rerun()

# 역할 공개 및 라운드 시작
elif st.session_state.game_phase == 'role_reveal':
    game = st.session_state.game
    
    if not st.session_state.round_data_initialized:
        # 역할 배정
        game.assign_roles()
        
        # 주제와 단어 선택
        chosen_topic = random.choice(list(game.topics.keys()))
        secret_word = random.choice(game.topics[chosen_topic])
        game.chosen_topic = chosen_topic
        st.session_state.secret_word = secret_word
        
        # 주제별 임베딩 초기화
        game.object_word_embeddings = compute_secret_embeddings(list(game.topics["object"]))
        game.food_word_embeddings = compute_secret_embeddings(list(game.topics["food"]))
        game.job_word_embeddings = compute_secret_embeddings(list(game.topics["job"]))
        game.place_word_embeddings = compute_secret_embeddings(list(game.topics["place"]))
        game.character_word_embeddings = compute_secret_embeddings(list(game.topics["character"]))
        
        # 플레이어 순서 설정
        players_order = game.players.copy()
        random.shuffle(players_order)
        if players_order[0].is_liar:
            liar_player = players_order.pop(players_order.index(game.liar))
            insert_position = random.randint(1, len(players_order))
            players_order.insert(insert_position, liar_player)
        st.session_state.players_order = players_order
        
        # 라이어 상태 초기화
        st.session_state.round_data_initialized = True
        st.session_state.show_predicted_words = False
    
    # 정보 표시
    st.write(f"### 라운드 {game.current_round}")
    display_game_info()
    
    # 라이어가 인간인 경우 현재 플레이어 바로 직전에 예측 단어 표시
    human_player = next(p for p in game.players if p.is_human)
    current_player_idx = st.session_state.current_player_idx if hasattr(st.session_state, 'current_player_idx') else 0
    players_order = st.session_state.players_order
    
    if human_player.is_liar:
        # 첫 번째 플레이어 전에 예측 단어를 보여주기 위한 로직
        if current_player_idx == 0 and not st.session_state.show_predicted_words:
            # 이전 플레이어들의 설명 수집
            aggregated_comments = " ".join(st.session_state.descriptions.values())
            
            # 예측 단어 가져오기
            st.session_state.ai_predicted_words = game.predict_secret_word_from_comments(aggregated_comments)
            
            # 메인 화면에 예측 단어 표시
            if st.session_state.ai_predicted_words:
                processed_words = process_predicted_words(st.session_state.ai_predicted_words)
                st.write("### 시스템 예측 단어들")
                for word, score in list(processed_words.items())[:5]:
                    st.write(f"{word}: {score:.4f}")
            
            # 예측 단어를 한 번만 보여주도록 설정
            st.session_state.show_predicted_words = True
    
    if st.button("설명 단계로"):
        st.session_state.game_phase = 'explanation'
        st.rerun()

# [나머지 코드는 동일하게 유지]
# (이전 코드의 explanation, voting, result, game_over 섹션들을 그대로 복사)
