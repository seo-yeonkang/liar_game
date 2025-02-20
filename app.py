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
    st.session_state.ai_predicted_words = None  # 새로 추가된 상태
    st.session_state.start_time = None  # 타이머 시작 시간

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
        
        # 라이어가 인간인 경우 예측 단어 초기화
        human_player = next(p for p in game.players if p.is_human)
        if human_player.is_liar:
            st.session_state.ai_predicted_words = None
        
        st.session_state.round_data_initialized = True
    
    # 정보 표시
    st.write(f"### 라운드 {game.current_round}")
    display_game_info()
    
    # 라이어가 인간인 경우 예측 단어 표시
    human_player = next(p for p in game.players if p.is_human)
    if human_player.is_liar and st.session_state.ai_predicted_words is None:
        aggregated_comments = " ".join(st.session_state.descriptions.values())
        st.session_state.ai_predicted_words = game.predict_secret_word_from_comments(aggregated_comments)
        
        # 메인 화면에 예측 단어 표시
        if st.session_state.ai_predicted_words:
            processed_words = process_predicted_words(st.session_state.ai_predicted_words)
            st.write("### 시스템 예측 단어들")
            for word, score in list(processed_words.items())[:5]:
                st.write(f"{word}: {score:.4f}")
    
    if st.button("설명 단계로"):
        st.session_state.game_phase = 'explanation'
        st.rerun()

# 설명 단계
elif st.session_state.game_phase == 'explanation':
    game = st.session_state.game
    display_game_info()
    current_player = st.session_state.players_order[st.session_state.current_player_idx]
    
    st.write("### 설명 단계")
    st.write("각 플레이어는 제시어에 대해 한 문장씩 설명해주세요.")
    
    # 라이어가 인간인 경우 예측 단어 표시
    human_player = next(p for p in game.players if p.is_human)
    if human_player.is_liar and st.session_state.ai_predicted_words is not None:
        processed_words = process_predicted_words(st.session_state.ai_predicted_words)
        st.write("### 시스템 예측 단어들")
        for word, score in list(processed_words.items())[:5]:
            st.write(f"{word}: {score:.4f}")
    
    # 현재까지의 설명들 표시
    if st.session_state.descriptions:
        st.write("\n### 지금까지의 설명:")
        for name, desc in st.session_state.descriptions.items():
            st.write(f"{name}: {desc}")
    
    # 현재 플레이어의 설명 처리
    st.write(f"\n### {current_player.name}의 차례")
    
    if current_player.is_human:
        if current_player.name not in st.session_state.descriptions:
            # 타이머 설정
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()
            
            # 남은 시간 계산
            elapsed_time = time.time() - st.session_state.start_time
            remaining_time = max(0, 60 - int(elapsed_time))
            
            st.write(f"남은 시간: {remaining_time}초")
            
            # 60초 제한 타이머
            if remaining_time > 0:
                explanation = st.text_input("당신의 설명을 입력하세요", max_chars=100)
                if st.button("설명 제출"):
                    if explanation.strip():
                        st.session_state.descriptions[current_player.name] = explanation
                        st.session_state.current_player_idx += 1
                        st.session_state.start_time = None  # 타이머 초기화
                        if st.session_state.current_player_idx >= len(game.players):
                            st.session_state.game_phase = 'voting'
                        st.rerun()
            else:
                st.error("시간이 초과되었습니다!")
                explanation = "시간 초과"
                st.session_state.descriptions[current_player.name] = explanation
                st.session_state.current_player_idx += 1
                st.session_state.start_time = None  # 타이머 초기화
                if st.session_state.current_player_idx >= len(game.players):
                    st.session_state.game_phase = 'voting'
                st.rerun()
    else:
        if current_player.name not in st.session_state.descriptions:
            aggregated_comments = " ".join(st.session_state.descriptions.values())
            if current_player.is_liar:
                explanation, _ = game.generate_ai_liar_description(aggregated_comments)
            else:
                explanation = game.generate_ai_truth_description(st.session_state.secret_word)
            st.session_state.descriptions[current_player.name] = explanation
            
        st.write(f"AI의 설명: {st.session_state.descriptions[current_player.name]}")
        if st.button("다음 플레이어"):
            st.session_state.current_player_idx += 1
            if st.session_state.current_player_idx >= len(game.players):
                st.session_state.game_phase = 'voting'
            st.rerun()

# 나머지 코드는 이전과 동일하게 유지 (투표, 결과, 게임 종료 단계)
# ... (이전 코드와 동일)

# 마지막 부분에 다음 코드 추가
elif st.session_state.game_phase == 'game_over':
    game = st.session_state.game
    
    st.write("### 게임 종료!")
    st.write("\n### 최종 점수:")
    for player in game.players:
        st.write(f"{player.name}: {player.score}점")
    
    # 승자 결정
    max_score = max(player.score for player in game.players)
    winners = [player.name for player in game.players if player.score == max_score]
    if len(winners) == 1:
        st.write(f"\n최종 승자: {winners[0]}!")
    else:
        st.write(f"\n최종 승자: {', '.join(winners)} (공동 승자)!")
    
    if st.button("새 게임 시작"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
