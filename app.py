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
    
    if human_player.is_liar:
        # 설명 단계에서 예측 단어를 표시하도록 이동
        st.session_state.ai_predicted_words = None
    
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
    
    # 인간 라이어인 경우 예측 단어 보여주기
    human_player = next(p for p in game.players if p.is_human)
    if human_player.is_liar and st.session_state.ai_predicted_words is None:
        aggregated_comments = " ".join(st.session_state.descriptions.values())
        st.session_state.ai_predicted_words = game.predict_secret_word_from_comments(aggregated_comments)
    
    if current_player.is_human:
        if current_player.name not in st.session_state.descriptions:
            explanation = st.text_input("당신의 설명을 입력하세요")
            if st.button("설명 제출"):
                st.session_state.descriptions[current_player.name] = explanation
                st.session_state.current_player_idx += 1
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

# 투표 단계
elif st.session_state.game_phase == 'voting':
    game = st.session_state.game
    display_game_info()
    
    st.write("### 투표 단계")
    st.write("모든 설명:")
    for name, desc in st.session_state.descriptions.items():
        st.write(f"{name}: {desc}")
    
    human_player = next(p for p in game.players if p.is_human)
    if human_player.name not in st.session_state.votes:
        vote_options = [p.name for p in game.players if p != human_player]
        human_vote = st.selectbox("라이어라고 생각하는 플레이어를 선택하세요", vote_options)
        if st.button("투표"):
            st.session_state.votes[human_player.name] = human_vote
            # AI 플레이어들의 투표
            for player in game.players:
                if not player.is_human:
                    vote = game.generate_ai_vote(player, st.session_state.descriptions)
                    st.session_state.votes[player.name] = vote
            st.session_state.game_phase = 'result'
            st.rerun()

# 결과 단계
elif st.session_state.game_phase == 'result':
    game = st.session_state.game
    display_game_info()
    
    st.write("### 투표 결과")
    # 투표 결과 집계
    vote_counts = {}
    for vote in st.session_state.votes.values():
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    
    for name, count in vote_counts.items():
        st.write(f"{name}: {count}표")
    
    # 라이어 공개
    st.write(f"\n실제 라이어는 {game.liar.name}입니다!")
    st.write(f"제시어는 '{st.session_state.secret_word}'였습니다!")
    
    # 점수 계산
    if 'points_calculated' not in st.session_state:
        highest_votes = max(vote_counts.values())
        top_candidates = [name for name, cnt in vote_counts.items() if cnt == highest_votes]
        
        if game.liar.name in top_candidates:
            for player in game.players:
                if not player.is_liar:
                    player.score += 1
            if game.liar.is_human:
                st.write("당신이 라이어입니다! 제시어를 맞춰보세요:")
                # 추측 단어 입력 추가
                liar_guess = st.text_input("당신이 생각하는 제시어는?")
                if st.button("제출"):
                    if liar_guess.lower() == st.session_state.secret_word.lower():
                        game.liar.score += 3
                        st.write("정답입니다! 승리하셨습니다! 3점을 획득하셨습니다!")
                        st.session_state.points_calculated = True
                    else:
                        st.write("틀렸습니다. 패배하셨습니다.")
                        st.session_state.points_calculated = True
            else:
                liar_guess = game.liar_guess_secret()
                if liar_guess.lower() == st.session_state.secret_word.lower():
                    game.liar.score += 3
                    st.write(f"{game.liar.name}이(가) 제시어를 맞추어 3점을 획득했습니다!")
                else:
                    st.write(f"{game.liar.name}이(가) 제시어를 맞추지 못했습니다.")
                st.session_state.points_calculated = True
        else:
            game.liar.score += 1
            st.write(f"라이어가 지목되지 않아 {game.liar.name}이(가) 1점을 획득했습니다!")
            st.session_state.points_calculated = True
    
    # 점수 계산 후 다음 라운드로 넘어가기
    if st.button("다음 라운드"):
        # 라운드 관련 상태 초기화
        game.current_round += 1
        st.session_state.descriptions = {}
        st.session_state.votes = {}
        st.session_state.current_player_idx = 0
        st.session_state.round_data_initialized = False
        if 'points_calculated' in st.session_state:
            del st.session_state.points_calculated
        
        if game.current_round <= game.total_rounds:
            st.session_state.game_phase = 'role_reveal'
        else:
            st.session_state.game_phase = 'game_over'
        st.rerun()

# 게임 종료
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
