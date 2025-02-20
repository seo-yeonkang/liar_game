# liar_game.py

import random
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from player import Player
from scipy.spatial.distance import cosine
from ai_utils_bert import embedding_model, download_models, compute_secret_embeddings, gpt_generate_response, util, openai
from evaluation import recall_k, MRR, NDCG
import math

class LiarGame:
    def __init__(self, players, total_rounds=3):
        self.players = players
        self.total_rounds = total_rounds
        self.current_round = 1
        self.liar = None
        self.liar_count=0 # 사용자가 liar일 때는 모델 성능 지표에서 빼야하기 때문에 추가가

        # 주제별로 최소 20개의 secret 단어 후보를 정의합니다.
        self.topics = {
            "place": [
                "공원", "도서관", "해변", "산", "도시", "마을", "강", "호수", "광장", "카페",
                "식당", "학교", "병원", "극장", "박물관", "시장", "공항", "체육관", "지하철역", "호텔"
            ],
            "food": [
                "초콜릿", "피자", "라면", "스시", "햄버거", "김치", "비빔밥", "떡볶이", "파스타", "스테이크",
                "샐러드", "치킨", "감자튀김", "샌드위치", "토스트", "오믈렛", "초밥", "케이크", "아이스크림", "컵라면"
            ],
            "job": [
                "의사", "변호사", "요리사", "교사", "프로그래머", "디자이너", "엔지니어", "간호사", "회계사", "군인",
                "경찰", "소방관", "조종사", "비서", "관리자", "연구원", "작가", "예술가", "음악가", "배우"
            ],
            "object": [
                "컴퓨터", "휴대폰", "책상", "의자", "시계", "텔레비전", "냉장고", "전자레인지", "세탁기", "전구",
                "수도꼭지", "마우스", "책상", "프린터", "카메라", "스피커", "이어폰", "헤드폰", "책", "노트"
            ],
            "character": [
                "해리포터", "슈퍼맨", "아이언맨", "스파이더맨", "신데렐라", "닥터 스트레인지", "가모라", "타노스", "배트맨", "원더우먼",
                "플래시", "캡틴 아메리카", "토르", "헐크", "로켓 라쿤", "데드풀", "엑스맨", "스파이더우먼", "레드후드", "닉 퓨리"
            ]
        }   
        '''
        각 주제별로 하나의 리스트로 모아 임베딩을 미리 계산 (3개로 간추림)
        '''
        place_secret_words=list(self.topics["place"])
        self.place_word_embeddings=compute_secret_embeddings(place_secret_words)
        food_secret_words=list(self.topics["food"])
        self.food_word_embeddings=compute_secret_embeddings(food_secret_words)
        job_secret_words=list(self.topics["job"])
        self.job_word_embeddings=compute_secret_embeddings(job_secret_words)

        # 어떤 주제를 뽑았는지를 알려주는 벼수
        self.chosen_topic=None

        
        # 모든 주제의 단어들을 하나의 리스트로 모아 임베딩 미리 계산 (AI 내부 비교용)
        all_secret_words = [word for words in self.topics.values() for word in words]
        self.secret_word_embeddings = compute_secret_embeddings(all_secret_words)

    def assign_roles(self):
        """
        각 라운드 시작 시 모든 플레이어의 역할을 초기화한 뒤,
        플레이어 중 무작위로 한 명을 라이어로 지정합니다.
        """
        for player in self.players:
            player.is_liar = False
        self.liar = random.choice(self.players)
        self.liar.is_liar = True
        print(f"[DEBUG] 이번 라운드 라이어는 {self.liar.name}입니다.")

    def predict_secret_word_from_comments(self, comments):
        """
        이전 플레이어들의 설명(코멘트)을 임베딩한 후,
        모든 후보 단어(전체 목록) 중 가장 유사도가 높은 단어를 예측합니다.
        (참고용; 투표나 점수 계산에는 사용하지 않습니다.)
        """
        comment_embedding = embedding_model.encode(comments, convert_to_tensor=True)
        
        # 딕셔너리로 유사도 저장
        word_dict={}

        # 각 주제에 맞는 기존 embedding 값을 가져오기 위한 작업업
        attribute_name=f"{self.chosen_topic}_word_embeddings" # 문자열 생성
        get_self=getattr(self,attribute_name) # 동적으로 객체의 속성을 가져옴

        for word, emb in get_self.items():
            similarity = util.cos_sim(comment_embedding, emb)
            
            # 모델 평가를 위해 딕셔러니로 유사도를 저장장 
            word_dict[word]=similarity

        # 유사도를 저장한 딕셔너리를 value값(유사도)에 따라 정렬
        word_dict=dict(sorted(word_dict.items(),key=lambda x:x[1], reverse=True))
        return word_dict

    def predict_word_for_explanation(self, explanation, topic):
        """
        주어진 설명(explanation)을 임베딩한 후,
        해당 라운드의 주제(topic) 후보 단어들 중 가장 유사한 단어를 예측하여 반환합니다.
        이 값은 투표 참고용으로만 사용됩니다.
        """
                
        explanation_embedding = embedding_model.encode(explanation, convert_to_tensor=True)
        best_word = None
        best_similarity = -1
        
        # 각 주제에 맞는 기존 embedding 값을 가져오기 위한 작업업
        attribute_name=f"{self.chosen_topic}_word_embeddings" # 문자열 생성
        get_self=getattr(self,attribute_name) # 동적으로 객체의 속성을 가져옴

        for word, emb in get_self.items():
            similarity = util.cos_sim(explanation_embedding, emb).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_word = word
        return best_word

    def generate_ai_truth_description(self, secret_word):
        """
        진실 플레이어 AI의 설명을 생성합니다.
        GPT API를 통해 secret_word와 관련된 구체적인 힌트를 포함한 한 문장 설명을 요청합니다.
        """
        system_prompt = (
            "You are a truthful player in the Liar Game. The rules of the game are as follows:\n"
            "1. All truthful players share the same secret word, but the liar does not know it.\n"
            "2. Each player must describe the secret word in one sentence without directly revealing it.\n"
            "3. Descriptions should be subtle and not too obvious, so the liar cannot easily guess the word.\n"
            "4. After all players give their descriptions, they will discuss and vote to identify the liar.\n"
            "5. If the liar is correctly identified, they will have one last chance to guess the secret word.\n"
            "6. If the liar guesses correctly, they win. If they fail, the truthful players win.\n\n"
            f"Your secret word for this round is '{secret_word}'. "
            "Write a one-sentence description of this word using abstract and metaphorical expressions. "
            "Do not mention the secret word directly. "
            "Do not give any clear hints that would help the liar. "
            "Make your description sound natural and logical. "
            "Respond in Korean."
        )
        description = gpt_generate_response(system_prompt)
        return description if description else "설명을 생성하는 데 실패했습니다."

    def generate_ai_liar_description(self, previous_comments):
        """
        라이어 플레이어 AI의 설명을 생성합니다.
        이전 플레이어들의 설명을 취합하여 내부적으로 예측한 제시어를 참고하지만,
        실제 제시어 단서는 주지 않고 일반적인 설명을 생성하도록 합니다.
        """
        if previous_comments.strip():
            # return 값이 단일 단어에서, 모든 코사인 유사도 값을 가지는 딕셔너리로 바뀜뀜
            predicted_dict = self.predict_secret_word_from_comments(previous_comments)
            #가장 유사도가 높은 단어를 추출출
            predicted_secret=list(predicted_dict.keys())[0]

            system_prompt = (
                "You are the liar player in the Liar Game. The rules of the game are as follows:\n"
                "1. Truthful players share the same secret word, but the liar does not know it.\n"
                "2. Each player must describe the secret word in one sentence without directly revealing it.\n"
                "3. As the liar, you do not know the actual secret word, but you can infer it from previous players' descriptions.\n"
                "4. After all players give their descriptions, they will discuss and vote to identify the liar.\n"
                "5. If the liar is correctly identified, they will have one last chance to guess the secret word.\n"
                "6. If the liar guesses correctly, they win. If they fail, the truthful players win.\n\n"
                f"Based on the previous players' descriptions, your predicted secret word is '{predicted_secret}'. "
                "However, you do not know if this is the actual word. "
                "Write a one-sentence description that sounds like you are not the liar. "
                "Make your description logical and natural, but do not reveal too much detail. "
                "Avoid making your description too vague or meaningless. "
                f"Do NOT mention '{predicted_secret}' directly. "
                "Respond in Korean."
            )
        else:
            system_prompt = (
                "당신은 라이어 플레이어입니다."
                "최대한 플레이어들에게 들키지 않도록 자연스럽게 작성하세요."
                "한 문장으로 설명을 작성하세요."
            )
        description = gpt_generate_response(system_prompt)
        return description if description else "설명을 생성하는 데 실패했습니다.", predicted_dict


    # 훈련된 모델 로드
    download_models()

    MODEL_PATH = "./trained_model"  # 학습한 모델이 저장된 폴더 경로
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    '''
    def get_embedding(self, sentence):
        """
        단일 문장의 임베딩을 생성하는 함수.
        여기서는 CLS 토큰의 임베딩을 사용합니다.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, 
                                padding="max_length", max_length=128)
        with torch.no_grad():
            # 만약 모델이 roberta 기반이면, 내부의 roberta 모듈을 사용합니다.
            if hasattr(self.model, "roberta"):
                outputs = self.model.roberta(**inputs, output_hidden_states=True)
            # 만약 BERT 기반이면, 내부의 bert 모듈을 사용합니다.
            elif hasattr(self.model, "bert"):
                outputs = self.model.bert(**inputs, output_hidden_states=True)
            else:
                # 만약 다른 구조라면 fallback (하지만 이 경우에는 추가 조정 필요)
                outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    def generate_ai_vote(self, voter, descriptions):
        """
        AI 플레이어가 라이어로 의심되는 사람에게 투표하는 로직.
        기존에는 후보들의 모든 pairwise 유사도를 계산했지만,
        여기서는 후보(자신 제외) 각 플레이어의 설명 임베딩과, 
        나머지 후보들의 평균 임베딩과의 코사인 거리를 계산합니다.
        거리가 클수록 집단 평균과 동떨어져 있으므로 의심받을 확률이 높습니다.
        """
        # 자신을 제외한 후보 리스트
        candidate_names = [name for name in descriptions if name != voter.name]
    
        # 각 후보의 설명 임베딩 계산
        candidate_embeddings = {}
        for name in candidate_names:
            candidate_embeddings[name] = self.get_embedding(descriptions[name])
    
        suspicion_scores = {}
        for name in candidate_names:
            # 나머지 후보들의 임베딩을 모아서 평균 임베딩 계산
            other_embeddings = [candidate_embeddings[other] for other in candidate_names if other != name]
            if other_embeddings:
                avg_embedding = np.mean(other_embeddings, axis=0)
            else:
                avg_embedding = candidate_embeddings[name]
            # 코사인 거리는 값이 클수록 두 벡터 간 차이가 큼 (즉, 후보가 집단 평균과 동떨어짐)
            distance = cosine(candidate_embeddings[name], avg_embedding)
            suspicion_scores[name] = distance
    
        print("Suspicion scores:", suspicion_scores)

        # 선택: 거리가 클수록 의심받을 확률이 높으므로, softmax 적용 (여기서는 단순하게 거리값 사용)
        scores_tensor = torch.tensor(list(suspicion_scores.values()))
        probs = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
        chosen_candidate = random.choices(candidate_names, weights=probs, k=1)[0]
    
        return chosen_candidate
    '''
    def compute_sts_similarity(self, sentence1, sentence2):
        """
        두 문장의 의미적 유사도를 평가하는 함수 (KLUE RoBERTa 활용)
        """
        inputs = self.tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            output = self.model(**inputs).logits
        similarity_score = output.item()  # 모델의 출력값 (보통 0~5 점수)
        return similarity_score / 5  # 정규화 (0~1)
    
    def generate_ai_vote(self, voter, descriptions):
        """
        AI 플레이어가 라이어로 의심되는 사람에게 투표하는 로직.
        KLUE RoBERTa 기반 문장 유사도를 사용하여 의미적으로 다른 설명을 한 플레이어를 찾음.
        """
        # 자신을 제외한 후보 리스트
        candidate_names = [name for name in descriptions if name != voter.name]

        # 문장 유사도 계산 (STS 활용)
        inverse_similarities = []

        for name in candidate_names:
            sims = []
            for other_name in candidate_names:
                if name != other_name:
                    similarity = self.compute_sts_similarity(descriptions[name], descriptions[other_name])
                    sims.append(similarity)
        
            avg_sim = sum(sims) / len(sims) if sims else 0
            inverse_sim = 1 - avg_sim  # 유사도가 낮을수록 의심도가 높음
            inverse_similarities.append(inverse_sim)
        print(inverse_similarities)
    
        # Softmax 적용하여 확률 변환
        scores_tensor = torch.tensor(inverse_similarities)
        probs = torch.nn.functional.softmax(scores_tensor, dim=0)  # 합이 1이 되도록 변환

        # 확률을 기반으로 랜덤 투표
        chosen_candidate = random.choices(candidate_names, weights=probs.tolist(), k=1)[0]

        return chosen_candidate
    

    def liar_guess_secret(self):
        """
        라이어 플레이어가 secret_word를 추측합니다.
        GPT API를 사용하여 단서를 주지 않고 직감에 따라 단어를 하나 추측합니다.
        """
        prompt = (
            "당신은 라이어 플레이어입니다. 지금부터 제시어를 추측하세요. "
            "당신은 제시어에 대한 구체적인 정보를 모릅니다. 오직 당신의 직감을 기반으로 단어를 하나 추측하세요."
        )
        guess = gpt_generate_response(prompt, max_tokens=10, temperature=0.5)
        return guess.strip() if guess else ""

    def start_round(self):
        """
        한 라운드를 진행합니다.
          1. 매 라운드마다 역할(라이어/진실 플레이어)을 랜덤으로 재배정합니다.
          2. 랜덤으로 주제를 선택한 후, 해당 주제의 후보 단어 중 하나를 secret_word로 선정합니다.
          3. 각 플레이어에게 역할 및 정보를 안내합니다.
             - 진실 플레이어: 주제와 secret_word 모두 안내.
             - 라이어 플레이어: 주제는 안내하되 secret_word는 제공하지 않음.
          4. 플레이어들의 설명 순서는 랜덤으로 진행되되, 라이어 플레이어(인간 포함)는 첫 번째가 되지 않도록 합니다.
             (인간 라이어인 경우, 자신의 차례 전에 내부 참고용으로 예측된 단어를 출력합니다.)
          5. 모든 플레이어의 설명이 수집되면, 각 설명에 대해 주제 후보 단어 중 가장 적합한 단어를 예측하여 (참고용) 출력합니다.
          6. 이후 투표 시, 각 플레이어의 설명에서 예측된 단어와 실제 secret_word의 유사도를 비교해,
             가장 유사도가 낮은 플레이어에게 투표합니다.
          7. 투표 결과에 따라 점수를 업데이트합니다.
        """
        # 매 라운드 시작 시 역할 재배정
        self.assign_roles()
        
        print(f"\n=== {self.current_round} 라운드 ===")
        # 랜덤 주제 선택 후, 해당 주제의 후보 단어 중 하나를 secret_word로 선정
        chosen_topic = random.choice(list(self.topics.keys()))
        secret_word = random.choice(self.topics[chosen_topic])

        # 어떤 주제를 선택했는지 알 수 있도록 만듦
        self.chosen_topic=chosen_topic
       
        
        # 각 플레이어에게 역할 및 정보 안내
        for player in self.players:
            if player.is_human:
                if player.is_liar:
                    print(f"{player.name}: 당신은 **라이어**입니다! 주제는 '{chosen_topic}'입니다.")
                else:
                    print(f"{player.name}: 당신의 주제는 '{chosen_topic}'이고, 제시어는 '{secret_word}' 입니다.")
            else:
                if player.is_liar:
                    print(f"{player.name}: 역할이 배정되었습니다. (라이어, 주제: {chosen_topic})")
                else:
                    print(f"{player.name}: 역할이 배정되었습니다. (진실 플레이어)")
        
        # 플레이어들의 설명 순서를 랜덤하게 정하되, 라이어 플레이어가 첫 번째가 되지 않도록 설정
        players_for_comments = self.players.copy()
        random.shuffle(players_for_comments)
        if players_for_comments[0].is_liar:
            liar_player = players_for_comments.pop(players_for_comments.index(self.liar))
            insert_position = random.randint(1, len(players_for_comments))
            players_for_comments.insert(insert_position, liar_player)
        
        # 각 플레이어의 설명 수집
        print("\n각 플레이어는 제시어에 대해 한 문장씩 설명해주세요.")
        descriptions = {}
        predicted_dict= {} # 단어들의 코사인 유사도가 포함함
        
        skip_model_evaluate=0

        for player in players_for_comments:
            aggregated_comments = " ".join(descriptions.values())
            if player.is_human:
                if player.is_liar:

                    # 인간 라이어의 경우, 자신의 차례 전에 내부 참고용 예측 단어 출력
                    self.liar_count+=1
                    skip_model_evaluate=1 # model 평가에서 이번 반복은 제외

                    if aggregated_comments.strip():
                        predicted_secret = self.predict_word_for_explanation(aggregated_comments, chosen_topic)
                    else:
                        predicted_secret = "N/A"
                    print(f"[내부 참고용] 예측된 단어: {predicted_secret}")
                    desc = input(f"{player.name}의 설명: ")
                    descriptions[player.name] = desc
                else:
                    desc = input(f"{player.name}의 설명: ")
                    descriptions[player.name] = desc
            else:
                if player.is_liar:
                    # return 값이 2개가 됨!
                    desc, predicted_dict = self.generate_ai_liar_description(aggregated_comments)
                else:
                    desc = self.generate_ai_truth_description(secret_word)
                print(f"{player.name}의 설명: {desc}")
                descriptions[player.name] = desc


        # 모든 플레이어의 설명에 대해, 각 설명에서 주제 후보 단어 중 가장 적합한 단어를 예측하여 참고용으로 출력
        print("\n[참고용] 각 플레이어의 설명으로 예측한 단어:")
        for name, explanation in descriptions.items():
            predicted = self.predict_word_for_explanation(explanation, chosen_topic)
            print(f"{name}: {predicted}")


        '''
        모델 평가 방식 추가한 코드!!
        '''
        print("AI 라이어가 모델을 기반으로 가장 유사하다고 판단한 단어들들")
        print(list(predicted_dict.keys())[:5]) # 디버깅
        print(list(predicted_dict.values())[:5]) # 디버깅

        if skip_model_evaluate==0:
            k=3 # 1 < k < 주제에 있는 단어의 개수수
            recall_k_result=recall_k(predicted_dict,secret_word,k) # 0 /1
            MRR_result=MRR(predicted_dict,secret_word)
            if MRR_result==0:
                print("ERROR NOT WORKING")
            NDCG_result=NDCG(predicted_dict,secret_word)
            if NDCG_result==0:
                print("ERROR NOT WORKING")

        # 디버그용
        '''
        real_result=0
        predicted_list=list(predicted_dict.keys())
        for i in range(1,len(predicted_list)+1):
            if predicted_list[i-1]==secret_word:
                real_result=i
                break
        print(list(predicted_dict.keys()))
        print(f'실제 단어의 위치: {real_result} , R@K 결과: {recall_k_result}') # 몇번째에 있냐
        print(f'MRR 결과 : {MRR_result} == 실제 결과: {1/real_result}') 
        print(f'NDCG 결과 : {NDCG_result} == 실제 결과: {1/math.log2(real_result+1)}')
        ''' 
        
        # 투표 진행 (변경된 메커니즘 적용)
        print("\n투표 시간입니다. 각 플레이어는 라이어로 의심되는 사람의 이름을 선택해주세요.")
        votes = {}
        for player in self.players:
            if player.is_human:
                vote = input(f"{player.name}, 라이어로 의심되는 플레이어의 이름을 입력하세요: ")
            else:
                vote = self.generate_ai_vote(player, descriptions)
                print(f"{player.name}의 투표: {vote}")
            votes[vote] = votes.get(vote, 0) + 1
        
        # 투표 결과 출력
        print("\n투표 결과:")
        for name, count in votes.items():
            print(f"{name}: {count}표")
        
        # 점수 계산
        highest_votes = max(votes.values())
        top_candidates = [name for name, cnt in votes.items() if cnt == highest_votes]
        if self.liar.name in top_candidates:
            for player in self.players:
                if not player.is_liar:
                    player.score += 1
            liar_guess = self.liar_guess_secret()  # 내부적으로 추측만 수행 (출력하지 않음)
            if liar_guess.lower() == secret_word.lower():
                self.liar.score += 3
                print(f"\n[결과] {self.liar.name}은(는) 제시어를 올바르게 추측하여 3점을 획득했습니다!")
            else:
             
                print(f"\n[결과] {self.liar.name}은(는) 제시어를 추측하지 못했습니다.")
        else:
            self.liar.score += 1
            print(f"\n[결과] 라이어가 지목되지 않아 {self.liar.name}이(가) 1점을 획득했습니다.")
        
        # 현재 점수 출력
        print("\n현재 점수:")
        for player in self.players:
            print(f"{player.name}: {player.score}점")
        
        self.current_round += 1

        # 사용자가 liar였을 경우 모델 평가에서 제외외
        if skip_model_evaluate==1: 
            return 0,0,0

        # AI가 liar 였을 경우우
        return recall_k_result, MRR_result, NDCG_result

    def play_game(self):
        """
        전체 게임을 진행합니다.
        매 라운드마다 역할을 재배정한 후 정해진 총 라운드 수만큼 라운드를 반복하고,
        마지막에 최종 점수를 출력하며 승자를 결정합니다.
        """
        print("라이어 게임을 시작합니다!")

        recall_k_score=0
        MRR_score=0
        NDCG_score=0
        while self.current_round <= self.total_rounds:
            recall_k_result, MRR_result, NDCG_result=self.start_round()
            
            # 평가지표에 따라 결과 저장장
            recall_k_score+=recall_k_result
            MRR_score+=MRR_result
            NDCG_score+=NDCG_result
        print("모델 평가 지표 결과")
        print(f"Recall K의 결과: {recall_k_score/(self.total_rounds-self.liar_count)}")
        print(f"MRR의 결과: {MRR_score/(self.total_rounds-self.liar_count)}")
        print(f"NDCG의 결과: {NDCG_score/(self.total_rounds-self.liar_count)}")
        print(f"사용자 liar 횟수: {self.liar_count}")


        print("\n게임 종료!")
        print("\n최종 점수:")
        for player in self.players:
            print(f"{player.name}: {player.score}점")
        max_score = max(player.score for player in self.players)
        winners = [player.name for player in self.players if player.score == max_score]
        if len(winners) == 1:
            print(f"\n최종 승자: {winners[0]}!")
        else:
            print(f"\n최종 승자: {', '.join(winners)} (공동 승자)!")
