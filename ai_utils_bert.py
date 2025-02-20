# ai_utils_bert.py
import openai
from sentence_transformers import util 
from transformers import AutoTokenizer, AutoModel
import torch
import os
import gdown
import streamlit as st


# OpenAI API 키 설정 (본인의 API 키로 대체하세요)
api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")

if api_key:
    openai.api_key = api_key
else:
    st.warning("API 키를 입력해주세요. OpenAI 웹사이트에서 발급 가능합니다.")

    

def download_models():
    # 모델을 저장할 디렉토리 생성
    os.makedirs('./bert', exist_ok=True)
    os.makedirs('./trained_model', exist_ok=True)

    # Google Drive 공유 링크 (실제 ID로 교체 필요)
    bert_url = "https://drive.google.com/drive/folders/1lVOc6aRU6N6Ml6HKQb-Eqm2Rzo6p52E7?usp=sharing"
    trained_url = "https://drive.google.com/drive/folders/1ERxkidgaClnkCllwvhRHSEh73Ls3HKqQ?usp=sharing"
    
    if not os.path.exists('./bert/config.json'):
        print("BERT 모델 다운로드 중...")
        gdown.download_folder(bert_url, output='./bert', quiet=False)
        print("BERT 모델 다운로드 완료!")
        
    if not os.path.exists('./trained_model/config.json'):
        print("학습된 모델 다운로드 중...")
        gdown.download_folder(trained_url, output='./trained_model', quiet=False)
        print("학습된 모델 다운로드 완료!")


class BERTEmbeddingModel:
    def __init__(self, model_path):
        download_models()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def encode(self, text, convert_to_tensor=True):
        """
        문자열 또는 문자열 리스트를 받아 모델의 마지막 은닉 상태에서 평균 풀링하여
        임베딩 벡터를 생성합니다.
        """
        if isinstance(text, list):
            embeddings = [self._encode_single(t) for t in text]
            return torch.stack(embeddings)
        else:
            return self._encode_single(text)

    def _encode_single(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 마지막 은닉 상태: (batch_size=1, seq_len, hidden_size)
        token_embeddings = outputs.last_hidden_state  
        attention_mask = inputs["attention_mask"]
        # 패딩 토큰을 제외하고 평균 풀링
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.squeeze(0)

# 전역 변수 embedding_model을 BERT 모델로 초기화합니다.
embedding_model = BERTEmbeddingModel('./bert')
# -----------------------------------------------------------------------------

def compute_secret_embeddings(secret_words):
    """
    secret_words 리스트의 각 단어에 대해 임베딩을 계산하여
    딕셔너리 형태로 반환합니다.
    """
    embeddings = {}
    for word in secret_words:
        embeddings[word] = embedding_model.encode(word, convert_to_tensor=True)
    return embeddings

def gpt_generate_response(system_prompt, max_tokens=60, temperature=0.7):
    """
    주어진 시스템 프롬프트를 사용해 GPT API를 호출하고 응답 텍스트를 생성합니다.
    실패 시 None을 반환합니다.
    """
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return None
