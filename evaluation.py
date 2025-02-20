#from liar_game import LiarGame
import math

def recall_k(predicted_dict,secret_word,k):
    '''
    딕셔너리에서 상위 k개 내의 secret_word가 존재하냐!
    '''
    predicted_list=list(predicted_dict.keys())[:k]
    return secret_word in predicted_list

def MRR(predicted_dict,secret_word):
    predicted_list=list(predicted_dict.keys())
    for i in range(1,len(predicted_list)+1):
        if predicted_list[i-1]==secret_word:
            return 1/i

    return 0

def NDCG(predicted_dict,secret_word):
    predicted_list=list(predicted_dict.keys())
    for i in range(1,len(predicted_list)+1):
        if predicted_list[i-1]==secret_word:
            return 1/math.log2(i+1)
    return 0