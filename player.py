# player.py

class Player:
    def __init__(self, name, is_human=False):
        self.name = name            # 플레이어 이름
        self.is_liar = False        # 라이어 여부 (게임 시작 시 한 명만 True로 설정)
        self.score = 0              # 누적 점수
        self.is_human = is_human    # 인간 플레이어 여부 (True이면 사용자 입력, False이면 AI)
