class StatusCheck:
    check_str = {
        0: "집중",
        1: "사람 인식 X",
        2: "주의력 산만",
        3: "졸음"
    }
    check_weight = {
        0: 2,
        1: -2,
        2: -2,
        3: -2
    }

    # 하나의 싱글톤 인스턴스를 생성
    # 이미 생성된 인스턴스가 있다면 재사용
    def __new__(cls, *args, **kwargs):

        # *args와 **kwargs는 무슨의미일까?
        # 여러 가변인자를 받겠다고 명시하는 것이며, *args는 튜플형태로 전달, **kwargs는 키:값 쌍의 사전형으로 전달된다.
        # def test(*args, **kwargs):
        #    print(args)
        #    print(kwargs)
        # test(5,10,'hi', k='v')
        # 결과 : (5, 10, 'hi') {'k': 'v'}

        if not hasattr(cls, 'instance'):
            cls.instance = super(StatusCheck, cls, *args, **kwargs).__new__(cls, *args, **kwargs)
        return cls.instance

    def __init__(self):
        # 고개 좌우
        self.turn_head_LR = False
        # 고개 상하
        self.turn_head_UD = False
        # 자세 인식 안됨
        self.no_body = False
        # 눈 계속 감음
        self.close_eyes = False
        # 눈 인식 안됨
        self.no_eyes = False

    def check(self):
        # 눈과 몸이 안보이면
        if self.no_eyes and self.no_body:
            # "사람 인식 X"
            return 1
        # 고개를 움직이면
        elif self.turn_head_UD or self.turn_head_LR:
            # "주의력 산만"
            return 2
        # 눈을 감으면
        elif self.close_eyes:
            # "졸음"
            return 3
        else:
            # "집중"
            return 0


if __name__ == '__main__':
    # 싱글톤 테스트
    sc = StatusCheck()
    sc2 = StatusCheck()
    print(sc, sc2, sc == sc2)
