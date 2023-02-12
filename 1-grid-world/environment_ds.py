import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 50  # Grid 한 칸당 pixel 수
HEIGHT = 5  # Grid world 세로 칸 수
WIDTH = 5  # Grid world 가로 칸 수


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('DeepSARSA')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        
        self.counter = 0
        self.rewards = []       # 목적지와 장애물의 정보가 담긴 dictionary list
        self.goal = []
        
        self.set_reward([0, 1], -1)
        self.set_reward([1, 3], -1)
        self.set_reward([2, 2], -1)
        self.set_reward([4, 4], 1)
        
        # # 장애물 설정
        # self.set_reward([0, 1], -1)
        # self.set_reward([1, 2], -1)
        # self.set_reward([2, 3], -1)
        # # 목표 지점 설정
        # self.set_reward([4, 4], 1)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # Grid 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./img/rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("./img/triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("./img/circle.png").resize((30, 30)))

        return rectangle, triangle, circle

    def reset_reward(self):     # 장애물의 위치 초기화
        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        
        self.set_reward([0, 1], -1)
        self.set_reward([1, 3], -1)
        self.set_reward([2, 2], -1)
        self.set_reward([4, 4], 1)
        
        # self.set_reward([0, 1], -1)
        # self.set_reward([1, 2], -1)
        # self.set_reward([2, 3], -1)
        # self.set_reward([4, 4], 1)

    def set_reward(self, state, reward):        # 목적지와 장애물의 위치 및 reward 값 부여
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0:          # 목적지일 경우
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2])    # Circle

            self.goal.append(temp['figure'])

        elif reward < 0:        # 장애물일 경우
            temp['direction'] = -1      # -1이면 오른쪽으로 이동, +1이면 왼쪽으로 이동
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])     # Triangle

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        # rewards엔 목적지와 장애물의 state(좌표 값), coords(canvas 상의 위치), reward 값, 이미지가 담김
        self.rewards.append(temp)

    # new methods

    def check_if_reward(self, state):       # 현 state가 목적지 혹은 장애물인지 확인
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] == 1:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list       # 목적지 여부와 reward를 return

    def coords_to_state(self, coords):      # Canvas 상의 위치를 state 좌표 값으로 변환
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):        # Agent를 초기 위치로 옮긴 후 장애물 위치 초기화
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:       # 두 step에 한 번씩 장애물들의 위치 이동
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action)     # action대로 이동 후
        check = self.check_if_reward(self.coords_to_state(next_coords))     # reward가 있는지 확인
        done = check['if_goal']
        reward = check['rewards']

        self.canvas.tag_raise(self.rectangle)

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.rectangle))
        agent_x = location[0]
        agent_y = location[1]

        states = list()     # 장애물/목적지 간의 상대위치, 목적지의 이동방향, rewards

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)
            if reward['reward'] < 0:                # 장애물에 대해선,
                states.append(-1)                   # reward -1 추가
                states.append(reward['direction'])  # 장애물이 움직이는 direction 추가
            else:                                   # 목적지에 대해선,
                states.append(1)                    # reward +1 추가

        return states       # states = [delta x, delta y, -1, direction, ...]

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1:     # 목적지는 위치 변경 없이 저장
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)      
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)    # 장애물들은 위치를 옮긴 후 저장
        return new_rewards

    def move_const(self, target):

        s = self.canvas.coords(target['figure'])    # target의 coordinate

        base_action = np.array([0, 0])

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:   # 장애물이 map의 오른쪽 끝에 도달할 경우 이동방향 전환
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle  # target이 장애물인데 목적지의 위치에 있을 경우
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def move(self, target, action):     # action으로 전달된 index대로 target 이동
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # Up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # Down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # Right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # Left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(0.05)
        self.update()