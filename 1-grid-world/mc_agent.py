import numpy as np
import random
from collections import defaultdict
from environment_mc import Env


# Monte_Carlo 에이전트 (모든 episode 각각의 sample로부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01       # alpha
        self.discount_factor = 0.9      # gamma
        self.epsilon = 0.1              # epsilon
        self.samples = []               # Episode sequence를 저장
        self.value_table = defaultdict(float)

    # 메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트
    def update(self):
        G_t = 0
        visit_state = []
        
        ##################################################################################
        # <<끝까지 진행된 episode에 대해 지나온 state에 대한 value function을 거꾸로 계산>>
        for reward in reversed(self.samples):
            state = str(reward[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = reward[1] + self.discount_factor * G_t
                value = self.value_table[state]
                self.value_table[state] = (value +
                                           self.learning_rate * (G_t - value))
        ##################################################################################

    # Q-function에 따라 epsilon greedy하게 action 선택
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # random actions
            action = np.random.choice(self.actions)
        else:
            # Q-function에 따른 action
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # 가능한 다음 모든 state들을 반환
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state


# Main function
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태를 초기화
        state = env.reset()
        action = agent.get_action(state)

        while True:
            env.render()

            # 다음 state로 이동
            # Reward는 숫자이고, 완료 여부는 boolean
            next_state, reward, done = env.step(action)
            agent.save_sample(next_state, reward, done)

            # 다음 action 받아옴
            action = agent.get_action(next_state)

            # Episode가 완료됐을 때, Q-function 업데이트
            if done:
                agent.update()
                agent.samples.clear()
                break