import numpy as np
import random
from collections import defaultdict
from environment_ss import Env


class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01       # alpha
        self.discount_factor = 0.9      # gamma
        self.epsilon = 0.1              # epsilon
        
        # 상하좌우로의 action에 대한 Q-value
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 sample로부터 Q-value function을 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        ##################################################################################
        # <<각 time-step마다 s,a,r,s',a'를 sampling하여 Q-value function을 업데이트>>
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q
        ##################################################################################

    # Q-value function에 따라 epsilon greedy하게 action 선택
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # random actions
            action = np.random.choice(self.actions)
        else:
            # Q-function에 따른 action
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태를 초기화
        state = env.reset()
        action = agent.get_action(str(state))

        while True:
            env.render()

            # 다음 state로 이동
            # Reward는 숫자이고, 완료 여부는 boolean
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택
            next_action = agent.get_action(str(next_state))

            # <s,a,r,s',a'>로 Q-value function을 업데이트
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            # 모든 Q-value를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
