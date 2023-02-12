import numpy as np
import random
from environment_ql import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01       # alpha
        self.discount_factor = 0.9      # gamma
        self.epsilon = 0.1              # epsilon
        
        # 상하좌우로의 action에 대한 Q-value
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'>의 sample로부터 Q-value function 업데이트
    def learn(self, state, action, reward, next_state):
        ##################################################################################
        # <<SARSA와 달리, a'를 behavior policy에 따라 sampling하지 않고 greedy한 alternative policy로 결정>>
        current_q = self.q_table[state][action]
        # Bellman optimality equation을 사용한 Q-value function의 업데이트
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)
        ##################################################################################

    # Q-value function에 따라 epsilon greedy하게 action 선택
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # random actions
            action = np.random.choice(self.actions)
        else:
            # Q-value function에 따른 action
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
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태를 초기화
        state = env.reset()

        while True:
            env.render()

            # 다음 state로 이동
            # Reward는 숫자이고, 완료 여부는 boolean
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            # <s,a,r,s'>로 Q-value function을 업데이트
            agent.learn(str(state), action, reward, str(next_state))
            
            state = next_state
            
            # 모든 Q-value를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break