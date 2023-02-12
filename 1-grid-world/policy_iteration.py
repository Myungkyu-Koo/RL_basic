# -*- coding: utf-8 -*-
import random
from environment_pi import GraphicDisplay, Env


class PolicyIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        # Value-function을 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 상 하 좌 우 동일한 확률로 policy 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
        # 목적지에서는 정지
        self.policy_table[2][2] = []
        # Lambda
        self.discount_factor = 0.9

    # Policy iteration
    def policy_evaluation(self):

        next_value_table = [[0.00] * self.env.width
                                    for _ in range(self.env.height)]

        # 모든 state에 대해서 Bellman Expectation equation을 계산
        for state in self.env.get_all_states():
            value = 0.0
            if state == [2, 2]:     # 목적지의 value-function 값은 0으로 고정
                next_value_table[state[0]][state[1]] = value
                continue
            
            ##############################################################################################
            # <<Bellman expectation equation을 통해 현 policy에 대한 모든 state의 value function 업데이트>>
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *          # Bellman Expectation equation
                          (reward + self.discount_factor * next_value))

            # Expected value function 값을 대입
            next_value_table[state[0]][state[1]] = round(value, 2)
            ##############################################################################################

        self.value_table = next_value_table     # Simultaneous update

    # 현재 value-function에 대해서 greedy policy improvement
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            # 반환할 policy 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 action에 대해서 [reward + (lambda * next state's value-function)] 계산
            for index, action in enumerate(self.env.possible_actions):
                # Improvement 시에는 현 policy와 관계없이 가능한 다음 state의 value function을 기반으로 greedy하게
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # Return이 최대인 action의 index(최대가 복수라면 모두)를 추출
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 특정 action을 택할 확률 계산
            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 state에서 policy에 따른 action을 반환
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # policy에 담긴 action 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # state에 따른 policy 반환
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # value-function의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()