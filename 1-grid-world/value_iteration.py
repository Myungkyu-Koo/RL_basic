# -*- coding: utf-8 -*-
from environment_vi import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        # Value-function을 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # Lambda
        self.discount_factor = 0.9

    # Value iteration
    def value_iteration(self):
        
        # 다음 value-function 초기화
        next_value_table = [[0.0] * self.env.width
                                    for _ in range(self.env.height)]
        
        # 모든 state에 대해서 Bellman Expectation equation을 계산
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # value function을 위한 빈 리스트
            value_list = []
            
            ##############################################################################################
            # <<현 policy와 무관하게, Bellman optimality equation을 통해 value function을 최대화하도록 업데이트>>
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))
              
            # Maximum value function 값을 대입
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
            ##############################################################################################
        
        self.value_table = next_value_table

    # 현재 value-function에 대해서 greedy policy improvement
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # 모든 action에 대해서 [reward + (lambda * next state's value-function)] 계산
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            # Return이 최대인 action의 index(최대가 복수라면 모두)를 추출
            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    # value-function의 값을 반환
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()