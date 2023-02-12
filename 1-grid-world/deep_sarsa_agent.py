"""
5x5 grid world에서 장애물 1, 2, 3, 목적지 4가 각각 (0, 1), (1, 2), (2, 3), (4, 4)에서 시작해
좌우로 번갈아가며 움직일 때 agent와 각 장애물/목적지 간의 위치간격 및 장애물의 이동하는 방향 정보를
DNN에 입력하여 agent의 action에 대한 Q-value를 [stay, up, down, right, left]의 순서로 도출
"""

import copy
import pylab
import random
import numpy as np
from environment_ds import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000

# Grid world 예제에서의 deep-SALSA 에이전트
class DeepSARSAgent:
    def __init__(self):
        self.load_model = True
        # 에이전트가 가능한 모든 action 정의
        self.action_space = [0, 1, 2, 3, 4]
        # State의 크기와 action의 크기 정의
        self.action_size = len(self.action_space)
        
        # state는 agent와 장애물 및 목적지 간의 좌표 차이 및 direction, rewards로 구성되어 있음
        # [delta x_1, delta y_1, -1, direct_1, ... , delta x_4, delta y_4, +1]의 순서
        self.state_size = 15
        
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        
        self.model = self.build_model()
        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model_ds/deep_sarsa_trained.h5')

    # Map의 전체 state 정보가 입력, 해당 states에 대한 각 action의 Q-value function이 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Epsilon-greedy method
    def get_action(self, state):        # 전달된 state를 인공신경망에 입력하여 다음 action 반환
        if np.random.rand() <= self.epsilon:
            # Random actions
            return random.randrange(self.action_size)
        else:
            # Q-value function에 따른 action
            state = np.float32(state)
            q_values = self.model.predict(state)
            # Predicted Q-value가 가장 높은 방향을 index 형태로 return
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        ##################################################################################
        # <<움직이는 장애물과 목적지의 정보를 입력하여 agent의 현 state에 대한 Q-value function을
        # 출력하는 DNN을 생성 후 SALSA의 Q-value function을 target으로 설정해 학습>>
        target = self.model.predict(state)[0]
        
        # Q-value update of SALSA
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])
        
        # 출력 값 reshape
        target = np.reshape(target, [1, 5])
        # 인공신경망 업데이트
        self.model.fit(state, target, epochs=1, verbose=0)
        ##################################################################################


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSARSAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, 15])
        # Initially, [0, 1, -1, -1,   1, 2, -1, -1,   2, 3, -1, -1,   4, 4, +1]

        while not done:
            global_step += 1

            # 현재 state에 대한 action 선택
            action = agent.get_action(state)
            # 선택한 action으로 환경에서 한 time-step 진행 후 sample 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            # Sample로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:        # 목적지에 도달할 경우 step 함수에서 done <= True
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("C:/RL_study/save_graph_ds/deep_sarsa_.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights("C:/RL_study/save_model_ds/deep_sarsa.h5")