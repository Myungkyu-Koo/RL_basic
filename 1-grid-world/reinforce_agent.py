"""
5x5 grid world에서 장애물 1, 2, 3, 목적지 4가 각각 (0, 1), (1, 2), (2, 3), (4, 4)에서 시작해
좌우로 번갈아가며 움직일 때 agent와 각 장애물/목적지 간의 위치간격 및 장애물의 이동하는 방향 정보를
DNN에 입력하여 agent의 action에 대한 probability, 즉 policy 자체를 출력
"""

import copy
import pylab
import numpy as np
from environment_re import Env
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 2500

tf.compat.v1.disable_v2_behavior()

# Grid world 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        self.load_model = True
        # 에이전트가 가능한 모든 action 정의
        self.action_space = [0, 1, 2, 3, 4]
        # State의 크기와 action의 크기 정의
        self.action_size = len(self.action_space)
        
        # State는 agent와 장애물 및 목적지 간의 좌표 차이 및 direction, rewards로 구성되어 있음
        # [delta x_1, delta y_1, -1, direct_1, ... , delta x_4, delta y_4, +1]의 순서
        self.state_size = 15
        
        self.discount_factor = 0.99 
        self.learning_rate = 0.001
        
        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model_re/reinforce_trained.h5')
    
    # Map의 전체 state 정보 입력, 해당 states에 대한 각 action의 시행확률이 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
    def optimizer(self):
        action = K.placeholder(shape=[None, 5])
        discounted_rewards = K.placeholder(shape=[None, ])
        
        ##################################################################################
        # <<REINFORCE의 score function을 통한 policy function 업데이트>>
        # Cross-entropy loss function 계산
        # model.output이 theta, action_prob이 phi_theta
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)
        
        # 정책신경망을 업데이트하는 훈련함수 생성
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(loss, self.model.trainable_weights)
        train = K.function([self.model.input, action, discounted_rewards], [loss], updates=updates)
        ##################################################################################

        return train

    # 정책신경망으로 action 선택
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # Return 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드의 state, action, reward를 모두 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([np.array(self.states), np.array(self.actions),
                        np.array(discounted_rewards)])
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = ReinforceAgent()

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

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 정책신경망 업데이트
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score,2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph_re/reinforce.png")
            agent.model.save_weights("./save_model_re/reinforce.h5")