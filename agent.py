import numpy as np
class Agent:
    def __init__(self, Q, mode="mc_control", nA=6, alpha = 0.01, gamma = 0.99):
        self.Q = Q
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.trajectory = []


    def get_probs(self,Q_s, epsilon, nA):
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s



    def select_action(self, state, eps):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #####################################
        # replace this with your code !!!!!!!
        probs = self.get_probs(self.Q[state], eps, self.nA)
        action = np.random.choice(np.arange(self.nA), p=probs) if state in self.Q else np.random.choice(self.nA)
        ####################################

        return action

    def update_Q(self,episode, uQ, alpha, gamma): # Monte-Carlo Control 을 위한 Q 업데이트 함수입니다.


        for s, a, r in episode:
            idex = next(i for i, x in enumerate(episode) if x[0] == s)
            G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode[idex:])])
            uQ[s][a] = uQ[s][a] + alpha * (G - uQ[s][a])

        return uQ

    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.mode == 'mc_control':
            self.alpha=0.001 # 기본설정된 0.01로 여러번 시도하였으나 모두 실패하여 learning rate를 조정하기로 하였습니다.
            if done: # 최종 reward를 달성하거나, 200번을 모두 수행하였을때 trajectory에 저장된 experience를 바탕으로 기대값을 계산하여 Q를 업데이트합니다.
                self.trajectory.append((state, action, reward))
                self.Q = self.update_Q(self.trajectory,self.Q,self.alpha,self.gamma)
                self.trajectory=[]
            else:
                self.trajectory.append((state,action,reward))
        elif self.mode == 'q_learning':
            new_value = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
            self.Q[state][action] = new_value



