import numpy as np


def zero_delay(action):
    return 0


def obs_delay(obs):
    assert 0


class Delay:
    def __init__(self, delay_value, action=False, obs=False, delay_known=False, p=0.2):
        self.action = action
        self.obs = obs
        self.action_delay_fn = zero_delay
        self.delay_known = delay_known
        #self.delay_value = delay_value
        self.orig_delay_value = delay_value
        self.action_ix = 0
        self.p = p

        if self.action:
            self.action_delay_fn = self.action_delay
            print('Using stochastic action delay')
        self.observation_delay = zero_delay
        if self.obs:
            self.obs_delay = obs_delay

    def action_delay(self, action):
        action_ix = 0
        rand = np.random.rand()
        if rand < self.p:
            action_ix = -1
        elif rand > 1 - self.p:
            action_ix = 1
        return action_ix

    def find_executed_action(self, executed_actions, pending_actions, random_walk=False):
        action = pending_actions[0]
        self.cnt += 1
        action_ix = 0
        # Random walk step
        if random_walk:
            action_ix = self.action_delay_fn(action)
        '''
        for i in range(1, min(delay+1, len(pending_actions))):
            #print(pending_actions, i)
            if self.action_delay(pending_actions[i]) <= delay - i:
               executed_action = pending_actions[i]
        '''
        #action_ix = self.action_ix
        #print('action_ix={}'.format(action_ix))
        if action_ix < 0:
            return executed_actions[action_ix]
        else:
            return pending_actions[action_ix]

