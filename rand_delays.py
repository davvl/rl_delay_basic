import numpy as np


def zero_delay(action):
    return 0


def action_delay(action):
    # action in {0,1}
    return 3*action + 1


def obs_delay(obs):
    assert 0


class Delay:
    def __init__(self, delay_value, action=False, obs=False, delay_known=False):
        self.action = action
        self.obs = obs
        self.action_delay = zero_delay
        self.delay_known = delay_known
        self.delay_value = delay_value

        if self.action:
            self.action_delay = action_delay
            print('Using stochastic action delay')
        self.observation_delay = zero_delay
        if self.obs:
            self.obs_delay = obs_delay

    def find_executed_action(self, pending_actions):
        action = pending_actions[0]
        delay = self.action_delay(action)
        executed_action = action

        for i in range(1, min(delay+1, len(pending_actions))):
            #print(pending_actions, i)
            if self.action_delay(pending_actions[i]) <= delay - i:
                executed_action = pending_actions[i]
        return executed_action
