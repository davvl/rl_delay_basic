import numpy as np
from collections import deque
from delayed_env import DelayedEnv
from dqn_agents import reshape_state
from rand_delays import Delay

MAX_DELAY_SHIFT=1

class RandDelayedEnv(DelayedEnv):
    def __init__(self, orig_env, delay_value, action=False, obs=False, p=0.2):
        super().__init__(orig_env, delay_value)
        self.pending_obs = deque()
        self.past_actions = deque(np.zeros(self.delay_value, dtype=np.uint8))
        self.delay_RV = deque(np.zeros(self.delay_value, dtype=np.uint8))
        self.p = p
        self.obs_up = obs
        self.action_up = action
        self.action_ix = 0
        self.waiting_time = 0

    def switch_action_ix(self):
        if self.action_ix == 0:
            self.action_ix = -1
        else:
            self.action_ix = 0

    def action_delay(self, action=None):
        self.waiting_time -= 1
        if self.waiting_time <= 0:
            return self.action_ix
        rand = np.random.rand()
        if rand < self.p:
            #action_ix = -1
            self.switch_action_ix()
            self.waiting_time = 500
        #elif rand > 1 - self.p:
        #    action_ix = 0
        return self.action_ix

    def find_executed_action(self, past_actions, pending_actions, ix=0, random_walk=False):
        # Random walk step
        if random_walk:
            new_action_delay_shift = self.action_delay()
            self.delay_RV.append(new_action_delay_shift)

        action_ix = -MAX_DELAY_SHIFT + ix
        #if not random_walk:
            #print('LA')
        for i in ix + np.arange(-MAX_DELAY_SHIFT+1, MAX_DELAY_SHIFT+1):
            if i >= len(self.delay_RV):
                #print('Warning: i is bigger than len(delay_RV). Add restriction to i values')
                i = len(self.delay_RV)-1
            #print(i, self.delay_RV, random_walk)
            eff_range = self.delay_RV[i] + i
            if eff_range <= ix:
                action_ix = i

        if random_walk:
            self.delay_RV.popleft()

        '''
        for i in range(1, min(delay+1, len(pending_actions))):
            #print(pending_actions, i)
            if self.action_delay(pending_actions[i]) <= delay - i:
               executed_action = pending_actions[i]
        '''

        if action_ix < 0:
            return past_actions[action_ix]
        else:
            return pending_actions[action_ix]

    def step(self, action):
        if self.delay_value > 0 and self.action_up:
            self.pending_actions.append(action)
            if len(self.pending_actions) - 1 >= self.delay_value:
                # Update pending actions if last taken action has shorter delays than previous ones
                executed_action = self.find_executed_action(self.past_actions, self.pending_actions, random_walk=True)
                past_action = self.pending_actions.popleft()
            else:
                curr_state = reshape_state(self.get_curr_state(), self.is_atari_env, self.state_size)
                executed_action = self.trained_non_delayed_agent.act(curr_state)
            self.past_actions.append(past_action)
            self.past_actions.popleft()
        else:
            executed_action = action
        #print(self.executed_actions, self.pending_actions)
        # At undelayed env
        next_state, reward, done, info = self.orig_env.step(executed_action)
        info['executed_action'] = executed_action

        if self.obs_up:
            assert not self.action_up
            self.pending_actions.append(executed_action)
            self.pending_actions.popleft()
            self.pending_obs.append((next_state, reward, done, info))
            if len(self.pending_obs) > self.delay_value:
                # Return to delayed agent
                next_state, reward, done, info = self.pending_obs.popleft()
            else:
                print('small pending_obs queue < delay_value')
        return next_state, reward, done, info

    def get_pending_actions(self):
        pending_actions = super(RandDelayedEnv, self).get_pending_actions()
        return pending_actions.copy()
        #return deque(list(pending_actions)[:-1])

    def get_past_actions(self):
        return self.past_actions