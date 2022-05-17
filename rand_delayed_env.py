import numpy as np
from collections import deque
from delayed_env import DelayedEnv
from dqn_agents import reshape_state
from rand_delays import Delay


class RandDelayedEnv(DelayedEnv):
    def __init__(self, orig_env, delay_value, delay):
        super().__init__(orig_env, delay_value)
        self.delay = delay
        self.pending_obs = deque()

    def step(self, action):
        if self.delay_value > 0 and self.delay.action:
            self.pending_actions.append(action)
            if len(self.pending_actions) - 1 >= self.delay_value:
                # Update pending actions if last taken action has shorter delays than previous ones
                executed_action = self.delay.find_executed_action(self.pending_actions)
                self.pending_actions.popleft()
                #executed_action = self.pending_actions.popleft()
            else:
                curr_state = reshape_state(self.get_curr_state(), self.is_atari_env, self.state_size)
                executed_action = self.trained_non_delayed_agent.act(curr_state)
        else:
            executed_action = action

        # At undelayed env
        next_state, reward, done, info = self.orig_env.step(executed_action)
        #print(executed_action)
        info['executed_action'] = executed_action

        '''if self.delay.obs:
            self.pending_obs.append((next_state, reward, done, info))
            if len(self.pending_obs) > self.delay_value:
                # Return to delayed agent
                next_state, reward, done, info = self.pending_obs.popleft()
            else:
                print('small pending_obs queue < delay_value')
        '''
        return next_state, reward, done, info

    def get_pending_actions(self):
        pending_actions = super(RandDelayedEnv, self).get_pending_actions()
        return pending_actions.copy()
        #return deque(list(pending_actions)[:-1])
