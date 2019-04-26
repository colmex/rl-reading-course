## The intention of this script is to compare two ways to do value function updates with n-step Temporal Difference
#
# We will be comparing how quickly and closely they estimate the value of the states of a random walk.

import random

random.seed(a=2)

class RandomWalk(object):
    def __init__(self):
        self.states = ['0', 'A','B','C','D','E', 'T']
        self.state = random.choice(['A', 'B', 'C', 'D', 'E'])

    def terminal(self):
        return self.state == 'T' or self.state == '0'

    def step(self):
        current_state_index = self.states.index(self.state)

        if random.randint(0,1):
            self.state = self.states[current_state_index + 1]
            if self.state == 'T':
                return 1
        else:
            self.state = self.states[current_state_index - 1]

        return 0

if __name__ == "__main__":
    step_size = 0.15
    n_step = 3
    gamma = 1
    value = {}
    initial_value = 0

    for episode in range(20000):
        env = RandomWalk()

        state = env.state

        history = [(state, None)]
        while not env.terminal():
            reward = env.step()

            history.append((env.state, reward))

            tau = len(history) - n_step - 1

            if tau >= 0:
                G = sum([  (gamma ** index) * frame[1] for index, frame in enumerate(history[tau + 1:])])

                terminal = bool(sum([ frame[0] in ['0', 'T']  for frame in history[tau + 1:]]))

                if not terminal:
                    G = G + value.get(history[tau + n_step][0], initial_value) * (gamma ** n_step)

                value[history[tau][0]] = value.get(history[tau][0], initial_value) + (step_size * ( G - value.get(history[tau][0], initial_value)))

                print(value)
                print(history)
                print("\n\n\n\n")
