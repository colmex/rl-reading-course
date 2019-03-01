import random, csv

random.seed()

class TestBed:
    def __init__(self, agent, stationary=True, arms=10):
        self.stationary = stationary
        self.agent = agent
        if stationary:
            raise NotImplementedError
        else:
            self.arms = []
            for arm in range(arms):
                arm = {
                    'mean': 0,
                    'variance': 1,
                }

                self.arms.append(arm)

    def run_test(self, steps, runs):
        all_runs = []
        for run in range(runs):
            run_data = []
            for _ in range(steps):
                action = self.agent.pick_action()

                reward = self._pull(action)
                self.agent.process_reward(action, reward)

                run_data.append({
                    'reward': reward,
                    'optimal': action in self._get_optimal_action(),
                })
            all_runs.append(run_data)

        step_summary = []
        for step in range(steps):
            rewards = []
            optimal = []
            for run in all_runs:
                rewards.append(run[step]['reward'])
                optimal.append(run[step]['optimal'])

            average_reward = sum(rewards)/len(rewards)
            percent_optimal = sum(optimal)/float(len(optimal))
            step_summary.append({
                'average_reward': average_reward,
                'percent_optimal': percent_optimal,
            })

        return step_summary

    def _get_optimal_action(self):
        best_actions = []
        for arm, info in enumerate(self.arms):
            if not best_actions:
                best_actions.append((arm, info))
                continue

            if best_actions[0][1]['mean'] < info['mean']:
                best_actions = [(arm, info)]
            elif best_actions[0][1]['mean'] == info['mean']:
                best_actions.append((arm, info))
            
        random.shuffle(best_actions)
        return list(map(lambda x: x[0], best_actions))

    def _pull(self, arm):
        if self.stationary:
            raise NotImplementedError
        else:
            mean = self.arms[arm]['mean']
            std = self.arms[arm]['variance'] ** 2
            
            retval = random.gauss(mean, std)

            self.arms[arm]['mean'] = mean + random.gauss(0, 0.01)

            return retval

class Method:
    def pick_action(self):
        raise NotImplementedError

    def process_reward(self):
        raise NotImplementedError

class SampleAverageArms(Method):
    def __init__(self, starting_values, arms=10):
        self.arms = []

        for arm in range(arms):
            self.arms.append({
                'value': float(starting_values),
                'number_of_rewards': 0
            })
    def pick_action(self):
        best_actions = []
        for arm, info in enumerate(self.arms):
            if not best_actions:
                best_actions.append((arm, info))
                continue

            if best_actions[0][1]['value'] < info['value']:
                best_actions = [(arm, info)]
            elif best_actions[0][1]['value'] == info['value']:
                best_actions.append((arm, info))
            
        random.shuffle(best_actions)
        return best_actions[0][0]

    def process_reward(self, arm, reward):
        picked_arm = self.arms[arm]
        self.arms[arm]['number_of_rewards'] += 1
        self.arms[arm]['value'] = picked_arm['value'] + (float(reward) - picked_arm['value'])/picked_arm['number_of_rewards']

class EpsilonGreedy(Method):
    def __init__(self, starting_values, step_size, epsilon, arms=10):
        self.epsilon = epsilon
        self.step_size = step_size
        self.arms = []
        for arm in range(arms):
            self.arms.append({
                'value': float(starting_values),
                'number_of_rewards': 0
            })

    def pick_action(self):
        if random.random() > self.epsilon:
            best_actions = []
            for arm, info in enumerate(self.arms):
                if not best_actions:
                    best_actions.append((arm, info))
                    continue

                if best_actions[0][1]['value'] < info['value']:
                    best_actions = [(arm, info)]
                elif best_actions[0][1]['value'] == info['value']:
                    best_actions.append((arm, info))
                
            random.shuffle(best_actions)
            return best_actions[0][0]
        else:
            return random.randint(0,9)

    def process_reward(self, arm, reward):
        picked_arm = self.arms[arm]
        self.arms[arm]['number_of_rewards'] += 1
        self.arms[arm]['value'] = picked_arm['value'] + (float(reward) - picked_arm['value'])*self.step_size

if __name__ == '__main__':
    testbed = TestBed(
        SampleAverageArms(0),
        stationary=False
    )
    with open('sample_average.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for index, step in enumerate(testbed.run_test(10000, 1000)):
            writer.writerow([ index, step['average_reward'], step['percent_optimal']])

    testbed = TestBed(
        EpsilonGreedy(0, 0.1, 0.1),
        stationary=False
    )

    with open('episolon_greedy.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for index, step in enumerate(testbed.run_test(10000, 1000)):
            writer.writerow([ index, step['average_reward'], step['percent_optimal']])
