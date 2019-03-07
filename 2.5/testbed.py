import random, csv

random.seed()

class TestBed:
    def __init__(self, agent, stationary=True, arms_count=10):
        self.stationary = stationary
        self.agent = agent
        self.arms_count = arms_count
        self._reset_environment()

    def _reset_environment(self):
        if self.stationary:
            raise NotImplementedError
        else:
            self.arms = []
            for arm in range(self.arms_count):
                arm = {
                    'mean': 0,
                    'variance': 1,
                }

                self.arms.append(arm)

    def run_test(self, steps, runs):
        all_runs = []
        for run in range(runs):
            run_data = []
            self._reset_environment()
            self.agent.reset_agent()
            for _ in range(steps):
                action = self.agent.pick_action()
                optimal_actions = self._get_optimal_action()

                reward = self._pull(action)
                self.agent.process_reward(action, reward)

                run_data.append({
                    'reward': reward,
                    'optimal': action in optimal_actions,
                    'optimal_actions': optimal_actions,
                    'action': action,
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

            # Had to bump up the variance on the random walk of the arms because they weren't changing enough for the two algorithms to have any difference
            self.arms[arm]['mean'] = mean + random.gauss(0, 1)

            return retval

class Method:
    def __init__(self, starting_values, epsilon, arms_count=10):
        self.starting_values = starting_values
        self.arms_count = arms_count
        self.epsilon = epsilon

        self.reset_agent()

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
            return random.randint(0,len(self.arms) - 1)

    def process_reward(self):
        raise NotImplementedError

    def reset_agent(self):
        raise NotImplementedError

# Only did incrementally computed because it's equivalent to sample average
class SampleAverageArms(Method):
    def reset_agent(self):
        self.arms = []
        for arm in range(self.arms_count):
            self.arms.append({
                'value': float(self.starting_values),
                'number_of_rewards': 0
            })

    def process_reward(self, arm, reward):
        picked_arm = self.arms[arm]
        self.arms[arm]['number_of_rewards'] += 1
        self.arms[arm]['value'] = picked_arm['value'] + (float(reward) - picked_arm['value'])/picked_arm['number_of_rewards']

# Uses a step value to determine the learning rate
class StepValue(Method):
    def __init__(self, starting_values, epsilon, step_size, arms=10):
        self.step_size = step_size

        super().__init__(starting_values, epsilon, arms)
        
    def reset_agent(self):
        self.arms = []
        for arm in range(self.arms_count):
            self.arms.append({
                'value': float(self.starting_values),
            })

    def process_reward(self, arm, reward):
        picked_arm = self.arms[arm]
        self.arms[arm]['value'] = picked_arm['value'] + (float(reward) - picked_arm['value'])*self.step_size

if __name__ == '__main__':
    testbed = TestBed(
        SampleAverageArms(0, 0.1),
        stationary=False
    )
    with open('sample_average.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for index, step in enumerate(testbed.run_test(10000, 1000)):
            writer.writerow([ index, step['average_reward'], step['percent_optimal']])

    testbed = TestBed(
        StepValue(0, 0.1, 0.1),
        stationary=False
    )

    with open('step_value.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for index, step in enumerate(testbed.run_test(10000, 1000)):
            writer.writerow([ index, step['average_reward'], step['percent_optimal']])
