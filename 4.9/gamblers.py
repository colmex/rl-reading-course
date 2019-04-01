import pprint

def action_max(values, state, initial_value, p_head):
    maximum_value = None
    maximum_action = None
    for action in range(1, min(state, 100 - state) + 1):
        if state + action >= 100:
            head_state_value = 1
        else:
            head_state_value = values.get(state + action, initial_value)

        new_value = p_head * ( head_state_value) + (1 - p_head) * values.get(state - action, initial_value)

        if maximum_value is None:
            maximum_value = new_value
            maximum_action = action
        elif maximum_value < new_value:
            maximum_value = new_value
            maximum_action = action

    print(maximum_value, maximum_action, state, initial_value, p_head)

    return maximum_value, maximum_action

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    values = {
        0: 0
    }

    theta = 0.000000000000000000001

    p_head = 0.4
    initial_value = 0
    while True:
        delta = 0
        for state in range(1, 100):
            value = values.get(state, initial_value)
            values[state], _ = action_max(values, state, initial_value, p_head)

            delta = max(delta, abs(value - values.get(state, initial_value)))

        pp.pprint(values)
        if delta < theta:
            break

    policy = {
        state: action_max(values, state, initial_value, p_head)[1]
        for state in range(1, 100)
    }

    pp.pprint(policy)
