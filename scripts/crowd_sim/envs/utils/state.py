class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states

class ObservableState_noV(object):
    def __init__(self, px, py, radius):
        self.px = px
        self.py = py
        self.radius = radius

        self.position = (self.px, self.py)


    def __add__(self, other):
        return other + (self.px, self.py, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.radius]])


class JointState_noV(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState_noV)

        self.self_state = self_state
        self.human_states = human_states
        

def tensor_to_joint_state(state):
    robot_state, human_states = state

    robot_state = robot_state.cpu().squeeze().data.numpy()
    robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                            robot_state[5], robot_state[6], robot_state[7], robot_state[8])
    if human_states is None:
        human_states = []
    else:
        human_states = human_states.cpu().squeeze(0).data.numpy()
        human_states = [ObservableState(human_state[0], human_state[1], human_state[2], human_state[3],
                                        human_state[4]) for human_state in human_states]

    return JointState(robot_state, human_states)
