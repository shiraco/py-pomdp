"""Loading POMDP environment files and policy files into python objects.
Contains methods to perform POMDP tasks, like finding the optimal action and
updating the belief.

TODO(mbforbes): Check model after construction to provide sanity
     check for specified pomdp environment (e.g. observation and
     transition probabilities sum to 1.0)

"""

__author__ = 'mbforbes'

from xml.etree import ElementTree

import numpy as np


class POMDP(object):
    """Class that a user should interact with. Contains a POMDP environment and
    policy.

    Attributes:
        pomdpenv:    POMDPEnvironment
        pomdppolicy: POMDPPolicy
        belief:      np.array

    """

    def __init__(self, pomdp_env_filename: str, pomdp_policy_filename: str, prior: np.array):
        """

        Args:
            pomdp_env_filename:
            pomdp_policy_filename:
            prior:
        """
        self.pomdpenv = POMDPEnvironment(pomdp_env_filename)
        self.pomdppolicy = POMDPPolicy(pomdp_policy_filename)
        self.belief = prior

    def get_action_str(self, action_num: int) -> str:
        """Return a string representing the action with the given num.

        This is the name given to it in the pomdp environment file.

        Args:
            action_num:

        Returns:
            str:

        """
        return self.pomdpenv.actions[action_num]

    def get_belief_str(self) -> str:
        """Return a string representing the belief.

        Returns:
            str: belief_str

        """
        res = '['
        for num in self.belief:
            for val in num:
                res = res + str(val) + ', '
        return res[:-2] + ']'

    def get_best_action(self):
        """Return tuple (best_action_num, expected_reward_for_this_action).

        Returns:
            best_action:

        """
        return self.pomdppolicy.get_best_action(self.belief)

    def get_obs_num(self, obs_name: str) -> int:
        """Get the observation number.

        This is the observation named obs_name corresponds to.

        Args:
            obs_name:

        Returns:
            int:

        """
        return self.pomdpenv.observations.index(obs_name)

    def update_belief(self, action_num: int, observation_num: int) -> None:
        """

        Args:
            action_num:
            observation_num:

        Returns:
            None

        """
        self.belief = self.pomdpenv.update_belief(
            self.belief, action_num, observation_num)

    def belief_dump(self) -> None:
        """Used for debugging a two state POMDP.

        Sets the belief to a whole bunch of different values and outputs
        the optimal action for each.

        Returns:
            None

        """
        # adjust to change granularity
        increment = 0.01

        # save old belief
        old_belief = self.belief

        # test and display!
        pieces = int(1.0 / increment)
        for x in range(pieces):
            b1 = x * increment
            b2 = 1.0 - b1
            self.belief = np.array([[b1], [b2]])
            best_action, reward = self.get_best_action()
            print(b1, b2, '\t', self.get_action_str(best_action))

        # restore to old belief
        self.belief = old_belief


class POMDPEnvironment(object):
    """Class of POMDP environment.

    Attributes:
        contents: temporal data from file records
        discount (float):
        values (str):
        states (list<str>):
        actions (list<str>):
        observations (list<str>):
        T (dict<(int, int, int), float>):
            T[(action, start_state, next_state)] = prob
        Z (dict<(int, int, int), float>):
            Z[(action, next_state, obs)] = prob
        R (dict<(int, int, int, int), float>):
            R[(action, start_state, next_state, obs)] = prob

    """

    def __init__(self, filename: str):
        """Parse .pomdp file and loads info into this object's fields.

        Attributes:
            filename:
                discount (float):
                values (str):
                states (list<str>):
                actions (list<str>):
                observations (list<str>):
                T (dict):
                O (dict):
                R (dict):

        """
        f = open(filename, 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith('#') or x.isspace()))
        ]

        # set up transition function T, observation function Z, and
        # reward R
        self.T = {}
        self.Z = {}
        self.R = {}

        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('states'):
                i = self.__get_states(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('T'):
                i = self.__get_transition(i)
            elif line.startswith('O'):
                i = self.__get_observation(i)
            elif line.startswith('R'):
                i = self.__get_reward(i)
            else:
                raise Exception('Unrecognized line: ' + line)

        # cleanup
        f.close()

    def __get_discount(self, i: int) -> int:
        """Get discount from file and set to the self.discount.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        """
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i: int) -> int:
        """Get value from file and set to the self.value.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        """
        # Currently just supports "values: reward". I.e. currently meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_states(self, i: int) -> int:
        """Get states from file and set to the self.states.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        """
        # TODO: support states as number
        line = self.contents[i]
        self.states = line.split()[1:]
        return i + 1

    def __get_actions(self, i: int) -> int:
        """Get actions from file and set to the self.actions.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        """
        line = self.contents[i]
        self.actions = line.split()[1:]
        return i + 1

    def __get_observations(self, i: int) -> int:
        """Get observations from file and set to the self.observations.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        """
        line = self.contents[i]
        self.observations = line.split()[1:]
        return i + 1

    def __get_transition(self, i: int) -> int:
        """Get transition, T from file and set to the self.T.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        Raises:
            Exception:
                * 'Cannot parse line'

        """
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            prob = float(pieces[3])
            self.T[(action, start_state, next_state)] = prob
            return i + 1

        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i + 1]
            prob = float(next_line)
            self.T[(action, start_state, next_state)] = prob
            return i + 2

        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.T[(action, start_state, j)] = prob
            return i + 2

        elif len(pieces) == 1:
            next_line = self.contents[i + 1]
            if next_line == 'identity':
                # case 4: T: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        prob = 1.0 if j == k else 0.0
                        self.T[(action, j, k)] = prob
                return i + 2

            elif next_line == 'uniform':
                # case 5: T: <action>
                # uniform
                prob = 1.0 / float(len(self.states))
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        self.T[(action, j, k)] = prob
                return i + 2

            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.states)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.T[(action, j, k)] = prob
                    next_line = self.contents[i + 2 + j]
                return i + 1 + len(self.states)

        else:
            raise Exception('Cannot parse line ' + line)

    def __get_observation(self, i: int):
        """Get observation, O from file and set to the self.Z.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        Raises:
            Exception:
                * 'Cannot parse line'

        """

        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            prob = float(pieces[3])
            self.Z[(action, next_state, obs)] = prob
            return i + 1

        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            next_line = self.contents[i + 1]
            prob = float(next_line)
            self.Z[(action, next_state, obs)] = prob
            return i + 2

        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            next_state = self.states.index(pieces[1])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Z[(action, next_state, j)] = prob
            return i + 2

        elif len(pieces) == 1:
            next_line = self.contents[i + 1]
            if next_line == 'identity':
                # case 4: O: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        prob = 1.0 if j == k else 0.0
                        self.Z[(action, j, k)] = prob
                return i + 2

            elif next_line == 'uniform':
                # case 5: O: <action>
                # uniform
                prob = 1.0 / float(len(self.observations))
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        self.Z[(action, j, k)] = prob
                return i + 2

            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.observations)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.Z[(action, j, k)] = prob
                    next_line = self.contents[i + 2 + j]
                return i + 1 + len(self.states)

        else:
            raise Exception('Cannot parse line: ' + line)

    def __get_reward(self, i: int) -> int:
        """Get reword, R from file and set to the self.R.

        Notes:
            Wild card * are allowed when specifying a single reward probability.
            They are not allowed when specifying a vector or matrix of probabilities.

        Args:
            i: current line number

        Returns:
            int: next line number to read

        Raises:
            Exception:
                * 'Cannot parse line'

        """
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 5 or len(pieces) == 4:
            # case 1:
            # R: <action> : <start-state> : <next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            start_state_raw = pieces[1]
            next_state_raw = pieces[2]
            obs_raw = pieces[3]
            prob = float(pieces[4]) if len(pieces) == 5 \
                else float(self.contents[i + 1])
            self.__reward_ss(
                action, start_state_raw, next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) == 5 else i + 2

        elif len(pieces) == 3:
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.R[(action, start_state, next_state, j)] = prob
            return i + 2

        elif len(pieces) == 2:
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i + 1]
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.R[(action, start_state, j, k)] = prob
                next_line = self.contents[i + 2 + j]
            return i + 1 + len(self.states)

        else:
            raise Exception('Cannot parse line: ' + line)

    def __reward_ss(self, a, start_state_raw, next_state_raw, obs_raw, prob) -> None:
        """reward_ss means we're at the start state of the unrolling of the
        reward expression.

        start_state_raw could be * or the name of the real start state.

        """
        if start_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ns(a, i, next_state_raw, obs_raw, prob)
        else:
            start_state = self.states.index(start_state_raw)
            self.__reward_ns(a, start_state, next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, start_state, next_state_raw, obs_raw, prob) -> None:
        """reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start state,

        and next_state_raw could be * or the name of the real next state.

        """
        if next_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ob(a, start_state, i, obs_raw, prob)
        else:
            next_state = self.states.index(next_state_raw)
            self.__reward_ob(a, start_state, next_state, obs_raw, prob)

    def __reward_ob(self, a, start_state, next_state, obs_raw, prob) -> None:
        """reward_ob means we're at the observation of the unrolling of the
        reward expression. start_state is the number of the real start state,
        next_state is the number of the real next state, and.

        obs_raw could be * or the name of the real observation.

        """
        if obs_raw == '*':
            for i in range(len(self.observations)):
                self.R[(a, start_state, next_state, i)] = prob
        else:
            obs = self.observations.index(obs_raw)
            self.R[(a, start_state, next_state, obs)] = prob

    def update_belief(self, prev_belief: np.array, action_num: int, observation_num: int) -> np.array:
        """

        Notes:
            Note that a POMDPEnvironment doesn't hold beliefs, so this takes and returns a belief vector.

        Args:
            prev_belief:
            action_num:
            observation_num:

        Returns:
            np.array:

        """
        b_new_nonnormalized = []
        for s_prime in range(len(self.states)):
            p_o_prime = self.Z[(action_num, s_prime, observation_num)]
            summation = 0.0
            for s in range(len(self.states)):
                p_s_prime = self.T[(action_num, s, s_prime)]
                b_s = float(prev_belief[s])
                summation += p_s_prime * b_s
            b_new_nonnormalized.append(p_o_prime * summation)

        # normalize
        b_new = []
        total = sum(b_new_nonnormalized)
        for b_s in b_new_nonnormalized:
            b_new.append([b_s / total])
        return np.array(b_new)

    def print_summary(self) -> None:
        """Print POMDPEnvironment summary.

        Returns:
            None

        """
        print('discount:', self.discount)
        print('values:', self.values)
        print('states:', self.states)
        print('actions:', self.actions)
        print('observations:', self.observations)
        print('')
        print('T:', self.T)
        print('')
        print('Z:', self.Z)
        print('')
        print('R:', self.R)


class POMDPPolicy(object):
    """Class of POMDP policy.

    Attributes:
        action_nums:   The full list of action (numbers) from the alpha
                       vectors. In other words, this saves the action
                       number from each alpha vector and nothing else,
                       but in the order of the alpha vectors.

        pMatrix:       The policy matrix, constructed from all of the
                       alpha vectors.

    """

    def __init__(self, filename: str):
        """Get action_nums and pMatrix from file and set to the
        self.action_nums and self.pMatrix.

        Args:
            filename:

        """

        tree = ElementTree.parse(filename)
        root = tree.getroot()
        avec = list(root)[0]
        alphas = list(avec)
        self.action_nums = []
        val_arrs = []
        for alpha in alphas:
            self.action_nums.append(int(alpha.attrib['action']))
            vals = []
            for val in alpha.text.split():
                vals.append(float(val))
            val_arrs.append(vals)
        self.pMatrix = np.array(val_arrs)

    def get_best_action(self, belief: np.array) -> tuple:
        """Get best_action and expected-reward-for-this-action.

        Args:
            belief:

        Returns:
             tuple: (best-action-num, expected-reward-for-this-action).

        """
        res = self.pMatrix.dot(belief)
        highest_expected_reward = res.max()
        best_action = self.action_nums[res.argmax()]
        return best_action, highest_expected_reward
