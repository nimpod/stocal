"""Polymerization

....
"""

import stocal
from stocal import MassAction, Event, TransitionRule, Process, multiset


''' Rules '''
class Dilution(TransitionRule):
    Transition = stocal.MassAction

    def novel_reactions(self, species):
        yield self.Transition([species], [], 0.001)

class Polymerization(TransitionRule):
    Transition = stocal.MassAction

    def novel_reactions(self, k, l):
        yield self.Transition([k,l] [k+l], 10.0)

class DirectionalPolymerization(TransitionRule):
    Transition = stocal.MassAction

    def novel_reactions(self, k, l):
        yield self.Transition([k,l], [k+l], 5.0)
        yield self.Transition([k,l], [l+k], 5.0)

class NondirectionalPolymerization(TransitionRule):
    Transition = stocal.MassAction

    def novel_reactions(self, k, l):
        yield self.Transition([k,l], [Polymer(k+l)], 10.0)

class Hydrolysis(TransitionRule):
    Transition = stocal.MassAction

    def novel_reactions(self, k):
        for i in range(1, len(k)):
            c = 10.0 * i * (len(k) - i)
            yield self.Transition([k], [k[:i], k[i:]], c)

class Complex(tuple):
    @property
    def normalized(self):
        return tuple(sorted(self))

    def __eq__(self, other):
        return self.normalized == other.normalized

    def __ne__(self, other):
        return not self==other

    def __hash__(self):
        return hash(self.normalized)

class Polymer(str):
    @property
    def normalized(self):
        return min(self, ''.join(reversed(self)))

    def __eq__(self, other):
        return self.normalized == other.normalized

    def __ne__(self, other):
        return not self==other

    def __hash__(self):
        return hash(self.normalized)


species = ['A', 'B']
initial_state = {'A': 100}

process = stocal.Process(
    transitions=[ 
        stocal.Event({}, {'A': 1}, 0.0, 1.0),
        stocal.Event({}, {'B': 1}, 0.0, 1.0)
    ],
    rules=[
        Dilution(),
        DirectionalPolymerization(),
        Hydrolysis()
    ]
)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    traj = process.trajectory({}, tmax=100)
    plt.step(traj.times, traj.species["x"])
    plt.show()
