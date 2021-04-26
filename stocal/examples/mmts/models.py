#! /usr/bin/env python3
""" Miscellaneous Models Test Suite (MMTS)

This is a collection of models that don't belong to the DSMTS library, but they 
are worth including to allow a more diverse set of models to use for testing.
"""

import abc
import stocal
from stocal import MassAction, Event, TransitionRule, Process, multiset
try:
    import numpy as np
except ImportError:
    logging.error("mmts.models suite requires numpy.")
    sys.exit(1)

from stocal._utils import with_metaclass


class MMTS_Test(with_metaclass(abc.ABCMeta, object)):
    tmax = 100.
    dt = 1.

    def __call__(self, sampler, delta_t=0, tmax=None):
        """Sample species along sampler every delta_t time units

        species is a list of species labels that should be sampled.

        Returns a tuple of two elements, the first is a list of all firing
        times, the second a dictionary that holds for each species the
        list of copy numbers at each corresponding time point. If delta_t
        is given, it specifies the interval at which the sampler is sampled.
        """
        def every(sampler, delta_t, tmax):
            sampler.tmax = sampler.time
            while sampler.time < tmax:
                transitions = {}
                if sampler.steps and sampler.step >= sampler.steps:
                    break
                sampler.tmax += delta_t
                for trans in sampler:
                    transitions[trans] = transitions.get(trans, 0) + 1
                yield transitions

        delta_t = delta_t or self.dt
        tmax = tmax if tmax is not None else self.tmax

        times = [sampler.time]
        counts = {s: np.array([sampler.state[s]]) for s in self.species}
        it = every(sampler, delta_t, tmax) if delta_t else iter(sampler)
        for _ in it:
            times.append(sampler.time)
            for s in self.species:
                counts[s] = np.append(counts[s], [sampler.state[s]])
        return np.array(times), counts

    @abc.abstractproperty
    def process(self): pass

    @abc.abstractproperty
    def initial_state(self): pass

    @abc.abstractproperty
    def species(self): pass


class heteropolymer(MMTS_Test):
    import stocal.examples.pre2017 as pre2017

    MMTS_Test.tmax = 500            # [50, 100, 500, 1000]
    alpha = pre2017.alpha = 1.e-7       # [1.e-10, 1.e-9, 1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3]
    beta = pre2017.beta = 1000**-2       # [10000**-2, 1000**-2, 100**-2]    
    initial_state = pre2017.initial_state = {c: 3000 for c in 'ab'}     # [1000, 2100, 3000, 5000, 10000, 20000, 50000, 100000, 2000000]
    process = pre2017.process
    species = [s for s in pre2017.initial_state]


class brusselator(MMTS_Test):
    import stocal.examples.brusselator as brusselator

    species = ['x', 'y', 'c', 'd']
    process = brusselator.process
    initial_state = {}


class polymerization(MMTS_Test):
    import stocal.examples.polymerization as poly

    species = poly.species
    process = poly.process
    initial_state = poly.initial_state


class toggleswitch(MMTS_Test):
    species = [
        'u',    # promoter for repressor U
        'V',    # repressor V
        'U',    # repressor U
        'x',    # suppressed promoter u
        'y',    # suppressed promoter v
        's',    # inducer for repressor V
        't',    # inducer for repressor U
        'S',    # bound repressor V
        'T'     # bound repressor U
    ]

    process = stocal.Process([
        stocal.MassAction({"v": 1},         {"v": 1, "V": 1},   0.3),       # gene expression of repressor 1 | v --k-> v + V
        stocal.MassAction({"u": 1},         {"u": 1, "U": 1},   0.3),       # gene expression of repressor 2 | u --k-> u + U
        stocal.MassAction({"V": 4, "u": 1}, {"x": 1},           0.70),      # repression of promoter 1 | n*V + u --k+-> x         10 --> 5
        stocal.MassAction({"U": 4, "v": 1}, {"y": 1},           0.70),      # repression of promoter 2 | n*U + v --k+-> y
        stocal.MassAction({"x": 1},         {"V": 4, "u": 1},   .12),       # degradation of promoter 1 suppression | x --(k-)-> n*V + u
        stocal.MassAction({"y": 1},         {"U": 4, "v": 1},   .12),       # degradation of promoter 2 suppression | y --(k-)-> n*U + v
        stocal.MassAction({"V": 1},         {},                 0.0025),    # Decay of repressor | v + V --k-> v
        stocal.MassAction({"U": 1},         {},                 0.0025),    # Decay of repressor | u + U --k-> u
        stocal.MassAction({"s": 1, "V": 1}, {"S": 1},           10.),       # Inducer for repressor V | s + V --k-> S
        stocal.MassAction({"t": 1, "U": 1}, {"T": 1},           10.),       # Inducer for repressor U | t + U --k-> T
        stocal.MassAction({"S": 1},         {"s": 1, "V": 1},   0.05),      # Decay of Inducer and repressor | S --k-> s + V
        stocal.MassAction({"T": 1},         {"t": 1, "U": 1},   0.05),      # Decay of Inducer and repressor | T --k-> t + U
    ])

    initial_state = {"v": 10, "u": 10}


'''
class heteropolymer_02(heteropolymer):
    """ long tmax, small inital_state """
    MMTS_Test.tmax = 1000
    initial_state = {c: 2000 for c in 'ab'}

class heteropolymer_03(heteropolymer):
    """ short tmax, large inital_state """
    MMTS_Test.tmax = 50
    initial_state = {c: 10000 for c in 'ab'}

class heteropolymer_04(heteropolymer):
    """ high alpha """
    alpha = 1.e-3       # [1.e-10, 1.e-9, 1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3]

class heteropolymer_05(heteropolymer):
    """ high beta """
    beta = 100**-2      # [10000**-2, 1000**-2, 100**-2]

class heteropolymer_06(heteropolymer):
    """ original default paramters from pre2017.py file - be warned, "simulations will take a significant amount of time." """
    MMTS_Test.tmax = 100.0
    initial_state = {c: 200000 for c in 'ab'}

class heteropolymer_07(heteropolymer):
    """ parameters from experiment B in <https://doi.org/10.1103/PhysRevE.96.062407> """
    MMTS_Test.tmax = 1000.0
    alpha = 1.e-7      # [1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3]
    beta = 100**-2
    initial_state = {c: 2100 for c in 'ab'}

class heteropolymer_08(heteropolymer):
    """ parameters from experiment C in <https://doi.org/10.1103/PhysRevE.96.062407> """
    MMTS_Test.tmax = 100.0
    alpha = 1.e-10
    beta = 1.e-7    # inbetween 1000**-2 and 10000**-2 (i.e. 1.e-6 and 1.e-8)
    initial_state = {c: 200000 for c in 'ab'}
'''


'''
class MMTS_001_01(MMTS_Test):
    species = ['A', 'B', 'C']
    process = stocal.Process([
        stocal.MassAction(['A', 'B', 'B'], ['C'], 0.01),
        stocal.MassAction(['C'], ['A', 'B', 'B'], 0.01)])
    initial_state = {'A': 100, 'B': 100, 'C': 0}

class MMTS_001_02(MMTS_001_01):    
    species = ['A', 'B']
    process = stocal.Process([
        stocal.MassAction(['A'], ['B'], 0.1)])
    initial_state = {'A': 10000, 'B': 0}

class MMTS_001_03(MMTS_001_01):
    species = ['A', 'B']
    process = stocal.Process([
        stocal.MassAction(['A'], ['B'], 0.1),
        stocal.MassAction(['B'], ['A'], 0.1)])
    initial_state = {'A': 10000, 'B': 0}

class MMTS_001_04(MMTS_001_01):
    species = ['A', 'B', 'C']
    process = stocal.Process([
        stocal.MassAction(['A', 'B'], ['C'], 0.001),
        stocal.MassAction(['C'], ['A', 'B'], 0.001)])
    initial_state = {'A': 100, 'B': 100, 'C': 0}
'''


'''
class SBML_00019(DSMTS_Test):
    species = ['S1', 'S2', 'S3', 'S4']
    process = stocal.Process([
        stocal.MassAction(['S1', 'S2'], ['S3'], 1000.0),
        stocal.MassAction(['S3'], ['S1', 'S2'], 0.9),
        stocal.MassAction(['S3'], ['S1', 'S4'], 0.7)])
    initial_state = {'S1': 0.002, 'S2': 0.002, 'S3': 0, 'S4': 0}

class SBML_00017(DSMTS_Test):
    species = ['S1', 'S2', 'S3', 'S4']
    process = stocal.Process([
        stocal.MassAction(['S1', 'S2'], {'S3': 1, 'S4': 2}, 7.5)
        stocal.MassAction(['S3', 'S4'], ['S1', 'S2'])
    ])
    initial_state = {'S1': 0.1, 'S2': 0.1, 'S3': 0.1, 'S4': 0.1}

class SBML_00018(DSMTS_Test):
    species = ['S1', 'S2', 'S3', 'S4']
    process = stocal.Process([
        stocal.MassAction(['S1'], ['S1'], 0.75),
        stocal.MassAction(['S2'], ['S2'], 0.25),
        stocal.MassAction(['S2'], ['S3', 'S4'], 0.4),
        stocal.MassAction(['S3', 'S4'], ['S2'], 0.1)
    ])
    initial_state = {'S1': 0.0001, 'S2': 0.0002, 'S3': 0, 'S4': 0}

class SBML_00080(DSMTS_Test):
    def multiply():
        Transition = stocal.MassAction
        def novel_reactions(self, x, y):
            yield self.Transition([x*y], ['S5'], 1.0)

    def func1():
        Transition = stocal.MassAction
        def novel_reactions(self, x, y):
            yield self.Transition([x/(1+y)], ['S4'], 1.0)
    
    P1 = 2.5
    species = ['S1', 'S2', 'S3', 'S4', 'S5']
    process = stocal.Process(
        transitions=[
            stocal.MassAction(['S1'], ['S3'], 0.1),
            stocal.MassAction(['S3'], ['S2'], 0.15*process.state['S5'])
        ],
        rules=[
            multiply(),
            func1()
        ]
    )
    initial_state = {'S1': 1.0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0}

class SBML_00128(DSMTS_Test):
    species = ['S1', 'S2', 'S3', 'S4']
    process = stocal.Process([
        stocal.MassAction(['S1', 'S2'], {'S3': 1, 'S4': 2}, 750),
        stocal.MassAction(['S3', 'S4'], ['S1', 'S2'], 250)
    ])
    initial_state = {'S1': 0.001, 'S2': 0.001, 'S3': 0.001, 'S4': 0.002}

class SBML_00190(DSMTS_Test):
    species = ['S1', 'S2']
    process = stocal.Process([
        stocal.MassAction(['S1'], ['S2'], 1.5*0.05)
    ])
    initial_state = {'S1': 10, 'S2': 0}

class SBML_00204(DSMTS_Test):
    species = ['S1', 'S2', 'S3', 'S4']
    process = stocal.Process([
        stocal.MassAction(['S1', 'S2'], ['S3', 'S4'], 7500000),
        stocal.MassAction(['S3', 'S4'], ['S1', 'S2'], 2500000)
    ])
    initial_state = {'S1': 0.000001, 'S2': 0.0000015, 'S3': 0.0000020, 'S4': 0.000001}

class SBML_00491(MMTS_Test):
    species = ['S', 'F', 'G', 'P', 'T', 'R', 'X']
    process = stocal.Process([
        stocal.MassAction(['S', 'S', 'F'], ['F'], 0.1),
        stocal.MassAction(['F', 'G'], ['G'], 0.1),
        stocal.MassAction(['G', 'P'], ['G', 'P'], 0.1),
        stocal.MassAction(['G', 'P'], ['G', 'P', 'T'], 0.1),
        stocal.MassAction(['T', 'R'], ['T', 'R'], 0.1),
        stocal.MassAction(['T', 'R'], ['T', 'R'], 0.1),
        stocal.MassAction(['T'], [], 0.1),
        stocal.MassAction(['S'], [], 0.1),
        stocal.MassAction(['X'], [], 0.1)
    ])
    initial_state = {'S': 1, 'F': 1, 'G': 1, 'P': 1, 'T': 1, 'R': 1, 'X': 1}
'''

