import sys
import os
import logging
import logging.config
import argparse
import subprocess
import time
import warnings
import statistics
import csv
import seaborn
import pandas
import psutil
import matplotlib.colors as mcolors

from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from collections import namedtuple
from math import sqrt
from importlib import import_module
from rich.console import Console
from colorama import Fore
from collections import OrderedDict

from multiprocessing import Process, Queue, Lock
from inspect import isclass, isabstract
from itertools import product, islice, cycle

try:
    import numpy as np
    import jinja2
except ImportError:
    logging.error("Example benchmarking.py requires numpy and jinja2.")
    sys.exit(1)


console = Console()
warnings.filterwarnings("ignore", category=Warning)     # ignore depreceation warnings because the warning messages about using AndersonMethod screws up console output
details_debugmode3 = ' [bold white]Simulation [bold green]%s[bold white]/[bold green]%s [bold white]complete ([bold green]%s%s[bold white]) [bold red]%s [bold yellow]%s'
details_debugmode4 = ' [bold white]Simulation [bold green]%s[bold white]/[bold green]%s [bold white]complete ([bold green]%s%s[bold white]) [bold red]%s [bold yellow]%s [bold black](%s, %ss, %s, %s, %s)'
details_columns = "{:>100}".format('(RPS, runtime, #reactions, tmax, final_state)')


def abbreviate_algo_name(algo_name):
    """ Converts name of algorithm to an abbreviated version (e.g. DirectMethod = DM) """
    return ''.join([c for c in algo_name if c.isupper() or c.isdigit()])

def get_modelname_and_algoname_from_config(config):
    """ Returns the model name & algorithm name as tuple, from a config file """
    return (
        str(config[0].__name__),
        abbreviate_algo_name(str(config[1].__name__))
    )

def add_textbox(ax, text, pos):
    """ add a textbox to a matplotlib plot """
    ax.text(pos[0], pos[1], text, horizontalalignment="left", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.6), fontsize=8)
    return ax

def calcprogress(sim_count, sim_max):
    return '{:.4f}'.format(round((sim_count/sim_max)*100, 4))



class Stats(namedtuple('_Stats', ('runs', 'times', 'tmax', 'rps', 'runtime', 'reactions', 'final_states', 'simulated_time', 'unique_species', 'unique_species_len', 'inactive', 'overactive', 'config'))):
    """Simulation result statistics

    Stats collects simulation statistics for use in the DataStore,
    where each algorithm/model configuration has one associated
    Stats instance.
    
    Stats groups the number of runs, a sequence of trajectory time
    points, and various arrays of performance metric data.

    Stats.stdev returns a dictionary of standard deviation sequences
    for each species.
    """
    @property
    def stdev(self):
        """Dictionary of standard deviation sequences for each species"""
        return {
            s: (values/(self.runs-1))**.5
            for s, values in self.M2.items()
        }


class DataStore(object):
    """Persistent store for aggregated data
    
    This class provides a data store that maintains statistics of
    simulation results throughout multiple incarnations of the
    benchmarking script.

    A DataStore accepts individual simulation results for given
    configurations, and allows retrieval of the aggregated statistics
    for a given configuration.
    """

    def __init__(self, path):
        import errno
        self.path = path
        try:
            os.makedirs(self.path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    def __iter__(self):
        """Access all data files and statistics in the data store"""
        import pickle
        for dirpath, _, filenames in os.walk(self.path):
            for name in filenames:
                fname = os.path.join(dirpath, name)
                if fname.endswith('.dat'):
                    try:
                        with open(fname, 'rb') as fstats:
                            config = pickle.load(fstats).config
                        yield fname, self.get_stats(config)
                    except Exception as exc:
                        print(exc)
                        logging.warn("Could not access data in %s", fname)
                        logging.info(exc, exc_info=True)
                        yield fname, None

    def get_path_for_config(self, config):
        """Retrieve path of datafile for a given configuration"""
        model, algo = config
        prefix = '-'.join((algo.__name__, model.__name__))
        return os.path.join(self.path, prefix+'.dat')

    def feed_result(self, result, config):
        """Add a single simulation result for a given configuration

        feed_result uses an online algorithm to update mean and
        standard deviation with every new result fed into the store.
        (The online aggregation is adapted from
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)
        
        At times defined in self.checkpoints, a dump of the current
        statistics is memorized in Stats.conv_mean and Stats.conv_stdev.
        """
        
        import pickle
        from shutil import copyfile
        from math import log, floor

        # extract data from simulation...
        rps_val = result[1]
        runtime_val = result[2]
        reactions_val = result[3]
        unique_species_val = result[5]
        unique_species_len_val = len(result[5])
        final_states_val = result[6]        
        simulated_time_val = result[7]
        tmax_val = result[8]
        result = result[0]
        
        fname = self.get_path_for_config(config)
        if os.path.exists(fname):
            with open(fname, 'rb') as fstats:
                stats = pickle.load(fstats)

                # get values from stats
                N = stats.runs + 1
                times = stats.times
                rps = stats.rps
                runtime = stats.runtime                
                reactions = stats.reactions
                simulated_time = stats.simulated_time
                unique_species_len = stats.unique_species_len
                final_states = stats.final_states

                tmax = stats.tmax
                unique_species = stats.unique_species

                overactive = stats.overactive
                inactive = stats.inactive

                # update with values from most recent simulation
                rps = np.append(rps, rps_val)
                runtime = np.append(runtime, runtime_val)
                reactions = np.append(reactions, reactions_val)
                simulated_time = np.append(simulated_time, simulated_time_val)
                unique_species_len = np.append(unique_species_len, unique_species_len_val)
                final_states = np.append(final_states, [v for v in final_states_val])

                if simulated_time_val < tmax_val:
                    overactive += 1
                if reactions_val == 0:
                    inactive += 1

                copyfile(fname, fname+'~')
        else:
            N = 1
            times = result
            rps = np.array([rps_val])
            runtime = np.array([runtime_val])
            reactions = np.array([reactions_val])
            simulated_time = np.array([simulated_time_val])
            unique_species_len = np.array([unique_species_len_val])
            final_states = np.array([v for v in final_states_val])

            tmax = tmax_val
            unique_species = np.array([v for v in unique_species_val])

            overactive = 0
            inactive = 0
            if simulated_time_val < tmax_val:
                overactive += 1
            if reactions_val == 0:
                inactive += 1

        with open(fname, 'wb') as outfile:
            stats = Stats(N, times, tmax, rps, runtime, reactions, final_states, simulated_time, unique_species, unique_species_len, inactive, overactive, config)
            outfile.write(pickle.dumps(stats))

        if os.path.exists(fname+'~'):
            os.remove(fname+'~')

    def get_stats(self, config):
        """Read stats for a given configuration"""
        import pickle
        fname = self.get_path_for_config(config)
        with open(fname, 'rb') as fstats:
            return pickle.load(fstats)


def run_simulation(Model, Algorithm):
    """Perform single simulation of Model using Algorithm.

    Returns the result of a single simulation run.
    """

    def plot_trajectory(data):
        plt.style.use('seaborn-poster')
        fig, ax = plt.subplots()
        
        time = data[0][0]
        vals = data[0][1]
        for species,copy_numbers in vals.items():
            ax.plot(time, copy_numbers, label=species, alpha=0.75)
        plt.title(Algorithm.__name__)
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.ylabel('# molecules')
        plt.xlabel('time (seconds)')
        add_textbox(ax, f"RPS={round(data[1],1)}\n runtime={round(data[2],3)}\n reactions={data[3]}\nspecies={len(data[4])}\n state={data[5]}", (-0.15,1.04))
        plt.show()

    # setup model and algorithm
    model = Model()
    trajectory = Algorithm(model.process, model.initial_state, tmax=model.tmax)
    
    # perform simulation
    time_before = time.perf_counter()
    result = model(trajectory)
    time_after = time.perf_counter()

    # measure performance
    runtime = time_after - time_before
    num_of_reactions = trajectory.step
    rps = float(num_of_reactions)/float(runtime)
    num_of_unique_species = trajectory.state.keys()
    final_state = trajectory.state.values()

    #plot_trajectory((result, rps, runtime, trajectory.step, num_of_unique_species, trajectory.state))
    
    return (result, rps, runtime, num_of_reactions, trajectory.state, num_of_unique_species, final_state, trajectory.time, model.tmax)


def run_in_process(queue, locks, args):
    """Worker process for parallel execution of simulations.
    
    The worker continuously fetches a simulation configuration from
    the queue, runs the simulation and feeds the simulation result
    into the data store. The worker stops if it fetches a single None
    from the queue.
    """
    while True:
        config = queue.get()
        if not config:
            break
        try:
            result = run_simulation(*config)
        except Exception as exc:
            print(f'Could not not run simulation for {str(config)} due to exception: {exc}')

        with locks[config]:
            try:
                args.store.feed_result(result, config)
            except Exception as exc:
                print(f'Could not not store result for {str(config)} due to exception: {exc}')
            
            model_name, algo_name = get_modelname_and_algoname_from_config(config)
            try:
                rps, runtime, reactions, tmax, state = result[1], result[2], result[3], config[0].tmax, result[4]
                if args.debugmode == 3:
                    console.rule(' [bold white]Simulation complete [bold red]%s [bold yellow]%s' % (model_name, algo_name), align='left', style='black')
                elif args.debugmode == 4:
                    console.rule(' [bold white]Simulation complete [bold red]%s [bold yellow]%s [bold black](%s, %s, %s, %s, %s)' % (model_name, algo_name, rps, runtime, reactions, tmax, state), align='left', style='black')
            except Exception as exc:
                print(f'Could not access simulation result for {str(config)} due to exception: {exc}')
    
    logging.debug(f'Worker finished', align='left', style='black')


def run_benchmarking(args):
    """Perform benchmarking simulations.
    
    Run simulations required for the store to hold aggregregated
    statistics from args.N samples for each given algorithm and model
    combination. If args.model is not given, models classes are
    loaded from stocal.examples.dsmts.models. If args.algo is not given,
    algorithms are loaded from stocal.algorithms.

    If args.cpu is given and greater than 1, simulations are performed
    in parallel.
    """

    def get_implementations(module, cls):
        return [
            member for member in module.__dict__.values()
            if isclass(member)
            and issubclass(member, cls)
            and not isabstract(member)
        ]

    def configurations(N):
        """ generate required simulation configurations """
        import random
        required = {
            config: (max(0, N-args.store.get_stats(config).runs)
                        if os.path.exists(args.store.get_path_for_config(config))
                        else N)
            for config in product(args.models, args.algo)
        }
        configs = list(required)
        while configs:
            next_config = random.choice(configs)
            required[next_config] -= 1
            if not required[next_config]:
                configs.remove(next_config)
            yield next_config

    def run_sequential(sim_count, sim_max):
        """ Function to execute simulations sequentially
        
        Creates simulation configuration file for each
        algorithm & model combination. Runs the simulation,
        and feeds the simulation result to the DataStore.

        There are different debugmodes (i.e. different ways
        of viewing details about the ongoing simulations):
            1: Numerical progress
            2: Progress bar
            3: Details about each simulation
            4: More details about each simulation
            5: Show nothing
        """

        def add_result(config, args, result=None):
            """ run simulation and add result to datastore """
            try:
                if result is None:
                    result = run_simulation(*config)
                    args.store.feed_result(result, config)
                else:
                    args.store.feed_result(result, config)
                return result
            except Exception as exc:
                print(f'Could not run simulation for {str(config)} due to exception: {exc}')
            return None
        
        if args.debugmode == 2:
            progressbar = tqdm(total=sim_max, unit="simulation", leave=True, ncols=150, bar_format="%s{desc}%s{percentage:0.4f}%s%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.LIGHTGREEN_EX, '%|', Fore.LIGHTGREEN_EX, Fore.CYAN))
        if args.debugmode == 4:
            print(details_columns)
        
        for config in configurations(args.N):
            model_name, algo_name = get_modelname_and_algoname_from_config(config)

            # Run simulation, and add result to DataStore... using a specific debugmode
            if args.debugmode == 2:
                progressbar.set_description('{desc: <38}'.format(desc='Simulating %s using %s' % (model_name, algo_name)))
                add_result(config, args)
                sim_count += 1
                progressbar.update(1)
            elif args.debugmode == 3:
                with console.status(f"[bold green] Running next simulation... ", spinner='line') as status:
                    add_result(config, args)
                    sim_count += 1
                    console.rule(details_debugmode3 % (str(sim_count), sim_max, calcprogress(sim_count, sim_max), '%', model_name, algo_name), align='left', style='black')
            elif args.debugmode == 4:
                with console.status(f"[bold green] Running next simulation... ", spinner='line') as status:
                    result = add_result(config, args)
                    sim_count += 1
                    try:
                        rps, runtime, reactions, tmax, finalstate = result[1], result[2], result[3], config[0].tmax, result[4]
                        console.rule(details_debugmode4 % (str(sim_count), sim_max, calcprogress(sim_count, sim_max), '%', model_name, algo_name, rps, runtime, reactions, tmax, finalstate), align='left', style='black')
                    except Exception as exc:
                        console.rule(details_debugmode3 % (str(sim_count), sim_max, calcprogress(sim_count, sim_max), '%', model_name, algo_name), align='left', style='black')
            elif args.debugmode == 5:
                add_result(config, args)
            else:
                add_result(config, args)
                sim_count += 1
                sys.stdout.write(f'{sim_count}/{sim_max} ({calcprogress(sim_count, sim_max)}%) \r')
       
        if args.debugmode == 2:
            progressbar.close()

    def run_parallel(sim_count, sim_max):
        """ Function that prepares to execute simulations parallely
        
        Creates a queue of simulation processes, which is sent to the
        run_in_process function
        """
        
        queue = Queue(maxsize=args.cpu)
        locks = {
            config: Lock()
            for config in product(args.models, args.algo)
        }
        processes = [Process(target=run_in_process, args=(queue, locks, args)) for _ in range(args.cpu)]
        for proc in processes:
            proc.start()
            console.rule(f'Started {proc}', align='left', style='black')
        console.rule(f'{args.cpu} processes started.', align='left', style='black')
        if args.debugmode == 4:
            print(details_columns)
        for config in configurations(args.N):
            queue.put(config)
            sim_count += 1
            sys.stdout.write(f'Started {sim_count} simulations out of {sim_max} ({calcprogress(sim_count, sim_max)}%) \r')
        console.rule(f'All jobs requested', align='left', style='black')
        for _ in processes:
            queue.put(None)
            console.rule(f'Shutdown signal sent', align='left', style='black')
        queue.close()
        for proc in processes:
            console.rule(f'Waiting for {proc}', align='left', style='black')
            proc.join()
    
    # collect algorithms to benchmark
    if not args.algo:
        from stocal import algorithms
        args.algo = get_implementations(algorithms, algorithms.StochasticSimulationAlgorithm)
    
    # collect models for benchmarking
    if not args.ts:
        if not args.models:
            from stocal.examples.dsmts import models as dsmts
            args.models = get_implementations(dsmts, dsmts.DSMTS_Test)
    else:
        if not args.models:
            from stocal.examples.mmts import models as mmts
            from stocal.examples.dsmts import models as dsmts

            if args.ts.upper() == 'MMTS':
                args.models = get_implementations(mmts, mmts.MMTS_Test)
            elif args.ts.upper() == 'DSMTS':
                args.models = get_implementations(dsmts, dsmts.DSMTS_Test)
            elif args.ts.upper() == 'ALL':
                args.models = get_implementations(mmts, mmts.MMTS_Test) + get_implementations(dsmts, dsmts.DSMTS_Test)

    sim_count = 0
    sim_max = args.N * len(args.models) * len(args.algo)
    
    if args.cpu > 1:
        run_parallel(sim_count, sim_max)
    else:
        run_sequential(sim_count, sim_max)
    
    console.rule(f'Done', align='left', style='black')


def report_benchmarking(args, frmt='png', template='doc/benchmarking.tex'):
    """ Generate results from benchmarking data. 
    
    These results include:
     Figures for each config file,
     LaTeX report that collates all these figures,
     Comparison of algorithm performance on a per model basis
    """

    def camel_case_split(identifier):
        import re
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return ' '.join(m.group(0) for m in matches)
    
    def splitpath(path, maxdepth=20):
        ( head, tail ) = os.path.split(path)
        return splitpath(head, maxdepth - 1) + [ tail ] \
            if maxdepth and head and head != path \
            else [ head or tail ]
    
    def export_to_csv(fpath, data_columns, data_rows):
        try:
            with open(fpath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_columns)
                writer.writeheader()
                for row in data_rows:
                    writer.writerow(row)
        except IOError:
            print('Failed to export data')


    data_cols = ['model', 'algorithm', 'samples', 'tmax', 'rps_avg', 'rps_stdev', 'rps_min', 'rps_max', 'runtime_avg', 'runtime_stdev', 'runtime_min', 'runtime_max', 'reactions_avg', 'reactions_stdev', 'reactions_min', 'reactions_max', 'unique_species_len_avg', 'unique_species_len_stdev', 'unique_species_len_min', 'unique_species_len_max', 'simulated_time_avg', 'inactive', 'inactive_rate', 'overactive', 'overactive_rate']
    data_rows = []

    figures = {}
    # generate figures for the entire data store
    for fname, stats in args.store:
        if stats:
            figname = fname[:-len('.dat')]+'.'+ frmt
            if not os.path.exists(figname) or os.path.getmtime(fname) > os.path.getmtime(figname):
                # only if .dat newer than
                logging.debug("Generate figure for %s", fname)
                try:
                    generate_figure(stats, figname)
                except Exception as exc:
                    print(exc)
                    logging.warning("Could not generate figure for %s", fname)
                    logging.info(exc, exc_info=True)

            model, algo = stats.config
            algo_name = camel_case_split(algo.__name__).replace('_', ' ')

            # latex requires forwardslashes ('/') as seperators to define a path to a file
            figname_latex = '/'.join(splitpath(figname))
            figures[algo_name] = sorted(figures.get(algo_name, [])+[figname_latex])

            data_rows.append({
                'model': str(model.__name__),
                'algorithm': str(abbreviate_algo_name(algo.__name__)),
                'samples': str(stats.runs),
                'tmax': str(model.tmax),
                'rps_avg': np.mean(stats.rps),
                'rps_stdev': np.std(stats.rps),
                'rps_min': np.min(stats.rps),
                'rps_max': np.max(stats.rps),
                'runtime_avg': np.mean(stats.runtime),
                'runtime_stdev': np.std(stats.runtime),
                'runtime_min': np.min(stats.runtime),
                'runtime_max': np.max(stats.runtime),
                'reactions_avg': np.mean(stats.reactions),
                'reactions_stdev': np.std(stats.reactions),
                'reactions_min': np.min(stats.reactions),
                'reactions_max': np.max(stats.reactions),
                'unique_species_len_avg': np.mean(stats.unique_species_len),
                'unique_species_len_stdev': np.std(stats.unique_species_len),
                'unique_species_len_min': np.min(stats.unique_species_len),
                'unique_species_len_max': np.max(stats.unique_species_len),
                'simulated_time_avg': np.mean(stats.simulated_time),
                'inactive': stats.inactive,
                'inactive_rate': str((stats.inactive/stats.runs)*100) + '%',
                'overactive': stats.overactive,
                'overactive_rate': str((stats.overactive/stats.runs)*100) + '%'
            })
            export_to_csv(args.store.path+'\\data.csv', data_cols, data_rows)

    # generate the main benchmarking results from all data
    generate_final_results(args, stats, data_rows)

    # populate latex template
    latex_jinja_env = jinja2.Environment(
        block_start_string = '\BLOCK{',
        block_end_string = '}',
        variable_start_string = '\VAR{',
        variable_end_string = '}',
        comment_start_string = '\#{',
        comment_end_string = '}',
        line_statement_prefix = '%%',
        line_comment_prefix = '%#',
        trim_blocks = True,
        autoescape = False,
        loader = jinja2.FileSystemLoader(os.path.abspath('.'))
    )
    template = latex_jinja_env.get_template(template)
    context = {
        'version': os.path.basename(args.store.path).replace('_', '\_'),
        'methods': figures,
    }

    tex_file = 'benchmarking-%s.tex' % (context['version'])

    reportfile = args.reportfile or tex_file
    with open(reportfile, 'w') as report:
        report.write(template.render(**context))

    with open(tex_file, 'r+') as fname:
        content = fname.read()
        fname.seek(0)
        fname.truncate()
        # check for old version of revtex4 (pdflatex command cant convert .tex file if old old revtex version is being used)
        fname.write(content.replace('revtex4-1', 'revtex4-2'))


def generate_final_results(args, stats, data_rows):
    """ Generate final results from the benchmarking data """

    plt.style.use('default')     # (default, ggplot, classic, bmh, fast, fivethirtyeight, seaborn, seaborn-bright, seaborn-dark, seaborn-poster, seaborn-pastel) 
    algcolors = [mcolors.CSS4_COLORS[name] for name in ['turquoise', 'lime', 'orange', 'cornflowerblue', 'plum', 'lightcoral']]
    fpath = default_store       # 'benchmarking_data\\heteropolymer\\test'
    
    def barchart1(fname, models, algs, data):
        """ plot the performance of each algorithm, averaged across all models """

        if len(models) > 1:
            means = np.array([np.mean(data[key]) for key in data])
            stdevs = np.array([np.std(data[key]) for key in data])
            algcolors_crop = algcolors[:len(algs)]

            df = pandas.DataFrame ({'Alg':  algs, 'Mean': means.tolist(), 'Stdev': stdevs.tolist(), 'Color': algcolors_crop})
            df = df.sort_values(by=['Mean'])

            fig = plt.figure(figsize=(10,7))

            #plt.bar(df.Alg, df.Mean, err=df.Stdev, color=df.Color, alpha=0.5, capsize=5)
            plt.bar(df.Alg, df.Mean, color=df.Color, alpha=0.5, capsize=5)
            plt.ylabel('RPS')
            plt.title('Mean RPS for each algorithm, across all models')
            plt.grid(True, axis='y', color='#999999', alpha=0.15)
            plt.xticks(rotation="0")
            plt.tight_layout()
            plt.savefig('%s\\%s.png' % (fpath, fname))
            plt.close()

    def barchart2(fname, models, algs):
        """ make 1 big bar chart showing the performance of each alg & model combination """
        
        if len(models) > 1:
            df = pandas.read_csv(fpath + '\\data.csv')
            index = 'algorithm'
            columns = 'model'
            avg_pivot = df.pivot(index, columns, 'rps_avg')
            std_pivot = df.pivot(index, columns, 'rps_stdev')
            avg_pivot.plot(kind='bar', yerr=std_pivot, width=0.8, alpha=0.75, capsize=0, figsize=(30, 12))
            plt.legend(loc='upper left')
            plt.grid(True, axis='y', color='#999999', alpha=0.15)
            plt.xlabel('')
            plt.ylabel('RPS', size=19)
            plt.xticks(size=19, rotation="0")
            plt.yticks(size=13)
            plt.title('Mean RPS for each algorithm & model combination', size=26)
            plt.savefig('%s\\%s.png' % (fpath, fname))
            plt.close()

    def barchart_per_model(fname, models, algs, data):
        """make individual barcharts for each model, showing the performance each alg achieved with error bar"""
        
        metrics = ['RPS']

        if len(algs) > 1:
            for model_name,v in data.items():
                df = pandas.DataFrame({'Alg': algs, 'Mean': v[0], 'Stdev': v[2], 'Color': algcolors[:len(algs)]})
                df = df.sort_values(by=['Mean'])

                fig = plt.figure(figsize=(10,7))
                plt.title('Performance of each algorithm - %s - (%s samples each)' % (model_name, stats.runs), fontsize=13)
                plt.bar(df.Alg, df.Mean, yerr=df.Stdev, color=df.Color, alpha=0.5, capsize=5)
                plt.grid(True, axis='y', color='#999999', alpha=0.15)
                plt.ylabel('RPS')
                plt.xticks(rotation="0", fontsize=9)
                plt.yticks(fontsize=9)
                #plt.tight_layout()
                plt.savefig('%s\\xxx%s%s%s.png' % (fpath, model_name, fname, metrics[0]))
                plt.close()

    def scatterplot(fname, models, algs, data):
        """ scatter plot comparing the best observed performance across all algs vs. performance of each alg """
        
        plt.style.use('default')

        if len(models) > 1 and len(algs) > 1:
            data2 = {k: v for k, v in sorted(data.items(), key=lambda item: item[1][1])}
            model_names = [m[0] for m in data2.items()]
            rps_best = [m[1] for m in data2.values()]

            L = len(algs)
            if L>=1 and L<=3:
                rows,cols = 1,3
            elif L>= 4 and L <= 6:
                rows,cols = 2,3
            elif L>= 7 and L <= 9:
                rows,cols = 3,3
            elif L>= 10 and L <= 12:
                rows,cols = 4,3

            fig, ax = plt.subplots(rows, cols, sharex='col', figsize=(15,15))
            
            a = 0
            for row in range(rows):
                for col in range(cols):
                    rps_eachmodel = [m[0][a] for m in data2.values()]
                    ax[row, col].plot(model_names, rps_eachmodel, color='r', alpha=0.7, marker='o', linestyle='', markersize=4)
                    ax[row, col].plot(model_names, rps_best, color='g', alpha=0.7, marker='o', linestyle='', markersize=4)
                    #ax[row, col].set_xticklabels([])    # remove labels from x-axis
                    ax[row, col].tick_params(axis="x", rotation=90, labelsize=7)    # keep labels on x-axis and rotate them 90 degrees
                    ax[row, col].set_yticks(np.arange(0, max(rps_best)*1.15, 10000))
                    ax[row, col].set_title(algs[a])
                    a += 1
                    if a == len(model_names)-1:
                        break
                else:
                    continue
                break
            #plt.setp(ax[-1, :], xlabel='Each model')
            #plt.setp(ax[:, 0], ylabel='RPS')
            plt.subplots_adjust(bottom=0.15, top=0.93)
            plt.savefig('%s\\%s.png' % (fpath, fname))
            plt.close()

    def heatmap(fname, models, algs, data):
        """ create a heatmap showing the performance of each alg & model combination, presenting the data as colour coded numerical values """
        
        if len(models) > 1:
            plt.style.use('default')

            metrics = ['RPS', 'Reactions', 'Runtime']
            cmaps = ['viridis', 'BrBG', 'coolwarm']
            roundings = [1, 0, 4]

            for i in range(len(metrics)):
                values = np.array([model[i] for model in data.values()])
                values = np.around(values, roundings[i])
                
                fig, ax = plt.subplots(figsize=(35,15))
                colormap = ax.imshow(values, cmap=cmaps[i])

                # We want to show all ticks...
                ax.set_xticks(np.arange(len(models)))
                ax.set_yticks(np.arange(len(algs)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(models)
                ax.set_yticklabels(algs)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Loop over data dimensions and create text annotations.
                for a in range(len(algs)):
                    for m in range(len(models)):
                        text = ax.text(m, a, values[a, m], ha="center", va="center", color="w", fontsize=14)

                plt.title('%s for each algorithm & model combination - (%s samples each)' % (metrics[i], stats.runs), fontsize=30)
                plt.yticks(size=20)
                plt.xticks(size=13)
                plt.savefig('%s\\%s-%s.png' % (fpath, fname, metrics[i]))
                plt.close()
    
    def boxplot_per_model(fname, models, algs, one_figure, sort_by_performance):
        """ Make boxplots for each performance metric, showing how spread the values are for each algorithm against a particular model """
            
        seaborn.set(style='whitegrid')
        
        metrics = ['RPS', 'Reactions', 'Runtime', 'Species']
        data = [[], [], [], []]
        locations = [(121), (322), (324), (326)]
        orientations = ['v', 'h', 'h', 'h'] if one_figure else ['v', 'v', 'v', 'v']
        algcolors_crop = algcolors[:len(algs)]
        
        for model in models:
            if one_figure:
                fig,ax = plt.subplots(figsize=(19,10))
            
            [d.clear() for d in data]

            # find data for this model
            for f,stats in args.store:
                if stats and stats.config[0].__name__ == model:
                    if one_figure:
                        fig.suptitle('Performance metrics - %s - (%s samples for each alg)' % (stats.config[0].__name__, stats.runs), fontsize=14)
                    if len(data[0]) < len(algs):
                        data[0].append([arr for arr in stats.rps])
                        data[1].append([arr for arr in stats.reactions])
                        data[2].append([arr for arr in stats.runtime])
                        data[3].append([arr for arr in stats.unique_species_len])
            
            # generate 1 plot for each metric
            for loc, metric, dt, ori in zip(locations, metrics, data, orientations):
                df = pandas.DataFrame({'Alg': algs, 'Values': dt, 'Mean': float(0), 'Color': algcolors_crop})

                # compute mean values and put into the dataframe
                for i in range(len(dt)):
                    df.at[i, 'Mean'] = np.mean(dt[i])
                
                # sort the dataframe by the algs mean performance
                if sort_by_performance:
                    df = df.sort_values(by=['Mean'])
                
                if one_figure:
                    plt.subplot(loc)
                else:
                    fig = plt.figure(figsize=(3,13))

                categories = df['Alg'].to_list()
                vals = df['Values'].to_list()
                orderedcolors = df['Color'].to_list()
                
                if metric == 'RPS':
                    seaborn.boxplot(data=vals, orient=ori, palette=df.Color, width=0.85, linewidth=1.4)
                    plt.xticks(np.arange(len(categories)), categories)
                    if one_figure is False:
                        plt.ylim(0, 15000)
                elif metric == 'Reactions':
                    seaborn.boxplot(data=vals, orient=ori, palette=df.Color, width=0.85, linewidth=1.4)
                    if one_figure is False:
                        plt.xticks(np.arange(len(categories)), categories)
                        plt.ylim(0, 500000)
                    else:
                        plt.yticks(np.arange(len(categories)), categories)
                elif metric == 'Runtime':
                    seaborn.boxplot(data=vals, orient=ori, palette=df.Color, width=0.85, linewidth=1.4)
                    if one_figure is False:
                        plt.xticks(np.arange(len(categories)), categories)
                        plt.ylim(0, 2000)
                    else:
                        plt.yticks(np.arange(len(categories)), categories)
                elif metric == 'Species':
                    df = df.explode('Values')
                    df['Values'] = df['Values'].astype('float')
                    seaborn.barplot(data=vals, orient=ori, palette=orderedcolors)
                    if one_figure is False:
                        plt.xticks(np.arange(len(categories)), categories)
                        plt.ylim(0, 10)
                    else:
                        plt.yticks(np.arange(len(categories)), categories)
                
                plt.xlabel('')
                plt.ylabel('')
                
                if one_figure:
                    plt.xticks(fontsize=11)
                    plt.yticks(fontsize=11)
                    plt.title(metric, fontsize=13)
                else:
                    seaborn.despine(offset=10, trim=True)
                    plt.xticks([])
                    plt.yticks(fontsize=5)
                    plt.savefig('%s\\xxx%s%s%s.png' % (fpath, model, fname, metric))
                    plt.close()

            if one_figure:
                ax.xaxis.grid(True)
                plt.tight_layout()
                plt.subplots_adjust(top=0.90, wspace=0.10, hspace=0.30)
                plt.savefig('%s\\xxx%s%s.png' % (fpath, model, fname))
            plt.close()

    def barchartstemp(fname, models, algs, sort_by_performance):        
        """ make individual barcharts for each model, showing the performance each alg achieved with error bar """
        
        seaborn.set_style('whitegrid', {'grid.color': 'lightgrey', 'grid.linestyle': '-', 'xtick.major.size': 5, 'ytick.major.size': 5})

        metrics = ['RPS', 'Reactions', 'Runtime', 'Species']
        data = [[], [], [], []]
        locations = [(121), (322), (324), (326)]
        algcolors_crop = algcolors[:len(algs)]
        
        for model in models:            
            [d.clear() for d in data]

            # find data for this model
            for f,stats in args.store:
                if stats and stats.config[0].__name__ == model:
                    if len(data[0]) < len(algs):
                        data[0].append([arr for arr in stats.rps])
                        data[1].append([arr for arr in stats.reactions])
                        data[2].append([arr for arr in stats.runtime])
                        data[3].append([arr for arr in stats.unique_species_len])
            
            # generate 1 plot for each metric
            for loc, metric, dt in zip(locations, metrics, data):
                df = pandas.DataFrame({'Alg': algs, 'Values': dt, 'Mean': float(0), 'Stdev': float(0), 'Color': algcolors_crop})

                # compute mean values and put into the dataframe
                for i in range(len(dt)):
                    df.at[i, 'Mean'] = np.mean(dt[i])
                    df.at[i, 'Stdev'] = np.std(dt[i])
                
                # sort the dataframe by the algs mean performance
                if sort_by_performance:
                    df = df.sort_values(by=['Mean'])
                
                fig = plt.figure(figsize=(4,10))
                
                if metric == 'RPS':
                    plt.bar(df.Alg, df.Mean, yerr=df.Stdev, color=df.Color, alpha=0.65, capsize=0)
                    plt.ylim(0, 6000)
                elif metric == 'Reactions':
                    plt.bar(df.Alg, df.Mean, yerr=df.Stdev, color=df.Color, alpha=0.65, capsize=0)
                    plt.ylim(0, 2500000)
                elif metric == 'Runtime':
                    plt.bar(df.Alg, df.Mean, yerr=df.Stdev, color=df.Color, alpha=0.65, capsize=0)
                    plt.ylim(0, 20000)
                elif metric == 'Species':
                    plt.bar(df.Alg, df.Mean, yerr=df.Stdev, color=df.Color, alpha=0.65, capsize=0)
                    plt.ylim(0, 11)
                
                plt.xlabel('alpha = 1.e-9', fontsize=12)
                plt.ylabel(metric, fontsize=10)
                plt.yticks(fontsize=10)
                plt.xticks(rotation="0", fontsize=10)
                plt.tight_layout()
                plt.savefig('%s\\xxx%s%s%s.png' % (fpath, model, fname, metric))
                plt.close()
                
            plt.close()

    def algrankings(fname, models, algs, data):
        """ get rankings of algorithms for each model """
        
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        cols = [ordinal(n) for n in range(1,len(algs)+1)]

        frames = []
        for model_name,v in data.items():
            df = pandas.DataFrame({'Alg':  algs, 'Mean': v[0]})
            df = df.sort_values(by=['Mean'], ascending=False)
            df.rename(columns={'Alg': model_name, 'Mean': model_name}, inplace=True)
            df = df.transpose().head(1)
            #df.columns = [n+1 for n in range(len(algs))]
            df.columns = cols
            frames.append(df)
        
        result = pandas.concat(frames)
        result.to_csv('%s\\%s.csv' % (fpath, fname))

        print(result[cols].apply(pandas.Series.value_counts))

    
    # collect data from 'data_rows'
    rps_values = [float(row['rps_avg']) for row in data_rows]
    rps_stdev_values = [float(row['rps_stdev']) for row in data_rows]
    reactions_values = [float(row['reactions_avg']) for row in data_rows]
    runtime_values = [float(row['runtime_avg']) for row in data_rows]
    models = list(OrderedDict.fromkeys([row['model'] for row in data_rows]))
    algs = list(OrderedDict.fromkeys([row['algorithm'] for row in data_rows]))

    # group rps values by models {'DSMTS_001_01': [], 'DSMTS_001_03': [], ...}
    data_grouped_by_model = {key: [[0]*len(algs), 0, [0]*len(algs)] for key in models}    
    m=0
    for model in models:
        data_grouped_by_model[model][0] = rps_values[m:len(rps_values):len(models)]
        data_grouped_by_model[model][1] = max(data_grouped_by_model[model][0])
        data_grouped_by_model[model][2] = rps_stdev_values[m:len(rps_stdev_values):len(models)]
        m += 1

    # group rps values by algs {'AFRM': [], 'AM': [], ...}
    data_grouped_by_alg = {key: [[0]*len(models), [0]*len(models), [0]*len(models)] for key in algs}    
    a=0
    for alg in algs:
        data_grouped_by_alg[alg][0] = rps_values[a:a+len(models)]
        data_grouped_by_alg[alg][1] = reactions_values[a:a+len(models)]
        data_grouped_by_alg[alg][2] = runtime_values[a:a+len(models)]
        a += len(models)
    
    ''' ------ MAKE PLOTS ------ '''
    # create barchart showing alg performance per model
    barchart_per_model('-AlgPerformance', models, algs, data_grouped_by_model)
    
    # create boxplot showing alg performance per model, shows equivilant data to the barchart above, but includes outliers and indicates spread
    boxplot_per_model('-AlgPerformance', models, algs, one_figure=True, sort_by_performance=True)

    # create scatter plot of algorithm performance profiles
    scatterplot('xxxScatters', models, algs, data_grouped_by_model)

    # create heatmaps
    heatmap('xxxHeatMap', models, algs, data_grouped_by_alg)

    # rank the algorithms by their performance
    algrankings('xxxAlgRankings', models, algs, data_grouped_by_model)


def generate_figure(stats, fname):
    """ Generate figure for given stats and save it to fname. """

    try:
        from matplotlib import pyplot as plt
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
        })
    except ImportError:
        logging.error("benchmarking.py requires matplotlib.")
        sys.exit(1)
    
    # setup...
    model, algo = stats.config
    fig = plt.figure(figsize=(14,9))
    title = '%s %s (%d samples)' % (model.__name__, algo.__name__, stats.runs)
    fig.suptitle(title)

    #rows, columns
    gridsize = (3,3)

    # Make scatter plot, takes up first 2 rows, and all 3 columns    
    ax = plt.subplot2grid(shape=gridsize, loc=(0, 0), rowspan=2, colspan=3)
    ax.ticklabel_format(useOffset=False, style='plain')     # disable scientific notation on x-axis and y-axis
    plt.title("RPS as a function of the algorithm runtime (x-axis) and the #reactions (y-axis)")
    scatterplot = plt.scatter(x=stats.runtime, y=stats.reactions, c=stats.rps, alpha=0.70, cmap="RdYlGn")
    plt.colorbar(scatterplot, ax=ax, format='RPS = %d')
    plt.axvline(0, c=(.5, .5, .5), ls='--')
    plt.axhline(0, c=(.5, .5, .5), ls='--')
    plt.xlabel('Algorithm runtime (seconds)')
    plt.ylabel('# reactions')

    # Histogram of RPS values
    ax = plt.subplot2grid(shape=gridsize, loc=(2, 0))
    plt.hist(stats.rps, bins=50, edgecolor='black', alpha=0.70)
    plt.xlabel('RPS')
    plt.ylabel('Frequncies')

    # Histogram of reactions
    ax = plt.subplot2grid(shape=gridsize, loc=(2, 1))
    plt.hist(stats.reactions, bins=50, edgecolor='black', alpha=0.70)
    plt.xlabel('# Reactions')

    # Histogram of runtime
    ax = plt.subplot2grid(shape=gridsize, loc=(2, 2))
    plt.hist(stats.runtime, bins=50, edgecolor='black', alpha=0.70)
    plt.xlabel('Algorithm runtime (seconds)')

    # Histogram of final state of all species
    '''
    ax = plt.subplot2grid(shape=gridsize, loc=(2, 2))
    L = len(model.species)
    for i in range(L):
        plt.hist(stats.final_states[i:stats.runs:L], bins=100, alpha=0.9, label=model.species[i])
    plt.yticks()
    #plt.ylim((None, 50))
    plt.xlabel('Final states of species')
    plt.legend(loc='upper right')
    '''

    # Line plot of average trajectory
    '''
    ax = plt.subplot2grid(shape=gridsize, loc=(2, 2))    
    colormaps = [plt.cm.winter, plt.cm.copper]
    for (species, values), cm in zip(stats.mean.items(), colormaps):
        low = values - stats.stdev[species]**0.5
        high = values + stats.stdev[species]**0.5
        ax.fill_between(stats.times, low, high, facecolor=cm(0.99), alpha=0.3)
        ax.plot(stats.times, values, color=cm(0.99), alpha=.67, label=species)
        plt.xlim(left=0)    # begin x-axis labels at 0
        ax.ticklabel_format(useOffset=False)        # disable scientific notation
    add_textbox(ax, 'Average trajectory', (0.25, 0.90))
    plt.xlabel('time')
    plt.ylabel('# molecules')
    plt.legend(loc='lower left')
    plt.savefig('trajectory - %s - %s' % (model.__name__, algo.__name__))
    plt.close()
    '''

    # add a textbox displaying average values for main performance metrics
    add_textbox(ax, f"RPS={round(np.mean(stats.rps),1)}\n runtime={round(np.mean(stats.runtime),3)}\n reactions={np.mean(stats.reactions)}", (-2.9,3.8))
    plt.subplots_adjust(hspace=0.35, wspace=0.20)
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    def import_by_name(name):
        """import and return a module member given by name

        e.g. 'stocal.algorithms.DirectMethod' will return the class
        <class 'stocal.algorithms.DirectMethod'>
        """
        module, member = name.rsplit('.', 1)
        mod = import_module(module)
        return getattr(mod, member)
        
    git_label = subprocess.check_output(["git", "describe", "--always"]).decode('utf-8').strip()
    default_store = os.path.join('benchmarking_data', git_label)

    #parser = argparse.ArgumentParser(prog=sys.argv[0], description=console.print("\n[bold green]STOCAL BENCHMARKING CLI [bold black] - Perform stochastic simulations and generate SSA benchmarking data"), epilog="""If --dir is provided, it specifies a directory used to hold benchmarking data.""")
    #console.print('-------------------------------------------------------------------------------------------')
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="STOCAL BENCHMARKING CLI - Perform stochastic simulations and generate SSA benchmarking data", epilog="""If --dir is provided, it specifies a directory used to hold benchmarking data.""")

    # global options
    parser.add_argument('--dir', dest='store', type=DataStore, default=DataStore(default_store), help='directory for/with simulation results')
    subparsers = parser.add_subparsers(help='benchmarking sub-command')

    # parser for the "run" command
    parser_run = subparsers.add_parser('run', help='run simulations to generate benchmarking data')
    parser_run.add_argument('N', type=int, help='number of simulations to be performed in total')
    parser_run.add_argument('--ts', type=str, choices=['DSMTS', 'MMTS', 'ALL'], help='specify a test suite to use')
    parser_run.add_argument('--algo', type=import_by_name, action='append', help='specify algorithm to be benchmarked')
    parser_run.add_argument('--model', type=import_by_name, action='append', dest='models', help='specify model to be benchmarked')
    parser_run.add_argument('--cpu', metavar='N', type=int, default=1, help='number of parallel processes')
    parser_run.add_argument('--debugmode', type=int, default=1, choices=[1,2,3,4,5], help='see different info during simulation')
    parser_run.set_defaults(func=run_benchmarking)

    # parser for the "report" command
    parser_report = subparsers.add_parser('report', help='generate figures from generated data')
    parser_report.add_argument('--file', action='store', dest='reportfile', default='', help='file name of generated report')
    parser_report.set_defaults(func=report_benchmarking)
    
    # parse and act
    #logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    args.func(args)
