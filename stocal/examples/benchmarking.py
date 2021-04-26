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

def abbreviate_algo_name(algo_name):
    """ converts name of algorithm to an abbreviated version (e.g. DirectMethod = DM) """
    return ''.join([c for c in algo_name if c.isupper() or c.isdigit()])

def add_textbox(ax, text, pos):
    """ add a textbox to a matplotlib plot """
    ax.text(pos[0], pos[1], text, horizontalalignment="left", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.6), fontsize=8)
    return ax



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


def run_simulation(Model, Algorithm, max_steps=1000000):
    """Perform single simulation of Model using Algorithm.

    Returns the result of a single simulation run.
    """
    
    def plot_trajectory(data):
        """ little optional function to show a time vs. #molecules plot """
        plt.style.use('seaborn-poster')
        fig, ax = plt.subplots()
        
        time = data[0][0]
        state = data[0][1]
        for species,copy_numbers in state.items():
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
    rps = float(trajectory.step)/float(runtime)
    unique_species = trajectory.state.keys()
    final_state = trajectory.state.values()

    #plot_trajectory((result, rps, runtime, trajectory.step, unique_species, trajectory.state))
    
    return (result, rps, runtime, trajectory.step, trajectory.state, unique_species, final_state, trajectory.time, model.tmax)


def run_in_process(queue, locks, store):
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
            logging.warning("Could not run simulation for %s", str(config))
            logging.info(exc, exc_info=True)

        with locks[config]:
            try:
                store.feed_result(result, config)
                logging.debug("Stored result for %s", str(config))   
            except Exception as exc:
                logging.warning("Could not store result for %s", str(config))
                logging.info(exc, exc_info=True)

    logging.debug("Worker finished")
    


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

    def add_result(config, args):
        """ run a simulation and add result to datastore """
        try:
            result = run_simulation(*config)
            args.store.feed_result(result, config)
            return result
        except Exception as exc:
            logging.warning("Could not run simulation for %s", str(config))
            logging.info(exc, exc_info=True)
        
        return None        

    def get_implementations(module, cls):
        return [
            member for member in module.__dict__.values()
            if isclass(member)
            and issubclass(member, cls)
            and not isabstract(member)
        ]

    def configurations(N):
        """generate required simulation configurations
        """
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
    
    def choose_debug_mode_for_parallel(args, config, sim_count, sim_max, result=None):
        """Different modes for viewing simulation progression/metainfo in the console, if parallel processing is being used...
        	1: show numerical progression (default)
			3: show details of each simulation
			4: show nothing
        """
        
        model_name = str(config[0].__name__)
        algo_name = abbreviate_algo_name(str(config[1].__name__))

        if args.debugmode == 3:
            if result is None:
                console.rule(f' [bold white]simulation [bold green]{str(sim_count)}[bold white]/[bold green]{str(sim_max)} [bold white]complete ([bold green]{calcprogress(sim_count, sim_max)}%[bold white]) [bold red]{model_name} [bold yellow]{algo_name}', align='left', style='black')
            else:
                console.rule(f' [bold white]simulation [bold green]{str(sim_count)}[bold white]/[bold green]{str(sim_max)} [bold white]complete ([bold green]{calcprogress(sim_count, sim_max)}%[bold white]) [bold red]{model_name} [bold yellow]{algo_name} [bold black]({result[1]}, {result[2]}, {result[3]}, {result[4]}, {config[0].tmax})', align='left', style='black')
        
        elif args.debugmode == 4:
            pass
        
        else:   # default if no debugmode is specified
            sys.stdout.write(f'{sim_count}/{sim_max} ({calcprogress(sim_count, sim_max)}%)\r')

    def choose_debug_mode_for_nonparallel(args, sim_count, sim_max):
        """Different modes for viewing simulation progression/metainfo in the console, if parallel processing isn't being used...
        	1: show numerical progression (default)
			2: show progress bar
			3: show details of each simulation
			4: show nothing
        """
        
        if args.debugmode == 2:
            with tqdm(total=sim_max, unit="simulation", leave=True, ncols=150, bar_format="%s{desc}%s{percentage:0.4f}%s%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.LIGHTGREEN_EX, '%|', Fore.LIGHTGREEN_EX, Fore.CYAN)) as progressbar:
                for config in configurations(args.N):
                    progressbar.set_description('{desc: <35}'.format(desc='Simulating %s using %s' % (config[0].__name__, abbreviate_algo_name(config[1].__name__))))          
                    add_result(config, args)
                    sim_count += 1
                    progressbar.update(1)
                progressbar.close()

        elif args.debugmode == 3:
            console.rule(f'                                                        [bold black](rps, runtime, reactions, simulated_time, tmax, final_state)', align='left', style='black')
            with console.status(f"[bold green] Running next simulation... ", spinner='line') as status:
                for config in configurations(args.N):
                    result = add_result(config, args)
                    sim_count += 1
                    if result is None:
                        console.rule(f' [bold white]Simulation [bold green]{str(sim_count)}[bold white]/[bold green]{str(sim_max)} [bold white]complete ([bold green]{calcprogress(sim_count, sim_max)}%[bold white]) [bold red]{str(config[0].__name__)} [bold yellow]{abbreviate_algo_name(str(config[1].__name__))}', align='left', style='black')
                    else:
                        console.rule(f' [bold white]Simulation [bold green]{str(sim_count)}[bold white]/[bold green]{str(sim_max)} [bold white]complete ([bold green]{calcprogress(sim_count, sim_max)}%[bold white]) [bold red]{str(config[0].__name__)} [bold yellow]{abbreviate_algo_name(str(config[1].__name__))} [bold black]({result[1]}, {result[2]}s, {result[3]}, {result[7]}s, {config[0].tmax}s, {result[4]})', align='left', style='black')

        elif args.debugmode == 4:
            for config in configurations(args.N):
                add_result(config, args)
        
        else:   # default if no debugmode is specified
            for config in configurations(args.N):
                add_result(config, args)
                sim_count += 1
                sys.stdout.write(f'{sim_count}/{sim_max} ({calcprogress(sim_count, sim_max)}%)\r')

    def calcprogress(sim_count, sim_max):
        return '{:.4f}'.format(round((sim_count/sim_max)*100, 4))

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

    # perform simulations in parallel if args.cpu > 1, otherwise just run the simulation normally
    if args.cpu > 1:
        queue = Queue(maxsize=args.cpu)
        locks = {
            config: Lock()
            for config in product(args.models, args.algo)
        }
        processes = [Process(target=run_in_process, args=(queue, locks, args.store)) for _ in range(args.cpu)]
        for proc in processes:
            proc.start()
        logging.debug("%d processes started." % args.cpu)
        for config in configurations(args.N):
            queue.put(config)
            sim_count += 1
            choose_debug_mode_for_parallel(args, config, sim_count, sim_max)
        logging.debug("All jobs requested.")
        for _ in processes:
            queue.put(None)
            logging.debug("Shutdown signal sent.")
        queue.close()
        for proc in processes:
            print(f"Waiting for {proc}")
            proc.join()
    else:
        choose_debug_mode_for_nonparallel(args, sim_count, sim_max)
    logging.debug("Done.")


def report_benchmarking(args, frmt='png', template='doc/benchmarking.tex'):
    """ Generate results from benchmarking data. 
    
    These results include:
     Figures for each .dat file,
     LaTeX report that collates all these figures,
     Comparison of algorithm performance on a per model basis) 
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
    algcolors = ['r', 'g', 'b', 'c', 'm', 'y'] + [mcolors.CSS4_COLORS[name] for name in ['lightcoral', 'limegreen', 'cornflowerblue', 'plum', 'orange', 'aquamarine']]
    fpath = default_store
    
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

    def barcharts(fname, models, algs, data):
        """make individual barcharts for each model, showing the performance each alg achieved with error bar"""

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
                plt.savefig('%s\\xxx%s%s.png' % (fpath, model_name, fname))
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
    
    def boxplots(fname, models, algs):
        """ Make boxplots for each performance metric, showing how spread the values are for each algorithm against a particular model """
        
        seaborn.set(style='whitegrid')
        
        metrics = ['RPS', 'Number of reactions', 'Algorithm runtime', 'Number of unique species']
        data = [[], [], [], []]
        locations = [(121), (322), (324), (326)]
        algcolors_crop = algcolors[:len(algs)]
        
        for model in models:
            fig,ax = plt.subplots(figsize=(19,10))
            [d.clear() for d in data]

            # find data for this model
            for f,stats in args.store:
                if stats and stats.config[0].__name__ == model:
                    fig.suptitle('Performance metrics - %s - (%s samples for each alg)' % (stats.config[0].__name__, stats.runs), fontsize=14)
                    if len(data[0]) < len(algs):
                        data[0].append([arr for arr in stats.rps])
                        data[1].append([arr for arr in stats.reactions])
                        data[2].append([arr for arr in stats.runtime])
                        data[3].append([arr for arr in stats.unique_species_len])

            # generate 1 plot for each metric
            for loc, metric, dt in zip(locations, metrics, data):
                df = pandas.DataFrame({'Alg': algs, 'Values': dt, 'Mean': float(0), 'Color': algcolors_crop})
                for i in range(len(dt)):
                    df.at[i, 'Mean'] = np.mean(dt[i])
                df = df.sort_values(by=['Mean'])
                orderedcolors = df['Color'].to_list()
                
                plt.subplot(loc)
                categories = df['Alg'].to_list()
                vals = df['Values'].to_list()

                if metric == 'RPS':
                    seaborn.boxplot(data=vals, orient='v', palette=df.Color, width=0.7)
                    plt.xticks(np.arange(len(categories)), categories)

                elif metric == 'Number of reactions':
                    seaborn.boxplot(data=vals, orient='h', palette=df.Color, width=0.7)
                    plt.yticks(np.arange(len(categories)), categories)

                elif metric == 'Algorithm runtime':
                    seaborn.boxplot(data=vals, orient='h', palette=df.Color, width=0.7)
                    plt.yticks(np.arange(len(categories)), categories)

                elif metric == 'Number of unique species':
                    df = df.explode('Values')
                    df['Values'] = df['Values'].astype('float')
                    seaborn.barplot(data=vals, orient='h', palette=orderedcolors)
                    plt.yticks(np.arange(len(categories)), categories)
                
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.xlabel('')
                plt.ylabel('')
                plt.title(metric, fontsize=13)

            ax.xaxis.grid(True)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, wspace=0.10, hspace=0.30)
            plt.savefig('%s\\xxx%s%s.png' % (fpath, model, fname))
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
    
    # ------ MAKE PLOTS ------
    barchart1('xxxEveryModel-PerformanceOfEachAlg', models, algs, data_grouped_by_alg)
    barchart2('xxxAlgModelCombinations', models, algs)
    #barcharts('-PerformanceRPS', models, algs, data_grouped_by_model)
    boxplots('-PerformanceMetrics', models, algs)
    scatterplot('xxxScatters', models, algs, data_grouped_by_model)
    heatmap('xxxHeatMap', models, algs, data_grouped_by_alg)
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
    parser_run.add_argument('--debugmode', type=int, default=1, choices=[1,2,3,4], help='see different info during simulation')
    parser_run.set_defaults(func=run_benchmarking)

    # parser for the "report" command
    parser_report = subparsers.add_parser('report', help='generate figures from generated data')
    parser_report.add_argument('--file', action='store', dest='reportfile', default='', help='file name of generated report')
    parser_report.set_defaults(func=report_benchmarking)
    
    # parse and act
    #logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    args.func(args)
