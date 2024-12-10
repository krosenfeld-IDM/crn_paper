
"""
Example 2) Static networks with SIR disease dynamics + vaccination
"""

# %% Imports and settings
import os
import starsim as ss
import matplotlib.pyplot as plt
import sciris as sc
import pandas as pd
import numpy as np
import networkx as nx
from networkx.utils import pairwise
from crn_paper import paths
from plotting import plot_scenarios, plot_graph

sc.options(interactive=True) # Assume not running interactively

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

rngs = ['centralized', 'multi']

debug = True
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [250, 25][debug]

basedir = os.path.join(paths.src.as_posix(), 'figs')

def run_sim(n_agents, idx, cov, rand_seed, rng, network=None, eff=0.8, fixed_initial_prevalence=False, pars=None, sir_pars=None):

    ppl = ss.People(n_agents)
    if network is None:
        G=nx.barabasi_albert_graph(n=n_agents, m=1, seed=rand_seed)
        G.name = 'Barabasi-Albert'
    elif callable(network):
        G = network(n_agents, rand_seed) # Call with n and random seed if callable
    elif isinstance(network, nx.Graph):
        G = network.copy()
    else:
        print('Check the network parameter')

    lbl = f'Sim {idx}: agents={n_agents}, cov={cov}, seed={rand_seed}, net={G.name}, rng={rng}'
    print('Starting', lbl)

    networks = ss.StaticNet(G)

    default_sir_pars = {
        'beta': ss.beta(75),
        'dur_inf': ss.expon(scale=ss.dur(30/365)),
        'init_prev': ss.bernoulli(p=0),  # Will seed manually
        'p_death': ss.bernoulli(p=0.05), # 5% chance of death
    }
    sir_pars = sc.mergedicts(default_sir_pars, sir_pars)
    sir = ss.SIR(sir_pars)

    default_pars = {
        'start': 2020,
        'stop': 2020.5,
        'dt': 1/365,
        'rand_seed': rand_seed,
        'verbose': 0,
        'slot_scale': 10 # Increase slot scale to reduce repeated slots
    }
    pars = sc.mergedicts(default_pars, pars)

    if cov > 0:
        # Create the product - a vaccine with specified efficacy
        MyVaccine = ss.sir_vaccine(pars=dict(efficacy=eff))

        # Create the intervention
        MyIntervention = ss.campaign_vx(
            years = pars['start'] + 5/365, # 5 days after start
            prob = cov,
            product=MyVaccine
        )
        pars['interventions'] = [ MyIntervention ]

    sim = ss.Sim(n_agents=n_agents, networks=networks, diseases=sir, pars=pars, label=lbl)

    sim.init()

    # Infect agent zero to start the simulation
    # WARNING: Graph algorithms may place agent 0 non-randomly
    if fixed_initial_prevalence:
        uids = np.arange(n_agents//10)
    else:
        uids = np.array([n_agents//2])
    sim.diseases.sir.set_prognoses(uids=ss.uids(uids), sources=None)

    sim.run()

    sim.diseases['sir'].log.line_list.to_csv( os.path.join('figs', f'll_{cov}.csv') )

    df = pd.DataFrame( {
        'time': sim.results.timevec,
        'Susceptible': sim.results.sir.n_susceptible,
        'Infected': sim.results.sir.n_infected,
        'Recovered': sim.results.sir.n_recovered,
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['network'] = G.name
    df['eff'] = eff
    df['rng'] = rng
    df['n_agents'] = n_agents

    print('Finishing', lbl)

    return df


def sweep_cov(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    figdir = os.path.join(basedir, 'SIR_coverage' if not debug else 'SIR_coverage-debug')
    sc.path(figdir).mkdir(parents=True, exist_ok=True)

    cov_levels = [0.05, 0.80] + [0] # Must include 0 as that's the reference level

    results = []
    times = {}
    for rng in rngs:
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for rs in range(n_seeds):
            for cov in cov_levels:
                cfgs.append({'cov':cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents, 'network':None, 'fixed_initial_prevalence':False}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))

    return df


class Grid2D:
    def __init__(self, m, n):
        self.G = nx.empty_graph(0)
        cols = np.arange(n)
        rows = np.arange(m)
        self.nodes = np.arange(m*n)

        self.Xp, self.Yp = np.meshgrid(cols, rows, indexing='ij')

        self.G.add_nodes_from(self.nodes)

        self.G.add_edges_from((i*m+j, pi*m+j) for pi, i in pairwise(rows) for j in cols)
        self.G.add_edges_from((i*m+j, i*m+pj) for i in rows for pj, j in pairwise(cols))
        return

    def plot(self, ax=None):
        if ax == None:
            fig, ax = plt.subplots()
        ax.scatter(self.Xp.flatten(), self.Yp.flatten(), s=10, marker='o')

        nx.draw(self.G, pos={i:(self.Xp.flatten()[i], self.Yp.flatten()[i]) for i in self.nodes}, ax=ax)
        return ax

def sweep_network(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    figdir = os.path.join(basedir, 'SIR_network' if not debug else 'SIR_network-debug')
    sc.path(figdir).mkdir(parents=True, exist_ok=True)

    print('Overriding n_agents to 1,000')
    n_agents = 1_000

    cov_levels = [0, 0.05, 0.80] # Must include 0 as that's the reference level
    efficacy = 0.5 # 0.8, 0.3

    results = []
    times = {}

    s = int(np.floor(np.sqrt(n_agents)))
    n_agents = s * s

    for rng in rngs:
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for rs in range(n_seeds):
            graphs = {
                'Barabasi-Albert (m=1)':        (nx.barabasi_albert_graph(n=n_agents, m=1, seed=rs), {'beta': ss.beta(140)}), # 115
                'Erdos-Renyi (p=4/N)':          (nx.fast_gnp_random_graph(n=n_agents, p=4/n_agents, seed=rs), {'beta': ss.beta(10)}),
                'Watts-Strogatz (k=4, p=0.20)': (nx.connected_watts_strogatz_graph(n=n_agents, k=4, p=0.20, seed=rs), {'beta': ss.beta(14)}),
                'Grid 2D':                      (Grid2D(m=s, n=s).G, {'beta': ss.beta(18.5)})
            }
            for name, (G,sir_pars) in graphs.items():
                G.name = name

                if rng == rngs[0] and rs == 0 and n_agents <= 1_000:
                    fig = plot_graph(G)
                    fig.savefig( os.path.join(figdir, f'Graph {name.replace("/", "_")}.png') )
                    plt.close(fig)

                for cov in cov_levels:
                    cfgs.append({'network':G, 'sir_pars': sir_pars, 'cov':cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})

        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents, 'eff': efficacy, 'fixed_initial_prevalence': False, 'pars': {'stop':2022}}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))
    return df


def sweep_n(n_seeds=default_n_rand_seeds):
    figdir = os.path.join(basedir, 'SIR_n' if not debug else 'SIR_n-debug')
    sc.path(figdir).mkdir(parents=True, exist_ok=True)

    n_agents_levels = [10, 100, 1000]
    if not debug:
        n_agents_levels += [10_000]#, 100_000]

    efficacy = 0.7
    cov_levels = [0, 0.05, 0.90] # Must include 0 as that's the reference level

    results = []
    times = {}

    for rng in rngs:
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for n_agents in n_agents_levels:
            for rs in range(n_seeds):
                for cov in cov_levels:
                    cfgs.append({'n_agents':n_agents, 'cov':cov, 'rand_seed':rs, 'network':None, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'eff':efficacy, 'fixed_initial_prevalence':True}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)

    # Normalize
    for col in ['Susceptible', 'Infected', 'Recovered']:
        df[col] /= df['n_agents']

    df.to_csv(os.path.join(figdir, 'results.csv'))

    return df


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    results = {}
    if args.plot:
        for d in ['SIR_network', 'SIR_coverage', 'SIR_n']:
            fn = os.path.join(args.plot, d, 'results.csv')
            try:
                print('Reading CSV file', fn)
                results[d] = pd.read_csv(fn, index_col=0)
            except:
                print(f'Unable to read {fn}')
    else:
        print('Running scenarios')
        results['SIR_n'] = sweep_n(n_seeds=args.s)
        results['SIR_network'] = sweep_network(n_agents=args.n, n_seeds=args.s)
        results['SIR_coverage'] = sweep_cov(n_agents=args.n, n_seeds=args.s)

    if 'SIR_n' in results:
        figdir = os.path.join(basedir, 'SIR_n' if not debug else 'SIR_n-debug')
        plot_scenarios(results['SIR_n'], figdir, channels=['Recovered'], var1='n_agents', var2='cov', slice_year = -1) # slice_year = 2020.05

    if 'SIR_network' in results:
        figdir = os.path.join(basedir, 'SIR_network' if not debug else 'SIR_network-debug')
        plot_scenarios(results['SIR_network'], figdir, channels=['Recovered'], var1='network', var2='cov')

    if 'SIR_coverage' in results:
        figdir = os.path.join(basedir, 'SIR_coverage' if not debug else 'SIR_coverage-debug')
        plot_scenarios(results['SIR_coverage'], figdir, channels=['Susceptible', 'Infected', 'Recovered'], var1='cov', var2='channel')

    print('Done')
