"""
Static networks with SIR disease dynamics // vaccination
"""

# %% Imports and settings
import os
import starsim as ss
import scipy.stats as sps
import matplotlib.pyplot as plt
import sciris as sc
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx

from plotting import plot_scenarios, plot_graph

sc.options(interactive=False) # Assume not running interactively

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

rngs = ['centralized', 'multi'] # 'single', 

n = 1_000 # Agents
n_rand_seeds = 10
intv_cov_levels = [0.01, 0.10, 0.25, 0.90] + [0] # Must include 0 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'SIR')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n, idx, intv_cov, rand_seed, rng, network=None):

    ppl = ss.People(n)
    # watts_strogatz_graph, erdos_renyi_graph, grid_2d_graph, configuration_model, line_graph, barabasi_albert_graph, scale_free_graph, complete_graph, maybe a tree
    #G = nx.gnp_random_graph(n, p=3/n, seed=rand_seed) #G=nx.erdos_renyi_graph(n, p=3/n, seed=rand_seed)
    # G=nx.complete_graph(n)
    if network is None:
        G=nx.barabasi_albert_graph(n=n, m=1, seed=rand_seed)
        G.name = 'Barabasi-Albert'
    elif callable(network):
        G = network(n, rand_seed) # Call with n and random seed if callable
    elif isinstance(network, nx.Graph):
        G = network.copy()
    else:
        print('Check the network parameter')

    lbl = f'Sim {idx}: agents={n}, intv_cov={intv_cov}, seed={rand_seed}, net={G.name}, rng={rng}'
    print('Starting', lbl)

    networks = ss.StaticNet(G)

    sir_pars = {
        'beta': 30,
        'dur_inf': sps.weibull_min(c=1, scale=30/365), # When c=1, it's an exponential
        #'dur_inf': sps.weibull_min(c=3, scale=33.5/365), # Can check sir_pars['dur_inf'].mean()
        'init_prev': 0,  # Will seed manually
        'p_death': 0, # No death
    }

    sir = ss.SIR(sir_pars)

    pars = {
        'start': 2020,
        'end': 2020.5,
        'dt': 1/365,
        'rand_seed': rand_seed,
        'verbose': 0,
        'remove_dead': True,
        'slot_scale': 10 # Increase slot scale to reduce repeated slots
    }

    if intv_cov > 0:
        # Create the product - a vaccine with 80% efficacy
        MyVaccine = ss.sir_vaccine(pars=dict(efficacy=0.8))

        # Create the intervention
        MyIntervention = ss.campaign_vx(
            years = 2020.1, #[pars['start']],
            prob = intv_cov,
            annual_prob = False,
            product=MyVaccine
        )
        pars['interventions'] = [ MyIntervention ]

    sim = ss.Sim(people=ppl, networks=networks, diseases=sir, pars=pars, label=lbl)
    sim.initialize()

    # Infect agent zero to start the simulation
    # WARNING: Graph algorithms may place agent 0 non-randomly
    sim.diseases.sir.set_prognoses(sim, uids=np.array([0]), source_uids=None)

    sim.run()

    df = pd.DataFrame( {
        'year': sim.yearvec,
        'sir.n_susceptible': sim.results.sir.n_susceptible,
        'sir.n_infected': sim.results.sir.n_infected,
        'sir.n_recovered': sim.results.sir.n_recovered,
    })
    df['intv_cov'] = intv_cov
    df['rand_seed'] = rand_seed
    df['network'] = G.name
    df['rng'] = rng

    if rand_seed == 0: # Will repead for coverage sweep, but that's okay!
        fig = plot_graph(G)
        fig.savefig(os.path.join(figdir, f'graph_{n}_{G.name}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f'Mean duration of infection is {sir_pars["dur_inf"].mean() * 365} days')

    print('Finishing', lbl)

    return df

def sweep_cov():
    results = []
    times = {}
    for rng in rngs:
        ss.options(rng=rng)
        cfgs = []
        for rs in range(n_rand_seeds):
            for intv_cov in intv_cov_levels:
                cfgs.append({'intv_cov':intv_cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'sweep_cov.csv'))
    return df

def sweep_network():
    results = []
    times = {}

    for rng in rngs:
        ss.options(rng=rng)
        cfgs = []
        for rs in range(n_rand_seeds):
            graphs = {
                'Barabasi-Albert':  nx.barabasi_albert_graph(n=n, m=1, seed=rs),
                'Erdos-Renyi':      nx.gnp_random_graph(n=n, p=3/n, seed=rs), #G=nx.erdos_renyi_graph(n, p=3/n, seed=rand_seed)
                'Watts-Strogatz':   nx.watts_strogatz_graph(n=n, k=3, p=0.25, seed=rs), # Small world
                'Complete':         nx.complete_graph(n=n),
            }

            for name, G in graphs.items():
                G.name = name
                cfgs.append({'network':G, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n, 'intv_cov': 0}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'sweep_network.csv'))
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(args.plot, index_col=0)
        raise Exception('TODO')
    else:
        print('Running scenarios')
        #dfc = sweep_cov()
        dfn = sweep_network()

    #plot_scenarios(df, figdir, sweep_var='network', reference_val=???)
    
    #df = run_sim(n, idx=0, intv_cov=0, rand_seed=0, rng='multi')
    #print(df)

    print('Done')
