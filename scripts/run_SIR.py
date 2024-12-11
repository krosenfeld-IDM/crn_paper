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
from crn_paper import paths
from plotting import plot_scenarios, plot_graph

sc.options(interactive=True)  # Assume not running interactively

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
        G = nx.barabasi_albert_graph(n=n_agents, m=1, seed=rand_seed)
        G.name = 'Barabasi-Albert'
    elif callable(network):
        G = network(n_agents, rand_seed)  # Call with n and random seed if callable
    elif isinstance(network, nx.Graph):
        G = network.copy()
    else:
        raise ValueError("Invalid 'network' parameter: must be None, callable, or an nx.Graph instance.")

    lbl = f'Sim {idx}: agents={n_agents}, cov={cov}, seed={rand_seed}, net={G.name}, rng={rng}'
    print('Starting', lbl)

    networks = ss.StaticNet(G)

    default_sir_pars = {
        'beta': 75,
        'dur_inf': ss.expon(scale=30 / 365),
        'init_prev': 0,  # Will seed manually
        'p_death': 0.05,  # 5% chance of death
    }
    sir_pars = sc.mergedicts(default_sir_pars, sir_pars)
    sir = ss.SIR(sir_pars)

    default_pars = {
        'start': 2020,
        'end': 2020.5,
        'dt': 1 / 365,
        'rand_seed': rand_seed,
        'verbose': 0,
        'slot_scale': 10  # Increase slot scale to reduce repeated slots
    }
    pars = sc.mergedicts(default_pars, pars)

    if cov > 0:
        # Create the product - a vaccine with specified efficacy
        MyVaccine = ss.sir_vaccine(pars=dict(efficacy=eff))

        # Create the intervention
        MyIntervention = ss.campaign_vx(
            years=pars['start'] + 5 / 365,  # 5 days after start
            prob=cov,
            product=MyVaccine
        )
        pars['interventions'] = [MyIntervention]

    sim = ss.Sim(people=ppl, networks=networks, diseases=sir, pars=pars, label=lbl)

    sim.initialize()

    # Infect agent zero to start the simulation
    if fixed_initial_prevalence:
        uids = np.arange(n_agents // 10)
    else:
        uids = np.array([n_agents // 2])
    sir.set_prognoses(uids=ss.uids(uids), source_uids=None)

    sim.run()

    sim.diseases['sir'].log.line_list.to_csv(os.path.join('figs', f'll_{cov}.csv'))

    df = pd.DataFrame({
        'year': sim.timevec,
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

    cov_levels = [0.05, 0.80, 0]  # Must include 0 as that's the reference level

    results = []
    times = {}
    for rng in rngs:
        ss.options(_centralized=(rng == 'centralized'))
        cfgs = []
        for rs in range(n_seeds):
            for cov in cov_levels:
                cfgs.append({'cov': cov, 'rand_seed': rs, 'rng': rng, 'idx': len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Plot from a cached CSV file', type=str)
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
            except FileNotFoundError:
                print(f'Unable to read {fn}')
    else:
        print('Running scenarios')
        results['SIR_coverage'] = sweep_cov(n_agents=args.n, n_seeds=args.s)

    print('Done')
