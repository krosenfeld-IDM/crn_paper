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

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [250, 25][debug]
basedir = os.path.join(paths.src.as_posix(), 'figs')

rngs = ['centralized', 'multi']


def run_sim(n_agents, idx, cov, rand_seed, rng, network=None, eff=0.8,
            fixed_initial_prevalence=False, pars=None, sir_pars=None):
    ppl = ss.People(n_agents)
    if network is None:
        G = nx.barabasi_albert_graph(n=n_agents, m=1, seed=rand_seed)
        G.name = 'Barabasi-Albert'
    elif callable(network):
        G = network(n_agents, rand_seed)  # Call with n and random seed
    elif isinstance(network, nx.Graph):
        G = network.copy()
        G.name = getattr(network, 'name', 'Custom-Graph')
    else:
        raise ValueError('Invalid network parameter')

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
    sir = ss.SIR(pars=sir_pars)

    default_pars = {
        'start': 2020,
        'end': 2020.5,
        'dt': 1 / 365,
        'rand_seed': rand_seed,
        'verbose': 0,
        'slot_scale': 10,  # Increase slot scale to reduce repeated slots
    }
    pars = sc.mergedicts(default_pars, pars)

    if cov > 0:
        # Create the product - a vaccine with specified efficacy
        MyVaccine = ss.sir_vaccine(pars={'efficacy': eff})

        # Create the intervention
        MyIntervention = ss.campaign_vx(
            years=pars['start'] + 5 / 365,  # 5 days after start
            prob=cov,
            product=MyVaccine
        )
        pars.update({'interventions': [MyIntervention]})

    sim = ss.Sim(people=ppl, networks=networks, diseases=sir, pars=pars, label=lbl)

    sim.initialize()

    # Infect agent zero to start the simulation
    if fixed_initial_prevalence:
        uids = np.arange(n_agents // 10)
    else:
        uids = np.array([n_agents // 2])
    sim.diseases.sir.set_prognoses(uids=uids, sources=None)

    sim.run()

    sim.diseases['sir'].log.line_list.to_csv(os.path.join('figs', f'll_{cov}.csv'))

    df = pd.DataFrame({
        'year': sim.yearvec,
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
