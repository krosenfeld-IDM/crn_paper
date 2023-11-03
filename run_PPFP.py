"""
Compare two HIV simulations, one baseline and the other with ART
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import argparse
import sciris as sc

from PPFP_demographics import PPFP

default_n_agents = 100

do_plot_longitudinal = True
do_plot_timeseries = True

ss.options(multistream = True) # Can set multistream to False for comparison

figdir = os.path.join(os.getcwd(), 'figs', 'PPFP')
sc.path(figdir).mkdir(parents=True, exist_ok=True)


def run_sim(n=25, rand_seed=0, intervention=False, analyze=False, lbl=None):
    ppl = ss.People(n)

    ppl.networks = ss.ndict(ss.maternal())

    pars = {
        'start': 1980,
        'end': 2020,
        'remove_dead': False, # So we can see who dies, sim results should not change with True
        #'interventions': [art] if intervention else [],
        'rand_seed': rand_seed,
        #'analyzers': [GraphAnalyzer()] if analyze else [],
    }

    ppfp_pars = {
        'efficacy': 0.95,
        'coverage': 0.01,
    }
    if intervention:
        preg = PPFP(ppfp_pars)
    else:
        ppfp_pars['coverage'] = 0
        preg = PPFP(ppfp_pars)

    sim = ss.Sim(people=ppl, diseases=[], demographics=[preg, ss.background_deaths()], pars=pars, label=lbl)
    sim.initialize()

    sim.run()

    return sim


def run_scenario(n=10, rand_seed=0, analyze=True):
    sims = sc.parallelize(run_sim,
                          kwargs={'n':n, 'analyze': analyze, 'rand_seed': rand_seed},
                          iterkwargs=[{'intervention':False, 'lbl':'Baseline'}, {'intervention':True, 'lbl':'Intervention'}], die=True)

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def plot_ts():
    # Plot timeseries summary
    fig, axv = plt.subplots(2,2, sharex=True)

    ax = axv[0,0]
    ax.plot(sim1.tivec, sim1.results.ppfp.new_ppfp, label=sim1.label)
    ax.plot(sim2.tivec, sim2.results.ppfp.new_ppfp, ls=':', label=sim2.label)
    ax.set_title('Num women starting PPFP')
    
    ax = axv[0,1]
    ax.plot(sim1.tivec, sim1.results.ppfp.n_ppfp, label=sim1.label)
    ax.plot(sim2.tivec, sim2.results.ppfp.n_ppfp, ls=':', label=sim2.label)
    ax.set_title('Num women on PPFP')

    ax = axv[1,0]
    ax.plot(sim1.tivec, sim1.results.n_alive, label=sim1.label)
    ax.plot(sim2.tivec, sim2.results.n_alive, ls=':', label=sim2.label)
    ax.set_title('Population size')

    ax = axv[1,1]
    ax.plot(sim1.tivec, sim1.results.new_deaths, label=sim1.label)
    ax.plot(sim2.tivec, sim2.results.new_deaths, ls=':', label=sim2.label)
    ax.set_title('New deaths')

    plt.legend()
    return fig


def analyze_people(sim):
    p = sim.people
    ever_alive = ss.false(np.isnan(p.age))
    years_lived = np.full(len(p), sim.ti+1) # Actually +1 dt here, I think
    years_lived[p.dead] = p.ti_dead[p.dead]
    years_lived = years_lived[ever_alive] # Trim, could be more efficient
    age_initial = p.age[ever_alive].values - years_lived
    age_initial = age_initial.astype(np.float32) # For better hash comparability, there are small differences at float64


    df = pd.DataFrame({
        'id': [hash((p.slot[i], age_initial[i], p.female[i])) for i in ever_alive], # if slicing, don't need ._view,
        'age_initial': age_initial,
        'years_lived': years_lived,
        #'ti_infected': p.hiv.ti_infected[ever_alive].values,
        #'ti_art': p.hiv.ti_art[ever_alive].values,
        'ti_dead': p.ti_dead[ever_alive].values,
        'ti_ppfp': sim.demographics.ppfp.ti_ppfp.values if 'ppfp' in sim.demographics else np.full(age_initial.size, fill_value=np.nan),

        # Useful for debugging, but not needed for plotting
        'slot': p.slot[ever_alive].values,
        'female': p.female[ever_alive].values,
    })
    df.replace(to_replace=ss.INT_NAN, value=np.nan, inplace=True)
    df['age_dead']     = df['age_initial'] + df['ti_dead']
    return df


def plot_longitudinal(sim1, sim2):

    df1 = analyze_people(sim1)
    df1['sim'] = 'Baseline'
    df2 = analyze_people(sim2)
    df2['sim'] = 'With PPFP'

    df = pd.concat([df1, df2]).set_index('id')

    df['ypos'] = pd.factorize(df.index.values)[0]
    N = df['sim'].nunique()
    height = 0.5/N

    fig, ax = plt.subplots(figsize=(10,6))

    # For the legend:
    plt.barh(y=0, left=0, width=1e-6, color='k', height=height, label='Alive')
    plt.barh(y=0, left=0, width=1e-6, color='r', height=height, label='PPFP')
    plt.scatter(y=0, x=0, color='c', marker='|', label='Death')

    for n, (lbl, data) in enumerate(df.groupby('sim')):
        yp = data['ypos'] + n/(N+1) # Leave space

        ti_initial = np.maximum(-data['age_initial'], 0)
        ti_final = data['ti_dead'].fillna(40)
        plt.barh(y=yp, left=ti_initial, width=ti_final - ti_initial, color='k', height=height)

        # PPFP
        ppfp = ~data['ti_ppfp'].isna()
        plt.barh(y=yp[ppfp], left=data.loc[ppfp]['ti_ppfp'], width=ti_final[ppfp]-data.loc[ppfp]['ti_ppfp'], color='r', height=height)

        # Dead
        dead = ~data['ti_dead'].isna()
        plt.scatter(y=yp[dead], x=data.loc[dead]['ti_dead'], color='c', marker='|')

    ax.set_xlabel('Simulation step')
    ax.set_ylabel('UID')
    ax.legend()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Rand seed', type=int, default=0)
    args = parser.parse_args()

    if args.plot:
        print('Reading files', args.plot)
        sim1 = sc.load(os.path.join(args.plot, 'sim1.obj'))
        sim2 = sc.load(os.path.join(args.plot, 'sim2.obj'))
    else:
        print('Running scenarios')
        [sim1, sim2] = run_scenario(n=args.n, rand_seed=args.s)

    if do_plot_longitudinal:
        plot_longitudinal(sim1, sim2)

    if do_plot_timeseries:
        plot_ts()

    plt.show()
    print('Done')