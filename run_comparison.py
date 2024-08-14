"""
Compare two HIV simulations, one baseline and the other with ART
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import argparse
import sciris as sc
from run_HIV import run_sim

sc.options(interactive=True)

default_n_agents = 100
# Three choices for network here, note that only the first two are stream safe
#net_idx = 1
#network = ['stable_monogamy', 'EmbeddingNet', 'hpv_network'][net_idx]
network = 'EmbeddingNet'
art_eff = 1
#net_pars = [None, dict(duration=ss.weibull(c=1.5, scale=10)), None][net_idx]

do_plot_graph = True
# Several choices for how to layout the graph when plotting
kind = ['radial', 'bipartite', 'spring', 'multipartite'][1]

do_plot_longitudinal = True
do_plot_timeseries = True

# on branch fix_348
#options.rng can take values in ['centralized', 'single', 'multi']
#ss.options(rng = 'multi')

figdir = os.path.join(os.getcwd(), 'figs', network)
sc.path(figdir).mkdir(parents=True, exist_ok=True)

class stable_monogamy(ss.SexualNetwork):
    """
    Very simple network for debugging in which edges are:
    1-2, 3-4, 5-6, ...
    """
    def __init__(self, **kwargs):
        # Call init for the base class, which sets all the keys
        super().__init__(**kwargs)
        return

    def initialize(self, sim):
        n = len(sim.people._uid_map)
        n_edges = n//2
        self.contacts.p1 = np.arange(0, 2*n_edges, 2) # EVEN
        self.contacts.p2 = np.arange(1, 2*n_edges, 2) # ODD
        self.contacts.beta = np.ones(n_edges)
        return


def run_scenario(n=10, rand_seed=0, analyze=True):
    sims = sc.parallelize(run_sim,
                          kwargs={'n_agents':n, 'analyze': analyze, 'rand_seed': rand_seed, 'return_sim': True, 'art_eff': art_eff},
                          iterkwargs=[{'cov':0.00, 'idx':0, 'rng':'multi'}, {'cov':0.10, 'idx':1, 'rng':'multi'}], die=False)

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def getpos(ti, g1, g2, guess=None, kind='bipartite'):

    n1 = dict(g1[ti].graph.nodes.data())
    n2 = dict(g2[ti].graph.nodes.data())
    nodes = sc.mergedicts(n2, n1)
    n = len(nodes)

    if kind == 'radial':
        pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}
        if guess:
            if len(guess) < n:
                pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}

    elif kind == 'spring':
        pos = nx.spring_layout(g1[ti].graph, k=None, pos=guess, fixed=None, iterations=50, threshold=0.0001, weight=None, scale=1, center=None, dim=2, seed=None)
        if guess:
            pos = sc.mergedicts(pos, guess)

    elif kind == 'multipartite':
        pos = nx.multipartite_layout(g1[ti].graph, subset_key='female', align='vertical', scale=10, center=None)
        if guess:
            pos = sc.mergedicts(pos, guess)

        if guess:
            for i in guess.keys():
                pos[i] = (pos[i][0], guess[i][1]) # Keep new x but carry over y

    elif kind == 'bipartite':
        pos = {i:(nd['age'], 2*nd['female']-1 + np.random.uniform(-0.3, 0.3)) for i, nd in nodes.items()}

        if guess:
            for i in guess.keys():
                pos[i] = (pos[i][0], guess[i][1]) # Keep new x but carry over y

    return pos


def plot_graph(sim1, sim2):
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs

    n = len(g1[-1].graph)
    el = n <= 25 # Draw edge labels

    fig, axv = plt.subplots(1, 2, figsize=(10,5))
    global ti
    timax = sim1.tivec[-1]

    global pos
    pos = {}
    pos[-1] = getpos(0, g1, g2, kind=kind)
    for ti in range(timax+1):
        pos[ti] = getpos(ti, g1, g2, guess=pos[ti-1], kind=kind)

    ti = -1 # Initial state is -1, representing the state before the first step

    def on_press(event):
        print('press', event.key)
        sys.stdout.flush()
        global ti, pos
        if event.key == 'right':
            ti = min(ti+1, timax)
        elif event.key == 'left':
            ti = max(ti-1, -1)

        # Clear
        axv[0].clear()
        axv[1].clear()

        g1[ti].plot(pos[ti], edge_labels=el, ax=axv[0])
        g2[ti].plot(pos[ti], edge_labels=el, ax=axv[1])
        fig.suptitle(f'Time is {ti} (use the arrow keys to change)')
        axv[0].set_title(sim1.label)
        axv[1].set_title(sim2.label)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)

    g1[ti].plot(pos[ti], edge_labels=el, ax=axv[0])
    g2[ti].plot(pos[ti], edge_labels=el, ax=axv[1])
    fig.suptitle(f'Time is {ti} (use the arrow keys to change)')
    axv[0].set_title(sim1.label)
    axv[1].set_title(sim2.label)
    return fig


def plot_ts():
    # Plot timeseries summary
    fig, axv = plt.subplots(2,2, sharex=True)
    #axv[0,0].plot(sim1.tivec, sim1.results.hiv.n_infected, label=sim1.label)
    #axv[0,0].plot(sim2.tivec, sim2.results.hiv.n_infected, ls=':', label=sim2.label)
    #axv[0,0].set_title('HIV number of infections')
    axv[0,0].plot(sim1.tivec, sim1.results.hiv.art_coverage, label=sim1.label)
    axv[0,0].plot(sim2.tivec, sim2.results.hiv.art_coverage, ls=':', label=sim2.label)
    axv[0,0].set_title('ART Coverage')

    axv[0,1].plot(sim1.tivec, sim1.results.hiv.cum_infections, label=sim1.label)
    axv[0,1].plot(sim2.tivec, sim2.results.hiv.cum_infections, ls=':', label=sim2.label)
    axv[0,1].set_title('Cumulative HIV infections')

    axv[1,0].plot(sim1.tivec, sim1.results.hiv.new_deaths.cumsum(), label=sim1.label)
    axv[1,0].plot(sim2.tivec, sim2.results.hiv.new_deaths.cumsum(), ls=':', label=sim2.label)
    axv[1,0].set_title('Cumulative HIV Deaths')

    axv[1,1].plot(sim1.tivec, sim1.results.hiv.prevalence, label=sim1.label)
    axv[1,1].plot(sim2.tivec, sim2.results.hiv.prevalence, ls=':', label=sim2.label)
    axv[1,1].set_title('HIV Prevalence')

    plt.legend()
    return fig


def analyze_people(sim):
    p = sim.people
    ever_alive = np.argwhere(np.isfinite(sim.people.age.raw)).flatten()

    last_year = np.full(len(ever_alive), sim.year)
    dead = ~p.alive.raw[ever_alive]
    last_year[dead] = sim.yearvec[p.ti_dead.raw[ever_alive][dead].astype(int)]
    first_year = np.maximum(sim.yearvec[0], last_year - p.age.raw[ever_alive])

    infected = np.isfinite(p.hiv.ti_infected.raw[ever_alive])
    infected_year = np.full(ever_alive.shape, np.nan)
    infected_year[infected] = sim.yearvec[p.hiv.ti_infected.raw[ever_alive][infected].astype(int)]

    art = np.isfinite(p.hiv.ti_art.raw[ever_alive])
    art_year = np.full(ever_alive.shape, np.nan)
    art_year[art] = sim.yearvec[p.hiv.ti_art.raw[ever_alive][art].astype(int)]

    dead = (p.hiv.ti_dead.raw[ever_alive] < sim.ti) # np.isfinite(p.hiv.ti_dead.raw[ever_alive])
    dead_year = np.full(ever_alive.shape, np.nan)
    dead_year[dead] = sim.yearvec[p.hiv.ti_dead.raw[ever_alive][dead].astype(int)]

    df = pd.DataFrame({
        'id': [hash((p.slot[i], first_year[i], p.female[i])) for i in ever_alive], # if slicing, don't need ._view,
        'first_year': first_year,
        'last_year': last_year,
        'infected_year': infected_year,
        'art_year': art_year,
        'dead_year': dead_year,

        # Useful for debugging, but not needed for plotting
        'slot': p.slot.raw[ever_alive],
        'female': p.female.raw[ever_alive],
    })
    #df.replace(to_replace=ss.INT_NAN, value=np.nan, inplace=True)
    #df['age_infected'] = df['age_initial'] + df['ti_infected']
    #df['age_art']      = df['age_initial'] + df['ti_art']
    #df['age_dead']     = df['age_initial'] + df['ti_dead']
    return df


def plot_longitudinal(sim1, sim2):

    df1 = analyze_people(sim1)
    df1['sim'] = 'Baseline'
    df2 = analyze_people(sim2)
    df2['sim'] = 'With ART'

    df = pd.concat([df1, df2]).set_index('id')

    df['ypos'] = pd.factorize(df.index.values)[0]
    N = df['sim'].nunique()
    height = 0.5/N

    fig, ax = plt.subplots(figsize=(10,6))

    # For the legend:
    fy = df['first_year'].min()
    plt.barh(y=0, left=fy, width=1e-6, color='k', height=height, label='Alive')
    plt.barh(y=0, left=fy, width=1e-6, color='m', height=height, label='Infected before birth')
    plt.barh(y=0, left=fy, width=1e-6, color='r', height=height, label='Infected')
    plt.barh(y=0, left=fy, width=1e-6, color='g', height=height, label='ART')
    plt.scatter(y=0, x=fy, color='c', marker='|', label='Death')

    for n, (lbl, data) in enumerate(df.groupby('sim')):
        yp = data['ypos'] + n/(N+1) # Leave space

        #ti_initial = np.maximum(-data['age_initial'], 0)
        #ti_final = data['ti_dead'].fillna(40)
        #plt.barh(y=yp, left=ti_initial, width=ti_final - ti_initial, color='k', height=height)
        plt.barh(y=yp, left=data['first_year'], width=data['last_year'] - data['first_year'], color='k', height=height)


        # Infected
        infected = ~data['infected_year'].isna()
        plt.barh(y=yp[infected], left=data.loc[infected]['infected_year'], width=data.loc[infected]['last_year'] - data.loc[infected]['infected_year'], color='r', height=height)

        # Infected before birth
        vertical = data['infected_year'] < data['first_year']
        plt.barh(y=yp[vertical], left=data.loc[vertical]['infected_year'], width=data.loc[vertical]['first_year'] - data.loc[vertical]['infected_year'], color='m', height=height)

        # ART
        art = ~data['art_year'].isna()
        plt.barh(y=yp[art], left=data.loc[art]['art_year'], width=data.loc[art]['last_year'] - data.loc[art]['art_year'], color='g', height=height)

        # Dead
        dead = ~data['dead_year'].isna()
        plt.scatter(y=yp[dead], x=data.loc[dead]['dead_year'], color='c', marker='|')

    ax.set_xlabel('Year')
    ax.set_ylabel('UID')
    ax.legend()

    return fig


if __name__ == '__main__':
    #ss.options(_centralized = True)

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

    if do_plot_graph:
        plot_graph(sim1, sim2)

    if do_plot_timeseries:
        plot_ts()

    plt.show()
    print('Done')