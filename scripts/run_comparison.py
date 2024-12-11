"""
Compare two HIV simulations, one baseline and the other with ART
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sciris as sc
from run_HIV import run_sim

sc.options(interactive=True)

default_n_agents = 100
network = 'EmbeddingNet'
art_eff = 1

do_plot_graph = True
kind = ['radial', 'bipartite', 'spring', 'multipartite'][1]

do_plot_longitudinal = True
do_plot_timeseries = True

figdir = os.path.join(os.getcwd(), 'figs', network)
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_scenario(n=10, rand_seed=0, analyze=True):
    sims = sc.parallelize(
        run_sim,
        kwargs=dict(n_agents=n, analyze=analyze, rand_seed=rand_seed, return_sim=True, art_eff=art_eff),
        iterkwargs=[{'cov': 0.00, 'idx': 0, 'rng': 'multi'}, {'cov': 0.10, 'idx': 1, 'rng': 'multi'}],
        die=False
    )

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def getpos(ti, g1, g2, guess=None, kind='bipartite'):
    nodes = sc.mergedicts(g2[ti].graph.nodes.data(), g1[ti].graph.nodes.data())
    n = len(nodes)

    if kind == 'radial':
        pos = {i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)}
        if guess and len(guess) < n:
            pos.update(guess)

    elif kind == 'spring':
        pos = nx.spring_layout(g1[ti].graph, pos=guess, fixed=None, iterations=50, threshold=1e-4)
        if guess:
            pos.update(guess)

    elif kind == 'multipartite':
        pos = nx.multipartite_layout(g1[ti].graph, subset_key='female', align='vertical', scale=10)
        if guess:
            pos.update(guess)
            for k in guess:
                pos[k] = (pos[k][0], guess[k][1])

    elif kind == 'bipartite':
        pos = {i: (nd['age'], 2 * nd['female'] - 1 + np.random.uniform(-0.3, 0.3)) for i, nd in nodes.items()}
        if guess:
            for k in guess:
                pos[k] = (pos[k][0], guess[k][1])

    return pos


def plot_graph(sim1, sim2):
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs

    timax = len(g1) - 1
    pos = {0: getpos(0, g1, g2, kind=kind)}
    for ti in range(1, timax + 1):
        pos[ti] = getpos(ti, g1, g2, guess=pos[ti - 1], kind=kind)

    def on_press(event):
        nonlocal ti
        if event.key == 'right':
            ti = min(ti + 1, timax)
        elif event.key == 'left':
            ti = max(ti - 1, 0)

        ax[0].clear()
        ax[1].clear()

        g1[ti].plot(pos[ti], ax=ax[0])
        g2[ti].plot(pos[ti], ax=ax[1])
        fig.suptitle(f'Time is {ti}')
        ax[0].set_title(sim1.label)
        ax[1].set_title(sim2.label)
        fig.canvas.draw()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.mpl_connect('key_press_event', on_press)
    return fig


def plot_ts(sim1, sim2):
    fig, axv = plt.subplots(2, 2, sharex=True)

    axv[0, 0].plot(sim1.t.yearvec, sim1.results.hiv.art_coverage, label=sim1.label)
    axv[0, 0].plot(sim2.t.yearvec, sim2.results.hiv.art_coverage, ls=':', label=sim2.label)
    axv[0, 0].set_title('ART Coverage')

    axv[0, 1].plot(sim1.t.yearvec, sim1.results.hiv.cum_infections, label=sim1.label)
    axv[0, 1].plot(sim2.t.yearvec, sim2.results.hiv.cum_infections, ls=':', label=sim2.label)
    axv[0, 1].set_title('Cumulative Infections')

    axv[1, 0].plot(sim1.t.yearvec, sim1.results.hiv.new_deaths.cumsum(), label=sim1.label)
    axv[1, 0].plot(sim2.t.yearvec, sim2.results.hiv.new_deaths.cumsum(), ls=':', label=sim2.label)
    axv[1, 0].set_title('Cumulative Deaths')

    axv[1, 1].plot(sim1.t.yearvec, sim1.results.hiv.prevalence, label=sim1.label)
    axv[1, 1].plot(sim2.t.yearvec, sim2.results.hiv.prevalence, ls=':', label=sim2.label)
    axv[1, 1].set_title('Prevalence')

    plt.legend()
    return fig


if __name__ == '__main__':
    args = sc.objdict()
    args.n = default_n_agents
    args.s = 0

    sim1, sim2 = run_scenario(n=args.n, rand_seed=args.s)

    if do_plot_longitudinal:
        plot_longitudinal(sim1, sim2)

    if do_plot_graph:
        plot_graph(sim1, sim2)

    if do_plot_timeseries:
        plot_ts(sim1, sim2)

    plt.show()
    print('Done')
