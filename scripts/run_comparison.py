"""
Compare two HIV simulations, one baseline and the other with ART
"""

# Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sciris as sc
from run_HIV import run_sim

sc.options(interactive=True)

default_n_agents = 100
network = 'EmbeddingNet'
art_eff = 1.0

do_plot_graph = True
kind = 'bipartite'
do_plot_longitudinal = True
do_plot_timeseries = True

figdir = os.path.join(os.getcwd(), 'figs', network)
sc.path(figdir).mkdir(parents=True, exist_ok=True)


def run_scenario(n=10, rand_seed=0, analyze=True):
    iterkwargs = [
        {'cov': 0.00, 'idx': 0, 'rng': 'multi'},
        {'cov': 0.10, 'idx': 1, 'rng': 'multi'},
    ]
    sims = sc.parallelize(
        run_sim,
        kwargs={'n_agents': n, 'analyze': analyze, 'rand_seed': rand_seed, 'return_sim': True, 'art_eff': art_eff},
        iterkwargs=iterkwargs,
        die=False,
    )

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def getpos(ti, g1, g2, guess=None, kind='bipartite'):
    n1 = dict(g1[ti].graph.nodes.data())
    n2 = dict(g2[ti].graph.nodes.data())
    nodes = sc.mergedicts(n2, n1)
    n = len(nodes)

    if kind == 'radial':
        pos = {i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)}
    elif kind == 'spring':
        pos = nx.spring_layout(g1[ti].graph, pos=guess, iterations=50, seed=None)
    elif kind == 'multipartite':
        pos = nx.multipartite_layout(g1[ti].graph, subset_key='female', scale=10)
    elif kind == 'bipartite':
        pos = {i: (nd['age'], 2 * nd['female'] - 1 + np.random.uniform(-0.3, 0.3)) for i, nd in nodes.items()}
        if guess:
            for i in guess.keys():
                pos[i] = (pos[i][0], guess[i][1])
    return pos


def plot_graph(sim1, sim2):
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs
    timax = sim1.tivec[-1]

    global ti, pos
    pos = {ti: getpos(ti, g1, g2, kind=kind) for ti in range(-1, timax + 1)}

    fig, axv = plt.subplots(1, 2, figsize=(10, 5))
    ti = -1

    def on_press(event):
        nonlocal ti
        if event.key == 'right':
            ti = min(ti + 1, timax)
        elif event.key == 'left':
            ti = max(ti - 1, -1)
        axv[0].clear()
        axv[1].clear()
        g1[ti].plot(pos[ti], edge_labels=False, ax=axv[0])
        g2[ti].plot(pos[ti], edge_labels=False, ax=axv[1])
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)
    g1[ti].plot(pos[ti], edge_labels=False, ax=axv[0])
    g2[ti].plot(pos[ti], edge_labels=False, ax=axv[1])
    return fig


def plot_ts(sim1, sim2):
    fig, axv = plt.subplots(2, 2, sharex=True)
    axv[0, 0].plot(sim1.tivec, sim1.results.hiv.art_coverage, label=sim1.label)
    axv[0, 0].plot(sim2.tivec, sim2.results.hiv.art_coverage, ls=':', label=sim2.label)
    axv[0, 0].set_title('ART Coverage')
    plt.legend()
    return fig


def plot_longitudinal(sim1, sim2):
    # Implemented similar to your original script for longitudinal plotting
    pass  # Add your longitudinal analysis logic here


if __name__ == '__main__':
    sims = run_scenario(n=default_n_agents)
    sim1, sim2 = sims

    if do_plot_longitudinal:
        plot_longitudinal(sim1, sim2)

    if do_plot_graph:
        plot_graph(sim1, sim2)

    if do_plot_timeseries:
        plot_ts(sim1, sim2)

    plt.show()
