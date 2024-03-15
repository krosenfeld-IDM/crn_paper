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

sc.options(interactive=False) # Assume not running interactively

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

rngs = ['centralized', 'single', 'multi']

n = 1_000 # Agents
n_rand_seeds = 25
intv_cov_levels = [0.01, 0.10, 0.25, 0.90] + [0] # Must include 0 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'SIR')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n, idx, intv_cov, rand_seed, rng):

    print(f'Starting sim {idx} with rand_seed={rand_seed} and intv_cov={intv_cov}, rng={rng}')

    ppl = ss.People(n)
    # watts_strogatz_graph, erdos_renyi_graph, grid_2d_graph, configuration_model, line_graph, barabasi_albert_graph, scale_free_graph, complete_graph, maybe a tree
    #G = nx.gnp_random_graph(n, p=3/n, seed=rand_seed) #G=nx.erdos_renyi_graph(n, p=3/n, seed=rand_seed)
    G=nx.barabasi_albert_graph(n=n, m=1, seed=rand_seed)
    #G=nx.complete_graph(n)
    networks = ss.StaticNet(G)

    sir_pars = {
        'beta': 30,
        'dur_inf': sps.weibull_min(c=1, scale=30/365), # When c=1, it's an exponential
        #'dur_inf': sps.weibull_min(c=3, scale=33.5/365), # Can check sir_pars['dur_inf'].mean()
        'init_prev': 0,  # Will seed manually
        'p_death': 0, # No death
    }
    if idx == 0:
        fig = plot_degree(G)
        fig.savefig(os.path.join(figdir, 'degree.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f'Mean duration of infection is {sir_pars["dur_inf"].mean() * 365} days')
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

    sim = ss.Sim(people=ppl, networks=networks, diseases=sir, pars=pars,
            label=f'Sim with {n} agents and intv_cov={intv_cov}')
    sim.initialize()

    # Infect agent zero to start the simulation
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
    df['rng'] = rng

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and intv_cov={intv_cov}, rng={rng}')

    return df

def run_scenarios():
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
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df

def plot_scenarios(df):
    d = pd.melt(df, id_vars=['year', 'rand_seed', 'intv_cov', 'rng'], var_name='channel', value_name='Value')
    d['baseline'] = d['intv_cov']==0
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['year', 'channel', 'rand_seed', 'intv_cov', 'rng'])[['Value']].reset_index('intv_cov')
    scn = scn.set_index(['year', 'channel', 'rand_seed', 'intv_cov', 'rng'])[['Value']].reset_index('intv_cov')
    mrg = scn.merge(bl, on=['year', 'channel', 'rand_seed', 'rng'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='year', y='Value', hue='intv_cov', col='channel', row='rng', row_order=rngs,
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='rng: {row_name}')
    g.set_xlabels('Year')
    g.figure.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('rng'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='year', y='Value - Reference', hue='intv_cov', col='channel',
                row='intv_cov', height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Year')
        g.figure.savefig(os.path.join(figdir, f'diff_{ms}.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['year'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='intv_cov', x='Value - Reference',
            col='channel', row='rng', row_order=rngs, height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='rng: {row_name}')
    g.set_xlabels(f'Value - Reference at year {tf}')
    g.figure.savefig(os.path.join(figdir, 'final.png'), bbox_inches='tight', dpi=300)

    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return


def plot_degree(G):
    # Code based on https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G, seed=10396954)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Graph")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    return fig



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(args.plot, index_col=0)
    else:
        print('Running scenarios')
        df = run_scenarios()

    plot_scenarios(df)
    
    #df = run_sim(n, idx=0, intv_cov=0, rand_seed=0, rng='multi')
    #print(df)

    print('Done')
