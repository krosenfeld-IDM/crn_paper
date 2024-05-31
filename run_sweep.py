"""
A simple beta sweep with rng using HIV as an example pathogen
"""

# %% Imports and settings
import os
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import seaborn as sns

n = 1_000 # Agents
n_rand_seeds = 25
xf_levels = [0.5, 0.8, 1.26, 2.0] + [1] # Must include 1 as that's the baseline | roughly np.logspace(np.log2(0.5), np.log2(20), 4, base=2)

rngs = ['centralized', 'multi']

figdir = os.path.join(os.getcwd(), 'figs', 'Sweep')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n, xf, rand_seed, rng):
    ppl = ss.People(n)

    rel_pars = {
        'male_shift': 5,
        'std': 3,
        'pars': {
            #'duration': ss.lognorm_mean(mean=15*xf, stdev=15*xf),  # Can vary by age, year, and individual pair
            'duration': ss.lognorm_ex(mean=15, stdev=15),  # Can vary by age, year, and individual pair
            'part_rates': 0.9,  # Participation rates - can vary by sex and year
            'rel_part_rates': 1.0,
            'debut': 16,  # Age of debut can vary by sex, year, and individual
            'rel_debut': 1.0,
        }} 
    networks = [ss.EmbeddingNet(**rel_pars), ss.MaternalNet()]

    hiv_pars = {
        'beta': {'embedding': [xf * 0.30, xf * 0.25], 'maternal': [xf * 0.2, 0]},
        #'beta': {'embedding': [0.30, 0.25], 'maternal': [0.2, 0]},
        'init_prev': 0.05
    }
    hiv = ss.HIV(hiv_pars)

    pregnancy = ss.Pregnancy(fertility_rate=20)

    pars = {
        'start': 1980,
        'end': 2020,
        'rand_seed': rand_seed,
    }
    sim = ss.Sim(people=ppl, diseases=[hiv], demographics=[pregnancy], networks=networks, pars=pars, label=f'Sim with {n} agents and xf={xf}')
    if rng == 'centralized':
        for dist in sim.dists.dists.values():
            dist.rng = np.random.mtrand._rand
    sim.initialize()
    sim.run()

    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'hiv.n_infected': sim.results.hiv.n_infected,
        'hiv.prevalence': sim.results.hiv.prevalence,
        'hiv.cum_deaths': sim.results.hiv.new_deaths.cumsum(),
        'pregnancy.cum_births': sim.results.pregnancy.births.cumsum(),
    })
    df['xf'] = xf
    df['rand_seed'] = rand_seed
    df['rng'] = rng

    return df

def run_scenarios(figdir):
    results = []
    times = {}
    for rng in rngs:
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for rs in range(n_rand_seeds):
            for xf in xf_levels:
                cfgs.append({'xf':xf, 'rand_seed':rs, 'rng':rng})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=cfgs, die=True)
        times[f'RNG={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df


def plot_scenarios(df, figdir):
    d = pd.melt(df, id_vars=['year', 'rand_seed', 'xf', 'rng'], var_name='channel', value_name='Value')
    d['baseline'] = d['xf']==1
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['year', 'channel', 'rand_seed', 'xf', 'rng'])[['Value']].reset_index('xf')
    scn = scn.set_index(['year', 'channel', 'rand_seed', 'xf', 'rng'])[['Value']].reset_index('xf')
    mrg = scn.merge(bl, on=['year', 'channel', 'rand_seed', 'rng'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='year', y='Value', hue='xf', col='channel', row='rng', row_order=rngs,
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='RNG: {row_name}')
    g.set_xlabels('Year')
    g.figure.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('rng'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='year', y='Value - Reference', hue='xf', col='channel', row='xf',
            height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Beta: {row_name}')
        g.figure.suptitle(ms)
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Value - Reference at Year')
        g.figure.savefig(os.path.join(figdir, f'diff_{ms}.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['year'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='xf', x='Value - Reference', col='channel', row='rng', row_order=rngs,
        height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='RNG: {row_name}')
    g.set_xlabels(f'Value - Reference at Year {tf}$')
    g.figure.savefig(os.path.join(figdir, 'final.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME function of beta
    dtf = d.set_index(['year', 'rand_seed']).sort_index().loc[tf]
    g = sns.relplot(kind='line', data=dtf.reset_index(), x='xf', y='Value', col='channel', row='rng', row_order=rngs,
        height=5, aspect=1.2, facet_kws=fkw, estimator=None, units='rand_seed', lw=0.25)
    g.set_titles(col_template='{col_name}', row_template='RNG: {row_name}')
    g.set_ylabels(f'Value at Year {tf}')
    g.figure.savefig(os.path.join(figdir, 'final_beta.png'), bbox_inches='tight', dpi=300)

    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Folder containing cached results.csv', type=str)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(os.path.join(args.plot, 'result.csv'), index_col=0)
        figdir = args.plot
    else:
        print('Running scenarios')
        df = run_scenarios(figdir)

    plot_scenarios(df, figdir)

    print('Done')