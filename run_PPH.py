"""
Postpartum hemorrhage (PPH)  simulation with an intervention like E-MOTIVE
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse
import sciris as sc

from PPH_demographics import PPH

PPH_INTV_EFFICACY = 0.6 # 60% reduction in maternal mortality due to PPH with intervention

default_n_agents = 1_000
default_n_rand_seeds = 250

do_plot_longitudinal = True
do_plot_timeseries = True

rngs = ['multi', 'centralized']

figdir = os.path.join(os.getcwd(), 'figs', 'PPH')
sc.path(figdir).mkdir(parents=True, exist_ok=True)


def run_sim(n_agents=default_n_agents, rand_seed=0, rng='multi', idx=0, pph_intv=False):
    # Make people using the distribution of the population by age/sex in 1995
    #age_data = pd.read_csv('test_data/nigeria_age.csv')
    #ppl = ss.People(n_agents, age_data=age_data)

    age_data = pd.read_csv('data/ssa_agedist.csv')
    pars = {
        'start': 1980,
        'end': 2020,
        'remove_dead': True,
        'rand_seed': rand_seed,
        #'slot_scale': 10,
    }

    asfr_data = pd.read_csv('data/ssa_asfr.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    preg_pars = {
        #'fertility_rate': 50, # per 1,000 live women. TODO: ASFR
        'fertility_rate': asfr_data, # per 1,000 live women.
        'maternal_death_rate': 1/1000, # live births, actually a probability
    }
    if pph_intv:
        preg_pars['maternal_death_rate'] *= 1-PPH_INTV_EFFICACY # Reduction in maternal mortality due to PPH with intervention
    preg = PPH(preg_pars)

    death_pars = {
        'death_rate': 10, # per 1,000. TODO: ASMR
    }
    deaths = ss.Deaths(death_pars)

    sim = ss.Sim(people=ppl, diseases=[], demographics=[preg, deaths], networks=ss.MaternalNet(), pars=pars) # Can add label
    sim.initialize()

    sim.run()

    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'pph.mother_died.cumsum': sim.results.pph.mother_died.cumsum(),
        'pph.births.cumsum': sim.results.pph.births.cumsum(),
        'pph.maternal_deaths.cumsum': sim.results.pph.maternal_deaths.cumsum(),
        'pph.infant_deaths.cumsum': sim.results.pph.infant_deaths.cumsum(),
    })
    df['pph_intv'] = pph_intv
    df['rand_seed'] = rand_seed
    df['rng'] = rng

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and pph_intv={pph_intv}, rng={rng}')

    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    times = {}
    for rng in rngs:
        ss.options(rng=rng)
        cfgs = []
        for rs in range(n_seeds):
            for pph_intv in [False, True]:
                cfgs.append({'pph_intv':pph_intv, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df


def plot_scenarios(df):
    d = pd.melt(df, id_vars=['year', 'rand_seed', 'pph_intv', 'rng'], var_name='channel', value_name='Value')
    d['baseline'] = d['pph_intv']==0
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['year', 'channel', 'rand_seed', 'pph_intv', 'rng'])[['Value']].reset_index('pph_intv')
    scn = scn.set_index(['year', 'channel', 'rand_seed', 'pph_intv', 'rng'])[['Value']].reset_index('pph_intv')
    mrg = scn.merge(bl, on=['year', 'channel', 'rand_seed', 'rng'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    cor = mrg.groupby(['year', 'channel', 'rng', 'pph_intv'])[['Value', 'Value_ref']].apply(lambda x: np.corrcoef(x['Value'], x['Value_ref'], rowvar=False)[0,1])
    cor.name = 'Pearson'

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='year', y='Value', hue='pph_intv', col='channel', row='rng', row_order=rngs,
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='rng: {row_name}')
    g.set_xlabels('Year')
    g.figure.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('rng'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='year', y='Value - Reference', hue='pph_intv', col='channel',
                row='pph_intv', height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Year')
        g.figure.savefig(os.path.join(figdir, f'diff_{ms}.png'), bbox_inches='tight', dpi=300)

    ## CORRELATION OVER TIME
    try:
        g = sns.relplot(kind='line', data=cor.to_frame(), x='year', y='Pearson', hue='rng', hue_order=rngs, col='channel',
                row='pph_intv', height=3, aspect=1.0, palette='Set1', errorbar='sd', lw=1, facet_kws=fkw)
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.figure.subplots_adjust(top=0.88)
        g.set_xlabels('Year')
        g.figure.savefig(os.path.join(figdir, 'cor.png'), bbox_inches='tight', dpi=300)
    except:
        print('CORRELATION OVER TIME did not work')

    ## FINAL TIME
    tf = df['year'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='pph_intv', x='Value - Reference',
            col='channel', row='rng', row_order=rngs, height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='rng: {row_name}')
    g.set_xlabels(f'Value - Reference at year {tf}')
    g.figure.savefig(os.path.join(figdir, 'final.png'), bbox_inches='tight', dpi=300)

    ## COR SCATTER FINAL TIME
    ctf = mtf.reset_index('rand_seed').set_index('pph_intv', append=True).sort_index()
    g = sns.relplot(data=ctf, kind='scatter', hue='rng', hue_order=rngs, style='rng', style_order=rngs, x='Value_ref', y='Value',
            col='channel', row='pph_intv', height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
    g.set_xlabels(f'Reference at year {tf}')
    g.set_ylabels(f'Value at year {tf}')
    g.figure.savefig(os.path.join(figdir, 'cor_final.png'), bbox_inches='tight', dpi=300)

    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(args.plot, index_col=0)
    else:
        print('Running scenarios')
        df = run_scenarios(n_agents=args.n, n_seeds=args.s)

    print(df)

    plot_scenarios(df)
    #plt.show()

    print('Done')
