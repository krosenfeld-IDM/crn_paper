"""
Example 3) Impact of voluntary medical male circumcision (VMMC) on HIV.
"""

# %% Imports and settings
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
from plotting import plot_scenarios
from analyzers import GraphAnalyzer
from hiv import HIV, ART, VMMC

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

sc.options(interactive=False) # Assume not running interactively

rngs = ['centralized', 'multi']

debug = False
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [500, 15][debug]

base_vmmc = 0.4
inc_vmmc_cov_levels = [base_vmmc + 0.1] + [0] # Must include 0 as that's the baseline
vmmc_eff = 0.6

figdir = os.path.join(os.getcwd(), 'figs', 'VMMC' if not debug else 'VMMC-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n_agents, idx, cov, rand_seed, rng, pars=None, hiv_pars=None, return_sim=False, analyze=False):

    age_data = pd.read_csv('data/ssa_agedist.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    def rel_dur(self, sim, uids):
        # Increase mean relationship duration with age, representing transition towards marital
        dur = 2*np.ones(len(uids))
        dur[sim.people.age[uids] > 20] = 4
        dur[sim.people.age[uids] > 25] = 8
        dur[sim.people.age[uids] > 30] = 12
        return dur

    en_pars = dict(duration=ss.weibull(c=2.5, scale=rel_dur)) # c is shape, gives a mean of about 0.9*scale years.

    networks = ss.ndict(ss.EmbeddingNet(en_pars), ss.MaternalNet())

    default_hiv_pars = {
        'beta': {'embedding': [0.010, 0.008], 'maternal': [0.2, 0]},
        'init_prev': np.maximum(5/n_agents, 0.02),
        'art_efficacy': 0.8,
        'VMMC_efficacy': 0.6,
    }
    hiv_pars = sc.mergedicts(default_hiv_pars, hiv_pars)
    hiv = HIV(hiv_pars)

    asfr_data = pd.read_csv('data/ssa_asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=asfr_data, rel_fertility=0.5)

    asmr_data = pd.read_csv('data/ssa_asmr.csv')
    deaths = ss.Deaths(death_rate=asmr_data)

    interventions = []

    default_pars = {
        'start': 1980,
        'end': 2070,
        'dt': 1/12,
        'rand_seed': rand_seed,
        'verbose': 0,
        'slot_scale': 10, # Increase slot scale to reduce repeated slots
        'analyzers': [GraphAnalyzer()] if analyze else [],
    }
    pars = sc.mergedicts(default_pars, pars)

    lbl = f'Sim {idx}: agents={n_agents}, cov={cov}, seed={rand_seed}, rng={rng}'
    print('Starting', lbl)

    # ART calibrated to be near 90-90-90 and 95-95-95, drifts upwards due to mortality effect
    interventions += [ ART(year=[2004, 2020, 2030], coverage=[0, 0.58, 0.62]) ]
    interventions += [ VMMC(year=[2007, 2020, 2025, 2030], coverage=[0, base_vmmc, cov, 0]) ]

    sim = ss.Sim(people=ppl, networks=networks, diseases=[hiv], demographics=[pregnancy, deaths], interventions=interventions, pars=pars, label=lbl)
    sim.initialize()
    sim.run()

    if return_sim:
        return sim

    df = pd.DataFrame( {
        'year': sim.yearvec,
        'Births': sim.results.pregnancy.births.cumsum(),
        'Deaths': sim.results.hiv.new_deaths.cumsum(),
        'Infections': sim.results.hiv.cum_infections,
        'Prevalence': sim.results.hiv.prevalence,
        'Prevalence 15-49': sim.results.hiv.prevalence_15_49,
        'ART Coverage': sim.results.hiv.art_coverage,
        'VMMC Coverage': sim.results.hiv.vmmc_coverage,
        'VMMC Coverage 15-49': sim.results.hiv.vmmc_coverage_15_49,
        'Population': sim.results.n_alive,
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['network'] = 'Embedding'
    df['eff'] = vmmc_eff
    df['rng'] = rng
    df['n_agents'] = n_agents

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and cov={cov}, rng={rng}')

    return df

def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    times = {}
    for rng in rngs:
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for rs in range(n_seeds):
            for cov in inc_vmmc_cov_levels:
                cfgs.append({'cov':cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))
    return df


def plot_sim_savings(df, figdir, channels=None):
    import seaborn as sns
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    sns.set_theme(font_scale=1.2, style='whitegrid')

    # Renaming
    df.replace({'rng': {'centralized':'Centralized', 'multi': 'CRN'}}, inplace=True)

    first_year = int(df['year'].iloc[0])
    assert df['year'].iloc[0] == first_year
    df['date'] = pd.to_datetime(365 * (df['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1)) #pd.to_datetime(df['year'], format='%f', unit='Y')

    covs = df['cov'].unique()
    covs.sort()
    rep = {c: f'{c:.0%}' if c > 0 else 'Reference' for c in covs}
    df.replace({'cov': rep}, inplace=True)
    cov_ord = [rep[c] for c in covs]
    first = cov_ord.pop(0)
    cov_ord.append(first)
    df['cov'] = pd.Categorical(df['cov'], categories=cov_ord) # Agh!

    df.rename(columns={'cov': 'Coverage'}, inplace=True)
    cov = 'Coverage'

    id_vars = ['date', 'rand_seed', cov, 'rng', 'network', 'eff', 'n_agents']
    d = pd.melt(df, id_vars=['date', 'rand_seed', cov, 'rng', 'network', 'eff', 'n_agents'], var_name='channel', value_name='Value')

    # Slice to the one channel specified by the user
    d = d.loc[d['channel'].isin(channels)]

    d['channel'] = pd.Categorical(d['channel'], categories = channels)

    d['baseline'] = d[cov]=='Reference'
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]

    id_vars.append('channel')
    bl = bl.set_index(id_vars)[['Value']].reset_index(cov)
    scn = scn.set_index(id_vars)[['Value']].reset_index(cov)

    id_vars.remove(cov)
    mrg = scn.merge(bl, on=id_vars, suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg['Cases Averted'] = mrg['Value_ref'] - mrg['Value']
    mrg = mrg.sort_index()
    mrg['Coverage'] = pd.Categorical(mrg['Coverage'], categories=cov_ord[:-1]) # Agh^2!

    id_vars = ['date', cov, 'rng', 'network', 'eff', 'n_agents', 'channel']

    # Standard error
    se = mrg.reset_index().groupby(id_vars, observed=True)[['Value', 'Value_ref']].apply(lambda x: np.std(x['Value_ref']-x['Value'], ddof=1) / np.sqrt(len(x)))
    se.name = 'Standard Error'

    sep = pd.pivot(se.reset_index(), index=['date', cov, 'network', 'eff', 'n_agents', 'channel'], columns='rng', values='Standard Error')
    sep['SE Ratio'] = sep['Centralized'] / sep['CRN']
    sep['Sim Savings'] = sep['SE Ratio']**2


    fig, ax = plt.subplots(figsize=(9, 3))
    g = sns.lineplot(data=sep, x='date', y='Sim Savings', hue='channel', ax=ax)
    g.set_xlabel('Year')
    g.set_ylabel('Fold Reduction in Replicates')
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    g.axes.xaxis.set_major_locator(locator)
    g.axes.xaxis.set_major_formatter(formatter)
    g.set_ylim(bottom=0)

    dates = sep.loc[~np.isnan(sep['Sim Savings'])].index.get_level_values('date')
    g.set_xlim(left=dt.datetime(year=2020, month=1, day=1), right=dates[-1])
    g.figure.savefig(os.path.join(figdir, 'sim_savings.pdf'), bbox_inches='tight', transparent=True)
    plt.close(g.figure)


    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', action=argparse.BooleanOptionalAction)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    if args.plot:
        fn = os.path.join(figdir, 'results.csv')
        print('Reading CSV file', fn)
        results = pd.read_csv(fn, index_col=0)
    else:
        print('Running scenarios')
        results = run_scenarios(n_agents=args.n, n_seeds=args.s)

    plot_sim_savings(results.copy(), figdir, channels=['Infections', 'Deaths'])
    plot_scenarios(results, figdir, channels=['Births', 'Deaths', 'Infections', 'Prevalence', 'Prevalence 15-49', 'ART Coverage', 'VMMC Coverage 15-49', 'Population'], var1='cov', var2='channel', slice_year = [2025, 2040, 2070]) # slice_year = 2030
    print('Done')
