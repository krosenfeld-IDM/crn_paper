"""
Exploring singlerng vs rng for a model HIV system sweeping coverage of ART.
"""

# %% Imports and settings
import os
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
from plotting import plot_scenarios
from analyzers import GraphAnalyzer

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

sc.options(interactive=False) # Assume not running interactively

rngs = ['centralized', 'multi'] # 'single', 

debug = False
default_n_agents = [10_000, 10_000][debug]
default_n_rand_seeds = [250, 3][debug] # 1000

base_vmmc = 0.4
inc_vmmc_cov_levels = [base_vmmc + 0.1] + [0] # Must include 0 as that's the baseline
vmmc_eff = 0.6

figdir = os.path.join(os.getcwd(), 'figs', 'VMMC_longer' if not debug else 'VMMC-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

class change_beta(ss.Intervention):
    def __init__(self, years, xbetas):
        super().__init__()
        self.name = 'Change Beta Intervention'
        self.years = years
        self.xbetas = xbetas
        self.change_inds = None
        return

    def apply(self, sim):
        if self.change_inds is None:
            self.change_inds = sc.findnearest(sim.yearvec, self.years)

        idx = np.where(sim.ti == self.change_inds)[0]
        if len(idx):
            xbeta = self.xbetas[idx[0]]

            for lbl, disease in sim.diseases.items():
                for blbl, betas in disease.pars.beta.items():
                    sim.diseases[lbl].pars.beta[blbl] = [b*xbeta for b in betas] # Note, multiplicative

        return


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
    #en_pars = dict(duration=ss.weibull(c=1.5, scale=8)) # c is shape, gives a mean of about 0.9*scale years.

    networks = ss.ndict(ss.EmbeddingNet(en_pars), ss.MaternalNet())

    default_hiv_pars = {
        #'beta': {'embedding': [0.015, 0.012], 'maternal': [0.2, 0]},
        #'beta': {'embedding': [0.012, 0.010], 'maternal': [0.2, 0]},
        'beta': {'embedding': [0.010, 0.008], 'maternal': [0.2, 0]},
        #'beta': {'embedding': [0.008, 0.006], 'maternal': [0.2, 0]},
        'init_prev': np.maximum(5/n_agents, 0.02),
        'art_efficacy': 0.8,
        'VMMC_efficacy': 0.6,
    }
    hiv_pars = sc.mergedicts(default_hiv_pars, hiv_pars)
    hiv = ss.HIV(hiv_pars)

    asfr_data = pd.read_csv('data/ssa_asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=asfr_data, rel_fertility=0.5)

    asmr_data = pd.read_csv('data/ssa_asmr.csv')
    deaths = ss.Deaths(death_rate=asmr_data)

    #y = [1995, 2000, 2005]
    #xb = [0.7, 0.7, 0.7] # Multiplicative reductions
    #interventions = [change_beta(y, xb)]
    interventions = []

    default_pars = {
        'start': 1980,
        'end': 2045, # 2030, 2045, 2075
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
    interventions += [ ss.hiv.ART(year=[2004, 2020, 2030], coverage=[0, 0.58, 0.62]) ] #0.55
    interventions += [ ss.hiv.VMMC(year=[2007, 2020, 2025, 2030], coverage=[0, base_vmmc, cov, 0]) ]

    sim = ss.Sim(people=ppl, networks=networks, diseases=[hiv], demographics=[pregnancy, deaths], interventions=interventions, pars=pars, label=lbl)
    sim.initialize()
    sim.run()

    if return_sim:
        return sim

    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'hiv.n_infected': sim.results.hiv.n_infected, # Optional, but mostly redundant with prevalence
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

    #plot_scenarios(results, figdir, channels=['Births', 'Deaths', 'Infections', 'Prevalence', 'Prevalence 15-49', 'ART Coverage', 'VMMC Coverage 15-49', 'Population'], var1='cov', var2='channel', slice_year = -1)

    figdir_2030 = os.path.join(figdir, '2030')
    sc.path(figdir_2030).mkdir(parents=True, exist_ok=True)
    plot_scenarios(results, figdir_2030, channels=['Births', 'Deaths', 'Infections', 'Prevalence', 'Prevalence 15-49', 'ART Coverage', 'VMMC Coverage 15-49', 'Population'], var1='cov', var2='channel', slice_year = 2030) # slice_year = 2030
    print('Done')
