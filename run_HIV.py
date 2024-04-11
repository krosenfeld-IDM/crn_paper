"""
Exploring singlerng vs rng for a model HIV system sweeping coverage of ART.
"""

# %% Imports and settings
import os
import starsim as ss
import scipy.stats as sps
import sciris as sc
import pandas as pd
import numpy as np
from plotting import plot_scenarios

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

sc.options(interactive=False) # Assume not running interactively

rngs = ['centralized', 'multi'] # 'single', 

debug = False
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [100, 10][debug]
intv_cov_levels = [0.01, 0.10, 0.25, 0.73] + [0] # Must include 0 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'HIV')
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
        """
        Applies the intervention to the simulation object.

        Args:
            sim (Simulation): The simulation object to apply the intervention to.

        Returns:
            None

        """
        if self.change_inds is None:
            self.change_inds = sc.findnearest(sim.yearvec, self.years)

        idx = np.where(sim.ti == self.change_inds)[0]
        if len(idx):
            xbeta = self.xbetas[idx[0]]

            for lbl, disease in sim.diseases.items():
                for blbl, betas in disease.pars.beta.items():
                    sim.diseases[lbl].pars.beta[blbl] = [b*xbeta for b in betas] # Note, multiplicative

        return


def run_sim(n_agents, idx, cov, rand_seed, rng, art_eff=0.96, pars=None, hiv_pars=None):

    ppl = ss.People(n_agents)

    en_pars = dict(duration=ss.weibull(c=1.5, scale=3)) # c is shape, gives a mean of about 0.9*scale years.
    networks = ss.ndict(ss.EmbeddingNet(en_pars), ss.MaternalNet())

    default_hiv_pars = {
        'beta': {'embedding': [0.02, 0.015], 'maternal': [0.2, 0]},
        'init_prev': np.maximum(5/n_agents, 0.01),
        'art_efficacy': art_eff,
    }
    hiv_pars = ss.omerge(default_hiv_pars, hiv_pars)
    hiv = ss.HIV(hiv_pars)

    pregnancy = ss.Pregnancy(fertility_rate=20)
    deaths = ss.Deaths(death_rate=10)

    y = [1990, 1995, 2000]
    xb = [0.7, 0.7, 0.7] # Multiplicative reductions
    cb = change_beta(y, xb)

    default_pars = {
        'start': 1980,
        'end': 2070,
        'dt': 1/12,
        'rand_seed': rand_seed,
        'verbose': 0,
        'remove_dead': True,
        'slot_scale': 10 # Increase slot scale to reduce repeated slots
    }
    pars = ss.omerge(default_pars, pars)

    lbl = f'Sim {idx}: agents={n_agents}, cov={cov}, seed={rand_seed}, rng={rng}'
    print('Starting', lbl)

    if cov > 0:
        pars['interventions'] = [ ss.hiv.ART(year=[2004, 2010, 2020], coverage=[0, cov/4, cov]) ]

    sim = ss.Sim(people=ppl, networks=networks, diseases=[hiv], demographics=[pregnancy, deaths], interventions=cb, pars=pars, label=lbl)

    if rng == 'centralized':
        for dist in sim.dists.dists.values():
            dist.rng = np.random.mtrand._rand

    sim.initialize()
    sim.run()

    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'hiv.n_infected': sim.results.hiv.n_infected, # Optional, but mostly redundant with prevalence
        'Births': sim.results.pregnancy.births.cumsum(),
        'Deaths': sim.results.hiv.new_deaths.cumsum(),
        'Prevalence': sim.results.hiv.prevalence,
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['network'] = 'Embedding'
    df['eff'] = art_eff
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
            for cov in intv_cov_levels:
                cfgs.append({'cov':cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=True, serial=False)
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    if args.plot:
        fn = os.path.join(args.plot, 'HIV', 'results.csv')
        print('Reading CSV file', fn)
        results = pd.read_csv(fn, index_col=0)
    else:
        print('Running scenarios')
        results = run_scenarios(n_agents=args.n, n_seeds=args.s)

    plot_scenarios(results, figdir, channels=['Births', 'Deaths', 'Prevalence'], var1='cov', var2='channel', slice_year = -1) # slice_year = 2020.05
    print('Done')
