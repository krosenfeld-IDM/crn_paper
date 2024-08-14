"""
Postpartum hemorrhage (PPH)  simulation with an intervention like E-MOTIVE
"""

# %% Imports and settings
import starsim as ss
import pandas as pd
import numpy as np
import os
import argparse
import sciris as sc

from PPH_demographics import PPH
from plotting import plot_scenarios

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False

covs = [0.1, 0.9] + [0]
PPH_INTV_EFFICACY = 0.6 # 60% reduction in maternal mortality due to PPH with intervention

default_n_agents = [100_000, 10_000][debug]
default_n_rand_seeds = [250, 3][debug]

rngs = ['centralized', 'multi']

figdir = os.path.join(os.getcwd(), 'figs', 'PPH' if not debug else 'PPH-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)
channels = ['Births', 'Maternal Deaths', 'Total Deaths'] # Set to None for all channels

class PPH_Intv(ss.Intervention):

    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        self.requires = PPH
        self.year = sc.promotetoarray(year)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.p_pphintv = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage))
        #def fun(self, sim, uids):
        #    return np.interp(sim.year, self.year, self.coverage)
        #self.p_pphintv = ss.bernoulli(p=fun)
        self.eff_pphintv = ss.bernoulli(p=PPH_INTV_EFFICACY)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_pphintv', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_mothers_saved', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        pph = sim.demographics['pph']
        maternal_deaths = (pph.ti_dead <= sim.ti).uids
        receive_pphintv = self.p_pphintv.filter(maternal_deaths)
        pph_deaths_averted = self.eff_pphintv.filter(receive_pphintv)
        pph.ti_dead[pph_deaths_averted] = np.nan # Used to request death
        sim.people.ti_dead[pph_deaths_averted] = np.nan # Used to actually "remove" people

        # Add results
        self.results['n_pphintv'][self.sim.ti] = len(receive_pphintv)
        self.results['n_mothers_saved'][self.sim.ti] = len(pph_deaths_averted)

        return len(pph_deaths_averted)


def run_sim(n_agents=default_n_agents, rand_seed=0, rng='multi', idx=0, cov=0):
    print(f'Starting sim {idx} with rand_seed={rand_seed} and cov={cov}, rng={rng}')

    # Make people using the distribution of the population by age
    pars = {
        'start': 2024,
        'end': 2030,
        #'remove_dead': True,
        'rand_seed': rand_seed,
        #'slot_scale': 10,
        'verbose': 0,
        'dt': 0.25,
    }

    age_data = pd.read_csv('data/ssa_agedist.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    asfr_data = pd.read_csv('data/ssa_asfr.csv')
    preg_pars = {
        #'fertility_rate': 50, # per 1,000 live women
        'fertility_rate': asfr_data, # per 1,000 live women.
        'maternal_death_prob': 1/1000, # Maternal death prob due to PPH per live birth (additive to demographic deaths)
        #'min_age': -1,
        #'max_age': 1000,
    }
    preg = PPH(preg_pars)


    asmr_data = pd.read_csv('data/ssa_asmr.csv')
    death_pars = {
        #'death_rate': 10, # per 1,000
        'death_rate': asmr_data, # rate per person
        'units': 1
    }
    deaths = ss.Deaths(death_pars)

    delay_start = 1 # years
    if cov > 0:
        interventions = PPH_Intv(
            year =      [pars['start'], pars['start']+delay_start-0.1, pars['start']+delay_start], 
            coverage =  [            0,                             0,                       cov] )
    else:
        interventions = None

    sim = ss.Sim(people=ppl, diseases=[], demographics=[preg, deaths], networks=ss.MaternalNet(), pars=pars, interventions=interventions) # Can add label
    sim.run()

    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'pph.mother_died.cumsum': sim.results.pph.mother_died.cumsum(),
        'Births': sim.results.pph.births.cumsum(),
        'CBR': sim.results.pph.cbr,
        'New Deaths': sim.results['new_deaths'],
        'Total Deaths': sim.results['cum_deaths'],
        'Maternal Deaths': sim.results.pph.maternal_deaths.cumsum(),
        'Infant Deaths': sim.results.pph.infant_deaths.cumsum(),
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['rng'] = rng
    df['network'] = 'None'
    df['eff'] = PPH_INTV_EFFICACY
    df['n_agents'] = n_agents

    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    times = {}
    for rng in rngs:
        #ss.options(rng=rng)
        ss.options(_centralized = rng=='centralized')
        cfgs = []
        for rs in range(n_seeds):
            for cov in covs:
                cfgs.append({'cov':cov, 'rand_seed':rs, 'rng':rng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False) # debug
        times[f'rng={rng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df


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
        df['network'] = 'None'
    else:
        print('Running scenarios')
        df = run_scenarios(n_agents=args.n, n_seeds=args.s)

    print(df)

    plot_scenarios(df, figdir, channels, var1='cov', var2='channel')

    print('Done')
