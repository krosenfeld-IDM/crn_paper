"""
Example 1) Postpartum hemorrhage (PPH) simulation with an intervention like E-MOTIVE
"""

# %% Imports and settings
import starsim as ss
import pandas as pd
import numpy as np
import os
import sciris as sc

from crn_paper.PPH_demographics import PPH
from crn_paper import paths
from plotting import plot_scenarios

# Suppress warnings from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True

covs = [0.1, 0.9] + [0]  # Covariate range for intervention coverage
PPH_INTV_EFFICACY = 0.6  # 60% reduction in maternal mortality due to PPH with intervention

default_n_agents = [100_000, 10_000][debug]
default_n_rand_seeds = [250, 3][debug]

rngs = ['centralized', 'multi']

figdir = os.path.join(paths.src.as_posix(), 'figs', 'PPH' if not debug else 'PPH-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)
channels = ['Births', 'Maternal Deaths', 'Total Deaths']  # Set to None for all channels

class PPH_Intv(ss.Intervention):
    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        super().__init__(**kwargs)  # Call parent constructor first
        self.requires = PPH
        self.year = sc.promotetoarray(year)
        self.coverage = sc.promotetoarray(coverage)
        self.p_pphintv = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.t.now('year'), self.year, self.coverage))
        self.eff_pphintv = ss.bernoulli(p=PPH_INTV_EFFICACY)

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_pphintv', sim.t.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_mothers_saved', sim.t.npts, dtype=int)
        self.initialized = True

    def apply(self, sim):
        if sim.t.now('year') < self.year[0]:
            return

        pph = sim.demographics['pph']
        maternal_deaths = pph.ti_dead[pph.ti_dead <= sim.t.ti].uids
        receive_pphintv = self.p_pphintv.filter(maternal_deaths)
        pph_deaths_averted = self.eff_pphintv.filter(receive_pphintv)

        pph.ti_dead[pph_deaths_averted] = np.nan  # Adjust for averted deaths
        sim.people.ti_dead[pph_deaths_averted] = np.nan

        # Update results
        self.results['n_pphintv'][sim.t.ti] = len(receive_pphintv)
        self.results['n_mothers_saved'][sim.t.ti] = len(pph_deaths_averted)

        return len(pph_deaths_averted)

def run_sim(n_agents=default_n_agents, rand_seed=0, rng='multi', idx=0, cov=0):
    print(f'Starting sim {idx} with rand_seed={rand_seed} and cov={cov}, rng={rng}')

    pars = {
        'start': 2024,
        'stop': 2030,
        'rand_seed': rand_seed,
        'verbose': 0,
        'dt': 0.25,
    }

    age_data = pd.read_csv(paths.src / 'data/ssa_agedist.csv')
    ppl = ss.People(n_agents=n_agents, age_data=age_data)

    asfr_data = pd.read_csv(paths.src / 'data/ssa_asfr.csv')
    preg_pars = {
        'fertility_rate': asfr_data,
        'maternal_death_prob': 1 / 1000,
    }
    preg = PPH(pars=preg_pars)

    asmr_data = pd.read_csv(paths.src / 'data/ssa_asmr.csv')
    death_pars = {'death_rate': asmr_data, 'rate_units': 1}
    deaths = ss.Deaths(pars=death_pars)

    delay_start = 1
    if cov > 0:
        interventions = PPH_Intv(
            year=[pars['start'], pars['start'] + delay_start - 0.1, pars['start'] + delay_start],
            coverage=[0, 0, cov]
        )
    else:
        interventions = None

    sim = ss.Sim(
        people=ppl,
        demographics=[preg, deaths],
        pars=pars,
        interventions=interventions
    )
    sim.run()

    df = pd.DataFrame({
        'year': sim.t.yearvec,
        'Births': sim.results['n_births'].cumsum(),
        'CBR': sim.results['cbr'],
        'New Deaths': sim.results['new_deaths'],
        'Total Deaths': sim.results['cum_deaths'],
        'Maternal Deaths': sim.results['n_maternal_deaths'].cumsum(),
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['rng'] = rng

    return df

def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    for rng in rngs:
        ss.options._centralized = (rng == 'centralized')
        cfgs = [{'cov': cov, 'rand_seed': rs, 'rng': rng, 'idx': idx} for idx, rs in enumerate(range(n_seeds)) for cov in covs]
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False)

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
        df = pd.read_csv(args.plot, index_col=0)
    else:
        df = run_scenarios(n_agents=args.n, n_seeds=args.s)

    plot_scenarios(df, figdir, channels)
