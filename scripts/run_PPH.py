"""
Example 1) Postpartum hemorrhage (PPH) simulation with an intervention like E-MOTIVE
"""

# %% Imports and settings
import starsim as ss
import pandas as pd
import numpy as np
import os
import argparse
import sciris as sc

from crn_paper.PPH_demographics import PPH
from crn_paper import paths
from plotting import plot_scenarios

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True

covs = [0.1, 0.9] + [0]
PPH_INTV_EFFICACY = 0.6  # 60% reduction in maternal mortality due to PPH with intervention

default_n_agents = [100_000, 10_000][debug]
default_n_rand_seeds = [250, 3][debug]

rngs = ['centralized', 'multi']

figdir = os.path.join(paths.src.as_posix(), 'figs', 'PPH' if not debug else 'PPH-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)
channels = ['Births', 'Maternal Deaths', 'Total Deaths']  # Set to None for all channels


class PPH_Intv(ss.Intervention):

    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        super().__init__(**kwargs)
        self.year = sc.promotetoarray(year)
        self.coverage = sc.promotetoarray(coverage)
        self.p_pphintv = ss.bernoulli(p=lambda sim, uids: np.interp(sim.year, self.year, self.coverage))
        self.eff_pphintv = ss.bernoulli(p=PPH_INTV_EFFICACY)

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results.define_results(
            ss.Result('n_pphintv', dtype=int, label='Number receiving intervention'),
            ss.Result('n_mothers_saved', dtype=int, label='Number of mothers saved'),
        )

    def step(self):
        sim = self.sim
        if sim.year < self.year[0]:
            return

        pph = sim.diseases['pph']
        maternal_deaths = pph.infected.uids
        receive_pphintv = self.p_pphintv.filter(maternal_deaths)
        pph_deaths_averted = self.eff_pphintv.filter(receive_pphintv)
        sim.people.resolve_deaths(pph_deaths_averted)

        # Add results
        self.results['n_pphintv'][sim.ti] = len(receive_pphintv)
        self.results['n_mothers_saved'][sim.ti] = len(pph_deaths_averted)


def run_sim(n_agents=default_n_agents, rand_seed=0, rng='multi', idx=0, cov=0):
    print(f'Starting sim {idx} with rand_seed={rand_seed} and cov={cov}, rng={rng}')

    pars = {
        'start': 2024,
        'stop': 2030,
        'rand_seed': rand_seed,
        'dt': 0.25,
    }

    age_data = pd.read_csv(paths.src / 'data/ssa_agedist.csv')
    people = ss.People(n_agents, age_data=age_data)

    asfr_data = pd.read_csv(paths.src / 'data/ssa_asfr.csv')
    pregnancy = PPH({'fertility_rate': asfr_data})

    asmr_data = pd.read_csv(paths.src / 'data/ssa_asmr.csv')
    deaths = ss.Deaths({'death_rate': asmr_data, 'rate_units': 1})

    interventions = None
    if cov > 0:
        interventions = PPH_Intv(
            year=[pars['start'], pars['start'] + 1 - 0.1, pars['start'] + 1],
            coverage=[0, 0, cov]
        )

    sim = ss.Sim(
        people=people,
        diseases=[pregnancy, deaths],
        pars=pars,
        interventions=interventions
    )
    sim.run()

    df = pd.DataFrame({
        'year': sim.yearvec,
        'Births': sim.results.births.cumsum(),
        'Maternal Deaths': sim.results['maternal_deaths'].cumsum(),
        'Total Deaths': sim.results['total_deaths'].cumsum(),
        'Coverage': cov,
        'Random Seed': rand_seed,
    })

    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    for rng in rngs:
        for seed in range(n_seeds):
            for cov in covs:
                results.append(run_sim(n_agents=n_agents, rand_seed=seed, rng=rng, cov=cov))

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    if args.plot:
        df = pd.read_csv(args.plot)
    else:
        df = run_scenarios(args.n, args.s)

    plot_scenarios(df, figdir, channels)
