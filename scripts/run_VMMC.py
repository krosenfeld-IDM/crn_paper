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
from crn_paper.analyzers import GraphAnalyzer
from crn_paper.hiv import HIV, ART, VMMC
from crn_paper import paths
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

sc.options(interactive=False)  # Assume not running interactively

rngs = ['centralized', 'multi']

debug = True
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [500, 15][debug]

base_vmmc = 0.4
inc_vmmc_cov_levels = [base_vmmc + 0.1, 0]  # Must include 0 as that's the baseline
vmmc_eff = 0.6

figdir = os.path.join(paths.src.as_posix(), 'figs', 'VMMC' if not debug else 'VMMC-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n_agents, idx, cov, rand_seed, rng, pars=None, hiv_pars=None, return_sim=False, analyze=False):
    age_data = pd.read_csv(paths.src / 'data/ssa_agedist.csv')
    ppl = ss.People(n_agents=n_agents, age_data=age_data)

    def rel_dur(self, sim, uids):
        dur = 2 * np.ones(len(uids))
        dur[sim.people.age[uids] > 20] = 4
        dur[sim.people.age[uids] > 25] = 8
        dur[sim.people.age[uids] > 30] = 12
        return dur

    en_pars = dict(duration=ss.weibull(c=2.5, scale=rel_dur))  # Correct Weibull args usage
    networks = ss.ndict(ss.EmbeddingNet(en_pars), ss.MaternalNet())

    default_hiv_pars = {
        'beta': {'embedding': [0.010, 0.008], 'maternal': [0.2, 0]},
        'init_prev': max(5 / n_agents, 0.02),
        'art_efficacy': 0.8,
        'VMMC_efficacy': vmmc_eff,
    }
    hiv_pars = sc.mergedicts(default_hiv_pars, hiv_pars)
    hiv = HIV(hiv_pars)

    asfr_data = pd.read_csv(paths.src / 'data/ssa_asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=asfr_data, rel_fertility=0.5)

    asmr_data = pd.read_csv(paths.src / 'data/ssa_asmr.csv')
    deaths = ss.Deaths(death_rate=asmr_data)

    interventions = [
        ART(year=[2004, 2020, 2030], coverage=[0, 0.58, 0.62]),
        VMMC(year=[2007, 2020, 2025, 2030], coverage=[0, base_vmmc, cov, 0])
    ]

    default_pars = {
        'start': 1980,
        'end': 2070,
        'dt': 1 / 12,
        'rand_seed': rand_seed,
        'verbose': 0,
        'slot_scale': 10,
        'analyzers': [GraphAnalyzer()] if analyze else [],
    }
    pars = sc.mergedicts(default_pars, pars)

    lbl = f'Sim {idx}: agents={n_agents}, cov={cov}, seed={rand_seed}, rng={rng}'
    print('Starting', lbl)

    sim = ss.Sim(people=ppl, networks=networks, diseases=[hiv], demographics=[pregnancy, deaths],
                 interventions=interventions, pars=pars, label=lbl)
    sim.initialize()
    sim.run()

    if return_sim:
        return sim

    results = sim.results.hiv
    df = pd.DataFrame({
        'year': sim.yearvec,
        'Births': sim.results.pregnancy.births.cumsum(),
        'Deaths': results.new_deaths.cumsum(),
        'Infections': results.cum_infections,
        'Prevalence': results.prevalence,
        'Prevalence 15-49': results.prevalence_15_49,
        'ART Coverage': results.art_coverage,
        'VMMC Coverage': results.vmmc_coverage,
        'Population': sim.results.n_alive,
    })
    df['cov'] = cov
    df['rand_seed'] = rand_seed
    df['rng'] = rng
    df['n_agents'] = n_agents

    print(f'Finished sim {idx}')
    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    for rng in rngs:
        ss.options(_centralized=(rng == 'centralized'))
        cfgs = [{'cov': cov, 'rand_seed': rs, 'rng': rng, 'idx': len(results)}
                for rs in range(n_seeds) for cov in inc_vmmc_cov_levels]
        results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'results.csv'))
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', action='store_true', help='Plot results from cached CSV')
    parser.add_argument('-n', type=int, default=default_n_agents, help='Number of agents')
    parser.add_argument('-s', type=int, default=default_n_rand_seeds, help='Number of seeds')
    args = parser.parse_args()

    if args.plot:
        results = pd.read_csv(os.path.join(figdir, 'results.csv'))
    else:
        results = run_scenarios(n_agents=args.n, n_seeds=args.s)

    plot_scenarios(results, figdir)
    print('Done')
