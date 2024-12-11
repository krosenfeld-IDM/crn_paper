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

sc.options(interactive=False)

rngs = ['centralized', 'multi']

debug = True
default_n_agents = [10_000, 1_000][debug]
default_n_rand_seeds = [500, 15][debug]

base_vmmc = 0.4
inc_vmmc_cov_levels = [base_vmmc + 0.1] + [0]  # Must include 0 as that's the baseline
vmmc_eff = 0.6

figdir = os.path.join(paths.src.as_posix(), 'figs', 'VMMC' if not debug else 'VMMC-debug')
sc.path(figdir).mkdir(parents=True, exist_ok=True)


def run_sim(n_agents, idx, cov, rand_seed, rng, pars=None, hiv_pars=None, return_sim=False, analyze=False):
    age_data = pd.read_csv(paths.src / 'data/ssa_agedist.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    def rel_dur(sim, uids):
        """ Increase mean relationship duration with age. """
        dur = 2 * np.ones(len(uids))
        age = sim.people.age[uids]
        dur[age > 20] = 4
        dur[age > 25] = 8
        dur[age > 30] = 12
        return dur

    en_pars = dict(duration=ss.weibull(c=2.5, scale=rel_dur))  # Use `scale` correctly.

    networks = ss.ndict(
        ss.EmbeddingNet(en_pars=en_pars),
        ss.MaternalNet()
    )

    default_hiv_pars = {
        'beta': {'embedding': [0.010, 0.008], 'maternal': [0.2, 0]},
        'init_prev': max(5 / n_agents, 0.02),
        'art_efficacy': 0.8,
        'vmmc_efficacy': 0.6,
    }
    hiv_pars = sc.mergedicts(default_hiv_pars, hiv_pars)
    hiv = HIV(hiv_pars)

    asfr_data = pd.read_csv(paths.src / 'data/ssa_asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=asfr_data, rel_fertility=0.5)

    asmr_data = pd.read_csv(paths.src / 'data/ssa_asmr.csv')
    deaths = ss.Deaths(death_rate=asmr_data)

    interventions = []

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

    interventions += [ART(year=[2004, 2020, 2030], coverage=[0, 0.58, 0.62])]
    interventions += [VMMC(year=[2007, 2020, 2025, 2030], coverage=[0, base_vmmc, cov, 0])]

    sim = ss.Sim(
        people=ppl,
        networks=networks,
        diseases=[hiv],
        demographics=[pregnancy, deaths],
        interventions=interventions,
        pars=pars,
        label=lbl
    )
    sim.initialize()
    sim.run()

    if return_sim:
        return sim

    df = pd.DataFrame({
        'year': sim.t.yearvec,
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
