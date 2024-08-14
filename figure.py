import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

vx_year = 2030
n_seeds = 40

pal = 'Set1'

pars = dict(
    n_agents = 1000,
    birth_rate = 20,
    death_rate = 15,
    networks = dict(
        type = 'random',
        n_contacts = 4
    ),
    diseases = dict(
        type = 'sir',
        dur_inf = 10,
        beta = 0.05,
    )
)

# Create the product - a vaccine
my_vaccine = ss.sir_vaccine(efficacy=0.5)

# Create the intervention
my_intervention = ss.routine_vx(
    start_year=vx_year,    # Begin vaccination in 2015
    prob=0.2,           # 20% coverage
    product=my_vaccine   # Use the MyVaccine product
)

for common_seeds in [False, True]:

    # Now create two sims: a baseline sim and one with the intervention
    sim_base = ss.Sim(pars=pars)
    msim_base = ss.MultiSim(base_sim=sim_base, reseed=False, iterpars={'rand_seed': list(range(n_seeds))})
    msim_base.run()

    cuminf_base = pd.DataFrame({i: s.results.sir.cum_infections for i,s in enumerate(msim_base.sims)}, index=pd.Index(msim_base.sims[0].results.yearvec, name='Year'))
    cuminf_base = cuminf_base.reset_index().melt(id_vars='Year', var_name='Replicate', value_name='Cumulative Deaths')
    cuminf_base['Scenario'] = 'Base'


    sim_intv = ss.Sim(pars=pars, interventions=my_intervention)
    if common_seeds:
        seeds = list(range(n_seeds))
    else:
        seeds = list(range(n_seeds, 2*n_seeds))
    msim_intv = ss.MultiSim(base_sim=sim_intv, reseed=False, iterpars={'rand_seed': seeds})
    msim_intv.run()

    cuminf_intv = pd.DataFrame({i: s.results.sir.cum_infections for i,s in enumerate(msim_intv.sims)}, index=pd.Index(msim_intv.sims[0].results.yearvec, name='Year'))
    cuminf_intv = cuminf_intv.reset_index().melt(id_vars='Year', var_name='Replicate', value_name='Cumulative Deaths')
    cuminf_intv['Scenario'] = 'Vaccine'


    cuminf = pd.concat([cuminf_base, cuminf_intv])

    for si, scens in enumerate([['Base'], ['Base', 'Vaccine']]):

        fig, ax = plt.subplots(1,2, width_ratios=[3, 1], sharey=True, figsize=(8, 5.5))

        sns.lineplot(cuminf, x='Year', y='Cumulative Deaths', hue='Scenario', hue_order=scens, units='Replicate', estimator=None, lw=1, palette=pal, ax=ax[0])
        if 'Vaccine' in scens:
            ax[0].axvline(x=vx_year, color='black', ls='--', lw=2, alpha=0.8)

        last_year = cuminf['Year'].max()
        cuminf_tf = cuminf.loc[cuminf['Year']==last_year]

        sns.kdeplot(data=cuminf_tf, y='Cumulative Deaths', hue='Scenario', hue_order=scens, legend=False, bw_adjust=0.8, palette=pal, ax=ax[1])
        sns.rugplot(data=cuminf_tf, y='Cumulative Deaths', hue='Scenario', hue_order=scens, legend=False, height=0.05, lw=1, palette=pal, ax=ax[1])
        ax[0].set_ylim([0, 1.1*cuminf['Cumulative Deaths'].max()])

        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[1].set_xticks([])

        fig.tight_layout()

        plt.savefig(f'fig_{common_seeds}_{si}.png', dpi=240)