"""
Implement a simple HIV module for the VMMC analysis
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['HIV', 'ART', 'VMMC']

class HIV(ss.Infection):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(**kwargs)
        self.default_pars(
            beta=1.0,  # Placeholder value
            art_efficacy=0.96,
            VMMC_efficacy=0.6,
            init_prev=ss.bernoulli(p=0.05),
            survival_without_art=ss.weibull(c=2, scale=13),
        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.add_states(
            ss.BoolArr('on_art', label='On ART'),
            ss.BoolArr('circumcised', label='Circumcised'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'),  # Time of HIV-caused death
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
        )

    def update_pre(self):
        people = self.sim.people
        self.rel_trans[self.infected & self.on_art] = 1 - self.pars['art_efficacy']
        self.rel_sus[people.male & self.circumcised] = 1 - self.pars['VMMC_efficacy']

        hiv_deaths = (self.ti_dead == self.sim.ti).uids
        people.request_death(hiv_deaths)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_deaths', dtype=int, label='Deaths'),
            ss.Result('art_coverage', dtype=float, label='ART Coverage'),
            ss.Result('vmmc_coverage', dtype=float, label='VMMC Coverage'),
            ss.Result('vmmc_coverage_15_49', dtype=float, label='VMMC Coverage 15-49'),
            ss.Result('prevalence_15_49', dtype=float, label='Prevalence 15-49'),
        )

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        people = self.sim.people
        inds = (people.age >= 15) & (people.age < 50)
        self.results.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        n_inf = np.count_nonzero(self.infected)
        self.results.art_coverage[ti] = np.count_nonzero(self.on_art) / n_inf if n_inf > 0 else 0
        self.results.vmmc_coverage[ti] = np.count_nonzero(self.circumcised) / np.count_nonzero(people.male)
        self.results.vmmc_coverage_15_49[ti] = np.count_nonzero(self.circumcised[inds]) / np.count_nonzero(people.male[inds])
        self.results.prevalence_15_49[ti] = np.count_nonzero(self.infected[inds]) / np.count_nonzero(people.alive[inds])

    def set_prognoses(self, uids, source_uids=None):
        super().set_prognoses(uids, source_uids)
        prog = self.pars.survival_without_art.rvs(uids)
        self.ti_dead[uids] = self.sim.ti + np.round(prog / self.sim.dt).astype(int)
