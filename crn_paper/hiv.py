"""
Implement a simple HIV module for the VMMC analysis
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['HIV', 'ART']

class HIV(ss.Infection):

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            beta=1.0,  # Placeholder value to be completed below
            art_efficacy=0.96,
            VMMC_efficacy=0.6,
            init_prev=ss.bernoulli(p=0.05),
            survival_without_art=ss.weibull(c=2, scale=13),
        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.define_states(
            ss.BoolArr('on_art', label='On ART'),
            ss.BoolArr('circumcised', label='Circumcised'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'),  # Time of HIV-caused death
        )
        return

    def step_state(self):
        people = self.sim.people
        self.rel_trans[self.infected & self.on_art] = 1 - self.pars.art_efficacy
        self.rel_sus[people.male & self.circumcised] = 1 - self.pars.VMMC_efficacy

        hiv_deaths = (self.ti_dead == self.sim.ti).uids
        people.request_death(hiv_deaths)
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('new_deaths', self.sim.t.npts, dtype=int, label='Deaths'),
            ss.Result('art_coverage', self.sim.t.npts, dtype=float, label='ART Coverage'),
            ss.Result('vmmc_coverage', self.sim.t.npts, dtype=float, label='VMMC Coverage'),
            ss.Result('vmmc_coverage_15_49', self.sim.t.npts, dtype=float, label='VMMC Coverage 15-49'),
            ss.Result('prevalence_15_49', self.sim.t.npts, dtype=float, label='Prevalence 15-49'),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        self.results['new_deaths'][ti] = np.count_nonzero(self.ti_dead == ti)
        n_inf = np.count_nonzero(self.infected)
        self.results['art_coverage'][ti] = np.count_nonzero(self.on_art) / n_inf if n_inf > 0 else 0
        self.results['vmmc_coverage'][ti] = np.count_nonzero(self.circumcised) / np.count_nonzero(self.sim.people.male)
        inds = (self.sim.people.age >= 15) & (self.sim.people.age < 50)
        self.results['vmmc_coverage_15_49'][ti] = (
            np.count_nonzero(self.circumcised[inds]) / np.count_nonzero(self.sim.people.male[inds])
        )
        self.results['prevalence_15_49'][ti] = (
            np.count_nonzero(self.infected[inds]) / np.count_nonzero(self.sim.people.alive[inds])
        )
        return

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.sim.ti

        prog = self.pars.survival_without_art.rvs(len(uids))
        self.ti_dead[uids] = self.sim.ti + np.round(prog / self.sim.t.dt).astype(int)  # Survival without treatment
        return

    def set_congenital(self, uids, sources):
        self.set_prognoses(uids, sources)


class ART(ss.Intervention):
    def __init__(self, year, coverage, pars=None, **kwargs):
        self.requires = HIV
        self.year = sc.toarray(year)
        self.coverage = sc.toarray(coverage)
        super().__init__()
        self.define_pars()
        self.update_pars(pars=pars, **kwargs)

        prob_art_at_init = lambda self, sim, uids: np.interp(sim.t.now('year'), self.year, self.coverage)
        self.prob_art_at_infection = ss.bernoulli(p=prob_art_at_init)
        self.prob_art_post_infection = ss.bernoulli(p=0)  # Set in init_pre
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Importance sampling
        cvec = np.interp(sim.t.yearvec, self.year, self.coverage)
        pvec = np.zeros_like(cvec)
        np.divide(np.diff(cvec), 1 - cvec[:-1], where=cvec[:-1] < 1, out=pvec[1:])
        pvec = np.clip(pvec, a_min=0, a_max=1)

        prob_art_post_init = lambda self, sim, uids, pvec=pvec: pvec[sim.ti]
        self.prob_art_post_infection.set(p=prob_art_post_init)

        self.define_results(
            ss.Result('n_art', sim.t.npts, dtype=int)
        )
        self.initialized = True
        return

    def step(self):
        if self.sim.t.now('year') < self.year[0]:
            return

        hiv = self.sim.people.hiv
        infected = hiv.infected.uids
        ti_delay = 1 + np.round(2 / self.sim.t.dt).astype(int)  # About 2 years to ART initiation
        recently_infected = infected[hiv.ti_infected[infected] == self.sim.ti - ti_delay]
        notrecent_noart = infected[(hiv.ti_infected[infected] < self.sim.ti - ti_delay) & (~hiv.on_art[infected])]

        n_added = 0
        if len(recently_infected):
            inds = self.prob_art_at_infection.filter(recently_infected)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = self.sim.ti
            hiv.ti_dead[inds] = np.nan
            n_added += len(inds)

        if len(notrecent_noart):
            inds = self.prob_art_post_infection.filter(notrecent_noart)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = self.sim.ti
            hiv.ti_dead[inds] = np.nan
            n_added += len(inds)

        # Add results
        self.results['n_art'][self.sim.ti] = np.count_nonzero(hiv.on_art)
        return n_added
