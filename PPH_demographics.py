"""
Pregnancy with avertable maternal mortality risk from postpartum hemorrhage (PPH)
"""

import numpy as np
import starsim as ss
import scipy.stats as sps

__all__ = ['PPH']

def p_infant_death(self, sim, uids):
    p_death = (1 - 0.94) * np.ones(len(uids)) # Base rate if infant mortality

    mother_died = self.mother_died[uids]
    p_death[mother_died] = 1 - 0.52 # Elevated rate if mother died

    return p_death

class PPH(ss.Pregnancy):
    # Postpartum hemorrhage (PPH)

    def __init__(self, pars=None, par_dists=None, metadata=None, **kwargs):
        super().__init__(pars, **kwargs)

        self.add_states(
            ss.State('mother_died', bool, False),  # Indicator (by child uid) of maternal mortality
        )
        self.p_infant_death = sps.bernoulli(p=p_infant_death) # Probability calculated below
        self.n_infant_deaths = 0 # Number of infant deaths on this timestep
        return

    def initialize(self, sim):
        super().initialize(sim)
        assert 'maternalnet' in sim.networks, 'PPH demographics requires the MaternalNet'
        return

    def init_results(self, sim):
        """
        Add tracking of infant deaths
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'infant_deaths', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'maternal_deaths', sim.npts, dtype=int, scale=True),
        ]
        return

    def update_states(self, sim):
        """
        Update states
        """
        deliveries = ss.true(self.pregnant & (self.ti_delivery <= sim.ti)) # Call before update_states

        super().update_states(sim)

        if np.any(deliveries):
            # Infant deaths due to death of mother
            mn = sim.networks['maternalnet'].to_df()
            infant_uids = mn.loc[mn['p1'].isin(deliveries)]['p2'].values # Find infants, not using find_contacts because that is bidirectional

            maternal_deaths = ss.true(self.ti_dead <= sim.ti)
            self.results['maternal_deaths'][sim.ti] = len(maternal_deaths)
            if np.any(maternal_deaths):
                infant_uids_mm = mn.loc[mn['p1'].isin(maternal_deaths)]['p2'].values # Find infants, not using find_contacts because that is bidirectional
                self.mother_died[infant_uids_mm] = True

            infant_deaths = self.p_infant_death.filter(infant_uids)
            self.results['infant_deaths'][sim.ti] = len(infant_deaths)
            if np.any(infant_deaths):
                sim.people.request_death(infant_deaths)

        return
