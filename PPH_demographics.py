"""
Pregnancy with avertable maternal mortality risk from postpartum hemorrhage (PPH)
"""

import numpy as np
import starsim as ss
import scipy.stats as sps

__all__ = ['PPH']

class PPH(ss.Pregnancy):
    # Postpartum hemorrhage (PPH) --> maternal and infant death

    def __init__(self, pars=None, par_dists=None, metadata=None, **kwargs):
        super().__init__(pars, **kwargs)

        self.p_infant_death = ss.bernoulli(p=0.5) # 50% chance of infant death if mother dies
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
                # Find infants, not using find_contacts because that is bidirectional
                # And only keep edges where dur >= 0, others are inactive
                infant_uids_mm = mn.loc[(mn['p1'].isin(maternal_deaths)) & (mn['dur'] >= 0)]['p2'].values

                infant_deaths = self.p_infant_death.filter(infant_uids_mm)
                self.results['infant_deaths'][sim.ti] = len(infant_deaths)
                if np.any(infant_deaths):
                    sim.people.request_death(infant_deaths)

        return
