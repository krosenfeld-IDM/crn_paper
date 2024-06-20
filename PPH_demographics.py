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
        super().__init__()

        self.default_pars(
            inherit = True,
            p_infant_death = ss.bernoulli(p=0.5), # 50% chance of infant death if mother dies
        )
        self.update_pars(pars, **kwargs)

        # For reporting
        self._possible_maternal_death_uids = None
        self._possible_infant_death_uids = None

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        assert 'maternalnet' in sim.networks, 'PPH demographics requires the MaternalNet'
        return

    def init_results(self):
        """
        Add tracking of infant deaths
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'infant_deaths', npts, dtype=int, scale=True),
            ss.Result(self.name, 'maternal_deaths', npts, dtype=int, scale=True),
        ]
        return

    def update_states(self):
        """ Update states """

        ti = self.sim.ti
        deliveries = (self.pregnant & (self.ti_delivery <= ti)).uids # Call before update_states

        self._possible_maternal_death_uids = None
        self._possible_infant_death_uids = None

        super().update_states()

        if np.any(deliveries):
            # Infant deaths due to death of mother
            mn = self.sim.networks['maternalnet'].to_df()

            maternal_deaths = (self.ti_dead <= ti).uids
            #####self.results['maternal_deaths'][sim.ti] = len(maternal_deaths)
            if np.any(maternal_deaths):
                self._possible_maternal_death_uids = maternal_deaths
                # Find infants, not using find_contacts because that is bidirectional
                # And only keep edges where dur >= 0, others are inactive
                infant_uids_mm = mn.loc[(mn['p1'].isin(maternal_deaths)) & (mn['dur'] >= 0)]['p2'].values

                infant_deaths = ss.uids(self.pars.p_infant_death.filter(infant_uids_mm))
                ###self.results['infant_deaths'][sim.ti] = len(infant_deaths)
                if np.any(infant_deaths):
                    self._possible_infant_death_uids = maternal_deaths
                    self.sim.people.request_death(infant_deaths)

        return

    def update_results(self):
        super().update_results()

        ti = self.sim.ti
        people = self.sim.people

        # Results must be tracked here for intervention impact to be properly recorded
        if self._possible_maternal_death_uids is not None:
            maternal_deaths = people.ti_dead[self._possible_maternal_death_uids] <= ti
            self.results['maternal_deaths'][ti] = np.count_nonzero(maternal_deaths)
        
        if self._possible_infant_death_uids is not None:
            infant_deaths = people.ti_dead[self._possible_infant_death_uids] <= ti
            self.results['infant_deaths'][ti] = np.count_nonzero(infant_deaths)
