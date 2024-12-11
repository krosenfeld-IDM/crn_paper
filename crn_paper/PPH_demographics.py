"""
Pregnancy with avertable maternal mortality risk from postpartum hemorrhage (PPH)
"""

import numpy as np
import starsim as ss

__all__ = ['PPH']

class PPH(ss.Pregnancy):
    # Postpartum hemorrhage (PPH) --> maternal and infant death

    def __init__(self, pars=None, par_dists=None, metadata=None, **kwargs):
        super().__init__(pars=pars, metadata=metadata, **kwargs)  # Ensure parent init is correctly called

        self.define_pars(
            inherit=True,
            p_infant_death=ss.bernoulli(p=0.5),  # 50% chance of infant death if mother dies
        )
        self.update_pars(pars, **kwargs)

        # For reporting
        self._possible_maternal_death_uids = None
        self._possible_infant_death_uids = None

    def init_pre(self, sim):
        super().init_pre(sim)
        if 'maternalnet' not in sim.networks:
            raise ValueError('PPH module requires a "maternalnet" network in the simulation.')
        return

    def init_results(self):
        """
        Add tracking of infant and maternal deaths.
        """
        super().init_results()
        npts = self.t.npts
        self.define_results(
            ss.Result(self.name, 'infant_deaths', shape=npts, dtype=int, scale=True, label='Infant Deaths'),
            ss.Result(self.name, 'maternal_deaths', shape=npts, dtype=int, scale=True, label='Maternal Deaths'),
        )
        return

    def update_states(self):
        """ Update states """

        ti = self.t.ti
        deliveries = (self.pregnant & (self.ti_delivery <= ti)).uids  # Track deliveries at this timestep

        self._possible_maternal_death_uids = None
        self._possible_infant_death_uids = None

        super().update_states()

        if np.any(deliveries):
            # Infant deaths due to death of the mother
            mn = self.sim.networks['maternalnet'].to_df()

            maternal_deaths = (self.ti_dead <= ti).uids
            if np.any(maternal_deaths):
                self._possible_maternal_death_uids = maternal_deaths

                # Find infants linked to maternal deaths
                valid_edges = (mn['p1'].isin(maternal_deaths)) & (mn['dur'] >= 0)
                infant_uids_mm = mn.loc[valid_edges, 'p2'].values

                infant_deaths = ss.uids(self.pars.p_infant_death.filter(infant_uids_mm))
                if np.any(infant_deaths):
                    self._possible_infant_death_uids = infant_deaths
                    self.sim.people.request_death(infant_deaths)

        return

    def update_results(self):
        """ Update results tracking """

        super().update_results()

        ti = self.t.ti
        people = self.sim.people

        if self._possible_maternal_death_uids is not None:
            maternal_deaths = people.ti_dead[self._possible_maternal_death_uids] <= ti
            self.results['maternal_deaths'][ti] = np.count_nonzero(maternal_deaths)
        
        if self._possible_infant_death_uids is not None:
            infant_deaths = people.ti_dead[self._possible_infant_death_uids] <= ti
            self.results['infant_deaths'][ti] = np.count_nonzero(infant_deaths)

        return
