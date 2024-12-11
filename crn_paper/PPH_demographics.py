"""
Pregnancy with avertable maternal mortality risk from postpartum hemorrhage (PPH)
"""

import numpy as np
import starsim as ss

__all__ = ['PPH']

class PPH(ss.Pregnancy):
    """
    Postpartum hemorrhage (PPH) - maternal and infant death.
    """

    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__(pars=pars, metadata=metadata, **kwargs)

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
        ss.check_requires(sim, ['maternalnet'])  # Validate required network exists
        return

    def init_results(self):
        """
        Add tracking of infant deaths and maternal deaths
        """
        super().init_results()
        npts = self.sim.t.npts
        self.define_results(
            ss.Result(self.name, 'infant_deaths', npts, dtype=int, scale=True, label='Infant Deaths'),
            ss.Result(self.name, 'maternal_deaths', npts, dtype=int, scale=True, label='Maternal Deaths'),
        )
        return

    def update_states(self):
        """ Update states """

        ti = self.sim.ti
        deliveries = (self.pregnant & (self.ti_delivery <= ti)).uids  # Get IDs of agents delivering this timestep

        # Clear previous tracking
        self._possible_maternal_death_uids = None
        self._possible_infant_death_uids = None

        # Update states
        super().update_states()

        if np.any(deliveries):
            # Infant deaths due to maternal death
            mn = self.sim.networks['maternalnet']  # Access network
            maternal_deaths = (self.sim.people.ti_dead[self.pregnant.uids] <= ti)
            if np.any(maternal_deaths):
                self._possible_maternal_death_uids = ss.uids(maternal_deaths)
                infants = mn.find_contacts(self._possible_maternal_death_uids)
                infant_uids_mm = infants['p2'][infants['dur'] >= 0]

                infant_deaths = self.pars.p_infant_death.filter(infant_uids_mm)
                if np.any(infant_deaths):
                    self._possible_infant_death_uids = infant_deaths
                    self.sim.people.request_death(infant_deaths)

        return

    def update_results(self):
        """
        Update tracked results at the end of each timestep.
        """
        super().update_results()

        ti = self.sim.ti
        people = self.sim.people

        # Track maternal deaths
        if self._possible_maternal_death_uids is not None:
            maternal_deaths = people.ti_dead[self._possible_maternal_death_uids] <= ti
            self.results['maternal_deaths'][ti] = np.count_nonzero(maternal_deaths)

        # Track infant deaths
        if self._possible_infant_death_uids is not None:
            infant_deaths = people.ti_dead[self._possible_infant_death_uids] <= ti
            self.results['infant_deaths'][ti] = np.count_nonzero(infant_deaths)

        return
