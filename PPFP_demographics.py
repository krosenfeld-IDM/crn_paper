"""
Define pregnancy for the MNCH example
"""

import numpy as np
import stisim as ss
import sciris as sc
import pandas as pd
import stisim as ss

__all__ = ['PPFP']


class PPFP(ss.Pregnancy):
    # Postpartum family planning (PPFP)

    def __init__(self, pars=None):
        super().__init__(pars)

        # Add ppfp
        self.ppfp = ss.State('ppfp', bool, False)  # True when pregnant women receive PPFP
        self.ti_ppfp = ss.State('ti_ppfp', int, ss.INT_NAN)  # Time of receiving PPFP

        self.pars = ss.omerge({
            'coverage': 0.15,
            'efficacy': 0.75,  # Rate ratio of fertility reduction for women on PPFP
        }, self.pars)

        self.rng_ppfp = ss.Stream(f'ppfp_{self.name}') # To determine which women receive the intervention
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'new_ppfp', sim.npts, dtype=int) # number of new women starting ppfp
        self.results += ss.Result(self.name, 'n_ppfp', sim.npts, dtype=int) # number of women on ppfp
        return

    def update_states(self, sim):
        """
        Update states
        """
        super().update_states(sim)

        # Add postpartum women to ppfp
        postpartum = ~self.pregnant & (self.ti_postpartum <= sim.ti)
        cov = self.pars.coverage
        uids = self.rng_ppfp.bernoulli_filter(ss.true(postpartum & ~self.ppfp), prob=cov)
        self.ppfp[uids] = True
        self.ti_ppfp[uids] = sim.ti

        return

    def make_pregnancies(self, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate key variables
        ppl = sim.people

        # If incidence of pregnancy is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        if self.pars.inci > 0:
            denom_conds = ppl.female & ppl.active & self.susceptible
            inds_to_choose_from = ss.true(denom_conds)

            inci = np.full(inds_to_choose_from.max()+1, fill_value=self.pars.inci)
            on_ppfp = ss.true(denom_conds & self.ppfp)
            inci[on_ppfp] *= 1-self.pars['efficacy']

            uids = self.rng_conception.bernoulli_filter(inds_to_choose_from, prob=inci[inds_to_choose_from])

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                new_slots = self.rng_choose_slots.integers(uids, low=sim.pars['n_agents'], high=sim.pars['slot_scale']*sim.pars['n_agents'], dtype=int)

                # Grow the arrays and set properties for the unborn agents
                new_uids = sim.people.grow(len(new_slots))

                sim.people.age[new_uids] = -self.pars.dur_pregnancy
                sim.people.slot[new_uids] = new_slots # Before sampling female_dist
                sim.people.female[new_uids] = self.female_dist.sample(uids)

                # Add connections to any vertical transmission layers
                # Placeholder code to be moved / refactored. The maternal network may need to be
                # handled separately to the sexual networks, TBC how to handle this most elegantly
                for lkey, layer in sim.people.networks.items():
                    if layer.vertical:  # What happens if there's more than one vertical layer?
                        durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy + self.pars.dur_postpartum)
                        layer.add_pairs(uids, new_uids, dur=durs)

                # Set prognoses for the pregnancies
                self.set_prognoses(sim, uids) # Could set from_uids to network partners?

        return

    def update_results(self, sim):
        self.results['n_ppfp'][sim.ti] = np.count_nonzero(self.ppfp)
        self.results['new_ppfp'][sim.ti] = np.count_nonzero(self.ti_ppfp == sim.ti)
        return