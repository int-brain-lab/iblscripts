"""
Import examples so as to run them as test.
Printing is used to request the import.
Examples have to be able to run upon adding.
"""
# Author: Olivier, Gaelle
import unittest

import matplotlib.pyplot as plt


class TestExamplesBehaviour(unittest.TestCase):

    def test_one(self):
        import examples.one.behavior.number_mice_inproject
        import examples.one.behavior.print_water_administrations
        import examples.one.behavior.water_administrations_add_new
        import examples.one.behavior.water_administrations_weekend
        import examples.one.behavior.plot_microphone_spectrogram
        import examples.one.behavior.plot_weight_curve

    def tearDown(self):
        plt.close()


class TestExamplesHistology(unittest.TestCase):

    def test_alyx_interactions(self):
        pass
        # TODO get track files somehwere
        # import examples.one.histology.register_lasagna_tracks_alyx
