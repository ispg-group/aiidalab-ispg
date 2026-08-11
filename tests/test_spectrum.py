"""Unit tests for the Spectrum class (spectrum.py).

Run with:  pytest test_spectrum.py -v
"""

import sys

import numpy as np
import pytest
from aiidalab_ispg.app.spectrum import BroadeningKernel, EnergyUnit, Spectrum
from inline_snapshot import snapshot

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def small_sample_grid(monkeypatch):
    """Use a small, reasonable number of x-axis points for every test.

    500 points (the production default) makes numerical regression
    snapshots huge and hard to review. 10 points is plenty to catch
    regressions while keeping snapshots readable.
    """
    monkeypatch.setattr(Spectrum, "N_SAMPLE_POINTS", 10)


def make_transitions(energies, osc_strengths):
    """Build a list-of-dicts 'transitions' structure like the real code expects."""
    return [{"energy": e, "osc_strength": f} for e, f in zip(energies, osc_strengths)]


@pytest.fixture
def single_transition():
    """A single excitation at 5.0 eV with osc. strength 0.5."""
    transitions = make_transitions([5.0], [0.5])
    return Spectrum(transitions, nsample=1)


@pytest.fixture
def two_transitions():
    """Two excitations, used for multi-peak / stick-spectrum tests."""
    transitions = make_transitions([4.0, 6.0], [0.3, 0.9])
    return Spectrum(transitions, nsample=1)


class TestInit:
    def test_stores_energies_and_osc_strengths(self):
        transitions = make_transitions([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        spec = Spectrum(transitions, nsample=3)

        np.testing.assert_allclose(spec.excitation_energies, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(spec.osc_strengths, [0.1, 0.2, 0.3])
        assert spec.nsample == 3

    def test_arrays_are_float_dtype(self):
        transitions = make_transitions([1, 2], [1, 0])
        spec = Spectrum(transitions, nsample=1)

        assert spec.excitation_energies.dtype == float
        assert spec.osc_strengths.dtype == float

    def test_empty_transitions_gives_empty_arrays(self):
        spec = Spectrum([], nsample=0)

        assert spec.excitation_energies.size == 0
        assert spec.osc_strengths.size == 0


class TestGetEnergyRangeEv:
    def test_typical_range_adds_padding(self):
        energies = np.array([3.0, 5.0])
        x_min, x_max = Spectrum.get_energy_range_ev(energies)

        assert x_max == pytest.approx(5.0 + 1.5)
        # x_min = 3.0 - 1.5 = 1.5, which is >= 1.0, so normal padding applies
        assert x_min == pytest.approx(3.0 - 1.5)

    def test_low_energy_uses_halved_padding(self):
        # en_min - 1.5 would be < 1.0, so the special-case branch kicks in:
        # x_min = en_min - en_min / 2.0
        energies = np.array([1.2, 4.0])
        x_min, x_max = Spectrum.get_energy_range_ev(energies)

        assert x_min == pytest.approx(1.2 - 1.2 / 2.0)
        assert x_max == pytest.approx(4.0 + 1.5)

    def test_single_energy(self):
        # 2.0 - 1.5 = 0.5, which is < 1.0, so the halved-padding branch applies
        energies = np.array([2.0])
        x_min, x_max = Spectrum.get_energy_range_ev(energies)

        assert x_min == pytest.approx(2.0 - 2.0 / 2.0)
        assert x_max == pytest.approx(2.0 + 1.5)

    def test_zero_energy_raises_assertion(self):
        energies = np.array([0.0, 3.0])
        with pytest.raises(AssertionError):
            Spectrum.get_energy_range_ev(energies)

    def test_negative_energy_raises_assertion(self):
        energies = np.array([-1.0, 3.0])
        with pytest.raises(AssertionError):
            Spectrum.get_energy_range_ev(energies)

    def test_x_min_less_than_x_max(self):
        energies = np.array([0.5, 0.9, 2.0])
        x_min, x_max = Spectrum.get_energy_range_ev(energies)
        assert x_min < x_max


class TestGetEnergyUnitFactor:
    def test_ev_factor_is_one(self):
        assert Spectrum.get_energy_unit_factor(EnergyUnit.EV) == 1.0

    def test_nm_factor(self):
        assert Spectrum.get_energy_unit_factor(EnergyUnit.NM) == pytest.approx(1239.8)

    def test_cm_factor(self):
        assert Spectrum.get_energy_unit_factor(EnergyUnit.CM) == pytest.approx(
            8065.547937
        )

    def test_unknown_unit_raises_keyerror(self):
        with pytest.raises(KeyError):
            Spectrum.get_energy_unit_factor("not_a_real_unit")


class TestBroadeningKernels:
    def test_gauss_peak_is_near_excitation_energy(self, single_transition):
        x = np.linspace(2.0, 8.0, 1000)
        y = np.zeros_like(x)
        single_transition._calc_gauss_spectrum(x, y, sigma=0.3)

        peak_x = x[np.argmax(y)]
        assert peak_x == pytest.approx(5.0, abs=0.05)
        assert np.all(y >= 0)

    def test_lorentzian_peak_is_near_excitation_energy(self, single_transition):
        x = np.linspace(2.0, 8.0, 1000)
        y = np.zeros_like(x)
        single_transition._calc_lorentzian_spectrum(x, y, tau=0.3)

        peak_x = x[np.argmax(y)]
        assert peak_x == pytest.approx(5.0, abs=0.05)
        assert np.all(y >= 0)

    def test_gauss_narrower_width_gives_taller_peak(self, single_transition):
        x = np.linspace(2.0, 8.0, 1000)

        y_narrow = np.zeros_like(x)
        single_transition._calc_gauss_spectrum(x, y_narrow, sigma=0.1)

        y_wide = np.zeros_like(x)
        single_transition._calc_gauss_spectrum(x, y_wide, sigma=1.0)

        assert y_narrow.max() > y_wide.max()

    def test_lorentzian_narrower_width_gives_taller_peak(self, single_transition):
        x = np.linspace(2.0, 8.0, 1000)

        y_narrow = np.zeros_like(x)
        single_transition._calc_lorentzian_spectrum(x, y_narrow, tau=0.1)

        y_wide = np.zeros_like(x)
        single_transition._calc_lorentzian_spectrum(x, y_wide, tau=1.0)

        assert y_narrow.max() > y_wide.max()

    def test_broadening_accumulates_onto_existing_y(self, single_transition):
        """y is modified in place / additively, not overwritten."""
        x = np.linspace(2.0, 8.0, 100)
        y = np.full_like(x, 10.0)
        single_transition._calc_gauss_spectrum(x, y, sigma=0.3)

        assert np.all(y >= 10.0)

    def test_two_transitions_gives_two_local_maxima(self, two_transitions):
        x = np.linspace(0.0, 10.0, 2000)
        y = np.zeros_like(x)
        two_transitions._calc_gauss_spectrum(x, y, sigma=0.1)

        # crude local-maxima detector
        is_peak = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        peak_xs = x[1:-1][is_peak]

        assert peak_xs.size >= 2
        assert np.any(np.isclose(peak_xs, 4.0, atol=0.05))
        assert np.any(np.isclose(peak_xs, 6.0, atol=0.05))


class TestGetSpectrum:
    def test_returns_four_arrays_same_length_xy(self, single_transition):
        x, y, x_stick, y_stick = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )

        assert len(x) == len(y) == Spectrum.N_SAMPLE_POINTS
        assert len(x_stick) == len(y_stick) == 1

    def test_spectrum_scales_with_nsamples(self, two_transitions):
        _, y1, _, _ = two_transitions.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )
        two_transitions.nsample = 2
        _, y2, _, _ = two_transitions.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )
        np.testing.assert_allclose(2 * y2, y1)

    def test_invalid_kernel_raises_value_error(self, single_transition):
        with pytest.raises(ValueError):
            single_transition.get_spectrum(
                kernel="not_a_kernel", width=0.3, x_unit=EnergyUnit.EV
            )

    def test_ev_unit_x_matches_default_range(self, single_transition):
        x, _, x_stick, _ = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )
        x_min, x_max = Spectrum.get_energy_range_ev(
            single_transition.excitation_energies
        )
        assert x.min() == pytest.approx(x_min)
        assert x.max() == pytest.approx(x_max)
        assert x_stick[0] == pytest.approx(5.0)

    def test_cm_unit_scales_linearly_from_ev(self, single_transition):
        x_ev, _, stick_ev, _ = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )
        x_cm, _, stick_cm, _ = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.CM
        )

        factor = Spectrum.get_energy_unit_factor(EnergyUnit.CM)
        np.testing.assert_allclose(x_cm, x_ev * factor)
        np.testing.assert_allclose(stick_cm, stick_ev * factor)

    def test_nm_unit_is_inverse_relationship(self, single_transition):
        x_nm, _, stick_nm, _ = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.NM
        )
        nm_factor = Spectrum.get_energy_unit_factor(EnergyUnit.NM)

        # stick position in nm should be factor / (energy in eV)
        expected_stick_nm = nm_factor / 5.0
        assert stick_nm[0] == pytest.approx(expected_stick_nm)

        # nm x-axis should be monotonically decreasing since it's 1/x of an
        # increasing eV axis
        assert np.all(np.diff(x_nm) < 0)

    def test_custom_x_min_x_max_are_respected(self, single_transition):
        x, _, _, _ = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS,
            width=0.3,
            x_unit=EnergyUnit.EV,
            x_min=1.0,
            x_max=10.0,
        )
        assert x.min() == pytest.approx(1.0)
        assert x.max() == pytest.approx(10.0)

    def test_stick_spectrum_normalized_to_max_of_broadened_spectrum(
        self, two_transitions
    ):
        _x, y, _x_stick, y_stick = two_transitions.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.1, x_unit=EnergyUnit.EV
        )
        # y_stick = osc_strengths * max(y) / max(osc_strengths)
        # so the stick belonging to the strongest oscillator should equal max(y)
        assert y_stick.max() == pytest.approx(y.max())

    def test_spectrum_is_non_negative(self, two_transitions):
        _, y, _, _ = two_transitions.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )
        assert np.all(y >= 0)


class TestConvertToNanometers:
    def test_conversion_formula(self, single_transition):
        x_ev = np.array([2.0, 4.0, 8.0])
        y_in = np.array([1.0, 2.0, 3.0])

        x_nm, y_out = single_transition._convert_to_nanometers(x_ev, y_in)

        nm_factor = Spectrum.get_energy_unit_factor(EnergyUnit.NM)
        np.testing.assert_allclose(x_nm, nm_factor / x_ev)
        # y is passed through unchanged by this helper
        np.testing.assert_allclose(y_out, y_in)


# ---------------------------------------------------------------------------
# Numerical regression tests (inline-snapshot)
# ---------------------------------------------------------------------------
#
# These pin down the exact numerical output of the two broadening kernels
# against a small, fixed x-grid (N_SAMPLE_POINTS == 10, see the
# `small_sample_grid` fixture above) so that any unintended change to the
# math (e.g. the COEFF constant, normalization factors, or the broadening
# formulas themselves) is caught.
#
# To (re)generate the snapshot values after an intentional change, run:
#   pytest test_spectrum.py -k TestNumericalRegression --inline-snapshot=fix
@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="requires python3.12 to get the same numerical values",
)
class TestNumericalRegression:
    @pytest.fixture
    def x_grid(self, single_transition):
        x_min, x_max = Spectrum.get_energy_range_ev(
            single_transition.excitation_energies
        )
        return np.linspace(x_min, x_max, num=Spectrum.N_SAMPLE_POINTS)

    def test_gauss_spectrum_single_transition(self, single_transition, x_grid):
        y = np.zeros_like(x_grid)
        single_transition._calc_gauss_spectrum(x_grid, y, sigma=0.3)

        assert y.tolist() == snapshot(
            [
                2.7197354939980887e-22,
                3.794816473791066e-20,
                1.5405963279145507e-18,
                1.819788484207561e-17,
                6.254418525839005e-17,
                6.254418525839014e-17,
                1.819788484207561e-17,
                1.5405963279145507e-18,
                3.794816473791086e-20,
                2.7197354939980887e-22,
            ]
        )

    def test_lorentzian_spectrum_single_transition(self, single_transition, x_grid):
        y = np.zeros_like(x_grid)
        single_transition._calc_lorentzian_spectrum(x_grid, y, tau=0.3)

        assert y.tolist() == snapshot(
            [
                1.1530718670422777e-18,
                1.8938528296070824e-18,
                3.654893818005765e-18,
                9.615984652673672e-18,
                5.211757427775058e-17,
                5.211757427775089e-17,
                9.615984652673672e-18,
                3.654893818005765e-18,
                1.8938528296070835e-18,
                1.1530718670422777e-18,
            ]
        )

    def test_gauss_spectrum_two_transitions(self, two_transitions, x_grid):
        x_min, x_max = Spectrum.get_energy_range_ev(two_transitions.excitation_energies)
        x = np.linspace(x_min, x_max, num=Spectrum.N_SAMPLE_POINTS)
        y = np.zeros_like(x)
        two_transitions._calc_gauss_spectrum(x, y, sigma=0.5)

        assert y.tolist() == snapshot(
            [
                2.918670225698297e-19,
                4.413168536498068e-18,
                1.9416412439076024e-17,
                2.4948110487800518e-17,
                1.2265913242499476e-17,
                2.877281367660313e-17,
                7.45912990467298e-17,
                5.824691594245453e-17,
                1.3239499412967427e-17,
                8.75601062896827e-19,
            ]
        )

    def test_lorentzian_spectrum_two_transitions(self, two_transitions, x_grid):
        x_min, x_max = Spectrum.get_energy_range_ev(two_transitions.excitation_energies)
        x = np.linspace(x_min, x_max, num=Spectrum.N_SAMPLE_POINTS)
        y = np.zeros_like(x)
        two_transitions._calc_lorentzian_spectrum(x, y, tau=0.5)

        assert y.tolist() == snapshot(
            [
                1.7715891714065612e-18,
                3.64556999017523e-18,
                1.3622430718031981e-17,
                3.132161312125551e-17,
                9.123311380683057e-18,
                1.500404924468646e-17,
                8.784181724983716e-17,
                3.7233769468592894e-17,
                8.536087091714532e-18,
                3.612201297543248e-18,
            ]
        )

    def test_get_spectrum_gauss_full_pipeline(self, single_transition):
        """Regression test for the full get_spectrum() output (broadened y,
        x-axis, and stick spectrum) in eV, using the small 10-point grid."""
        x, y, x_stick, y_stick = single_transition.get_spectrum(
            kernel=BroadeningKernel.GAUSS, width=0.3, x_unit=EnergyUnit.EV
        )

        assert x.tolist() == snapshot(
            [
                3.5,
                3.8333333333333335,
                4.166666666666667,
                4.5,
                4.833333333333333,
                5.166666666666666,
                5.5,
                5.833333333333333,
                6.166666666666666,
                6.5,
            ]
        )
        assert y.tolist() == snapshot(
            [
                2.7197354939980887e-22,
                3.794816473791066e-20,
                1.5405963279145507e-18,
                1.819788484207561e-17,
                6.254418525839005e-17,
                6.254418525839014e-17,
                1.819788484207561e-17,
                1.5405963279145507e-18,
                3.794816473791086e-20,
                2.7197354939980887e-22,
            ]
        )
        assert x_stick.tolist() == snapshot([5.0])
        assert y_stick.tolist() == snapshot([6.254418525839014e-17])

    def test_get_spectrum_lorentz_full_pipeline(self, single_transition):
        """Regression test for the full get_spectrum() output (broadened y,
        x-axis, and stick spectrum) in eV, using the small 10-point grid."""
        x, y, x_stick, y_stick = single_transition.get_spectrum(
            kernel=BroadeningKernel.LORENTZ, width=0.4, x_unit=EnergyUnit.EV
        )

        assert x.tolist() == snapshot(
            [
                3.5,
                3.8333333333333335,
                4.166666666666667,
                4.5,
                4.833333333333333,
                5.166666666666666,
                5.5,
                5.833333333333333,
                6.166666666666666,
                6.5,
            ]
        )
        assert y.tolist() == snapshot(
            [
                1.5256802432917473e-18,
                2.4935979234133957e-18,
                4.7570756148627734e-18,
                1.2047612955648624e-17,
                5.1547983302037483e-17,
                5.1547983302037705e-17,
                1.2047612955648624e-17,
                4.7570756148627734e-18,
                2.4935979234133972e-18,
                1.5256802432917473e-18,
            ]
        )
        assert x_stick.tolist() == snapshot([5.0])
        assert y_stick.tolist() == snapshot([5.1547983302037705e-17])
