"""Test code that transforms `spectrum_data` output from Atmospec AiiDA workflow
to a datastructure that is passed to the SpectrumWidget"""

import functools
import sys

import pytest
from inline_snapshot import snapshot

from aiidalab_ispg.app.spectrum import BroadeningKernel, EnergyUnit
from aiidalab_ispg.app.spectrum_widget import SpectrumWidget
from aiidalab_ispg.app.steps import ViewSpectrumStep, _get_conformer_transitions

# Apply tighter numerical tresholds by default
approx = functools.partial(pytest.approx, rel=1e-10, abs=1e-25)


def test_multiple_unoptimized_geometries(generate_atmospec_node):
    process = generate_atmospec_node(optimize=False, nstates=2, nconf=2, nwigner=0)

    conf_transitions = _get_conformer_transitions(process)

    ref = snapshot(
        [
            {
                "transitions": [
                    {
                        "energy": 0.9918730956021841,
                        "osc_strength": 0.0,
                        "geom_index": 0,
                    },
                    {
                        "energy": 1.1158572325524572,
                        "osc_strength": 0.01,
                        "geom_index": 0,
                    },
                    {
                        "energy": 0.9918730956021841,
                        "osc_strength": 0.0,
                        "geom_index": 1,
                    },
                    {
                        "energy": 1.1158572325524572,
                        "osc_strength": 0.01,
                        "geom_index": 1,
                    },
                ],
                "nsample": 2,
                "weight": 1.0,
            }
        ]
    )
    assert conf_transitions == ref

    widget = SpectrumWidget()
    widget.conformer_transitions = conf_transitions

    x, y, x_stick, y_stick = widget._compute_total_cross_section(
        kernel=BroadeningKernel.GAUSS, energy_unit=EnergyUnit.EV, width=0.05
    )
    x_ref = snapshot(
        [
            0.49593654780109203,
            0.7314832905512437,
            0.9670300333013955,
            1.2025767760515471,
            1.438123518801699,
            1.6736702615518506,
            1.9092170043020023,
            2.144763747052154,
            2.380310489802306,
            2.6158572325524574,
        ]
    )
    y_ref = snapshot(
        [
            3.650659056913137e-51,
            1.2869645461525309e-30,
            1.0435295529909635e-19,
            1.946192456943438e-18,
            8.348523600518966e-27,
            8.23713977594192e-45,
            1.8693303369171427e-72,
            9.757499433538044e-110,
            1.1714772998437452e-156,
            3.2349835935299014e-213,
        ]
    )
    x_stick_ref = snapshot(
        [0.9918730956021841, 1.1158572325524572, 0.9918730956021841, 1.1158572325524572]
    )
    y_stick_ref = snapshot([0.0, 1.946192456943438e-18, 0.0, 1.946192456943438e-18])
    assert x.tolist() == approx(x_ref)
    assert y.tolist() == approx(y_ref)
    assert x_stick.tolist() == approx(x_stick_ref)
    assert y_stick.tolist() == approx(y_stick_ref)

    step = ViewSpectrumStep()
    step.process_uuid = process.uuid
    assert step.state == step.State.SUCCESS
    assert step.spectrum.debug_output.value == ""


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="requires python3.12 to get the same numerical values",
)
def test_optimized_conformers_without_wigner_sampling(generate_atmospec_node):
    """Test a single point spectrum for multiple conformers without Wigner sampling"""
    process = generate_atmospec_node(optimize=True, nstates=1, nconf=3, nwigner=0)

    conf_transitions = _get_conformer_transitions(process)

    ref = snapshot(
        [
            {
                "transitions": [
                    {"energy": 0.9918730956021841, "osc_strength": 0.0, "geom_index": 0}
                ],
                "nsample": 1,
                "weight": 0.6795896535088463,
            },
            {
                "transitions": [
                    {"energy": 0.9918730956021841, "osc_strength": 0.0, "geom_index": 0}
                ],
                "nsample": 1,
                "weight": 0.2374469602303986,
            },
            {
                "transitions": [
                    {"energy": 0.9918730956021841, "osc_strength": 0.0, "geom_index": 0}
                ],
                "nsample": 1,
                "weight": 0.08296338626075511,
            },
        ]
    )

    assert conf_transitions == ref

    widget = SpectrumWidget()
    widget.conformer_transitions = conf_transitions

    x, y, x_stick, y_stick = widget._compute_total_cross_section(
        kernel=BroadeningKernel.GAUSS, energy_unit=EnergyUnit.EV, width=0.05
    )

    x_ref = snapshot(
        [
            0.49593654780109203,
            0.7177072753345467,
            0.9394780028680014,
            1.161248730401456,
            1.3830194579349107,
            1.6047901854683653,
            1.82656091300182,
            2.0483316405352747,
            2.2701023680687293,
            2.491873095602184,
        ]
    )
    y_ref = snapshot([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_stick_ref = snapshot([0.9918730956021841, 0.9918730956021841, 0.9918730956021841])
    y_stick_ref = snapshot([0.0, 0.0, 0.0])

    assert x.tolist() == approx(x_ref)
    assert y.tolist() == approx(y_ref)
    assert x_stick.tolist() == approx(x_stick_ref)
    assert y_stick.tolist() == approx(y_stick_ref)

    step = ViewSpectrumStep()
    step.process_uuid = process.uuid
    assert step.state == step.State.SUCCESS
    assert step.spectrum.debug_output.value == ""


def test_one_conformer_with_wigner_sampling(generate_atmospec_node):
    """Test a single point spectrum for multiple conformers without Wigner sampling"""
    process = generate_atmospec_node(optimize=True, nstates=2, nconf=1, nwigner=2)

    conf_transitions = _get_conformer_transitions(process)
    ref = snapshot(
        [
            {
                "transitions": [
                    {
                        "energy": 0.9918730956021841,
                        "osc_strength": 0.0,
                        "geom_index": 0,
                    },
                    {
                        "energy": 1.1158572325524572,
                        "osc_strength": 0.01,
                        "geom_index": 0,
                    },
                    {
                        "energy": 0.9918730956021841,
                        "osc_strength": 0.0,
                        "geom_index": 1,
                    },
                    {
                        "energy": 1.1158572325524572,
                        "osc_strength": 0.01,
                        "geom_index": 1,
                    },
                ],
                "nsample": 2,
                "weight": 1.0,
            }
        ]
    )

    assert conf_transitions == ref

    widget = SpectrumWidget()
    widget.conformer_transitions = conf_transitions

    x, y, x_stick, y_stick = widget._compute_total_cross_section(
        kernel=BroadeningKernel.GAUSS, energy_unit=EnergyUnit.EV, width=0.05
    )

    x_ref = snapshot(
        [
            0.49593654780109203,
            0.7314832905512437,
            0.9670300333013955,
            1.2025767760515471,
            1.438123518801699,
            1.6736702615518506,
            1.9092170043020023,
            2.144763747052154,
            2.380310489802306,
            2.6158572325524574,
        ]
    )
    y_ref = snapshot(
        [
            3.650659056913137e-51,
            1.2869645461525309e-30,
            1.0435295529909635e-19,
            1.946192456943438e-18,
            8.348523600518966e-27,
            8.23713977594192e-45,
            1.8693303369171427e-72,
            9.757499433538044e-110,
            1.1714772998437452e-156,
            3.2349835935299014e-213,
        ]
    )
    x_stick_ref = snapshot(
        [0.9918730956021841, 1.1158572325524572, 0.9918730956021841, 1.1158572325524572]
    )
    y_stick_ref = snapshot([0.0, 1.946192456943438e-18, 0.0, 1.946192456943438e-18])

    assert x.tolist() == approx(x_ref)
    assert y.tolist() == approx(y_ref)
    assert x_stick.tolist() == approx(x_stick_ref)
    assert y_stick.tolist() == approx(y_stick_ref)

    step = ViewSpectrumStep()
    step.process_uuid = process.uuid
    assert step.state == step.State.SUCCESS
    assert step.spectrum.debug_output.value == ""


def test_failed_workflow(generate_workchain_node):
    """Test a single point spectrum for multiple conformers without Wigner sampling"""
    process = generate_workchain_node(exit_code=1)

    step = ViewSpectrumStep()
    assert step.state == step.State.INIT

    step.process_uuid = process.uuid

    assert step.state == step.State.FAIL
    assert step.spectrum.debug_output.value.startswith("Workflow failed")
    assert step.spectrum.conformer_transitions is None

    # Reset the widget
    step.reset()
    assert step.process_uuid is None
    assert step.state == step.State.INIT
    assert step.spectrum.debug_output.value == ""
