"""Test code that transforms `spectrum_data` output from Atmospec AiiDA workflow
to a datastructure that is passed to the SpectrumWidget"""

import sys

import pytest
from aiidalab_ispg.app.steps import _get_conformer_transitions
from inline_snapshot import snapshot


def test_multiple_unoptimized_geometries(generate_atmospec_node):
    process = generate_atmospec_node(optimize=False, nstates=2, nconf=2, nwigner=0)
    for input in process.inputs:
        print(input, getattr(process.inputs, input))
    for output in process.outputs:
        print(output, getattr(process.outputs, output))

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
                ],
                "nsample": 1,
                "weight": 0.5,
            },
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
                ],
                "nsample": 1,
                "weight": 0.5,
            },
        ]
    )
    assert conf_transitions == ref


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
