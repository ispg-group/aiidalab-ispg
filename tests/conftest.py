from __future__ import annotations

import pytest
from aiidalab_ispg.workflows.utils import (
    extract_trajectory_arrays,
    structures_to_trajectory,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def generate_trajectory():
    """Return a ``TrajectoryData`` representing a water molecule."""
    from ase.build import molecule

    from aiida import orm

    def _generate_trajectory(
        num_steps: int = 1, array_data: orm.ArrayData | None = None
    ):
        struct = orm.StructureData(ase=molecule("H2O"))
        # structurelist = [struct.clone() for i in range(num_steps)]
        structures = {}
        for i in range(num_steps):
            structures[f"struct_{i}"] = struct.clone()
        return structures_to_trajectory(arrays=array_data, **structures)

    return _generate_trajectory


@pytest.fixture
def generate_workchain_node(aiida_localhost):
    """Fixture to generate a mock `WorkChainNode`"""
    from aiida import orm
    from aiida.common import LinkType
    from aiida.plugins.entry_point import format_entry_point_string

    def _generate_workchain_node(
        entry_point_name="ispg.atmospec", computer=None, inputs=None, outputs=None
    ) -> orm.WorkChainNode:
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """

        entry_point = format_entry_point_string("aiida.workflows", entry_point_name)

        node = orm.WorkChainNode(
            computer=computer or aiida_localhost, process_type=entry_point
        )
        node.set_process_state("finished")

        if inputs:
            for label, input_node in inputs.items():
                input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_WORK, link_label=label
                )

        # Node must be stored before adding outputs!
        node.store()

        if outputs:
            for output_label, output_node in outputs.items():
                output_node.store()
                output_node.base.links.add_incoming(
                    node, link_type=LinkType.RETURN, link_label=output_label
                )

        return node

    return _generate_workchain_node


@pytest.fixture
def generate_atmospec_node(generate_workchain_node, generate_trajectory):
    """Fixture to generate a mock `AtmospecWorkChainNode` for testing spectrum generation."""
    from aiida import orm

    def _generate_spectrum_data(nstates: int, nconf: int, nwigner: int) -> orm.List:
        single_point = {
            "oscillator_strengths": [0.01 * i for i in range(nstates)],
            "excitation_energies_cm": [8000.00 + (1000 * i) for i in range(nstates)],
        }
        if not nwigner:
            nsample = 1
        else:
            nsample = nwigner
        spectrum_data = []
        for conf in range(nconf):
            samples = [single_point for _ in range(nsample)]
            spectrum_data.append(samples)

        return orm.List(spectrum_data)

    def _generate_atmospec_node(
        optimize: bool, nstates: int, nconf: int, nwigner: int
    ) -> orm.WorkChainNode:
        inputs = {
            "optimize": orm.Bool(optimize),
            "nstates": orm.Int(nstates),
            "structure": generate_trajectory(nconf),
            "nwigner": orm.Int(nwigner),
        }

        outputs = {
            "spectrum_data": _generate_spectrum_data(
                nstates=nstates, nconf=nconf, nwigner=nwigner
            ),
        }
        if optimize:
            orca_params = {}
            for i in range(nconf):
                orca_params[f"orca_{i}"] = orm.Dict(
                    {"temperature": 300, "freeenergy": -20.0 + 0.001 * i}
                )
            array_data = extract_trajectory_arrays(**orca_params)
            outputs["relaxed_structures"] = generate_trajectory(
                nconf, array_data=array_data
            )

        node = generate_workchain_node(
            entry_point_name="ispg.atmospec", inputs=inputs, outputs=outputs
        )
        # Generate (partly) fake builder parameters
        bp = inputs.copy()
        bp.pop("structure")  # TrajectoryData is not serializable
        node.base.extras.set("builder_parameters", bp)
        return node

    return _generate_atmospec_node
