"""Small utility workflows and functions"""
import math

import numpy as np
from aiida.engine import WorkChain, calcfunction
from aiida.orm import (
    ArrayData,
    Dict,
    List,
    StructureData,
    TrajectoryData,
)

__all__ = [
    "add_orca_wf_guess",
    "ConcatInputsToList",
    "pick_structure_from_trajectory",
    "structures_to_trajectory",
    "extract_trajectory_arrays",
]

AUtoEV = 27.2114386245
AUtoKCAL = 627.04
KCALtoKJ = 4.183
AUtoKJ = AUtoKCAL * KCALtoKJ
EVtoKJ = AUtoKCAL * KCALtoKJ / AUtoEV


# Meta WorkChain for combining all inputs from a dynamic namespace into List.
# Used to combine outputs from several subworkflows into one output.
# It should be launched via run() instead of submit()
# NOTE: The code has special handling for Dict nodes,
# which otherwise fail with not being serializable,
# so we need the get the value with Dict.get_dict() first.
# We should check whether this is still needed in aiida-2.0
# Note we cannot make this more general since List and Dict
# don't have the .value attribute.
# https://github.com/aiidateam/aiida-core/issues/5313
class ConcatInputsToList(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace("ns", dynamic=True)
        spec.output("output", valid_type=List)
        spec.outline(cls.combine)

    def combine(self):
        input_list = [
            self.inputs.ns[k].get_dict()
            if isinstance(self.inputs.ns[k], Dict)
            else self.inputs.ns[k]
            for k in self.inputs.ns
        ]
        self.out("output", List(list=input_list).store())


@calcfunction
def pick_structure_from_trajectory(trajectory: TrajectoryData, index) -> StructureData:
    return trajectory.get_step_structure(index.value)


@calcfunction
def add_orca_wf_guess(orca_params: Dict) -> Dict:
    params = orca_params.get_dict()
    params["input_keywords"].append("MOREAD")
    params["input_blocks"]["scf"]["moinp"] = '"aiida_old.gbw"'
    return Dict(params)


# TODO: Switch to variadic arguments (supported since AiiDA 2.3)
@calcfunction
def structures_to_trajectory(arrays: ArrayData = None, **structures) -> TrajectoryData:
    """Concatenate a list of StructureData to TrajectoryData

    Optionally, set additional data as Arrays.
    """
    traj = TrajectoryData(list(structures.values()))
    if arrays is None:
        return traj

    for name in arrays.get_arraynames():
        traj.set_array(name, arrays.get_array(name))
    # Copy over extras as well, except for private ones like '_aiida_hash'
    extras = arrays.base.extras.all
    for key in list(extras.keys()):
        if key.startswith("_"):
            del extras[key]
    traj.base.extras.set_many(extras)
    return traj


def calc_boltzmann_weights(energies: list, T: float):
    """Compute Boltzmann weights for a list of energies.

    param energies: list of energies / kJ per mole
    param T: temperature / Kelvin
    returns: Boltzmann weights as numpy array
    """
    # Molar gas constant, Avogadro times Boltzmann
    R = 8.3144598
    RT = R * T
    E0 = min(energies)
    weights = [math.exp(-(1000 * (E - E0)) / RT) for E in energies]
    Q = sum(weights)
    return np.array([weight / Q for weight in weights])


@calcfunction
def extract_trajectory_arrays(**orca_output_parameters) -> ArrayData:
    """Extract Gibbs energies and other useful stuff from the list
    of ORCA output parameter dictionaries.

    Return ArrayData node, which will be appended to TrajectoryData node.
    """
    gibbs_energies = np.array(
        [params["freeenergy"] for params in orca_output_parameters.values()]
    )
    en0 = min(gibbs_energies)
    relative_gibbs_energies_kj = AUtoKJ * (gibbs_energies - en0)

    temperature = next(iter(orca_output_parameters.values()))["temperature"]

    boltzmann_weights = calc_boltzmann_weights(relative_gibbs_energies_kj, temperature)

    en = ArrayData()
    en.set_array("gibbs_energies_au", gibbs_energies)
    en.set_array("relative_gibbs_energies_kj", relative_gibbs_energies_kj)
    en.set_array("boltzmann_weights", boltzmann_weights)
    en.set_extra("temperature", temperature)

    # For the TrajectoryData viewer compatibility
    en.set_array("energies", relative_gibbs_energies_kj)
    en.set_extra("energy_units", "kJ/mole")
    return en
