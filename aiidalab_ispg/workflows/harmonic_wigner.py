"""AiiDA workflow for generating harmonic wigner sampling"""

from aiida.engine import calcfunction
from aiida.orm import StructureData, TrajectoryData
from aiidalab_ispg.wigner import Wigner

__all__ = [
    "generate_wigner_structures",
]


@calcfunction
def generate_wigner_structures(
    minimum_structure: StructureData,
    orca_output_dict: dict,
    nsample: int,
    low_freq_thr: float,
):
    seed = orca_output_dict.extras["_aiida_hash"]
    ase_molecule = minimum_structure.get_ase()
    frequencies = orca_output_dict["vibfreqs"]
    normal_modes = orca_output_dict["vibdisps"]

    wigner = Wigner(
        ase_molecule,
        frequencies,
        normal_modes,
        seed=seed,
        low_freq_thr=low_freq_thr.value,
    )

    wigner_list = [
        StructureData(ase=wigner.get_ase_sample()) for i in range(nsample.value)
    ]
    return TrajectoryData(structurelist=wigner_list)
