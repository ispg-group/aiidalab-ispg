import math

import bokeh.io
import ipywidgets as ipw
from aiida.plugins import DataFactory

StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")
CifData = DataFactory("core.cif")

# Energy units conversion factors
# TODO: Make this an Enum, or use a library
# atomic units to electronvolts
AUtoEV = 27.2114386245
AUtoKCAL = 627.04
KCALtoKJ = 4.183
AUtoKJ = AUtoKCAL * KCALtoKJ
EVtoKJ = AUtoKCAL * KCALtoKJ / AUtoEV

# Molar gas constant, Avogadro times Boltzmann
R = 8.3144598

# TODO: Make this configurable
# Safe default for 8 core, 32Gb machine
# TODO: Figure out how to make this work as a global keyword
# https://github.com/pzarabadip/aiida-orca/issues/45
MEMORY_PER_CPU = 3000  # Mb

# TODO: Use numpy here? Measure the speed...
# Energies expected in kJ / mole, Absolute temperature in Kelvins
def calc_boltzmann_weights(energies, T):
    RT = R * T
    E0 = min(energies)
    weights = [math.exp(-(1000 * (E - E0)) / RT) for E in energies]
    Q = sum(weights)
    return [weight / Q for weight in weights]


def get_formula(data_node):
    """A wrapper for getting a molecular formula out of the AiiDA Data node"""
    if isinstance(data_node, TrajectoryData):
        # TrajectoryData can only hold structures with the same chemical formula,
        # so this approach is sound.
        stepid = data_node.get_stepids()[0]
        return data_node.get_step_structure(stepid).get_formula()
    elif isinstance(data_node, StructureData):
        return data_node.get_formula()
    elif isinstance(data_node, CifData):
        return data_node.get_ase().get_chemical_formula()
    else:
        raise ValueError(f"Cannot get formula from node {type(data_node)}")


# https://stackoverflow.com/a/3382369
def argsort(seq):
    """Returns a list of indeces that sort the array"""
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


# This code was provided by a good soul on GitHub.
# https://github.com/bokeh/bokeh/issues/7023#issuecomment-839825139
class BokehFigureContext(ipw.Output):
    """Helper class for rendering Bokeh figures inside ipywidgets"""

    def __init__(self, fig):
        super().__init__()
        self._figure = fig
        self._handle = None
        self.on_displayed(lambda x: x.set_handle())

    def set_handle(self):
        self.clear_output()
        with self:
            self._handle = bokeh.io.show(self._figure, notebook_handle=True)

    def get_handle(self):
        return self._handle

    def get_figure(self):
        return self._figure

    def update(self):
        if self._handle is not None:
            bokeh.io.push_notebook(handle=self._handle)

    def remove_renderer(self, label: str, update=True):
        f = self.get_figure()
        renderer = f.select_one({"name": label})
        if renderer is None:
            return
        f.renderers.remove(renderer)
        if update:
            self.update()

    def clean(self):
        f = self.get_figure()
        labels = [r.name for r in f.renderers]
        for label in labels:
            self.remove_renderer(label, update=False)
        self.update()
