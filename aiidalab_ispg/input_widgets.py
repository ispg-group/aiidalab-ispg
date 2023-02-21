"""Common inputs widgets"""
import ipywidgets as ipw

from aiida.common import NotExistent
from aiida.orm import load_code

from aiidalab_widgets_base import ComputationalResourcesWidget
from .widgets import PCM_SOLVENT_LIST


class MoleculeSettings(ipw.VBox):
    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Molecule specification</h4>
        </div>"""
    )
    _DEFAULT_CHARGE = 0
    _DEFAULT_MULTIPLICITY = 1
    _DEFAULT_SOLVENT = "None"

    def __init__(self, **kwargs):
        self.multiplicity = ipw.BoundedIntText(
            min=1,
            max=7,
            step=1,
            description="Multiplicity",
            value=self._DEFAULT_MULTIPLICITY,
        )

        self.charge = ipw.IntText(
            description="Charge",
            disabled=False,
            value=self._DEFAULT_CHARGE,
        )

        style = {"description_width": "initial"}
        self.solvent = ipw.Dropdown(
            options=PCM_SOLVENT_LIST,
            value=self._DEFAULT_SOLVENT,
            description="PCM solvent",
            disabled=False,
            style=style,
        )
        super().__init__(
            children=[self.title, self.charge, self.multiplicity, self.solvent]
        )

    def reset(self):
        self.charge.value = self._DEFAULT_CHARGE
        self.charge.multiplicity.value = self._DEFAULT_MULTIPLICITY


class GroundStateSettings(ipw.VBox):
    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Ground state electronic structure</h4>
        </div>"""
    )

    def __init__(self, **kwargs):
        style = {"description_width": "initial"}

        self.method = ipw.Text(
            description="Ground state method",
            style=style,
        )

        self.basis = ipw.Text(description="Basis set")
        super().__init__(children=[self.title, self.method, self.basis])

    def reset(self):
        self.method.value = ""
        self.basis.value = ""


class CodeSettings(ipw.VBox):

    codes_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Codes</h4></div>"""
    )
    codes_help = ipw.HTML(
        """<div style="line-height: 140%; padding-top: 0px;
        padding-bottom: 10px"> Select the ORCA code.</div>"""
    )

    # In the order of priority, we will select the default ORCA code from these
    # First, we try to use SLURM on local machine, if available
    _DEFAULT_ORCA_CODES = ("orca@slurm", "orca@localhost")

    def __init__(self, **kwargs):

        self.orca = ComputationalResourcesWidget(
            default_calc_job_plugin="orca.orca",
            description="ORCA program",
        )
        self._set_default_codes()
        super().__init__(
            children=[
                self.codes_title,
                self.codes_help,
                self.orca,
            ],
            **kwargs,
        )

    def _set_default_codes(self):
        for code_label in self._DEFAULT_ORCA_CODES:
            try:
                self.orca.value = load_code(code_label).uuid
                return
            except (NotExistent, ValueError):
                pass

        if not self.orca.value:
            print("WARNING: ORCA code has not been found locally")

    def reset(self):
        self._set_default_codes()
