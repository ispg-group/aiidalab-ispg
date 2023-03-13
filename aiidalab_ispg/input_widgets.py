"""Common inputs widgets for workflow settings"""

from enum import Enum, unique

import ipywidgets as ipw
import traitlets

from aiida.common import NotExistent
from aiida.orm import load_code
from aiidalab_widgets_base import ComputationalResourcesWidget

# Taken from ORCA-5.0 manual, section 9.41
PCM_SOLVENT_LIST = (
    "None",
    "Water",
    "Acetone",
    "Acetonitrile",
    "Ammonia",
    "Benzene",
    "CCl4",
    "CH2Cl2",
    "Chloroform",
    "Cyclohexane",
    "DMF",
    "DMSO",
    "Ethanol",
    "Hexane",
    "Methanol",
    "Octanol",
    "Pyridine",
    "THF",
    "Toluene",
)


@unique
class ExcitedStateMethod(Enum):
    TDA = "TDA/TDDFT"
    TDDFT = "TDDFT"
    CCSD = "EOM-CCSD"
    ADC2 = "ADC2"


class MolecularGeometrySettings(ipw.VBox):

    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Molecular geometry</h4></div>"""
    )
    description = ipw.HTML(
        """<div style="line-height: 140%; padding-top: 0px; padding-bottom: 5px">
        By default, the workflow will optimize the provided geometry.<br>
        Select "Geometry as is" if this is not desired.</div>"""
    )

    def __init__(self, **kwargs):
        # Whether to optimize the molecule or not.
        self.geo_opt_type = ipw.ToggleButtons(
            options=[
                ("Geometry as is", "NONE"),
                ("Optimize geometry", "OPT"),
            ],
            value="OPT",
        )
        super().__init__(children=[self.title, self.geo_opt_type])


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
        style = {"description_width": "initial"}

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
        layout = ipw.Layout(max_width="280px")

        self.method = ipw.Text(
            description="Ground state method",
            style=style,
            layout=layout,
            tooltip="DFT funtional or MP2",
        )
        self.basis = ipw.Text(description="Basis set", layout=layout)

        super().__init__(children=[self.title, self.method, self.basis])

    def reset(self):
        self.method.value = ""
        self.basis.value = ""


class ExcitedStateSettings(ipw.VBox):
    """Widget for selecting ab initio level for excited states"""

    _DEFAULT_EXCITED_METHOD = ExcitedStateMethod.TDA
    _DEFAULT_FUNCTIONAL = "wb97X-D4"
    _DEFAULT_BASIS = "def2-SVP"
    _DEFAULT_NSTATES = 3

    qm_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Excited state QM method</h4>
        </div>"""
    )

    def __init__(self, **kwargs):
        style = {"description_width": "initial"}
        layout = ipw.Layout(max_width="280px")

        self.excited_method = ipw.Dropdown(
            options=[(method.value, method) for method in ExcitedStateMethod],
            value=self._DEFAULT_EXCITED_METHOD,
            description="Excited state method",
            style=style,
            layout=layout,
        )
        self.excited_method.observe(self._observe_excited_method, names="value")

        self.nstates = ipw.BoundedIntText(
            description="Number of excited states",
            tooltip="Number of excited states",
            disabled=False,
            value=self._DEFAULT_NSTATES,
            min=1,
            max=50,
            style=style,
            layout=layout,
        )

        self.ground_state_sync = ipw.Checkbox(
            value=True,
            description="Use ground state basis set and functional",
            indent=False,
        )
        self.ground_state_sync.observe(self._observe_gs_sync, "value")

        self.tddft_functional = ipw.Text(
            value=self._DEFAULT_FUNCTIONAL,
            description="TDDFT functional",
            style=style,
            layout=layout,
            disabled=True,
        )
        self.basis = ipw.Text(
            value=self._DEFAULT_BASIS,
            description="Basis set",
            style=style,
            layout=layout,
            disabled=True,
        )

        super().__init__(
            children=[
                self.qm_title,
                self.excited_method,
                self.nstates,
                self.ground_state_sync,
                self.tddft_functional,
                self.basis,
            ]
        )

    def _observe_gs_sync(self, change):
        if change["new"]:
            self.basis.disabled = True
            self.tddft_functional.disabled = True
        else:
            self.basis.disabled = False
            if self.excited_method.value not in (
                ExcitedStateMethod.ADC2,
                ExcitedStateMethod.CCSD,
            ):
                self.tddft_functional.disabled = False

    def _observe_excited_method(self, change):
        es_method = change["new"]
        if es_method in (ExcitedStateMethod.ADC2, ExcitedStateMethod.CCSD):
            self.tddft_functional.disabled = True
        elif not self.ground_state_sync.value:
            self.tddft_functional.disabled = False

    def reset(self):
        self.excited_method.value = _DEFAULT_EXCITED_METHOD
        self.method.value = self._DEFAULT_FUNCTIONAL
        self.basis.value = self._DEFAULT_BASIS
        self.nstate.value = self._DEFAULT_NSTATES


class WignerSamplingSettings(ipw.VBox):

    disabled = traitlets.Bool(default=False)

    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Wigner sampling</h4>
        </div>"""
    )

    _LOW_FREQ_THR_DEFAULT = 100  # cm^-1
    _NSAMPLES_DEFAULT = 1

    def __init__(self):
        style = {"description_width": "initial"}

        self.nwigner = ipw.BoundedIntText(
            value=self._NSAMPLES_DEFAULT,
            step=1,
            min=0,
            max=1000,
            style=style,
            description="Number of Wigner samples",
        )

        self.wigner_low_freq_thr = ipw.BoundedFloatText(
            value=self._LOW_FREQ_THR_DEFAULT,
            step=1,
            min=0,
            max=10000,
            style=style,
            description="Low-frequency threshold",
            title="Normal modes below this frequency will be ignored",
        )

        super().__init__([self.title, self.nwigner, self.wigner_low_freq_thr])

    @traitlets.observe("disabled")
    def _observer_disabled(self, change):
        if change["new"]:
            self.nwigner.disabled = True
            self.wigner_low_freq_thr.disabled = True
        else:
            self.nwigner.disabled = False
            self.wigner_low_freq_thr.disabled = False

    def reset(self):
        self.nwigner.value = self._NSAMPLES_DEFAULT
        self.wigner_low_freq_thr.value = self._LOW_FREQ_THR_DEFAULT


class CodeSettings(ipw.VBox):

    codes_title = ipw.HTML(
        """<div style="padding-top: 10px; padding-bottom: 0px">
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
                # self.codes_help,
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


class ResourceSelectionWidget(ipw.VBox):
    """Widget for the selection of compute resources."""

    title = ipw.HTML(
        """<div style="padding-top: 10px; padding-bottom: 0px">
        <h4>Resources</h4>
    </div>"""
    )
    prompt = ipw.HTML(
        """<div style="line-height:120%; padding-top:0px">
        <p style="padding-bottom:10px">
        Number of MPI tasks for this calculation.
        </p></div>"""
    )

    def __init__(self, **kwargs):
        extra = {
            "style": {"description_width": "initial"},
            # "layout": {"max_width": "200px"},
        }

        self.num_mpi_tasks = ipw.BoundedIntText(
            value=1, step=1, min=1, max=16, description="Number of MPI tasks", **extra
        )

        super().__init__(
            children=[
                self.title,
                self.num_mpi_tasks,
                # ipw.HBox(children=[self.prompt, self.num_mpi_tasks]),
            ]
        )

    def reset(self):
        self.num_mpi_tasks.value = 1
