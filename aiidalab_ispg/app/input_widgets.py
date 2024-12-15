"""Common inputs widgets for workflow settings"""

from enum import Enum, unique

import ipywidgets as ipw
import traitlets as tl

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
    ZINDO = "ZINDO/S"


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
        self.optimize = ipw.ToggleButtons(
            options=[
                ("Geometry as is", False),
                ("Optimize geometry", True),
            ],
            value=True,
        )
        super().__init__(children=[self.title, self.optimize])


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
        layout = ipw.Layout(max_width="250px")

        self.multiplicity = ipw.BoundedIntText(
            min=1,
            max=7,
            step=1,
            description="Multiplicity",
            value=self._DEFAULT_MULTIPLICITY,
            style=style,
            layout=layout,
        )

        self.charge = ipw.IntText(
            description="Charge",
            disabled=False,
            value=self._DEFAULT_CHARGE,
            style=style,
            layout=layout,
        )

        self.solvent = ipw.Dropdown(
            options=PCM_SOLVENT_LIST,
            value=self._DEFAULT_SOLVENT,
            description="PCM solvent",
            disabled=False,
            style=style,
            layout=layout,
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
        <h4>Excited state electronic structure</h4>
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
            layout=layout,
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
        self.excited_method.value = self._DEFAULT_EXCITED_METHOD
        self.method.value = self._DEFAULT_FUNCTIONAL
        self.basis.value = self._DEFAULT_BASIS
        self.nstate.value = self._DEFAULT_NSTATES


class WignerSamplingSettings(ipw.VBox):
    disabled = tl.Bool(default=False)

    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Harmonic Wigner sampling</h4>
        </div>"""
    )

    _LOW_FREQ_THR_DEFAULT = 100  # cm^-1
    _NSAMPLES_DEFAULT = 1

    def __init__(self):
        style = {"description_width": "initial"}
        layout = ipw.Layout(max_width="250px")

        self.nwigner = ipw.BoundedIntText(
            value=self._NSAMPLES_DEFAULT,
            step=1,
            min=0,
            max=1000,
            description="Number of Wigner samples",
            style=style,
            layout=layout,
        )

        self.wigner_low_freq_thr = ipw.BoundedFloatText(
            value=self._LOW_FREQ_THR_DEFAULT,
            step=1,
            min=0,
            max=10000,
            description="Low-frequency cutoff (cm⁻¹)",
            # NOTE: Tooltip does not show up
            tooltip="Normal modes below this frequency will be ignored",
            style=style,
            layout=layout,
        )

        super().__init__([self.title, self.nwigner, self.wigner_low_freq_thr])

    @tl.observe("disabled")
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


class RepresentativeSamplingSettings(ipw.VBox):
    disabled = tl.Bool(default=False)

    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Representative Sampling</h4>
        </div>"""
    )

    _NUM_CYCLES_DEFAULT = 1200
    _NSAMPLES_DEFAULT = 10
    _EXP_METHOD_DEFAULT = "ZIndo/S"

    def __init__(self):
        style = {"description_width": "initial"}
        layout = ipw.Layout(max_width="250px")

        # Checkbox to enable/disable representative sampling
        self.enable_rep_sampling = ipw.Checkbox(
            value=False,
            description="Enable Representative Sampling",
            indent=False,
            layout=layout,
        )
        self.enable_rep_sampling.observe(self._observe_enable_rep_sampling, "value")

        # Representative Sampling settings
        self.num_cycles = ipw.IntText(
            description="Number of annealing cycles",
            value=self._NUM_CYCLES_DEFAULT,  # Default value
            style=style,
            layout=layout,
            disabled=True,  # Initially disabled
        )

        self.sample_size = ipw.IntText(
            description="Reduced sample size",
            value=self._NSAMPLES_DEFAULT,  # Default value
            style=style,
            layout=layout,
            disabled=True,  # Initially disabled
        )

        self.exploratory_method = ipw.Text(
            description="Exp. method",
            value=self._EXP_METHOD_DEFAULT,  # Default value
            style=style,
            layout=layout,
            disabled=True,  # Initially disabled
        )

        # Update the super().__init__ call to include all the widgets
        super().__init__(
            [
                self.title,
                self.enable_rep_sampling,
                self.num_cycles,
                self.sample_size,
                self.exploratory_method,
            ]
        )

    def _observe_enable_rep_sampling(self, change):
        """Enable/disable representative sampling settings based on checkbox value."""
        enabled = change["new"]
        self.num_cycles.disabled = not enabled
        self.sample_size.disabled = not enabled
        self.exploratory_method.disabled = not enabled

    @tl.observe("disabled")
    def _observer_disabled(self, change):
        """Enable/disable all representative sampling widgets based on the 'disabled' trait."""
        is_disabled = change["new"]
        self.num_cycles.disabled = is_disabled
        self.sample_size.disabled = is_disabled
        self.exploratory_method.disabled = is_disabled
        self.enable_rep_sampling.disabled = is_disabled

    def reset(self):
        """Reset all settings to their default values."""
        self.num_cycles.value = self._NUM_CYCLES_DEFAULT
        self.sample_size.value = self._NSAMPLES_DEFAULT
        self.exploratory_method.value = self._EXP_METHOD_DEFAULT
        self.enable_rep_sampling.value = False  # Default is disabled


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
        super().__init__(
            children=[
                self.codes_title,
                # self.codes_help,
                self.orca,
            ],
            **kwargs,
        )
        # WARNING: The on_displayed method has been removed in ipywidgets 8.0!!!
        # https://github.com/jupyter-widgets/ipywidgets/issues/3451
        # https://github.com/jupyter-widgets/ipywidgets/pull/2021
        self.on_displayed(self._set_default_codes)

    # Extra dummy parameter is needed since this is called via on_displayed
    def _set_default_codes(self, _=None):
        for code_label in self._DEFAULT_ORCA_CODES:
            try:
                self.orca.value = load_code(code_label).uuid
                return
            except (NotExistent, ValueError):
                pass
            except tl.TraitError:
                # This can happen if one of the code/computers is not configured/enabled or hidden
                # In practice, this happened to me locally when importing from production DB.
                # https://github.com/ispg-group/aiidalab-ispg/issues/240
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
