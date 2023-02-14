"""Common inputs widgets"""
import ipywidgets as iwp

from .widgets import PCM_SOLVENT_LIST


class MoleculeDefinitionWidget(ipw.VBox):
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


class GroundStateDefinitionWidget(ipw.VBox):
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


class MoleculeDefinitionWidget(ipw.VBox):
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


class GroundStateDefinitionWidget(ipw.VBox):
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
