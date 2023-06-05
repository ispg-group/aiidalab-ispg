import ipywidgets as ipw

template = """
<div align="center">
    <img src="{appbase}/aiidalab_ispg/app/static/atmospec_logo-640.png" height="128px" width=453px">
</div>
<table>
<tr>
  <th style="text-align:center">ISPG Applications</th>
<tr>
  <td valign="top"><ul>
    <li><a href="{appbase}/notebooks/spectrum_widget.ipynb" target="_blank">Spectrum Widget</a></li>
  </ul></td>
  <td valign="top"><ul>
    <li><a href="{appbase}/notebooks/conformer_generation.ipynb" target="_blank">Conformer Sampling</a></li>
  </ul></td>
  <td valign="top"><ul>
    <li><a href="{appbase}/notebooks/optimization.ipynb" target="_blank">Conformer Optimization</a></li>
  </ul></td>
  <td valign="top"><ul>
    <li><a href="{appbase}/notebooks/atmospec.ipynb" target="_blank">ATMOSPEC</a></li>
  </ul></td>
</tr>
</table>
"""


def get_start_widget(appbase, jupbase, notebase):
    html = template.format(appbase=appbase, jupbase=jupbase, notebase=notebase)
    return ipw.HTML(html)
