# -*- coding: utf-8 -*-

import ipywidgets as ipw

template = """
<table>
<tr>
  <th style="text-align:center">DH Testing application</th>
<tr>
  <td valign="top"><ul>
    <li><a href="{appbase}/example.ipynb" target="_blank">Example notebook</a></li>
  </ul></td>
</tr>
</table>
"""


def get_start_widget(appbase, jupbase, notebase):
    html = template.format(appbase=appbase, jupbase=jupbase, notebase=notebase)
    return ipw.HTML(html)


# EOF
