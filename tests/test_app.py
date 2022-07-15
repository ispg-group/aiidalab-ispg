#!/usr/bin/env python
from selenium.webdriver.common.by import By


def test_atmospec(selenium, url):
    #selenium.get(url("http://localhost:8100/apps/apps/quantum-espresso/qe.ipynb"))
    selenium.get(url("apps/apps/aiidalab-ispg/atmospec.ipynb"))
    selenium.find_element(By.ID, "ipython-main-app")
    selenium.find_element(By.ID, "notebook-container")
    selenium.find_element(By.CLASS_NAME, "jupyter-widgets-view")
    selenium.get_screenshot_as_file("screenshots/atmospec.png")
