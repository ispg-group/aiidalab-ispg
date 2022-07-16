#!/usr/bin/env python
import time

from selenium.webdriver.common.by import By

# https://selenium-python.readthedocs.io/locating-elements.html

def test_atmospec_app_take_screenshot(selenium, url):
    selenium.get(url("apps/apps/aiidalab-ispg/atmospec.ipynb"))
    selenium.set_window_size(1920, 985)
    time.sleep(10)
    selenium.find_element(By.ID, "ipython-main-app")
    selenium.find_element(By.ID, "notebook-container")
    selenium.find_element(By.CLASS_NAME, "jupyter-widgets-view")
    selenium.get_screenshot_as_file("screenshots/atmospec-app.png")

def test_atmospec_generate_mol_from_smiles(selenium, url):
    selenium.get(url("apps/apps/aiidalab-ispg/atmospec.ipynb"))
    time.sleep(10)
    # TODO: Find the input SMILES element, and set it to "C"
    # and click "Generate molecule" button
    # This will likely fail since xtb is missing. Once we figure out
    # how to install it automatically, we can click on Confirm button
    # to get to the next step
    #generate_mol_button = selenium.find_element(By.XPATH, "//button[contains(.,'Generate molecule')]")
    # generate_mol_button.click()
    # Once we figure out how to install xtb automaticaly,
    # we can click the Confirm button to get to the next step
    confirm_button = selenium.find_element(By.XPATH, "//button[contains(.,'Confirm')]")
    confirm_button.location_once_scrolled_into_view  # scroll into view
    #confirm_button.click()
    selenium.get_screenshot_as_file("screenshots/atmospec-mol-confirmed.png")

