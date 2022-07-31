#!/usr/bin/env python
import time

from selenium.webdriver.common.by import By

# https://selenium-python.readthedocs.io/locating-elements.html


def test_atmospec_app_init(selenium, url):
    selenium.get(url("apps/apps/aiidalab-ispg/atmospec.ipynb"))
    selenium.set_window_size(1920, 1450)
    time.sleep(10)
    selenium.find_element(By.ID, "ipython-main-app")
    selenium.find_element(By.ID, "notebook-container")
    selenium.find_element(By.CLASS_NAME, "jupyter-widgets-view")
    selenium.get_screenshot_as_file("screenshots/atmospec-app.png")


def test_atmospec_generate_mol_from_smiles(selenium, url):
    selenium.get(url("apps/apps/aiidalab-ispg/atmospec.ipynb"))
    # selenium.set_window_size(1920, 1000)
    selenium.set_window_size(1920, 1450)
    smiles_textarea = selenium.find_element(By.XPATH, "//input[@placeholder='C=C']")
    smiles_textarea.send_keys("C")
    generate_mol_button = selenium.find_element(
        By.XPATH, "//button[contains(.,'Generate molecule')]"
    )
    generate_mol_button.click()

    # Once the structure is generated, proceed to the next workflow step
    time.sleep(1)
    selenium.get_screenshot_as_file("screenshots/atmospec-mol-generated.png")

    confirm_btn = selenium.find_element(By.XPATH, "//button[contains(.,'Confirm')]")
    confirm_btn.click()
    selenium.get_screenshot_as_file("screenshots/atmospec-mol-confirmed.png")

    # Test that we have indeed proceeded to the next step
    selenium.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")

    # Note, the element is found even if it is hidden behind fold
    # Can't actually click submit, obviously
    selenium.find_element(By.XPATH, "//button[contains(.,'Submit')]")
