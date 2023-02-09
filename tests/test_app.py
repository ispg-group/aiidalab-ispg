import requests
from pathlib import Path

import pytest

# https://selenium-python.readthedocs.io/locating-elements.html
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 1250


@pytest.mark.tryfirst
def test_post_install(notebook_service, aiidalab_exec, nb_user, appdir):
    aiidalab_exec("./post_install", workdir=appdir, user=nb_user)


def test_notebook_service_available(notebook_service):
    url, token = notebook_service
    response = requests.get(f"{url}/?token={token}")
    assert response.status_code == 200


def test_dependencies(notebook_service, aiidalab_exec, nb_user):
    aiidalab_exec("pip check", user=nb_user)


def test_conformer_generation_init(selenium_driver, final_screenshot):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Generate molecule']")


def test_conformer_generation_steps(
    selenium_driver, final_screenshot, generate_mol_from_smiles, check_first_atom
):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol_from_smiles(driver, "C")

    # Select the first atom
    selection = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[text()='Selection']"))
    )
    selection.click()
    check_first_atom(driver, "C")

    # Test different generation options
    driver.find_element(By.XPATH, "//option[@value='UFF']").click()
    driver.find_element(By.XPATH, "//option[@value='ETKDGv1']").click()
    generate_mol_from_smiles(driver, "N")
    check_first_atom(driver, "N")

    driver.find_element(By.XPATH, "//option[@value='MMFF94s']").click()
    driver.find_element(By.XPATH, "//option[@value='ETKDGv2']").click()
    generate_mol_from_smiles(driver, "O")
    check_first_atom(driver, "O")

    # Switch to `Download` tab in StructureDataViewer
    driver.find_element(By.XPATH, "//*[text()='Download']").click()
    download = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Download']"))
    )
    download.click()


def test_optimization_init(selenium_driver, final_screenshot):
    driver = selenium_driver("optimization.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Generate molecule']")


def test_optimization_steps(
    selenium_driver,
    final_screenshot,
    generate_mol_from_smiles,
    check_first_atom,
):
    driver = selenium_driver("optimization.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Generate molecule']")
    # Generate methane molecule
    generate_mol_from_smiles(driver, "C")
    driver.find_element(By.XPATH, "//*[text()='Selection']").click()
    check_first_atom(driver, "C")

    driver.find_element(By.XPATH, "//button[text()='Confirm']").click()
    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")


def test_spectrum_app_init(selenium_driver, final_screenshot):
    driver = selenium_driver("spectrum_widget.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Download spectrum']")


def test_atmospec_app_init(selenium_driver, final_screenshot):
    driver = selenium_driver("atmospec.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Refresh']")


def test_atmospec_steps(
    selenium_driver,
    screenshot_dir,
    final_screenshot,
    generate_mol_from_smiles,
    check_first_atom,
):
    driver = selenium_driver("atmospec.ipynb", wait_time=40.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol_from_smiles(driver, "C")
    check_first_atom(driver, "C")
    driver.get_screenshot_as_file(
        Path.joinpath(screenshot_dir, "atmospec-steps-mol-generated.png")
    )

    confirm = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Confirm']"))
    )
    confirm.click()
    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")
