import requests

import pytest

# https://selenium-python.readthedocs.io/locating-elements.html
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


def test_conformer_generation_init(selenium_driver):
    driver = selenium_driver(
        "conformer_generation.ipynb",
        wait_time=30.0,
        screenshot_name="conformer-generation-init.png",
    )
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Generate molecule']")


def test_conformer_generation_steps(
    selenium_driver, screenshot_dir, generate_mol_from_smiles, check_first_atom
):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol_from_smiles(driver, "C")

    # Select the first atom
    driver.find_element(By.XPATH, "//*[text()='Selection']").click()
    check_first_atom(driver, "C")

    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-generated.png"
    )

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
    driver.find_element(By.XPATH, "//button[text()='Download']").click()
    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-download-tab.png"
    )


def test_spectrum_app_init(selenium_driver, final_screenshot):
    driver = selenium_driver("spectrum_widget.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    final_screenshot["name"] = "spectrum-widget.png"
    driver.find_element(By.XPATH, "//button[text()='Download spectrum']")


def test_atmospec_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Refresh']")
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-app.png")


def test_atmospec_steps(
    selenium_driver, screenshot_dir, generate_mol_from_smiles, check_first_atom
):
    driver = selenium_driver("atmospec.ipynb", wait_time=40.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol_from_smiles(driver, "C")
    check_first_atom(driver, "C")
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-generated.png")

    driver.find_element(By.XPATH, "//button[text()='Confirm']").click()
    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")

    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-confirmed.png")
