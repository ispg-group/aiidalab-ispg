import requests
import time

import pytest
# https://selenium-python.readthedocs.io/locating-elements.html
from selenium.webdriver.common.by import By

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 1250

@pytest.mark.tryfirst
def test_post_install(notebook_service, docker_exec):
    docker_exec("./post_install", user="jovyan")


def test_notebook_service_available(notebook_service):
    url, token = notebook_service
    response = requests.get(f"{url}/?token={token}")
    assert response.status_code == 200


def test_dependencies(notebook_service, docker_exec):
    docker_exec("pip check", user="jovyan")


def test_conformer_generation_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[contains(.,'Generate molecule')]")
    driver.get_screenshot_as_file(f"{screenshot_dir}/conformer-generation-init.png")


def test_conformer_generation_steps(selenium_driver, screenshot_dir):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    smiles_textarea = driver.find_element(By.XPATH, "//input[@placeholder='C=C']")
    smiles_textarea.send_keys("C")

    generate_mol_button = driver.find_element(
        By.XPATH, "//button[contains(.,'Generate molecule')]"
    )
    generate_mol_button.click()
    time.sleep(5)
    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-generated.png"
    )

    # Switch to `Download` tab in StructureDataViewer
    driver.find_element(By.XPATH, "//*[text()='Download']").click()
    driver.find_element(By.XPATH, "//button[contains(.,'Download')]").click()
    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-download-tab.png"
    )


def test_spectrum_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("spectrum_widget.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[contains(.,'Download spectrum')]")
    driver.get_screenshot_as_file(f"{screenshot_dir}/spectrum-widget.png")


def test_atmospec_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[contains(.,'Refresh')]")
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-app.png")


def test_atmospec_steps(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=40.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # For some reason this test is stuck on the loading page
    smiles_textarea = driver.find_element(By.XPATH, "//input[@placeholder='C=C']")

    smiles_textarea.send_keys("C")
    generate_mol_button = driver.find_element(
        By.XPATH, "//button[contains(.,'Generate molecule')]"
    )
    generate_mol_button.click()

    # Once the structure is generated, proceed to the next workflow step
    time.sleep(5)
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-generated.png")

    confirm_btn = driver.find_element(By.XPATH, "//button[contains(.,'Confirm')]")
    confirm_btn.click()
    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")

    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-confirmed.png")
