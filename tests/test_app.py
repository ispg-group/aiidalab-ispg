import requests

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
    driver.find_element(By.XPATH, "//button[text()='Generate molecule']")
    driver.get_screenshot_as_file(f"{screenshot_dir}/conformer-generation-init.png")


def test_conformer_generation_steps(selenium_driver, screenshot_dir, generate_mol):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol(driver, "C")

    # Select the first atom
    driver.find_element(By.XPATH, "//*[text()='Selection']").click()
    driver.find_element(
        By.XPATH, "//label[text()='Selected atoms:']/following-sibling::input"
    ).send_keys("1")
    driver.find_element(By.XPATH, '//button[text()="Apply selection"]').click()
    driver.find_element(By.XPATH, "//div[starts-with(text(),'Id: 1; Symbol: C;')]")

    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-generated.png"
    )

    # Switch to `Download` tab in StructureDataViewer
    driver.find_element(By.XPATH, "//*[text()='Download']").click()
    driver.find_element(By.XPATH, "//button[text()='Download']").click()
    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-download-tab.png"
    )


def test_spectrum_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("spectrum_widget.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Download spectrum']")
    driver.get_screenshot_as_file(f"{screenshot_dir}/spectrum-widget.png")


def test_atmospec_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=30.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.find_element(By.XPATH, "//button[text()='Refresh']")
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-app.png")


def test_atmospec_steps(selenium_driver, screenshot_dir, generate_mol):
    driver = selenium_driver("atmospec.ipynb", wait_time=40.0)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate methane molecule
    generate_mol(driver, "C")
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-generated.png")

    driver.find_element(By.XPATH, "//button[text()='Confirm']").click()
    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")

    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-confirmed.png")
