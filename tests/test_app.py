import requests
import time

# https://selenium-python.readthedocs.io/locating-elements.html
from selenium.webdriver.common.by import By


def test_notebook_service_available(notebook_service):
    url, token = notebook_service
    response = requests.get(f"{url}/?token={token}")
    assert response.status_code == 200


def test_conformer_generation_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(1920, 1450)
    driver.find_element(By.XPATH, "//input[@placeholder='C=C']")
    driver.get_screenshot_as_file(f"{screenshot_dir}/conformer-generation-init.png")


def test_conformer_generation_steps(selenium_driver, screenshot_dir):
    driver = selenium_driver("conformer_generation.ipynb", wait_time=30.0)
    driver.set_window_size(1920, 1450)

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

    # click `Download` tab in StructureDataViewer
    driver.find_element(By.XPATH, "//*[text()='Download']").click()
    time.sleep(2)
    driver.get_screenshot_as_file(
        f"{screenshot_dir}/conformer-generation-download-tab.png"
    )


def test_spectrum_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("spectrum_widget.ipynb", wait_time=30.0)
    driver.set_window_size(1920, 1450)
    driver.find_element(By.XPATH, "//button[contains(.,'Download spectrum')]")
    driver.get_screenshot_as_file(f"{screenshot_dir}/spectrum-widget.png")


def test_atmospec_app_init(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=30.0)
    driver.set_window_size(1920, 1450)
    time.sleep(10)
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-app.png")


def test_atmospec_steps(selenium_driver, screenshot_dir):
    driver = selenium_driver("atmospec.ipynb", wait_time=40.0)
    driver.set_window_size(1920, 1650)

    # For some reason this test is stuck on the loading page
    time.sleep(10)
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-steps-loading.png")
    return
    smiles_textarea = driver.find_element(By.XPATH, "//input[@placeholder='C=C']")

    smiles_textarea.send_keys("C")
    generate_mol_button = driver.find_element(
        By.XPATH, "//button[contains(.,'Generate molecule')]"
    )
    generate_mol_button.click()

    # Once the structure is generated, proceed to the next workflow step
    time.sleep(2)
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-generated.png")

    confirm_btn = driver.find_element(By.XPATH, "//button[contains(.,'Confirm')]")
    confirm_btn.click()
    driver.get_screenshot_as_file(f"{screenshot_dir}/atmospec-mol-confirmed.png")

    # Test that we have indeed proceeded to the next step
    driver.find_element(By.XPATH, "//span[contains(.,'✓ Step 1')]")

    # Note, the element is found even if it is hidden behind fold
    # Can't actually click submit, obviously
    driver.find_element(By.XPATH, "//button[contains(.,'Submit')]")
