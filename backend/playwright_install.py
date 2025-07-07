import subprocess
import sys
import os

def install_playwright_browsers():
    """Install Playwright browsers for Heroku"""
    try:
        # Install browsers
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("Playwright browsers installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Playwright browsers: {e}")

if __name__ == "__main__":
    install_playwright_browsers()