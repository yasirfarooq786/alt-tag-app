#!/bin/bash
# Post-deployment script for Heroku

echo "Installing Playwright browsers..."
python playwright_install.py

echo "Setting up browser-use environment..."
export DISPLAY=:99