import sys
import os

# 1. Force the current folder into the python path
# This ensures 'scripts' is always found as a top-level package
sys.path.append(os.getcwd())

# 2. Import the service using the full package path
# This fixes the "Unknown request type" error by aligning the class names
from scripts.google_script import GoogleGenAIService
from sic_framework import SICComponentManager

if __name__ == "__main__":
    print("Starting Google GenAI Service from Runner...")
    SICComponentManager([GoogleGenAIService], name="GoogleGenAIService")