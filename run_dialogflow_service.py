import sys
import os

# 1. Force the current folder into the python path
sys.path.append(os.getcwd())

# 2. Import the standard Dialogflow component
# We import the internal 'DialogflowCXComponent' to run the service
from sic_framework.services.dialogflow_cx.dialogflow_cx import DialogflowCXComponent
from sic_framework import SICComponentManager

if __name__ == "__main__":
    print("Starting Dialogflow CX Service from Runner...")
    SICComponentManager([DialogflowCXComponent], name="DialogflowCX")