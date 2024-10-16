Matlab programs from EELS in the Electron Microscope 3rd Edition

Environment setup:
1. Set up virtual environment
   1. python3 -m venv venv
   2. source venv/bin/activate
2. Installed required libraries
   1. pip install -r requirement.txt
   
To run scripts:

Run scripts in terminal by simply enter:
python <script_name>.py
This way the parameters will be asked after

You can also call script in another python file:
from <ScriptName> import ScriptName
ScriptName(parameters)


