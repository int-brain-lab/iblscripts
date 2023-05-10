# Experiment Description GUI
This GUI is intended to simplify the categorization of an experiment and cleanly define what projects and procedures an 
experiment is for.

Assuming the following configurations on Windows 10:
* iblscripts has been cloned into the `C:\iblscripts` repo
* python v3.8 has already been installed

## Create the python virtual environment
Run the following commands from the non-administrative **Windows Powershell** prompt. Please modify the `<Username>` when 
appropriate:

```powershell
cd C:\iblscripts
C:\Users\<Username>\AppData\Local\Programs\Python\Python38\.\python.exe -m venv C:\iblscripts\venv
C:\iblscripts\venv\Scripts\.\Activate.ps1
python.exe -m pip install --upgrade pip wheel
cd deploy\project_procedure_gui
python.exe -m pip install --requirement pp_requirements.txt
```

## Run the Project Procedure GUI
Run the following commands from the non-administrative **Windows Powershell** prompt:

```powershell
C:\iblscripts\venv\Scripts\.\Activate.ps1
cd C:\iblscripts\deploy\project_procedure_gui
python experiment_form.py SW_023 Nate
```

## Troubleshooting

### qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
This error has cropped up on ubuntu machines. This error is due to an opencv incompatibility. Simply run the following command 
from the appropriate virtual environment to resolve:
* `pip install --upgrade  opencv-python==4.3.0.36 opencv-python-headless==4.3.0.36`