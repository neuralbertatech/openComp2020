# NeurAlbertaTech: Open Competiiton 2020

## Project Description
The NAT Brain Drone is a general purpose BCI classification controller that is specifically tailored to control a drone. This controller is fundamentally designed to be expandable and general purpose so that it can realistically control whatever you can connected to the computer. For more information about this project, visit its webpage natuab.ca/drone or view the submission video youtube.com.

## Requirements
* A 16 channel OpenBCI (Cyton + Daisy)
* A DJI Tello drone
* A MacOS machine (should also work on Linux and Windows, though it has not been tested)
* python >= 3.6.5

## Set Up
### Mac
To open terminal, press ` command + space ` and type ` terminal ` then press ` enter `

Navigate to your desired install location using
` cd [directory] ` (For example, ` cd Desktop `)

Clone this repo\n
` git clone https://github.com/neuralbertatech/openComp2020 `

Navigate into this repo\n
` cd openComp2020 `

Install homebrew\n
 ` ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" `

Install venv\n
 ` brew install venv `

Install pyenv\n
 ` brew install pyenv `

Configure a virtual environment to contain dependencies\n
```
 pyenv install 3.6.5
 pyenv local 3.6.5
 python3 -m venv venv
```

Activate the environment\n
` source venv/bin/activate `

Update Pip installer\n
` pip install --upgrade pip`

To install the required software run\n
` pip install -r requirements.txt `

### Windows
Not yet tested

### Linux
Not yet tested


## Run The Program
Activate the virtual environment with\n
` pyenv local 3.6.5 `

and\n
` source venv/bin/activate `


Then finally, run the command\n
` python masterController.py `

and follow the onscreen instructions.

If you are running the program after a training session, and have not yet moved the OpenBCI from where it was when the baseline was collected, you can reuse this data and skip the lengthy baseline collection process by calling\n
` python masterController.py nocol `

By default, the previous session will be saved to ~/TrainingData/masterControllerSessions. If you would like to use data in another directory, you can call\n
` python masterController.py nocol [directory relative to this file] `


## Known Issues
