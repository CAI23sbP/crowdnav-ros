# Installation

## Activate poetry shell
```
cd arena-rosnav # navigate to the arena-rosnav directory
poetry shell
```
## Make sure to source the workspace environment
```bash
cd ../.. # navigate to the catkin_ws directory
source devel/setup.zsh # if you use bash: source devel/setup.bash 
```
## Install RVO2 (make sure SARL* planner is downloaded)
If you have already went through the installation process of SARL*, you can skip this step.
```bash
roscd sarl_star_ros
cd ../Python-RVO2
pip install Cython
python setup.py build
python setup.py install
```
## Install Crowdnav dependencies
```bash
roscd crowdnav-ros
cd scripts
pip install -e .
```
