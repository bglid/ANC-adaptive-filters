# Python Implementation of DSP Adaptive Filters

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/adaptive_filter.svg)](https://pypi.org/project/adaptive_filter/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/bglid/adaptive_filter/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/bglid/adaptive_filter/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/bglid/adaptive_filter/releases)
[![License](https://img.shields.io/github/license/bglid/adaptive_filter)](https://github.com/bglid/adaptive_filter/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)



</div>

## LMS, RLS, NLMS, APA, and Frequency Domain Adaptive filters for Adaptive Noise Cancelling.

* ###### Python implementation used as a proof of concept to benchmark performance of different filter algorithms on noisy speech. Created as a final project for a masters course in DSP. Part of a project to implement [embedded adaptive filters using C++ and Bela](https://github.com/bglid/Bela-NLMS-ANC#).

* ###### *Note* - The frequency domain filters need some work. Currently their performance is not adequate for legit use-cases. 

 - - - 
 ## Project Background:
Adaptive Noise Cancellation (ANC) is a research topic that focuses on removing unwanted noise sources from a signal to recover the desired signal in a “less-noisy” form. An area that can be applied to the task of ANC is Adaptive Filters.

The goal of this project was to deepend knowledge of adaptive filters and related DSP topics through implementing various adaptive filter algorithms. In particular, seeing how well they can work with speech, and how well they can work in various settings, such as when needing to rely on ALE because no noise reference is available.

Additionally, out of my own personal interests, I wanted to work on a DSP topic that could be levereged into becoming an embedded project using C++ as well. 

The project has two forms: 
 - *Running the evaluation script* - runs a chosen algorithm over a various noise types, which can be found in the ```data/``` directory of this project. The data itself exists in the ```data/evaluation_data/``` directory, and can be adjusted if you wish to run the evaluation script on your own data, to test how an adaptive filter may perform across larger chunks of data - Just swap out the .wav files for your files! The evaluation script when executed will return a list of performance metrics, and is ideally run from the terminal. For more info, see the *"How to use: Evaluation Script"* listed below.

  - *Using the filters as a tool in your own python script*. The filters are designed to be imported and used as any other module. Considering packaging them is currently under review and a work in progress. However, if you wish to use them, feel free to clone this repo, or ```filter_model.py```, which contains the class definition for normal AFs, as well as the algorithm of your choosing. For an example of these filters in action, see the *"How to use: Filters"* below 

 - - - 
 ## Setup

*Note, requires python 3.9 or greater*

**If using python venv, run:**

```
python -m venv `<your_venv_name>`
```

*Bash/Zsh Activation*

```
#bash/zsh
source <your_venv_name>/bin/activate 
```

*Windows Activation*

```
<your_venv_name>\Scripts\activate.bat
```

**If using conda, run:**

```
conda create --name your_env_name python=3.9
conda activate your_env_name
```
*cloning the repository:*

```
git clone https://github.com/bglid/ANC-adaptive-filters.git
```

**Installing Dependencies:**

 - *Note* - this project uses poetry. For easiest install, it's recommended to install following the instructions below using the [`Makefile`](https://github.com/bglid/adaptive_filter/blob/master/Makefile) commands. For more information on the available commands, see the ```docs/``` directory.

 
[`Makefile`](https://github.com/bglid/adaptive_filter/blob/master/Makefile) commands for install:

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks could be installed after `git init` via

```bash
make pre-commit-install
```
</p>
</details>

 - - -
 ## How to use: Evalulation Script

**Running the Filter Evaluation**

To easily run the filter evaluation bash script, run the script from root, passing the filter algorithm you wish to evaluate. The available algorithms are ```["LMS", "NLMS", "RLS", "APA", "FDLMS", "FDNLMS"]``` *(Case sensitive)*. 

```
# To give permission, if needed
chmod +x ./scripts/run_eval.sh

# Basic example of running the eval script:
./scripts/run_eval.sh LMS

# which should output from the terminal:
-----Starting testing on air_conditioner with the LMS algorithm-----
Algorithm:      LMS
---------------------
params:
mu = 0.001
filter-order = 32
---------------------
...
```
**Changing filter parameters**

 - To change any filter params, such as mu, or to run other algorithms or other tests, pass extra flags in upon calling the script. To see a full list, pass ```-h```. You will need to pass an algorithm before ```-h``` to see the possible flags.
 ```
 # Example running another script, adjusting eval params
./scripts/run_eval.sh APA --block_size=3 --mu=0.005 --ale=True --ale_delay=24

# Which outputs at the terminal:
-----Starting testing on air_conditioner with the APA algorithm-----
Block-based Algorithm:  APA
---------------------
params:
mu = 0.005
filter-order = 32
Block-size = 3
---------------------
...
 
 ```

 - - - 
 ## How to use: Filters
```
from adaptive_filter.algorithms.nlms import NLMS 

# setting up filter
mu = 0.001
nlms_af = NLMS(mu=mu, n=32)

# Assuming signals are already present and named accordingly 
error, noise_estimate = nlms_af.filter(
    d=noisy_signal, x=noise, eval_at_sample=1000, clean_signal=clean_signal, return_metrics=False
)
```

 - - -
## 🛡 License


This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/bglid/adaptive_filter/blob/master/LICENSE) for more details.
 - - -

## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)


Library organization inspired and referenced from ``Padasip``

The MS-SNSD Dataset was used for evaluating filter algorithm performance:
```bibtext
@article{reddy2019scalable,
  title={A Scalable Noisy Speech Dataset and Online Subjective Test Framework},
  author={Reddy, Chandan KA and Beyrami, Ebrahim and Pool, Jamie and Cutler, Ross and Srinivasan, Sriram and Gehrke, Johannes},
  journal={Proc. Interspeech 2019},
  pages={1816--1820},
  year={2019}
}
```

## 📃 Citation
Hey! If you found any of this helpful, feel free to cite it, or just send me a message.
```bibtex
@misc{adaptive_filter,
  author = {bglid},
  title = {Python implementation of DSP adaptive filters},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bglid/adaptive_filter}}
}
```
