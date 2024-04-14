# Enhanced Language-Guided World Models: A Model-Based Approach to AI Control\nThis repository contains the code necessary for running our project's experiments. Here, we propose the concept of *Enhanced Language-Guided World Models* (LWMs). These models are designed to comprehend environment dynamics by reading language descriptions. There are two main phases of LWM learning - one being the learning of the language-guided world model by exploring the environment, the other being model-based policy learning via imitation learning or behavior cloning.\n\nTo learn more, visit [the project's website](https://language-guided-world-model.github.io/).\n\n![Example of LWM](teaser.gif)\n\n## Getting Started: Setup\n\nTo get started, create a conda environment\n\n```
conda create -n lwm python=3.9 && conda activate lwm
```
Next, install the necessary dependencies through pip:\n\n```
pip install -r requirements.txt
```
\nFeel free to download the 