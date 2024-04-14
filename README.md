# Enhanced Language-Guided World Models: A Model-Based Approach to AI Control\nThis repository contains the code necessary for running our project's experiments. Here, we propose the concept of *Enhanced Language-Guided World Models* (LWMs). These models are designed to comprehend environment dynamics by reading language descriptions. There are two main phases of LWM learning - one being the learning of the language-guided world model by exploring the environment, the other being model-based policy learning via imitation learning or behavior cloning.\n\nTo learn more, visit [the project's website](https://language-guided-world-model.github.io/).\n\n![Example of LWM](teaser.gif)\n\n## Getting Started: Setup\n\nTo get started, create a conda environment\n\n```
conda create -n lwm python=3.9 && conda activate lwm
```
Next, install the necessary dependencies through pip:\n\n```
pip install -r requirements.txt
```
\nFeel free to download the trajectory dataset from [this link](https://drive.google.com/file/d/1vUwP4EzBMrmDZWhmnXFie2woWbGzjKUq/view?usp=sharing), unzip the file (approximately 7 GB), and place the .pickle file inside `world_model/custom_dataset`\n\nPre-trained model checkpoints are available. Please follow the sections for various steps of the training process (train WM, then train policy). We offer some checkpoints for both the trained world model, as well as the expert EMMA policy. This allows for skipping of world model training. These checkpoints were utilized in the experiments described in the paper. Download the checkpoints [at this link](https://drive.google.com/file/d/1YiQyjeInXqztyffAbZ-8SDsnS2II8HUh/view?usp=sharing) and place them in an appropriate folder. \n\n## Training the World Model\n\nInitially change directory into world_model/\n
```
cd world_model
```
\nSubsequently, you can train the world model using the following bash script\n
```
bash scripts/train_wm.sh ${MODEL_NAME}
```
\nYou can replace `${MODEL_NAME}` with either `none` (observational, doesn't use language), `standardv2` (standard Transformer), `direct` (GPT-hard attention), `emma` (our proposed EMMA-LWM model), or `oracle` (oracle semantic-parsing). The script will generate a folder in `experiments/` containing model checkpoints. Full results are in Table 1 of the paper.\n\nTo interact with a trained world model, run:\n
```
bash scripts/play_wm.sh ${MODEL_NAME}
```
\n\nTo modify the game for visualization, change the `game_id` in `play_wm.py`. If you define a different seed for training the world model, make sure to define the same seed when playing (hard-coded in the current setup).\n\n## Downstream Policy Learning\n\n**Note:** *An expert policy is needed for filtered behavior cloning. Refer to the paper for more details.* Make sure you are in the `world_model/` directory. First, train an expert policy and save its checkpoints:\n\n
```
bash scripts/train_emma_policy.sh
```
\nPost learning a language-guided world model (as described in '[Training the World Model](#training-the-world-model)'), apply it to downstream policy learning on Messenger:\n

```
bash scripts/train_downstream.sh ${TRAIN_TASK} ${MODEL_NAME} ${SPLIT} ${GAME}
```
\n-${TRAIN_TASK} can be either `imitation` (Imitation Learning) or `filtered_bc` (Filtered Behavior cloning).\n-${MODEL_NAME} indicates any world model from [Training the World Model](#training-the-world-model) section that has been trained.\n-${SPLIT} specifies the difficulty of the task which could be `easy` (NewCombo), `medium` (NewAttr), or `hard`(NewAll).\n- ${GAME} is the game id on MESSENGER to evaluate on. It ranges from 0-29.\n-${ORACLE_CKPT} can be `half`, which sets the oracle weights to a training checkpoint where it was roughly halfway trained, or it could be any other string that sets it to the best oracle world model checkpoint on the hardest validation split.\n\nThis script will generate a folder in `experiments/` containing model checkpoints.\n\nCredits:\n- https://github.com/eloialonso/iris\n- https://github.com/karpathy/minGPT\n\nYou can cite our work as follows:\n\n```
@article{zhang2024languageguided
  title={Language-Guided World Models: A Model-Based Approach to AI Control},
  author={Zhang, Alex and Nguyen, Khanh and Tuyls, Jens and Lin, Albert and Narasimhan, Karthik},
  year={2024},
  journal={arXiv},
}
```