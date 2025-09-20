# MotionTrans-Pi0

This repository contains the Pi0-VLA code for [MotionTrans: Human VR Data Enable Motion-Level Learning for Robotic Manipulation Policies](https://github.com/michaelyuancb/motiontrans), which is a modification of the [original Pi0 codebase](https://github.com/Physical-Intelligence/openpi) from [Physics Intelligence](https://www.physicalintelligence.company/).

For more details of the whole project, please refer to the [main MotionTrans repository](https://github.com/michaelyuancb/motiontrans).

## Installation

Please follow the installation instructions in the original [Pi0 repository](https://github.com/Physical-Intelligence/openpi). 

## Data Preparation

Please follow the data preparation instructions in the original [MotionTrans repository](https://github.com/michaelyuancb/motiontrans). All data should be processed as `.zarr` files for Pi0-VLA training.

## Training and Evaluation

```
bash scripts_exp/train_cotrain.sh
bash scripts_exp/eval.sh
```

## Inference

First change the policy path in `Line 89` of `scripts\serve_policy.py`, and then run:
```
bash scripts_exp/serve_policy.sh
```
This will launch a web server in local. You can then open the client following the instructions in [MotionTrans repository](https://github.com/michaelyuancb/motiontrans) to use Pi0-VLA to control the real robot. 


## Acknowledgment

We thanks [Ruiqian Nai](https://richard-coder-nai.github.io/) and [Fanqi Lin](https://fanqi-lin.github.io/) for their great help on the development of this MotionTrans-Pi0-VLA codebase !

This repository is based on the code from [OneTwoVLA](hhttps://github.com/Fanqi-Lin/OneTwoVLA), [OpenPi](https://github.com/Physical-Intelligence/openpi) and[UMI](https://github.com/real-stanford/universal_manipulation_interface). We sincerely appreciate their contribution to the open-source community, which have significantly supported this project. We also sincerely thank our AI-collaborators [ChatGPT](https://openai.com/chatgpt), [Kimi](https://www.kimi.com/) and [Github Copilot](https://github.com/features/copilot) !!

## Citation

If you find this repository useful, please kindly acknowledge our work and cite the paper in our [main MotionTrans repository](https://github.com/michaelyuancb/motiontrans).