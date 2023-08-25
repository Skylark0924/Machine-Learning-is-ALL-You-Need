# Machine-Learning-Basic-Codes🏆

朱子云：

> 所谓致知在格物者，言欲致吾之知，在即物而穷其理也。盖人心之灵，莫不有知，而天下之物，莫不有理。惟于理有未穷，故其知有不尽也。是以大学始教，必使学者即凡天下之物，莫不因其已知之理而益穷之，以求至乎其极。至于用力之久，而一时豁然贯通焉，则众物之表里精粗无不到，而吾心之全体大用无不明矣。

📐📏
> **格物 (Ko Wu) which means 'investigate the essence of things' in English is a key method for study and better understanding of the knowledge.** It is proposed by ancient Chinese philosophers about 2000 years ago and has a profound impact on later generations. The spirit of Ko Wu asks us to not only learn how to use knowledge, but also clearly understand the intrinsic theory. Therefore, it is necessary to re-implement ML algorithms by ourselves to figure out what exactly they did and why they succeed.

This repository aims to implement popular Machine Learning and Deep Learning algorithms by **both pure python and use open-source frameworks**.

- Common Machine Learning Part: switch by **`use_sklearn` flag** in the main function；
- Deep Learning Part: **four** implement methods for each algorithm (`use_sklearn`, `use_keras`, `use_torch` and **`self_implement`**)；
- Applications Part: **RL + NLP + CV**
- New trend: **GNNs**

## Welcome everyone to help me finish this Ko Wu project by pulling requests or giving me some suggestions and issues!!!

## 关联知乎专栏 Associated Zhihu Blog

[RL in Robotics](https://zhuanlan.zhihu.com/c_1188392852261134336)

[Machine Learning 格物志](https://zhuanlan.zhihu.com/c_1236984830903996416)

## 代码目录 Code Catalog

### Regression
1. [Single Linear Regression](./01Single_Linear_Regression/1Single_Linear_Regression.py)
2. [Multiple Linear Regression](./02Multiple_Linear_Regression/2Multiple_Linear_Regression.py)

### Classification
3. [Logistic Regression](./03Logistic_Regression/3Logistic_Regression.py)
4. [KNN](./04K_Nearest_Neighbours/)
5. [Support Vector Machine](./05Support_Vector_Machine/)
6. [Naive Bayes](./06Naive_Bayes/)

### Regression & Classification
7. [Decision Tree](./07Decision_Trees/)
8. [Random Forest](./08Random_Forest/)

### Neural Network
9. [Feedforward Neural Network](./09Neural_Network/)
10. [Convolutional Neural Network](./10CNN/)
11. [LSTM](./11LSTM/)

### Unsupervised Learning
12. [PCA](./12PCA/)
13. [K-Means](./13Kmeans/)

### Ensemble Model
14. [Boosting](./14Boost/)

### Reinforcement Learning
1.  [**Value Based Methods**](./RL_DQN/): [Q-learning(Tabular)](./RL_DQN/Q_learning.py), [DQN](./RL_DQN/15DQN.py)
2.  [**Policy Based Methods**](./RL_PPO/): [Vanilla Policy Gradient](./RL_PPO/vanilla_PG.py), [TRPO](./RL_PPO/TRPO.py), [PPO](./RL_PPO/16PPO.py)
3.  [**Actor-Critic Structure**](./RL_Actor_Critic/): AC, [A2C](./RL_Actor_Critic/17Actor_Critic.py), A3C
4.  [**Deep Deterministic Policy Gradient**](./RL_DDPG): [DDPG](./RL_DDPG/18DDPG.py), [DDPG C++ (Undone)](./RL_DDPG/DDPG_LibTorch-master/), [TD3](./RL_DDPG/TD3.py)
5.  [**Soft Actor-Critic**](./RL_SAC/)

### Computer Vision
1. [ **GAN** ](./CV_GAN/)
2. [**Resnet**](./CV_Resnet/): [Pytorch version](./CV_Resnet/21Resnet.py), [libtorch C++ version](./CV_Resnet/Resnet_libtorch_C++/py_2_C.py)
3. [**VGG**](./CV_VGG/)
4. [**FlowNet**](./CV_FlowNet/)

### Natural Language Processing
1. [Attention mechanism](./NLP_Attention/)
2. [Transformer](./NLP_Transformer/)
3. BERT

### Graph Neural Networks 
1. [Graph Neural Network (GNN)](./Graph_GNN/)
2. Graph Convolutional Neural Network (GCN)
3. Graph Attention Networks (GAT)
4. GraphSAGE
5. GraphRNN
6. Variational Graph Auto-Encoders (GAE)

---

If you're interested in reinforcement learning, we encourage you to check out our latest library of reinforcement learning and imitation learning in (humanoid) robotics.

[![](img/image-20230825121432059.png)](https://github.com/Skylark0924/Rofunc)

[![Release](https://img.shields.io/github/v/release/Skylark0924/Rofunc)](https://pypi.org/project/rofunc/)
![License](https://img.shields.io/github/license/Skylark0924/Rofunc?color=blue)
![](https://img.shields.io/github/downloads/skylark0924/Rofunc/total)
[![](https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc?color=brightgreen)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed)
[![](https://img.shields.io/github/issues-raw/Skylark0924/Rofunc?color=orange)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue)
[![Documentation Status](https://readthedocs.org/projects/rofunc/badge/?version=latest)](https://rofunc.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat)](https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main)

> **Repository address: https://github.com/Skylark0924/Rofunc**
