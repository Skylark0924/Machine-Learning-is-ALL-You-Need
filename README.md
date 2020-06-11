# Machine-Learning-Basic-Codes🏆

朱子云：

> 所谓致知在格物者，言欲致吾之知，在即物而穷其理也。盖人心之灵，莫不有知，而天下之物，莫不有理。惟于理有未穷，故其知有不尽也。是以大学始教，必使学者即凡天下之物，莫不因其已知之理而益穷之，以求至乎其极。至于用力之久，而一时豁然贯通焉，则众物之表里精粗无不到，而吾心之全体大用无不明矣。

📐📏

本代码库通过 sklearn 标准机器学习库以及自己编写的类两种方式，实现了基本的机器学习算法。

- Common Machine Learning Part: 通过更改主函数中的 **use_sklearn flag** 即可切换；

- Deep Learning Part: **use_sklearn, use_keras, use_torch以及 Self-implement 四种实现方式**；

- Applications Part: **RL + NLP + CV**

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

### Natural Language Processing
1. Transformer

2. BERT
