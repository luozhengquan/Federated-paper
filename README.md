# 最新联邦学习论文梳理
（简单梳理不一定全，个人理解不一定准。仅供参考欢迎交流）
## CVPR2022
### Oral
1、Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning
联邦学习客户端之间数据异质性问题，导致客户端模型偏离理想的全局优化点并过度拟合到局部目标。直截了当的通过缩减局部训练轮次确实可以有效改善（原因是局部梯度减小，FedAVG当局部轮次为1时优化公式与数据集中式严格一致），但是严重阻碍收敛速度导致巨大收敛时间和通信开销。通过增加近端项（proximal terms）限制相对于全局模型的局部更新，虽然有效抑制了漂移但是也限制了局部收敛的潜力，同时减少了每轮通信所能收集的信息。
针对联邦数据异质性问题，核心在于提升模型的泛化性从而使得局部优化目标接近于全局目标。本文重新审视了数据和结构正则化方法在减少客户端漂移和提高FL性能方面的有效性，同时参考分布外泛化OOD理论以确定成功的FL优化的理论指标。另一方面，针对目前正则化可能在局部产生的大量资源开销问题，本文提出了基于蒸馏的正则化方法FedAlign，在促进局部泛化性的同时保持出色的资源效率。
FedAlign专注于对网络中最终块的 Lipschitz 常数的表示进行正则化，由于只关注最后一个块，所以可以在有效地规范网络中最容易过度拟合部分的同时将额外的资源需求保持在最低限度。


### Poster
1、ATPFL: Automatic Trajectory Prediction Model Design Under Federated Learning Framework 

2、Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage

3、CD2-pFed: Cyclic Distillation-Guided Channel Decoupling for Model Personalization in Federated Learning

4、Closing the Generalization Gap of Cross-Silo Federated Medical Image Segmentation

5、Differentially Private Federated Learning With Local Regularization and Sparsification

6、FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning

7、FedCorr: Multi-Stage Federated Learning for Label Noise Correction

8、FedDC: Federated Learning With Non-IID Data via Local Drift Decoupling and Correction

9、Federated Class-Incremental Learning

10、Federated Learning With Position-Aware Neurons

11、Fine-Tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning

12、Layer-Wised Model Aggregation for Personalized Federated Learning

13、Learn From Others and Be Yourself in Heterogeneous Federated Learning

14、ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning

15、Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning

16、Robust Federated Learning With Noisy and Heterogeneous Clients

17、RSCFed: Random Sampling Consensus Federated Semi-Supervised Learning
