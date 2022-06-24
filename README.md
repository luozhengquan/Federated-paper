# 最新联邦学习论文梳理
（简单梳理不一定全，个人理解不一定准。仅供参考欢迎交流）
## CVPR2022
### Oral
1、Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedAlign_title.PNG)

[code](https://github.com/mmendiet/FedAlign)

#### 文章总结
联邦学习客户端之间数据异质性问题，导致客户端模型偏离理想的全局优化点并过度拟合到局部目标。直截了当的通过缩减局部训练轮次确实可以有效改善（原因是局部梯度减小，FedAVG当局部轮次为1时优化公式与数据集中式严格一致），但是严重阻碍收敛速度导致巨大收敛时间和通信开销。通过增加近端项（proximal terms）限制相对于全局模型的局部更新，虽然有效抑制了漂移但是也限制了局部收敛的潜力，同时减少了每轮通信所能收集的信息。

针对联邦数据异质性问题，核心在于提升模型的泛化性从而使得局部优化目标接近于全局目标。本文重新审视了数据和结构正则化方法在减少客户端漂移和提高FL性能方面的有效性，同时参考分布外泛化OOD理论以确定成功的FL优化的理论指标。另一方面，针对目前正则化可能在局部产生的大量资源开销问题，本文提出了基于蒸馏的正则化方法FedAlign，在促进局部泛化性的同时保持出色的资源效率。

FedAlign专注于对网络中最终块的 Lipschitz 常数的表示进行正则化，由于只关注最后一个块，所以可以在有效地规范网络中最容易过度拟合部分的同时将额外的资源需求保持在最低限度。

#### 理论发现：

A、最近NAS和网络泛化方面的研究表示top Hessian特征值和Hessian trace可作为性能预测和网络通用性的评价指标，越小越好。而正则化方法在降低这两个指标方面最为有效特别是GradAug方法。

B、分布外泛化领域揭示了学习模型的跨域损失彼此一致时致使更好的OOD泛化，因此跨客户之间匹配 Hessians 的范数和方向可评估模型泛化性能

#### 实验验证：

A、在不同数据异质性条件下，结构正则化方法的性能优于标准FL算法。即使纯同质条件下，FedProx和MOON似乎近端项优化方向抑制方法会阻碍模型充分学习少量异构甚至同质数据的能力。

B、随着局部训练轮次增加，近端项方法遭遇了性能瓶颈，模型性能并没有随着更多局部训练而进一步提升。另一方面正则化方法的准确性不断提高，保持了高效训练的能力。

C、客户数量增加或者每轮训练仅采样部分客户进行优化，标准正则化方法在所有设置中都比 FedAvg 保持更好的准确性。

#### FedAlign

网络结构:

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedAlign_framework.PNG)

算法流程：

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedAlign_algorithm.PNG)

Objective：

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedAlign_Objective.PNG)

关键原则：

A、在内部对网络块的Liptschitz常数进行了正则化以促进模型内的平滑优化和一致性。

B、重用整个网络的中间特征作为最终块的输入以减小宽度，从而显着减少计算量。

FedAlign与传统的 FL 算法的关键区别在于，新的修正项促使本地客户端模型根据自己的数据学习良好的泛化表示，而不是强制本地模型接近全局模型。

实验效果：

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedAlign_performance.PNG)

#### 个人观点（仅供参考）

联邦学习数据端分布异质性问题->表象原因在于客户端模型优化方向不一致。

FedProx，MOON通过引入近端项抑制局部优化方向不偏离全局优化，但是带来了大量的计算开销和局部优化限制。

作者对于问题关键理解在于FL需要的是模型泛化性能提升，从而使得局部优化和全局优化从本质上得以靠近。

为此，作者通过引入top Hessian特征值，Hessian trace和跨客户端Hessians的范数和方向匹配等评价指标，评估网络泛化能力。大量实验表明（结构）正则化方法相比于近端项在泛化性上更有优势。

从而结合GradAug方法提出FedAlgin，并针对联邦计算场景下边缘设备资源有限的问题进行了计算量优化。

（核心：从近端项强制收束局部全局模型优化，转变为更好的本地模型泛化性）


### Poster
1、ATPFL: Automatic Trajectory Prediction Model Design Under Federated Learning Framework 

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22ATPFL_title.PNG)

#### 文章总结
作者通过分析和总结现有工作，通过总结TP模型的设计过程，收集了每个步骤的可用操作，并通过分析现有的 TP 工作确定了每个操作的局限性。将上述经验和知识整合到关系图中，从而为 TP 区域构建有效的搜索空间。此外，考虑到复杂的约束关系，操作之间的时间关系和技术连接作者设计了一种关系序列感知的搜索策略，该策略可以利用构建的关系图、图神经网络（GNN）结合递归神经网络（RNN）来学习所选操作序列的高级特征，从而为后续步骤的设计提供有效参考，实现了TP模型的自动设计，并通过合适的联合训练方法成功将联邦学习应用于轨迹预测（TP），通过充分利用具有丰富实际场景的分布式多源数据集来学习更强大的TP模型，实验结果表明ATPFL比单源数据集上训练的 TP 模型可以取得更好的结果。

#### 个人观点（仅供参考）
AutoML+FL应用型工作，FL部分为FedAvg, PerFedAvg, 和pFedMe的应用。



2、Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22APTFL_title.PNG)

#### 文章总结
作者提出了一种新的FL隐私泄露，即生成梯度泄漏（GGL）泄漏。方法利用从公共图像数据集中学习的生成对抗网络（GAN）的潜在空间作为先验来补偿梯度退化期间的信息损失。以对利用抗梯度信息退化进行FL隐私保护的防御机制，例如交互之前使用加性噪声或梯度压缩。本文实验性的说明了即使在某些防御设置下，从共享梯度中恢复高保真图像仍然是可行的。大型公共图像数据集学习的生成对抗网络 (GAN)的流形作为先验信息，提供了对自然图像空间的良好逼近。通过最小化 GAN 图像流形中的梯度匹配损失，该方法可以找到与客户的高质量私人训练数据高度相似的图像。


[Code](https://github.com/zhuohangli/GGL)

#### 个人观点（仅供参考）
针对使用梯度退化防御的FL进行隐私泄露攻击，本质就是建模梯度退化的求逆过程。



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
