# 最新联邦学习论文梳理
（简单梳理不一定全，个人理解不一定准。仅供参考欢迎交流）

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#CVPR2022">CVPR2022</a></li>
    <li><a href="#ICML2022">ICML2022</a></li>
    <li><a href="#ICLR2022">ICLR2022</a></li>
    <li><a href="#AAAI2022">AAAI2022</a></li>
  </ol>
</details>

## CVPR2022
### Oral
## 1、Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning
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
## 1、ATPFL: Automatic Trajectory Prediction Model Design Under Federated Learning Framework 

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22ATPFL_title.PNG)

#### 文章总结
作者通过分析和总结现有工作，通过总结TP模型的设计过程，收集了每个步骤的可用操作，并通过分析现有的 TP 工作确定了每个操作的局限性。将上述经验和知识整合到关系图中，从而为 TP 区域构建有效的搜索空间。此外，考虑到复杂的约束关系，操作之间的时间关系和技术连接作者设计了一种关系序列感知的搜索策略，该策略可以利用构建的关系图、图神经网络（GNN）结合递归神经网络（RNN）来学习所选操作序列的高级特征，从而为后续步骤的设计提供有效参考，实现了TP模型的自动设计，并通过合适的联合训练方法成功将联邦学习应用于轨迹预测（TP），通过充分利用具有丰富实际场景的分布式多源数据集来学习更强大的TP模型，实验结果表明ATPFL比单源数据集上训练的 TP 模型可以取得更好的结果。

AutoML+FL应用型工作，FL部分为FedAvg, PerFedAvg, 和pFedMe的应用。



## 2、Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22APTFL_title.PNG)

#### 文章总结
作者提出了一种新的FL隐私泄露，即生成梯度泄漏（GGL）泄漏。方法利用从公共图像数据集中学习的生成对抗网络（GAN）的潜在空间作为先验来补偿梯度退化期间的信息损失。以对利用抗梯度信息退化进行FL隐私保护的防御机制，例如交互之前使用加性噪声或梯度压缩。本文实验性的说明了即使在某些防御设置下，从共享梯度中恢复高保真图像仍然是可行的。大型公共图像数据集学习的生成对抗网络 (GAN)的流形作为先验信息，提供了对自然图像空间的良好逼近。通过最小化 GAN 图像流形中的梯度匹配损失，该方法可以找到与客户的高质量私人训练数据高度相似的图像。


[Code](https://github.com/zhuohangli/GGL)




## 3、CD^2-pFed: Cyclic Distillation-Guided Channel Decoupling for Model Personalization in Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22cd2-pfed_title.PNG)

#### 文章总结
个性化 FL方案以解决联邦数据分布异质性问题，本文提出了一种新颖的循环蒸馏引导的通道解耦框架，与之前建立分层个性化以克服跨客户端的非独立同分布数据的工作不同，本文尝试对模型通道进行个性化分配为私有和共享权重。为了进一步促进私有权重和共享权重之间的协作，本文提出了循环蒸馏在局部和和全局模型表示之间施加一致的正则化。在其指导之下所提出的通道解耦框架可以为不同类型的异质性提供更准确和通用的结果，例如特征偏斜、标签分布偏斜和概念转移。
网络结构:

![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22cd2-pfed_framework.PNG)


## 4、Closing the Generalization Gap of Cross-Silo Federated Medical Image Segmentation
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22CGG_title.PNG)
#### 文章总结
NVIDIA的工作，本文针对FL在医学图像分割任务中遇到客户端漂移问题，提出了FedSM框架生成个性化模型以很好地适应不同的数据分布，并生成一种新颖的模型选择器来为任何测试数据确定最接近的模型/数据分布，而不是找到一个适合所有客户数据分布的全局模型。



## 5、Differentially Private Federated Learning With Local Regularization and Sparsification
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22DPFLLRS_title.PNG)
#### 文章总结

用户级差分隐私 (DP)用于FL隐私保证是以严重降低精度为代价的，现有DP方法主要建立在基于高斯噪声扰动机制之上，具体来说，高斯机制需要将局部更新的l2 幅值限制到灵敏度阈值 S，并将与 S 成比例的噪声添加到高维局部更新中，这两个操作会导致较大的偏差（当 S 较小时）或较大的方差（当 S 较大时），从而减慢收敛速度并损害全局模型的性能。为此，作者提出了两种技术：有界局部更新正则化和局部更新稀疏化，以在不牺牲隐私的情况下提高模型质量。动机是在裁剪之前自然地减少局部更新的 l2 范数，从而使局部更新更适应裁剪操作，其中有界局部更新正则化（BLUR）为局部目标函数引入了一个正则化项，并明确地将局部更新的 l2 范数规范化为有界。局部更新稀疏化（LUS）将一些对局部模型性能影响不大的更新值归零，从而在不损害局部模型精度的情况下进一步降低局部更新的范数。作者对框架的收敛性进行了理论分析，并给出了严格的隐私保证，同时通过大量实验表明该框架显着改善了隐私效用权衡。

## 6、FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedCor_title.PNG)
#### 文章总结
从主动客户端选择思路解决联邦数据异质性问题，相比于统一选择策略只能取得边际改进不同，本文提出FedCor基于相关性的客户选择策略FL框架，具体而言首先使用高斯过程 (GP) 对客户端之间的损失相关性进行建模，基于GP导出客户端选择策略显着降低了每一轮的预期全局损失。



## 7、FedCorr: Multi-Stage Federated Learning for Label Noise Correction
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedCorr_title.PNG)
#### 文章总结
本文针对FL设置下存在的异构标签噪声问题（客户端数据可能具有标签噪声，并且不同的客户端可能具有截然不同的标签噪声水平。）提出了FedCorr多阶段框架通过利用在所有客户端上独立测量的模型预测子空间的维度来动态识别噪声客户端，然后根据每个样本的损失识别噪声客户端上的错误标签。为了处理数据异质性并提高训练稳定性，本文提出了一种基于估计局部噪声水平的自适应局部近端正则化项。全局模型在干净客户端上进一步微调，并为剩余的噪声客户端校正噪声标签。最后在进行全部客户端的联合FL训练以充分利用所有本地数据。

## 8、FedDC: Federated Learning With Non-IID Data via Local Drift Decoupling and Correction
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FedDC_title.PNG)
#### 文章总结
针对FL客户端数据分布异质性问题，提出了具有局部漂移解耦和校正的新型联邦学习算法FedDC，通过引入漂移变量将局部模型和全局模型在训练过程中解耦，减少了局部漂移对全局目标的影响，使模型快速收敛并达到更好的性能。实验结果和分析表明FedDC 在各种图像分类任务上获得了加速收敛和更好的性能。

## 9、Federated Class-Incremental Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FCIL_title.PNG)
#### 文章总结
本文针对FL存在的灾难性遗忘问题，目前大多数FL框架假设整个框架的对象类随着时间的推移是固定的，然而实际场景中局部客户端不断收集的新类并且仅有限的存储旧类，使得全局模型遭受对旧类的灾难性遗忘。为应对这一问题本文开发了一种新的全局-局部遗忘补偿 (GLFC) 模型，以学习一个全局类增量模型，以从局部和全局角度减轻灾难性遗忘。具体而言为了解决由于本地客户端的类不平衡导致的本地遗忘，本文设计了一个类感知梯度补偿损失和一个类语义关系蒸馏损失来平衡旧类的遗忘并提炼跨任务的一致的类间关系。为解决客户端之间数据异质性造成的全局遗忘，本文提出了一个代理服务器，它选择最好的旧的全局模型来辅助局部关系蒸馏。

## 10、Federated Learning With Position-Aware Neurons
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FLPAN_title.PNG)
#### 文章总结
神经网络具备置换不变性，这意味着隐藏的神经元可以在训练过程中错位而不影响局部性能。另一方面，FL跨客户端的样本是非独立同分布的，其进一步加剧了局部训练期间模型神经网络错位并导致基于坐标的参数平均权重发散。为此本文提出位置感知神经元 (PAN)将位置编码融合到神经元输出中并最大限度地减少错位的可能性。PAN 在应用于 FL 时与位置紧密耦合，使跨客户端的参数预先对齐并促进基于坐标的参数平均。


## 11、Fine-Tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22ftgm_title.PNG)
#### 文章总结
针对FL客户端数据异质性问题，本文提出了一种无数据知识提取方法来微调服务器中的全局模型（FedFTG），从而缓解了直接全局模型聚合引起的性能下降。具体来说，服务器在收集本地模型并将它们聚合为初步的全局模型后并不直接广播回每个客户端，而是使用从本地模型中提取的知识在服务器中微调这个初步的全局模型。为此本文开发了一种基于硬样本挖掘的无数据知识蒸馏方法，以有效地探索知识并将其转移到全局模型中。同时考虑到客户标签分布的变化，文章建议定制标签采样和类级集成，以促进更有效的知识利用。

## 12、Layer-Wised Model Aggregation for Personalized Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22LMAPFL_title.PNG)
#### 文章总结
目前大多数个性化联合学习应用加权聚合的方式来生成个性化模型，其中权重是通过校准整个模型参数或损失值的距离来确定的，尚未考虑层级对聚合过程的影响，导致模型收敛滞后和非 IID 数据集的个性化不足。为此，本文提出了pFedLA在服务器端为每个客户采用了一个专用的超网，该网络经过训练可以在层的粒度上识别相互贡献因素。同时，文章引入了一个参数化的机制来更新层化的聚合权重，以逐步利用客户端之间的相似性并实现精确的模型个性化。


## 13、Learn From Others and Be Yourself in Heterogeneous Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22FCCL_title.PNG)
#### 文章总结
针对FL数据异质性问题以及局部优化过程可能造成的模型灾难性遗忘问题，本文提出了FCCL联邦互相关和持续学习，利用未标记的公共数据进行通信并构建互相关矩阵来学习域转移下的可概括表示，同时在本地更新中利用知识蒸馏，在不泄露隐私的情况下提供域间和域内信息。

## 14、ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22ResSFL_title.PNG)
#### 文章总结
本文尝试克服分离式联合学习（SFL）的模型反转（MI）攻击，SFL（splitfed learning）是一个最新的分布式训练方案，网络被分割成客户端部分和服务器部分，其中多个客户端将中间激活（即特征图smashed data）而不是原始数据，发送到中心服务器，经由中心服务器网络前传反传再将对应梯度发回客户端，而FedAvg部分应用于客户端模型聚合。SFL目前框架无法应对训练期间的模型反转攻击，对此本文提出了ResSFL可在训练期间抵抗MI，具体做法通过攻击者感知的训练得出一个抗性特征提取器，并在标准的SFL训练之前使用该提取器初始化客户端模型。这种方法有助于降低由于在客户端对抗性训练中使用强反转模型而产生的计算复杂性，以及在早期训练时代发起的攻击的脆弱性。

## 15、Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22RADFL_title.PNG)
#### 文章总结
针对FL数据和设备异构性可能导致的灾难性遗忘，本文证明了self-attention-based架构（Transformer）对于分布变化更加稳健同时在缓解灾难性遗忘、加速收敛以及达到并行和串行FL方法的最佳优化方面的优势。

## 16、Robust Federated Learning With Noisy and Heterogeneous Clients
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22RFL_title.PNG)
#### 文章总结
这项工作针对客户端存在标签噪声问题，提出了RHFL（稳健的异质联合学习），对于异构模型之间的通信通过利用公共数据直接调整模型的反馈，对于客户端内部标签噪声应用一个稳健的耐噪声损失函数来减少其负面影响。同时设计了一个新颖的客户信心再加权方案，该方案在协作学习阶段自适应地给每个客户标上相应的权重。

## 17、RSCFed: Random Sampling Consensus Federated Semi-Supervised Learning
![](https://github.com/luozhengquan/Federated-paper/blob/main/image/CVPR22RSCFed_title.PNG)
#### 文章总结
联邦半监督学习（FSSL）通过训练完全标记、完全未标记、或部分标记的客户来获得一个全局模型，针对FSSL同样面对的数据非独立同分布问题，本文提出了一种随机抽样共识联合学习RSCFed，由于训练模型在有标签的客户和无标签的客户之间具有较大的偏差，直接汇总局部模型并不合理，具体做法首先通过对客户进行随机子抽样，提炼出几个子共识模型，然后将这些子共识模型汇总到全局模型中，为了进一步提高子共识模型的稳健性本文使用了基于距离加权的模型聚合方法。




## ICML2022
### Oral
## 1、Anarchic Federated Learning
## 2、FEDNEST: Federated Bilevel Optimization
## 3、The Poisson Binomial Mechanism for Unbiased Federated Learning with Secure Aggregation
## 4、Federated Reinforcement Learning: Communication-Efficient Algorithms and Convergence Analysis

### Spotlight
## 1、QSFL: A Two-Level Uplink Communication Optimization Framework for Federated Learning
## 2、Bitwidth Heterogeneous Federated Learning with Progressive Weight Dequantization
## 3、Multi-Level Branched Regularization for Federated Learning
## 4、Personalized Federated Learning via Variational Bayesian Inference
## 5、Disentangled Federated Learning for Tackling Attributes Skew via Invariant Aggregation and Diversity Transferring
## 6、Neurotoxin: Durable Backdoors in Federated Learning
## 7、ProgFed: Effective, Communication, and Computation Efficient Federated Learning by Progressive Training
## 8、FLPerf: A Comprehensive Benchmark for Federated Learning at Scale
## 9、Federated Learning with Positive and Unlabeled Data
## 10、Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
## 11、Federated Minimax Optimization: Improved Convergence Analyses and Algorithms
## 12、Proximal and Federated Random Reshuffling
## 13、Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning
## 14、FedNL: Making Newton-Type Methods Applicable to Federated Learning
## 15、Fast Composite Optimization and Statistical Recovery in Federated Learning
## 16、EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning
## 17、Understanding Clipping for Federated Learning: Convergence and Client-Level Differential Privacy
## 18、DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning
## 19、Federated Learning with Label Distribution Skew via Logits Calibration
## 20、FedNew: A Communication-Efficient and Privacy-Preserving Newton-Type Method for Federated Learning
## 21、Generalized Federated Learning via Sharpness Aware Minimization
## 22、Federated Learning with Partial Model Personalization
## 23、The Fundamental Price of Secure Aggregation in Differentially Private Federated Learning
## 24、DisPFL: Towards Communication-Efficient Personalized Federated learning via Decentralized Sparse Training
## 25、Personalized Federated Learning through Local Memorization
## 26、Resilient and Communication Efficient Learning for Heterogeneous Federated Systems
## 27、Architecture Agnostic Federated Learning for Neural Networks
## 28、Personalization Improves Privacy-Accuracy Tradeoffs in Federated Optimization
## 29、Neural Tangent Kernel Empowered Federated Learning
## 30、Accelerated Federated Learning with Decoupled Adaptive Optimization
## 31、Communication-Efficient Adaptive Federated Learning
## 32、Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification
## 33、The Poisson Binomial Mechanism for Unbiased Federated Learning with Secure Aggregation
## 34、ProxSkip: A Simple and Provably Effective Communication-Acceleration Technique for Federated Learning


## ICLR2022
### Oral
## 1、Minibatch vs Local SGD with Shuffling: Tight Convergence Bounds and Beyond 

### Spotlight
## 1、On Bridging Generic and Personalized Federated Learning for Image Classification 
## 2、Hybrid Local SGD for Federated Learning with Heterogeneous Communications 
## 3、Improving Federated Learning Face Recognition via Privacy-Agnostic Clusters 

### Poster 
## 1、FedBABU: Toward Enhanced Representation for Federated Image Classification 
## 2、What Do We Mean by Generalization in Federated Learning? 
## 3、Towards Model Agnostic Federated Learning Using Knowledge Distillation 
## 4、Divergence-aware Federated Self-Supervised Learning 
## 5、An Agnostic Approach to Federated Learning with Class Imbalance 
## 6、Recycling Model Updates in Federated Learning: Are Gradient Subspaces Low-Rank? 
## 7、Diverse Client Selection for Federated Learning via Submodular Maximization 
## 8、ZeroFL: Efficient On-Device Training for Federated Learning with Local Sparsity 
## 9、Diurnal or Nocturnal? Federated Learning of Multi-branch Networks from Periodically Shifting Distributions 
## 10、Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models 
## 11、Efficient Split-Mix Federated Learning for On-Demand and In-Situ Customization 
## 12、Acceleration of Federated Learning with Alleviated Forgetting in Local Training 
## 13、FedPara: Low-rank Hadamard Product for Communication-Efficient Federated Learning 
## 14、FedChain: Chained Algorithms for Near-optimal Communication Cost in Federated Learning 
## 15、Bayesian Framework for Gradient Leakage 
## 16、Federated Learning from Only Unlabeled Data with Class-conditional-sharing Clients 

## AAAI2022
## 1、HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images
## 2、Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning
## 3、Federated Learning for Face Recognition with Gradient Correction
## 4、SpreadGNN: Decentralized Multi-Task Federated Learning for Graph Neural Networks on Molecular Data
## 5、SmartIdx: Reducing Communication Cost in Federated Learning by Exploiting the CNNs Structures
## 6、Is Your Data Relevant?: Dynamic Selection of Relevant Data for Federated Learning
## 7、Critical Learning Periods in Federated Learning
## 8、Coordinating Momenta for Cross-silo Federated Learning
## 9、FedProto: Federated Prototype Learning across Heterogeneous Clients
## 10、FedInv : Byzantine-robust Federated Learning by Inversing Local Model Updates
## 11、FedSoft: Soft Clustered Federated Learning with Proximal Local Updating
## 12、Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better
## 13、FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition
## 14、SplitFed: When Federated Learning Meets Split Learning
## 15、Efficient Device Scheduling with Multi-Job Federated Learning
## 16、Implicit Gradient Alignment in Distributed and Federated Learning
## 17、Learning Advanced Client Selection Strategy for Federated Learning
## 18、Federated Nearest Neighbor Classification with a Colony of Fruit-Flies: With Supplement
## 19、Preserving Privacy in Federated Learning with Ensemble Cross-Domain Knowledge Distillation
## 20、CrowdFL : A Marketplace for Crowdsourced Federated Learning
## 21、Contribution-Aware Federated Learning for Smart Healthcare
## 22、Threats to Federated Learning
## 23、Asynchronous Federated Optimization
## 24、Preservation of Global Knowledge by Not-True Distillation in Federated Learning
