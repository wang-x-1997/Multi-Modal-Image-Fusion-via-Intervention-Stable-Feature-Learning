Source code of the paper ***"Multi-Modal Image Fusion via Intervention-Stable Feature Learning"*** which has been accepted by CVPR 2026.
- Xue Wang, Zheng Guan, Wenhua Qian, Chengchao Wang, RunZhuo Ma

# Abstract
Multi-modal image fusion integrates complementary information from different modalities into a unified representation. Current methods predominantly optimize statistical correlations between modalities, often capturing dataset-induced spurious associations that degrade under distribution shifts. In this paper, we propose an intervention-based framework inspired by causal principles to identify robust cross-modal dependencies. Drawing insights from Pearl's causal hierarchy, we design three principled intervention strategies to probe different aspects of modal relationships: i) complementary masking with spatially disjoint perturbations tests whether modalities can genuinely compensate for each other's missing information, ii) random masking of identical regions identifies feature subsets that remain informative under partial observability, and iii) modality dropout evaluates the irreplaceable contribution of each modality. Based on these interventions, we introduce a Causal Feature Integrator (CFI) that learns to identify and prioritize intervention-stable features maintaining importance across different perturbation patterns through adaptive invariance gating, thereby capturing robust modal dependencies rather than spurious correlations. Extensive experiments demonstrate that our method achieves SOTA performance on both public benchmarks and downstream high-level vision tasks. 

| ![The framework of ISFuse](image/1.png) |
|:-------------------------------------------:|
| **Figure 1.**  Comparison with SOTA method in training framework and performance |

# :triangular_flag_on_post: Testing
If you want to infer with our ISFuse and obtain the fusion results in our paper, please run ```test.py```.
Then, the fused results will be saved in the ```'./Fused/'``` folder.

# :triangular_flag_on_post: Training
You can modify the data path in ```train.py``` to retrain the model with your own dataset.
