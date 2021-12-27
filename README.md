# Entropy-Fuzzy-Support-Vector-Machine

This repo is the implementation of EFSVM (Entropy Fuzzy Support Vector Machine)

Reference:

[1] Qi Fan, Zhe Wang, Dongdong Li, Daqi Gao, Hongyuan Zha,
Entropy-based fuzzy support vector machine for imbalanced datasets,
Knowledge-Based Systems,
Volume 115,
2017,
Pages 87-99,
ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2016.09.032.
(https://www.sciencedirect.com/science/article/pii/S0950705116303495)
Abstract: Imbalanced problem occurs when the size of the positive class is much smaller than that of the negative one. Positive class usually refers to the main interest of the classification task. Although conventional Support Vector Machine (SVM) results in relatively robust classification performance on imbalanced datasets, it treats all samples with the same importance leading to the decision surface biasing toward the negative class. To overcome this inherent drawback, Fuzzy SVM (FSVM) is proposed by applying fuzzy membership to training samples such that different samples provide different contributions to the classifier. However, how to evaluate an appropriate fuzzy membership is the main issue to FSVM. In this paper, we propose a novel fuzzy membership evaluation which determines the fuzzy membership based on the class certainty of samples. That is, the samples with higher class certainty are assigned to larger fuzzy memberships. As the entropy is utilized to measure the class certainty, the fuzzy membership evaluation is named as entropy-based fuzzy membership evaluation. Therefore, the Entropy-based FSVM (EFSVM) is proposed by using the entropy-based fuzzy membership. EFSVM can pay more attention to the samples with higher class certainty, i.e. enhancing the importance of samples with high class certainty. Meanwhile, EFSVM guarantees the importance of the positive class by assigning positive samples to relatively large fuzzy memberships. The contributions of this work are: (1) proposing a novel entropy-based fuzzy membership evaluation method which enhances the importance of certainty samples, (2) guaranteeing the importance of the positive samples to result in a more flexible decision surface. Experiments on imbalanced datasets validate that EFSV outperforms the compared algorithms.
Keywords: Information entropy; Fuzzy support vector machine; Imbalanced dataset; Pattern recognition
