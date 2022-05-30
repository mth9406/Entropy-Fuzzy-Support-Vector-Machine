# Entropy-Fuzzy-Support-Vector-Machine

This repo is the implementation of EFSVM (Entropy Fuzzy Support Vector Machine)

Reference:

[1] Qi Fan, Zhe Wang, Dongdong Li, Daqi Gao, Hongyuan Zha, Entropy-based fuzzy support vector machine for imbalanced datasets,
Knowledge-Based Systems, Volume 115, 2017, Pages 87-99, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.09.032.
(https://www.sciencedirect.com/science/article/pii/S0950705116303495)

Keywords: Information entropy; Fuzzy support vector machine; Imbalanced dataset; Pattern recognition

# How to use:
First of all, install this package
```bash
!pip install git+https://github.com/mth9406/Entropy-Fuzzy-Support-Vector-Machine.git
```
Then, you can import the model as:
```bash
import efsvm.EFSVM as EFSVM
model = EFSVM(k= 8, m= 18, beta= 1/20, C=1.0)
```
You can fit the model as:
```bash
# X, y are your data
model.fit(X,y)
```
In addition, you can predict using the model as:
```bash
# X_new is the test data
y_pred = model.predict(X_new)
```
