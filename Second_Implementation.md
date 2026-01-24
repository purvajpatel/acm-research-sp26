Shreyaj Padigala: 2nd ForeQast Implementation

Author: Shreyaj Padigala

Project: Research implementation of HM-RNN for temporal hierarchy learning

Reference: Chung et al. - Hierarchical Multiscale Recurrent Neural Networks


What is a HM-RNN?:

An HM-RNN is a type of recurrent neural network that learns patterns at different time scales by 
deciding when to update its memory and when to keep it the same. A normal RNN updates its state 
at every time step, which can make it sensitive to short-term noise and harder to capture long-term 
trends. In contrast, an HM-RNN updates lower layers frequently and higher layers only when meaningful 
changes occur, letting it track both short-term movements and long-term structure. This is useful in 
finance because price data naturally has fast noise, medium-term trends, and slow market regimes, and 
HM-RNNs can model these layers separately instead of mixing them together.



This notebook implements a 3-layer HM-RNN from scratch to predict Ethereum prices.
The model learns hierarchical temporal patterns through explicit COPY, UPDATE, and FLUSH operations.

==================================================
TEST SET PERFORMANCE
==================================================
MSE:  178638.46
RMSE: 422.66
MAE:  378.89
MAPE: 13.93%
==================================================


Dataset: Same as implementation #1 (2017-2022 ETH/USD Price)

Boundary Activation Statistics:
==================================================
Layer 1 (Fast)      : 100.00% activation rate
Layer 2 (Medium)    : 100.00% activation rate
Layer 3 (Slow)      :  3.33% activation rate
==================================================

