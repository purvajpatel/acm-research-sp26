Link to Paper: https://www-sciencedirect-com.utd.idm.oclc.org/science/article/pii/S0169023X22000234

Steps to access paper:   Go to Eugene McDermott Library > Articles > Sign in with NetID > search up "Forecasting cryptocurrency prices using Recurrent Neural Network and Long Short-term Memory"




What is this article about:

This paper talks about using an RNN-LSTM to address the volitility and sequential nature of financial time series data. The significance of the architecture
lies in its use of a bidirectional LSTM with two hiddden layers within an RNN framework, which is specifically designed to solve the standard RNN issue of being
able to retain information and recognize historical price information over extended sequences.





How my implementation relates to the paper:

My implementation involves using an RNN-LSTM model to train a Kaggle data set containing Ethereum prices ($ETH) and trading volume each day from 2017 to 2023.

Link to data set: https://www.kaggle.com/datasets/varpit94/ethereum-data

Here are some specifics of my RNN-LSTM model:

  1) 3 LSTM Layers
  2) Drop-out Layers
  3) 10 Epochs Trained
  4) Batch Size of 32


Advantages of this implementation:

  1) It reduces the RNN Vanishing Graaient Issue by remembering long-term information better

Disadvantages of this implementation:

  2) This model involves many parameters which makes its prone to overfitting




