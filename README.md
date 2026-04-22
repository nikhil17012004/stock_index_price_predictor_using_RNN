# stock_index_price_predictor_using_RNN

A deep learning project that predicts the next day's High and Low prices of the BankNifty index using a Gated Recurrent Unit (GRU) neural network, engineered financial features, and a robust dual-stage normalization pipeline with time-series cross-validation.

- The historical data was downloaded from NSE website. The Banknifty_index.csv file has data till 19-02-2025

Approach:
- Predicts both High and Low simultaneously (2-output model)
- 5-Fold TimeSeriesSplit cross-validation
- Dual-stage scaling: StandardScaler -> MinMaxScaler on features AND targets
- ZScore computed only from train split and applied to test, no data leakage
- Rolling FFT Energy over 50-day windows - captures frequency-domain momentum

Features Engineered:
- Started with engineering 32 features across multiple categories.

  Used PFI to get the importance of features and get the most important features
  The method was:
    - Get the original MSE of the trained model on the test fold.
    - For each feature, randomly shuffle its values in the test set.
    - Measure the new MSE. If the MSE goes up a lot, that feature was important. If it stays the same, the feature was noise.

- The top 6 most important features were selected: (ZScore, SMA_20, EMA_50, FFTEnergy, DateEncoded, VPT)


Model Architecture:
Input: (21 timesteps × 6 features)
   GRU(units=736, return_sequences=False)
       Dropout(rate=0.0447)
           Dense(2, activation='linear')  

Optimizer : Adam (lr = 0.000343)
Loss      : Mean Squared Error (MSE)


Results:
RMSE: 492.81 
MAE: 364.44
