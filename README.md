# 📈 GRU–HHO Forex Prediction

A web application for Forex price forecasting using a hybrid model of Gated Recurrent Unit (GRU) and Harris Hawks Optimization (HHO).
The app allows users to load historical Forex data, train the model with customizable parameters, and visualize both training process and forecasting results on interactive charts.

## ✨ Features

- Dynamic Data Loading: Load 5 years of historical Forex data (EUR/USD, GBP/USD, etc.) directly from Yahoo Finance.

- Model Customization: Users can configure key GRU and HHO parameters.

- Real-time Training: Model training with live logs streamed from the backend.

- Interactive Visualization: Display historical data, training/testing splits, and forecasting results in interactive charts.

- Chart Controls: Zoom and pan functionality for deeper time-series analysis.

## 🧪 Methodology

The forecasting process using the GRU–HHO hybrid model consists of several key steps:

1. Data Preprocessing

    - Collect daily Forex data (e.g., GBP/IDR) from Yahoo Finance.
    
    - Normalize data using Min–Max scaling to fit within the range (0,1).
    
    - Split dataset into 70% training and 30% validation.
    
    - Create data patterns with 5 days input → 1 day output.

2. Model Design

    - GRU architecture: 5 input neurons, 1 output neuron, hidden units as hyperparameters.
    
    - Harris Hawks Optimization (HHO) used to optimize GRU weights at each epoch.

3. Training Process

    - Initialize GRU and HHO parameters (hidden units, batch size, epochs, hawks population, max iterations).
    
    - Train the GRU model iteratively while updating weights using HHO optimization.
    
    - Stop training when target MSE threshold is reached or max epochs are completed.

4. Validation

    - Evaluate model using Mean Square Error (MSE).
    
    - Compare multiple parameter combinations to find the best-performing model.

5. Forecasting

    - Apply the trained GRU–HHO model to generate predictions for future Forex prices.
    
    - Denormalize predicted values back to original scale.

## 📊 Results

Best training MSE: ~0.00119

Best validation MSE: ~0.00253

The model demonstrates effective forecasting performance with predictions close to actual values.

Example of predicted GBP/IDR values for the next 5 days:

| Date	    | Prediction |
|-----------|------------|
|11/02/2025 |	20,584.31  |
|12/02/2025	| 20,164.32  |
|13/02/2025	| 20,049.59  |
|14/02/2025	| 20,048.26  |
|17/02/2025	| 19,905.59  |

## ⚙️ How to Use

- You can acces https://mkii-forecast.vercel.app
  
- Choose forex pair you want, then load data.

- Configure the parameters as desired.

- Train the model to the desired result.

- Click 'Test Model' to see the performance.

- Click 'Predict Future' to see the prediction.


## 🛠️ Tech Stack

- **Languages**: Python 3.13, TypeScript  
- **Libraries**: NumPy, Pandas, Matplotlib  
- **Backend**: FastAPI  
- **Frontend**: Next.js (React, Tailwind CSS)  
- **Deployment**: Vercel, Railway  

## 📚 References

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.

Heidari, A. A., Mirjalili, S., Faris, H., et al. (2019). Harris hawks optimization: Algorithm and applications.

Junior, M. A., Appiahene, P., Appiah, O., & Bombie, C. N. (2023). Forex market forecasting using machine learning: Systematic Literature Review and meta-analysis.
