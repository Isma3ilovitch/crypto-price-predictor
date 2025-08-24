# Cryptocurrency Price Prediction & Portfolio Optimization

A deep learning model designed to predict short-term future prices of cryptocurrencies (like Bitcoin) to aid in portfolio optimization and trading strategy development. This model employs a hybrid architecture combining Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and Multi-Head Attention mechanisms for robust time-series forecasting.

## 🚀 Features

- **Hybrid AI Model:** Leverages CNN for feature extraction, LSTM for temporal dependency learning, and Attention mechanisms to focus on important time steps.
- **Technical Analysis Integration:** Utilizes a wide array of technical indicators (RSI, MACD, Bollinger Bands, OBV, ATR, EMAs, SMAs) as model features.
- **Continuous Training:** Includes a checkpointing system to save progress and allow training to be resumed after interruptions.
- **30-Minute Horizon Prediction:** Predicts the closing price 30 minutes into the future.

## 📊 Model Architecture

The model ingests a sequence of past 30 minutes of market data and technical indicators to predict the price 30 minutes ahead.

1.  **Input:** A window of 30 time steps, each containing multiple features (price data + technical indicators).
2.  **CNN Layer:** `Conv1D` layer processes the input to extract local patterns and features.
3.  **LSTM Layer:** Processes the sequential data to understand long-term dependencies and trends.
4.  **Attention Layer:** `MultiHeadAttention` layer helps the model weigh the importance of different time steps in the sequence.
5.  **Fully Connected Layers:** The outputs from CNN and Attention layers are concatenated and passed through Dense layers to produce the final prediction.

## 📁 Project Structure
-crypto-price-predictor/
-├── predict.ipynb # Main Jupyter notebook for data processing and model training
-├── requirements.txt # Python dependencies (see below)
-├── data/ # Directory for your input data (not included in repo)
-│ └── data.csv # Expected input data file
-├── checkpoint/ # Directory for training checkpoints (created automatically)
-└── saved_model/ # Directory for the final saved model (created automatically)

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/crypto-price-predictor.git
    cd crypto-price-predictor
    ```

2.  **Set up a Python environment** (recommended using `venv` or `conda`):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If you need to generate the `requirements.txt`, you can use:*
    ```bash
    pip install pandas numpy tensorflow scikit-learn ta-lib
    pip freeze > requirements.txt
    ```

4.  **Install TA-Lib:** The notebook tries to install it automatically, but it can be tricky. For a local setup, it's best to install it manually following the official instructions for your OS: [TA-Lib GitHub](https://github.com/mrjbq7/ta-lib#installation).

## 📖 Usage

### 1. Prepare Your Data
Your input data file (`data.csv`) should be placed in the project directory or its path updated in the notebook. It must contain the following columns:
- `Timestamp` (in milliseconds)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume USD`
- `Volume BTC` (will be dropped)
- `Symbol` (will be dropped)
- `Date` (will be dropped)

### 2. Run the Jupyter Notebook
Open the `predict.ipynb` notebook in Jupyter Lab, Google Colab, or your preferred environment.

1.  **Mount Google Drive (if using Colab):** The notebook is configured to read data from and save models to Google Drive. You will be prompted to authenticate.
2.  **Run all cells:** Execute the cells sequentially to:
    - Load and preprocess the data.
    - Calculate technical indicators.
    - Scale the data and create sequences.
    - Build and compile the hybrid model.
    - Train the model (loading checkpoints if they exist).
    - Save the final model.

### 3. Making Predictions & Portfolio Integration
After training, the model is saved in the `saved_model/` directory. To use it for live predictions and feed those signals into a portfolio optimizer:

1.  **Load the trained model** in a separate script or notebook:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('/path/to/saved_model/model.keras')
    ```
2.  **Preprocess live/streaming data** in the exact same way as the training data (same technical indicators, same scaler).
3.  **Use the model's prediction** (e.g., the predicted price change) as an alpha signal in your portfolio optimization algorithm (e.g., a mean-variance optimizer or a reinforcement learning agent).

## 📈 Example of Integration for Portfolio Optimization

A simple strategy could be:
```python
# Pseudocode
predicted_price = model.predict(latest_sequence)
current_price = get_current_price()

if predicted_price > current_price * (1 + threshold):
    signal = "BUY"
elif predicted_price < current_price * (1 - threshold):
    signal = "SELL"
else:
    signal = "HOLD"

# Feed this signal into your portfolio allocation logic
rebalance_portfolio(signal, predicted_volatility)
```
## 🔧 Dependencies

Core Python libraries required:

pandas

numpy

tensorflow

scikit-learn

ta-lib

See requirements.txt for specific versions.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer
This software is for educational and research purposes only. 
