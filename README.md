# Oil Well Anomaly Detection using Deep Learning

## Competition Submission: Advanced Time Series Analysis for Well Monitoring

This repository contains our solution for the oil well anomaly detection competition, implementing state-of-the-art deep learning approaches including Temporal Fusion Transformers (TFT) and traditional machine learning algorithms.

## 🎯 Project Overview

Our solution focuses on detecting anomalies in oil well operations using multivariate time series data from various sensors. We've implemented multiple approaches with a primary focus on the Temporal Fusion Transformer architecture, which has shown promising results in handling complex temporal dependencies.

### Key Features

- **Data Preprocessing Pipeline**
  - Robust handling of missing values
  - Feature engineering for temporal data
  - Data balancing techniques for improved model performance

- **Model Implementation**
  - Temporal Fusion Transformer (TFT) with PyTorch Lightning
  - Traditional ML algorithms for baseline comparison
  - Ensemble approaches for improved robustness

- **Evaluation Framework**
  - Cross-validation strategy
  - Comprehensive metric tracking
  - Performance visualization

## 📊 Dataset Structure

The dataset consists of various sensor measurements from oil wells, including:
- Pressure measurements (P-PDG, P-TPT, P-MON-CKP)
- Temperature readings (T-TPT, T-JUS-CKP)
- Temporal features (hour, day, month)

## 🛠️ Technical Implementation

### Requirements

```
pytorch-lightning>=2.0.0
pytorch-forecasting>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── tft_models/
│   └── ml_models/
├── src/
│   ├── tft_training.py
│   ├── ml_algorithms.py
│   ├── downsampling.py
│   └── view_data.py
└── requirements.txt
```

### Key Components

1. **Data Preprocessing (`view_data.py`)**
   - Data cleaning and validation
   - Feature engineering
   - Missing value handling

2. **TFT Implementation (`tft_training.py`)**
   - Custom PyTorch Lightning implementation
   - Configurable hyperparameters
   - Advanced logging and monitoring

3. **Traditional ML Approaches (`ml_algorithms.py`)**
   - Implementation of baseline models
   - Model comparison framework

4. **Data Balancing (`downsampling.py`)**
   - Implementation of various sampling techniques
   - Class balance optimization

## Data Analysis

The data analysis plots and visualizations are available in this [Google Drive folder](https://drive.google.com/drive/folders/1xTl5v202lrVRWizhBWFwjtucnz4wn4qy?usp=sharing). The plots include:

- Feature distributions for each dataset
- Correlation matrices
- Time series visualizations

The analysis code is available in `data_analysis.py`.

## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   ```bash
   python view_data.py
   ```

3. **Model Training**
   ```bash
   python tft_training.py
   ```

## Model Performance Comparison

### Temporal Fusion Transformer (TFT)

Performance across different datasets:
- **Balanced Drawn Data**: Best validation loss of 0.001
- **Balanced Simulated Data**: Best validation loss of 0.001
- **Balanced Well Data**: Higher loss (2.000), indicating potential challenges

### Traditional ML Models

1. **XGBoost**
   - Accuracy: 100%
   - F1-Score: 0.99-1.00
   - AUC-ROC: 0.9999
   - Key Features: Best overall performance, consistent across datasets

2. **LightGBM**
   - Accuracy: 100%
   - F1-Score: 0.99-1.00
   - AUC-ROC: 0.9999
   - Key Features: Fast training, matches XGBoost performance

3. **Random Forest**
   - Accuracy: 99%
   - F1-Score: 0.98-0.99
   - AUC-ROC: 0.9993
   - Key Features: Robust performance, good generalization

### Key Findings

1. **Overall Performance**
   - Both approaches achieve excellent performance (accuracy > 99%)
   - Traditional ML models show more consistent performance across all datasets
   - TFT performance varies significantly between datasets

2. **Dataset-Specific Performance**
   - Traditional ML: Consistent across all datasets
   - TFT: 
     - Excellent on drawn and simulated data
     - Struggles with well data (higher loss)

3. **Practical Considerations**
   - Traditional ML models are simpler to train
   - Traditional ML achieves comparable or better results
   - TFT requires more computational resources

### Recommendations

1. **For Immediate Deployment**
   Use LightGBM or XGBoost as they provide:
   - Best overall performance
   - More consistent results across datasets
   - Faster training time
   - Simpler implementation

2. **For Future Development**
   - Further investigate TFT's performance on well data
   - Consider ensemble approaches combining both TFT and traditional ML
   - Collect more well data to improve TFT performance
   - Experiment with different TFT architectures and hyperparameters

## 🤝 Contributing

While this is a competition submission, we welcome feedback and suggestions for improvement. Please feel free to open issues or submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- GOODLUCK ARCHIBONG
- ABDUSSAMAD IDRIS ALI
- KING FAHD UNIVERSITY OF PETROLEUM AND MINERALS

## 📚 References

1. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209.
2. Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30.

## 🙏 Acknowledgments

We would like to thank the competition organizers and our technical advisors for their support throughout this project.
