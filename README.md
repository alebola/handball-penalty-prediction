# 🤾 Handball Penalty Prediction: Shots vs Feints  

## 📌 Project Overview  
This project is based on my **Bachelor’s Thesis (TFG)** and focuses on predicting whether a handball penalty will result in a **shot or a feint**, **before the action actually happens**.  

To achieve this, the models analyze **video fragments truncated right at the critical decision point**, without ever seeing the final outcome. This setup ensures the task is truly **anticipatory prediction**, rather than mere classification of observed outcomes.  

The approach uses **deep learning models applied to Human Activity Recognition (HAR) video embeddings**, and the research followed a **long experimental phase with more than 20 experiments**, exploring:  
- Different sequence models (LSTM, GRU, TCN, Transformer).
- Experiments with embeddings extracted from **four different HAR architectures** (SlowFast, I3D, X3D, C2D).
- Multiple **temporal windows** and **offsets** in the embeddings.
- **Fusion of different HAR architectures** to identify which combinations provide richer and more discriminative representations.  

The final outcome is a robust methodology combining **Transformer models with SMOTE** for class balancing, achieving the best results across folds.  


## 🚀 Key Highlights  
- **End-to-end pipeline**: preprocessing from CSV embeddings, dataset structuring, balancing, training, and evaluation.  
- **Experiment-driven**: +20 experiments testing models, temporal configurations, and hyperparameters.  
- **Deep Learning focus**: compared **LSTM, GRU, TCN, and Transformer** architectures.  
- **Best performing solution**: Transformer + SMOTE with cross-validation.  


## 📂 Project Structure  
```
handball-penalty-prediction/
├── notebooks/
│   ├── 00_preprocessing.ipynb   # Preprocessing and dataset preparation
│   ├── 01_experiments.ipynb     # Experiments with LSTM, GRU, TCN, Transformer
├── run_transformer_smote.py     # Final script for best model (Transformer + SMOTE)
└── README.md                    # Project documentation

```


## 📊 Results  
- **Best model**: Transformer with SMOTE balancing.  
- Evaluated with **5-fold cross-validation** to ensure robustness.  
- Explored multiple **temporal windows** and **offsets**:  
  - Using the **last frames introduced noise**.  
  - Best performance came from embeddings with a **6-frame offset**.  
- **Fusion of embeddings (SlowFast + I3D)** provided the most discriminative representation.  
- **Final metrics (Transformer + SMOTE, SlowFast+I3D fusion, 6-frame offset):**  
  - **Accuracy:** 0.7847  
  - **F1-score:** 0.7322 
- **Conclusion**: The combination of **Transformer + SMOTE + fusion embeddings** achieved the highest and most consistent performance, outperforming LSTM, GRU, and TCN alternatives.  


## 🛠️ Technologies Used  
- **Python**  
- **PyTorch** (deep learning models, training, Transformer implementation)  
- **Scikit-learn** (metrics, preprocessing, cross-validation)  
- **Imbalanced-learn (SMOTE)** (balancing classes)  
- **Matplotlib** (visualization)  


## ⚡ How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/handball-penalty-prediction.git
   cd handball-penalty-prediction
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn imbalanced-learn matplotlib
   ```
3. Run the final script with your CSV embeddings:
   ```bash
   python run_transformer_smote.py --csv1 path/to/embedding1.csv --csv2 path/to/embedding2.csv --folds path/to/folds.csv
   ```
   - Use --csv2 for fusion experiments.
   - Add --no_smote if you want to disable SMOTE.
  

## 📌 Takeaways
- Designed, trained, and evaluated multiple sequence models on HAR embeddings.
- Showed how temporal offsets and data balancing affect performance.
- Achieved best performance with Transformer + SMOTE, proving the viability of Transformers in fine-grained sports video prediction tasks.
