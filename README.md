Markdown
# 💳 Credit Card Fraud Detection: Handling Extreme Data Imbalance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Data_Balancing-yellow)

## 📌 Project Overview
In the financial industry, detecting fraudulent credit card transactions is critical to preventing revenue loss and ensuring customer trust. However, fraud detection datasets are notoriously imbalanced. 

In this project, using the Kaggle Credit Card Fraud dataset (284,807 transactions), **only 0.172% of transactions were actual frauds**. Relying on standard metrics like `Accuracy` would yield a deceptive 99.8% score by simply predicting all transactions as "Normal". Therefore, this project focuses on **Recall**, **Precision**, and **PR-AUC** to evaluate model performance realistically, utilizing advanced balancing techniques like **SMOTE** and **Class Weighting**.

## 🏗️ Project Architecture
The repository is structured with a modular and clean pipeline, separating data preprocessing, model training, and evaluation logic.

```text
credit-card-fraud-detection/
├── data/                   # Dataset directory (git-ignored)
├── src/                    # Source code modules
│   ├── data_preprocessing.py   # Data splitting, RobustScaler implementation
│   ├── model_training.py       # Model building (RF, XGBoost) and SMOTE
│   └── evaluation.py           # Metrics calculation and performance reports
├── venv/                   # Virtual environment (git-ignored)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
🚀 Key Methodologies
Outlier Resistance: Time and Amount features were scaled using RobustScaler to minimize the impact of extreme transaction values.

Preventing Data Leakage: SMOTE (Synthetic Minority Over-sampling Technique) was applied strictly to the training set after the train-test split.

Algorithm Selection:

RandomForestClassifier with class_weight='balanced' as a robust baseline.

XGBClassifier combined with SMOTE for aggressive fraud pattern recognition.

📊 Model Performance & Business Impact
The models were evaluated based on their ability to minimize financial loss (High Recall) without overwhelming the system with false alarms (Reasonable Precision).

Model	Technique	Recall (Fraud Caught)	Precision	PR-AUC
Random Forest	Class Weights	79.0%	91.0%	0.8666
XGBoost	SMOTE	87.0%	75.0%	0.8714
Conclusion:
The XGBoost (SMOTE) model is the winning architecture for a real-world banking scenario. By boosting the Recall to 87%, the system successfully flags a significantly higher number of actual fraudulent transactions. While the Precision slightly drops to 75% (meaning a few false alarms), the cost of sending automated verification SMS to clients is vastly lower than the financial liability of missing a fraudulent transaction.

⚙️ Installation & Usage
1. Clone the repository and navigate to the project directory:

Bash
git clone [https://github.com/erenonder35/credit-card-fraud-detection.git](https://github.com/erenonder35/credit-card-fraud-detection.git)
cd credit-card-fraud-detection
2. Set up the virtual environment and install dependencies:

Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
3. Download the Data:
Download the creditcard.csv file from Kaggle and place it inside the data/ directory.

4. Run the Pipeline:
Execute the evaluation script. It will automatically call the preprocessing and training modules.

Bash
python src/evaluation.py
Developed by Eren Önder