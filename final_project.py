import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris.data', header=None, names=column_names)

binary_data = data[data['class'].isin(['Iris-setosa', 'Iris-versicolor'])]

X = binary_data.iloc[:, :-1].values
y = LabelEncoder().fit_transform(binary_data['class'])   # 0/1 labels


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_svm = X_scaled.copy()
X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
print(f"SVM shape: {X_svm.shape}, LSTM shape: {X_lstm.shape}")

def compute_metrics(y_true, y_pred):
    tp, fn = confusion_matrix[0][0], confusion_matrix[0][1]
    fp, tn = confusion_matrix[1][0], confusion_matrix[1][1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) 
    f1 = 2 * tp / (2 * tp + fp + fn)
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    bacc = (tpr + tnr) / 2
    tss = tpr - fpr
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'F1': f1,
        'Error Rate': error_rate,
        'BACC': bacc,
        'TSS': tss,
        'HSS': hss
    }

kf = KFold(splits=10, shuffle=True, random_state=42)

metrics_rf = []
metrics_svm = []
metrics_lstm = []
fold = 1

for train_idx, test_idx in kf.split(X_svm):
    print(f"\n Fold {fold} ")

    X_train_svm, X_test_svm = X_svm[train_idx], X_svm[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_lstm, X_test_lstm = X_lstm[train_idx], X_lstm[test_idx]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_svm, y_train)
    y_pred_rf = rf.predict(X_test_svm)
    m_rf = compute_metrics(y_test, y_pred_rf)
    metrics_rf.append(m_rf)
    print("Random Forest:", m_rf)

    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_svm, y_train)
    y_pred_svm = svm.predict(X_test_svm)
    m_svm = compute_metrics(y_test, y_pred_svm)
    metrics_svm.append(m_svm)
    print("SVM:", m_svm)

    lstm_model = Sequential([
        LSTM(16, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), activation='tanh'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train, epochs=30, batch_size=8, verbose=0)
    y_pred_prob_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm = (y_pred_prob_lstm > 0.5).astype(int).reshape(-1)
    m_lstm = compute_metrics(y_test, y_pred_lstm)
    metrics_lstm.append(m_lstm)
    print("LSTM:", m_lstm)

    fold += 1

df_rf = pd.DataFrame(metrics_rf)
df_svm = pd.DataFrame(metrics_svm)
df_lstm = pd.DataFrame(metrics_lstm)

print("\n Per-fold metrics: Random Forest ")
print(df_rf)
print("\n Average RF metrics:")
print(df_rf.mean())

print("\n Per-fold metrics: SVM ")
print(df_svm)
print("\n Average SVM metrics:")
print(df_svm.mean())

print("\n Per-fold metrics: LSTM ")
print(df_lstm)
print("\n Average LSTM metrics:")
print(df_lstm.mean())
