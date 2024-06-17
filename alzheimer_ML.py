import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# Load data
file_hc = 'C:/AZI/remember_healthy'
file_pa = 'C:/AZI/remember_patients'
healthy = loadmat(file_hc)['remember_healthy']
patients = loadmat(file_pa)['remember_patients']

subNum_healthy = healthy.shape[1]
subNum_patients = patients.shape[1]
chNum = 5  # EEG channels
lvl = 5  # wavelet decomposition level
wavelet_type = 'sym2' # wavelet filter
window_length_sec = 10  # length of each window in seconds

def preprocess(data, fs):
    # preprocessing of data including band pass filtering
    Wp = [0.5, 40]  # passband frequencies
    nyquist = fs / 2
    Wn = [wp / nyquist for wp in Wp]
    
    sos = butter(N=4, Wn=Wn, btype='band', output='sos') # designing the filter
    filtered_data = sosfiltfilt(sos, data) # applying the designed filter on the EEG signal

    return filtered_data

def extract_features(sig, fs):
    features = []
    v = np.var(sig) # variance
    s = np.mean(((sig - np.mean(sig)) / np.std(sig)) ** 3)  # skewness
    k = np.mean(((sig - np.mean(sig)) / np.std(sig)) ** 4)  # kurtosis
    f, Pxx = welch(sig, fs=fs) # power spectral density
    p = np.mean(Pxx)  # mean power of EEG
    C = pywt.wavedec(sig, wavelet_type, level=lvl) # wavelet coefficents
    a4, d4, d3, d2, d1 = C[0], C[1], C[2], C[3], C[4]
    features = [np.mean(a4), np.mean(d4), np.mean(d3), np.mean(d2), np.mean(d1), v, s, k, p]
    return features

# splitting the data(EEG) into ten-second windows, then filtering and extracting features for each window 
# the size of feature matrix would be a 3D matrix: (number of subjects, number of windows, number of features* number of EEG channels) 
def process_subjects(subjects, subNum): 
    subjects_features = []
    min_windows = np.inf  # minimum number of windows across all subjects
    for i in range(subNum):
        all_features = []
        dataset = subjects[0, i]
        t = dataset[:, 0]
        fs = 1 / (t[1] - t[0])
        window_samples = int(window_length_sec * fs)
        L = len(dataset)
        windows_count = (L - window_samples) // window_samples
        min_windows = min(min_windows, windows_count)
        
        for start in range(0, windows_count * window_samples, window_samples):
            window_features = []
            for j in range(1,chNum+1):
                sig = preprocess(dataset[start:start + window_samples, j], fs)
                features = extract_features(sig, fs)
                window_features.extend(features)
            all_features.append(window_features)
        subjects_features.append(np.array(all_features))
    
    # truncate each subject's features to have the same number of windows
    truncated_features = [features[:min_windows] for features in subjects_features]
    return np.array(truncated_features)

# normalizing the features to preapre them as the ML models' input
def normalizing(feature_train, feature_test):
    scaler = StandardScaler()
    feature_train = scaler.fit_transform(feature_train)
    feature_test = scaler.transform(feature_test)
    return feature_train, feature_test

# process healthy subjects
features_healthy = process_subjects(healthy, subNum_healthy)
# process patient subjects
features_patients = process_subjects(patients, subNum_patients)

# dividing the feature matrix into trainging and testing sets, then reshaping the size of feature matricies
def matrix_features(features, train_test_ratio = 0.8):
    number_subject = int(features.shape[0] * train_test_ratio)
    number_feature = features.shape[2]
    feature_train = features[0:number_subject,:,:]
    feature_test = features[number_subject:,:,:]
    return feature_train.reshape(-1, number_feature), feature_test.reshape(-1, number_feature)

X1, X3 = matrix_features(features_healthy)
X2, X4 = matrix_features(features_patients)

# stacking the feature matrices of health group and pations group in a single matrix and normalizing them
X_train, X_test = normalizing(np.vstack((X1, X2)), np.vstack((X3, X4)))
y_train = np.hstack((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
y_test = np.hstack((np.zeros(X3.shape[0]), np.ones(X4.shape[0])))

# Define hyperparameter grids
param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "LightGBM": {
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9]
    }
}

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42)
}

# Train and evaluate classifiers with hyperparameter tuning
results = []
plt.figure(figsize=(10, 8))
for name, clf in classifiers.items():
    print(f"--- {name} ---")
    param_grid = param_grids[name]
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    sensitivity = recall_score(y_test, y_pred) * 100  # Sensitivity is the same as recall
    specificity = (confusion_matrix(y_test, y_pred)[0, 0] / np.sum(confusion_matrix(y_test, y_pred)[0, :])) * 100
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    results.append({
        "Classifier": name,
        "Best Parameters": grid_search.best_params_,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "AUC": roc_auc
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results using tabulate`
print(tabulate(results_df, headers='keys', tablefmt='psql'))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Display DataFrame
print(results_df)
