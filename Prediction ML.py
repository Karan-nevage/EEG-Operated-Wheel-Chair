import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import scipy.signal

# Load the dataset
df = pd.read_csv('/dataset/right03.csv')  # Change file name as needed

# Assign labels based on the specified conditions
df.loc[:9999, 'label'] = '00'       # Below 10000
df.loc[10000:19999, 'label'] = '01' # Between 10000 and 20000
df.loc[20000:29999, 'label'] = '10' # Between 20000 and 30000
df.loc[30000:, 'label'] = '11'      # Above 30000

# Drop specific rows wich contains errored data
df.drop(df.index[150000:310001], inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(df.index[600000:750000], inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop the first column
df.drop(columns=df.columns[0], axis=1, inplace=True)

# Rename the columns
df.columns = ['raw_eeg', 'label']

# Extract data and labels
data = df['raw_eeg']
labels_old = df['label']

# Define filters
sampling_rate = 512
notch_freq = 50.0
lowcut, highcut = 0.5, 30.0
nyquist = 0.5 * sampling_rate
notch_freq_normalized = notch_freq / nyquist
b_notch, a_notch = scipy.signal.iirnotch(notch_freq_normalized, Q=0.05, fs=sampling_rate)
lowcut_normalized = lowcut / nyquist
highcut_normalized = highcut / nyquist
b_bandpass, a_bandpass = scipy.signal.butter(4, [lowcut_normalized, highcut_normalized], btype='band')

# Define feature extraction functions
def calculate_psd_features(segment, sampling_rate):
    f, psd_values = scipy.signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    alpha_indices = np.where((f >= 8) & (f <= 13))
    beta_indices = np.where((f >= 14) & (f <= 30))
    theta_indices = np.where((f >= 4) & (f <= 7))
    delta_indices = np.where((f >= 0.5) & (f <= 3))
    energy_alpha = np.sum(psd_values[alpha_indices])
    energy_beta = np.sum(psd_values[beta_indices])
    energy_theta = np.sum(psd_values[theta_indices])
    energy_delta = np.sum(psd_values[delta_indices])
    alpha_beta_ratio = energy_alpha / energy_beta
    return {
        'E_alpha': energy_alpha,
        'E_beta': energy_beta,
        'E_theta': energy_theta,
        'E_delta': energy_delta,
        'alpha_beta_ratio': alpha_beta_ratio
    }

def calculate_additional_features(segment, sampling_rate):
    f, psd = scipy.signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    peak_frequency = f[np.argmax(psd)]
    spectral_centroid = np.sum(f * psd) / np.sum(psd)
    log_f = np.log(f[1:])
    log_psd = np.log(psd[1:])
    spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
    return {
        'peak_frequency': peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_slope': spectral_slope
    }

# Process data segments
features = []
labels = []
for i in range(0, len(data) - 512, 256):
    segment = data.loc[i:i+512]
    segment = pd.to_numeric(segment, errors='coerce')
    segment = scipy.signal.filtfilt(b_notch, a_notch, segment)
    segment = scipy.signal.filtfilt(b_bandpass, a_bandpass, segment)
    segment_features = calculate_psd_features(segment, 512)
    additional_features = calculate_additional_features(segment, 512)
    segment_features = {**segment_features, **additional_features}
    features.append(segment_features)
    labels.append(labels_old[i])

# Create DataFrame with features and labels
columns = ['E_alpha', 'E_beta', 'E_theta', 'E_delta', 'alpha_beta_ratio', 'peak_frequency', 'spectral_centroid', 'spectral_slope']
df = pd.DataFrame(features, columns=columns)
df['label'] = labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('label', axis=1))
df_scaled = pd.DataFrame(X_scaled, columns=columns)
df_scaled['label'] = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled.drop('label', axis=1), df_scaled['label'], test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
svc = SVC(probability=True)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)
test_accuracy = model.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy:.2f}")

# Save the model and scaler
model_filename = 'model.pkl'
scaler_filename = 'scaler.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

# Evaluate model
y_true = df_scaled['label']
y_pred = model.predict(X_scaled)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision
precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

# Recall
recall = recall_score(y_true, y_pred, average='weighted')
print(f'Recall: {recall:.4f}')

# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true, y_pred)
print('Classification Report:')
print(class_report)

# Calibration curve
probabilities = model.predict_proba(X_test)[:, 1]
fop, mpv = calibration_curve(y_test, probabilities, n_bins=10)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(mpv, fop, marker='.')
plt.show()

# Calibrate the classifier
calibrator = CalibratedClassifierCV(model, cv=3)
calibrator.fit(X_train, y_train)
yhat = calibrator.predict(X_test)
fop, mpv = calibration_curve(y_test, yhat, n_bins=10)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(mpv, fop, marker='.')
plt.show()

# Evaluate calibrated classifier
accuracy = accuracy_score(y_test, yhat)
print(f"Classification Accuracy: {accuracy:.2f}")
