import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

data_set = pd.read_csv("FU.csv")
#print(data_set.head())

train_dataset = data_set.sample(frac=0.8, random_state=0)
test_dataset = data_set.drop(train_dataset.index)

#sns.pairplot(train_dataset[["Profit", "Power", "Reliability", "Work Hours", "Production Rate"]], diag_kind="kde")
#plt.show()
print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Profit')
test_labels = test_features.pop('Profit')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
normalizer.mean.numpy()


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split = 0.2,
    verbose=0,
    epochs=100
)
test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features,test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [Profit]']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Profit]')
plt.ylabel('Predictions [Profit]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Profit]')
_ = plt.ylabel('Count')
plt.show()

dnn_model.save('dnn_model.keras')

reloaded = tf.keras.models.load_model('dnn_model.keras')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [Profit]']).T
plt.show()

def predict_profit(power, reliability, work_hours, production_rate):
    input_data = np.array([[power, reliability, work_hours, production_rate]])
    prediction = dnn_model.predict(input_data)
    return prediction[0][0]

power = float(input("Enter Power (kW): "))
reliability = float(input("Enter Reliability (%): "))
work_hours = float(input("Enter Work Hours: "))
production_rate = float(input("Enter Production Rate (Units/Hr): "))
predicted_profit = predict_profit(power, reliability, work_hours, production_rate)
print(f"Predicted Profit: {predicted_profit:.2f}")