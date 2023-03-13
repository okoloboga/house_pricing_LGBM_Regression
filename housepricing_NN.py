import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
y = pd.read_csv('train.csv')

df.head(5)

# Sort by nulls in columns

pd.set_option('display.max_rows', df.shape[0])
pd.DataFrame(df.isnull().sum().sort_values(ascending = False))

# Delete columns with a lot of nulls (over50%)

df.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'],inplace=True)
a = df.columns[df.isnull().any()]

# In another columns replace nulls - mode

for i in a:
    df[i] = df[i].fillna(df[i].mode()[0])  
    
df.head(10)

# Extract SalePrice feature as target array

y = df['SalePrice']
del df['SalePrice']

# Transfom objects in coloumns to int64

label_encoder = preprocessing.LabelEncoder()

for i in df.columns:
    df[i] = label_encoder.fit_transform(df[i])
    df.drop(columns = [], inplace=True)
    
df.head(10)

# Convert all to [0 - 1]

y = (y - min(y)) / max(y) 

for i in df.columns:
    a = max(df[i])
    b = min(df[i])
    df[i] = (df[i] - b) / a
    
df.head(10)

# Train/test splitting

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 37)

# Build network model

model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam",
             loss="mean_squared_error",
             metrics=["mae"])    

history = model.fit(x_train,
                    y_train,
                    epochs = 150,
                    batch_size = 16,
                    validation_data = (x_test, y_test))

history_dict = history.history
history_dict.keys()

# Visualization of training process, epoch/loss

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualization of training process, epoch/accuracy

plt.clf()
acc = history_dict["mae"]
val_acc = history_dict["mae"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Prediction and accuracy

pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")

# Prediction and acc. on all data

pred_0 = model.predict(df)
test_loss, test_acc = model.evaluate(df, y)
print(f"test_acc: {test_acc}")

# Prediction on specific example

a = pd.read_csv('train.csv')
y_0 = a['SalePrice']

pred_1 = (pred_0 * max(y_0)) + min(y_0)
abs(1 - (pred_1[1337] / y_0[1337]))
