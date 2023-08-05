#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#%%
data = pd.read_csv("E:\Projects\HealthCare_ML_DL\heart_disease\heart.csv")
print(data.describe())

#%%

print(data.columns)

f = sns.countplot(x='target', data=data)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("");

#%%

plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red", s=60)
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], s=60)
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate");


#%%
X = data.loc[:,data.columns!='target']
y = data.iloc[:,-1]

#%%
print(data.isnull().sum())

#%%

feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
  feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
data["thal"] = data["thal"].apply(str)
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['3', '6', '7'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

data["sex"] = data["sex"].apply(str)
sex = tf.feature_column.categorical_column_with_vocabulary_list(
      'sex', ['0', '1'])
sex_one_hot = tf.feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

data["cp"] = data["cp"].apply(str)
cp = tf.feature_column.categorical_column_with_vocabulary_list(
      'cp', ['0', '1', '2', '3'])
cp_one_hot = tf.feature_column.indicator_column(cp)
feature_columns.append(cp_one_hot)

data["slope"] = data["slope"].apply(str)
slope = tf.feature_column.categorical_column_with_vocabulary_list(
      'slope', ['0', '1', '2'])
slope_one_hot = tf.feature_column.indicator_column(slope)
feature_columns.append(slope_one_hot)


# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
feature_columns.append(age_thal_crossed)

cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
feature_columns.append(cp_slope_crossed)

#%%

def create_dataset(dataframe, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) \
          .shuffle(buffer_size=len(dataframe)) \
          .batch(batch_size)
          
#%%
train, test = train_test_split(data, test_size=0.2)
train_ds = create_dataset(train)
test_ds = create_dataset(test)         

#%%
model = tf.keras.models.Sequential([
  tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dense(units=1, activation='sigmoid')
])

#%%

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=100, use_multiprocessing=True)

#%%
model.evaluate(test_ds)
#%%

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0, 1))
plt.legend(['train', 'test'], loc='upper left');

#%%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
model.save("heart_disease.h5")

#%%
new_model = tf.keras.models.load_model('heart_disease.h5')

#%%

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)

#%%

lin_model = LogisticRegression(solver='lbfgs')
lin_model.fit(x_train, y_train)
print("Linear Model Accuracy: ", lin_model.score(x_test, y_test))

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
print("K Nearest Neighbor Model Accuracy: ", knn_model.score(x_test, y_test))

svm_model = SVC(gamma='auto')
svm_model.fit(x_train, y_train)
print("Support Vector Machine Model Accuracy: ", svm_model.score(x_test, y_test))

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print("Naive Bayes Model Accuracy: ", nb_model.score(x_test, y_test))

tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
print("Decision Tree Model Accuracy: ", tree_model.score(x_test, y_test))

forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(x_train, y_train)
print("Random Forest Model Accuracy: ", forest_model.score(x_test, y_test))

#%%

votes = lin_model.predict(x_test) + svm_model.predict(x_test) + nb_model.predict(x_test) \
        + forest_model.predict(x_test) + tree_model.predict(x_test) + knn_model.predict(x_test) 
        
#%%
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix((y_test.values == 1.0),  (votes >= 4))
total = sum(sum(conf_mat))
sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total

print("Statistics for voting classifier, where simple majority rules:\n")
print(conf_mat)
print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
print('accuracy : ', accuracy)