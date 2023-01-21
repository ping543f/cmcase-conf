import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')



cmcdata = pd.read_csv("tabular-lstm--data_preview.csv")
cmcdata.head()
cmcdata.drop(cmcdata[cmcdata['CMCase'] < 0].index, inplace = True)
cmcdata.head()
len(cmcdata)


X = cmcdata[['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K']]
y = cmcdata['CMCase']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cols = X_train.columns
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores



TargetVariable=['CMCase']
Predictors=['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K'] # All feature

# Predictors=['A', 'F', 'H', 'G', 'D'] # RFR feature
# Predictors=['A', 'E', 'H', 'I', 'K'] # MEA feature

X=cmcdata[Predictors].values
y=cmcdata[TargetVariable].values
 
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)

X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(units=256, input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=512, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['MeanSquaredLogarithmicError'])
history = model.fit(X_train, y_train, batch_size = 20, epochs = 1000, verbose=1, validation_data=(X_test,y_test), shuffle=True, callbacks=[es])

y_pred = model.predict(X_test)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=20)
results


ifile = 'model_arch.jpg'
tf.keras.utils.plot_model (model, to_file = ifile, show_shapes = True, show_layer_names = True)

model.summary()
model.save('ann_cmcase.h5')


evs_no=sklearn.metrics.explained_variance_score(y_test,model.predict(X_test))
me_no=sklearn.metrics.max_error(y_test,model.predict(X_test))
mae_no=sklearn.metrics.mean_absolute_error(y_test,model.predict(X_test))
mse_no=sklearn.metrics.mean_squared_error(y_test,model.predict(X_test))
Mae_no=sklearn.metrics.median_absolute_error(y_test,model.predict(X_test))
r2_no=sklearn.metrics.r2_score(y_test,model.predict(X_test))



print('Explained Variance Score:',evs_no)
print('Max Error               :',me_no)
print('Mean Absolute Error     :',mae_no)
print('Mean Square Error       :',mse_no)
print('Median Absolute Error   :',Mae_no)
print('R2 Score                :',r2_no)



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

plt.plot(history.history['mean_squared_logarithmic_error'])
plt.plot(history.history['val_mean_squared_logarithmic_error'])
plt.grid()
plt.title('model Mean Squared Logarithmic Error')
plt.ylabel('mean_squared_logarithmic_error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


