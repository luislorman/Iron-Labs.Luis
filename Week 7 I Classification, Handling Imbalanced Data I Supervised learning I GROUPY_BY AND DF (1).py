#!/usr/bin/env python
# coding: utf-8

# In[456]:


import pandas as pd


# In[457]:


data_frame_df= pd.read_csv("DATA_Customer-Churn-Copy1.csv")
data_frame_df 


# # Scenario
# ## machine learning model that will help the company identify customers that are more likely to default/churn and thus prevent losses from such customers.

# ### Round 1
# 
# Import the required libraries and modules that you would need.
# Read that data into Python and call the dataframe churnData.
# Check the datatypes of all the columns in the data. You would see that the column TotalCharges is object type. Convert this column into numeric type using pd.to_numeric function.
# Check for null values in the dataframe. Replace the null values.
# Use the following features: tenure, SeniorCitizen, MonthlyCharges and TotalCharges:
# Scale the features either by using normalizer or a standard scaler.
# Split the data into a training set and a test set.
# (optional)Fit a logistic Regression model on the training data.
# Fit a Knn Classifier(NOT KnnRegressor please!)model on the training data.

# In[458]:


churnData = data_frame_df 


# In[459]:


churnData


# In[460]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import OneHotEncoder  ##. better to use dummy from pandas 
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import boxcox
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
pd.options.display.max_rows = 50
import pandas as pd
import numpy as np;
import scipy;
## plotting libraries
from matplotlib import pyplot as plt
import seaborn as sns
## stats Libraries
from scipy import stats
import statsmodels.api as sm
## Sklearn libraries
from sklearn import model_selection
from sklearn import metrics as metrics
from sklearn import preprocessing
from sklearn import linear_model as lm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[461]:


# TotalCharges is object type. Convert this column into numeric type using pd.to_numeric function.


# In[462]:


churnData.info()


# In[463]:


churnData["TotalCharges"] [488] # there is an string here. I was told by this formula (actually its not a string, but an space, however for the machine is considered as string):
                                # pd.to_numeric(numeric, downcast ='signed')


# In[464]:


churnData.loc[churnData["TotalCharges"].str.contains(" ") == True] # ensenhar rows que tienen espacio
# formula loc dice ensenhame (valor true), las rows donde haya valor espacio 


# In[465]:


churnData=churnData.replace(" ", np.nan) #reemplazar los espacios por nan y luego eliminar la NaN. 
#es mucho mas facil quitar un NaN, ya que e suna condicion


# In[466]:


churnData.dropna(inplace=True) # eliminar los NaN y con la parte true, me aseguro que se guarde en el dataframe
churnData.reset_index(inplace=True) # tengo que resetear el indice, xq al eliminar 11 rows el index ya no queda ordenado


# In[467]:


churnData["TotalCharges"]=pd.to_numeric(churnData["TotalCharges"]) # formula to turns this column from object into numerical


# In[468]:


churnData["TotalCharges"] # i check if its a numerical


# In[469]:


churnData


#  Check for null values in the dataframe. Replace the null values. Use the following features: tenure, SeniorCitizen, MonthlyCharges and TotalCharges: Split the data into a training set and a test set. Scale the features either by using normalizer or a standard scaler.  (optional)Fit a logistic Regression model on the training data. Fit a Knn Classifier(NOT KnnRegressor please!)model on the training data.

# In[470]:


numeric.isnull() #check if there is NaN values


# In[471]:


numeric.isnull().any() #check if there is any NaN value in this column/new dataframe


# In[472]:


churnData.isnull().any() # check if there is NaN values in the dataframe


# In[473]:


churnData.info()  # vuelvo a mirar los datatype


# In[474]:


#  new target columns tenure, SeniorCitizen, MonthlyCharges and TotalCharges


# In[475]:


# We must tunr them to numericals, since we want them to  Scale the features either by using normalizer or a standard scaler
# however in this case they are already turned, next step...
# we must split the data into test and train


# In[476]:


# hay que pasar la columna Churn (target variable)a codigo binario


# In[477]:


churnData["Churn"]


# In[478]:


pd.get_dummies(churnData["Churn"]) # miro su funciona pasar a codigo binario


# In[479]:


churnData=churnData.replace({'Churn': {"Yes": 1, "No": 0}}) # reemplazar valores //// si hubiese muchas variables categoricas, puedo crear una columna nueva y elimnar la vieja


# In[480]:


churnData


# # Now we do the split 

# In[481]:


x= churnData[["tenure","SeniorCitizen","MonthlyCharges","TotalCharges"]] # tengo que poner mas de un corchete, xq son varias columnas
y=churnData["Churn"]


# In[482]:


x # funciona


# In[483]:


# SPLIT we must define x /////// formula vija

#X = data.drop(columns=["Id"], axis = 1)
#y = np.log(data['Churn'])


# In[484]:


#now we do the split


# In[485]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

X_train = pd.DataFrame(X_train, columns=x.columns)
X_test  = pd.DataFrame(X_test, columns=x.columns)


#  Scale the features either by using normalizer or a standard scaler. Split the data into a training set and a test set. (optional)Fit a logistic Regression model on the training data. Fit a Knn Classifier(NOT KnnRegressor please!)model on the training data.

#  # normalizer or a standard scaler

# In[486]:


std_scaler=StandardScaler().fit(X_train)   ##. finding the parameters ( mean, variance from the training set )

X_train_scaled=std_scaler.transform(X_train)


# In[487]:


X_test_scaled=std_scaler.transform(X_test)


# ### En train se hace el modelo y en test se evalua

# ## Fit a Knn Classifier(NOT KnnRegressor please!)model on the training data.

# In[492]:


scaler = StandardScaler()
scaler.fit(x)
X_scaled = scaler.transform(x)
X_scaled_df = pd.DataFrame(X_scaled, columns = x.columns)
display(x.head())
print()
display(X_scaled_df.head())


#  ### dos tipos de KNN =  !!!!!!!!!!!!
# regresor = estimaciond euna variable continua. Precio, altura,etc //// Target variable forecast the price (continuos)
# 
# clasificador = clasifica, genero, binario /// Tell what kind of castumer (gender or studies) can purchase my product

# ## Explicacion
# Classification: logistic regression, KNN clasifier
# 
# Regression: linear regression, KNN regressor

# In[493]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2,weights='uniform')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_pred_train=model.predict(X_train)
#model.predict_proba(inputdata)


# ## classification_plot_confusion_matrix

# Classification Model confusion matrix for training and test set

# In[494]:


from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(1,2, figsize=(14,8))

plot_confusion_matrix(model,X_train,y_train,ax=ax[0], values_format = 'd')
ax[0].title.set_text("Train Set")

plot_confusion_matrix(model,X_test,y_test,ax=ax[1],values_format = 'd')
ax[1].title.set_text("Test Set")


# 0,0 (amarillo)= prediccion era no y coincidio q es 0 - 4126
# 
# 0,1 (y) 0  predicion era 0 pero el dato era 1 -792
# 
# 1 (x),1 = predicicon era positiva y resultado fue positivo -703
# 
# 1(x),0(y)= prediccion era positia pero el resultado es negativo - 4
# 
# Datos del train test (donde se crea el mdelo q q luego se aplica en el test)

#                | Positive Prediction | Negative Prediction
# Positive Class | True Positive (TP)  | False Negative (FN)
# Negative Class | False Positive (FP) | True Negative (TN)

# 

# 

# #### classification_model_evaluating

# Classification Model Metrics

# In[496]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def evaluate_classification_model(y_train, y_pred_train, y_test, y_pred_test):
    performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                               'Train': [accuracy_score(y_train, y_pred_train),
                                         precision_score(y_train, y_pred_train),
                                         recall_score(y_train, y_pred_train)],
                               'Test': [accuracy_score(y_test, y_pred_test),
                                        precision_score(y_test, y_pred_test),
                                        recall_score(y_test, y_pred_test)]})
    
    pd.options.display.float_format = '{:.2f}'.format

    df_train = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train})
    df_test  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test})

    return performance_df, df_train, df_test

## calling the function
error_metrics_df,y_train_vs_predicted,     y_test_vs_predicted=evaluate_classification_model(y_train, y_pred_train,
                                                    y_test, y_pred)
error_metrics_df


# ### Accuracy
# cantidad de veces que acertaste una afirmación, sobre el total de datos de entrada. 

# ### Precision
# Esta métrica se define como la cantidad de casos verdaderos positivos sobre la cantidad total de todo lo que dijiste que era positivo. En otras palabras, de todo lo que el algoritmo predijo como positivo, se evalúa cuánto de eso era cierto. Uno de los ejemplos propuesto nuevamente aquí es el de marcar un correo como spam, cuando realmente no lo era.

# ### Recall
# Se compara la cantidad de casos clasificados como verdaderos positivos sobre todo lo que realmente era positivo. Y a diferencia de la anterior (precision), antes comparábamos lo que el algoritmo dice con lo que es cierto, en cambio acá, lo que él dice contra lo que no dijo que era cierto. Quizás con el ejemplo publicado otra vez por estos señores, pueda quedar claro; donde pone qué pasaría si un clasificador deja de decir que hay un caso de fraude. También, volviendo al tema de salud, imagina un algoritmo que no dice que una persona está contagiada de algún virus, y en verdad si lo está.
# 
# 
# En este caso es mucho mas importante mejorar el recall, xq es peor mandar a alguien sano a tratamiento de cancer que alguien decir que alguien q esta enfermo, no lo esta.

# In[ ]:


para subirlo se podria, cambiar los 0 o nulls por el average de la columna ---> referencia a columna
churnData.loc[churnData["TotalCharges"]
              
              Pero lo mas interesante seria enplear la fomrula near neighbor, 
              en la cual bsuca los rows mas parecidos a los q tienen ese null data y pone esos datos


# In[367]:


#Recall: cuantos de los inputs positivos, identifico como negativos, es decir
mas de la mitad de los positivos, los clasifico como negativos.
 792 vs 703


# ## (optional)Fit a logistic Regression model on the training data. 

# no es una regresion es un KNN llamado regresion pero no funciona no es una regression (nombre q confunde)
# 
# en este caso al haber categorias si o no (pasadas posteriormente a codigo binario), solo s epuded usar un modelo categorico

# In[497]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix


log_model = LogisticRegression() 

## Data splitting
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)

trans = PowerTransformer()

trans.fit(X_train)

X_train_mod = trans.transform(X_train)
X_test_mod  = trans.transform(X_test)

log_model.fit(X_train_mod, y_train)

y_pred_train_log = log_model.predict(X_train_mod)
y_pred_test_log = log_model.predict(X_test_mod)

performance_log = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                               'Train': [accuracy_score(y_train, y_pred_train_log),
                                         precision_score(y_train, y_pred_train_log),
                                         recall_score(y_train, y_pred_train_log)],
                               'Test': [accuracy_score(y_test, y_pred_test_log),
                                        precision_score(y_test, y_pred_test_log),
                                        recall_score(y_test, y_pred_test_log)]})

display(performance_log)

print("Confusion matrix for the train set")
print(confusion_matrix(y_train,y_pred_train_log))
plot_confusion_matrix(log_model,X_train_mod,y_train, values_format = 'd')
plt.show()

print()
print()

print("Confusion matrix for the test set")
print(confusion_matrix(y_test, y_pred_test_log))
plot_confusion_matrix(log_model,X_test_mod,y_test, values_format = 'd')
plt.show()


# # Round 2
# 
# ### Fit a Decision Tree Classifier on the training data.
# 
# ### Check the accuracy on the test data.
# 
# Decision tree doesnt neeed scalation

# In[498]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')


# In[499]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=11)

# Bear in mind that sklearn uses a different function for decission trees used for 
# classification ( to predict a categorical feature ): DecisionTreeClassifier() 
model = DecisionTreeClassifier(max_depth=3) # DEPHT SON LOS NIVELES EN LOS Q SE DIVIDIRA EL ESQUEMA/GRAFICAS DE ABAJO - DEPENDE DEL NIVEL Q PONGAS EL RECALL DEL 
# CUADRO DE ABAJO IRA CAMBIANDO; ASI COMO LAS GRAFICAS DE LAS SIGUIENTES LINES DE CODIGO: TE INTERESA TENER UN RECALL;ETC ALTO CON BAJOS DEPTHS

model.fit(X_train, y_train)

y_pred_train_dt = model.predict(X_train)
y_pred_test_dt = model.predict(X_test)


performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                               'Train': [accuracy_score(y_train, y_pred_train_dt),
                                         precision_score(y_train, y_pred_train_dt),
                                         recall_score(y_train, y_pred_train_dt)],
                               'Test': [accuracy_score(y_test, y_pred_test_dt),
                                        precision_score(y_test, y_pred_test_dt),
                                        recall_score(y_test, y_pred_test_dt)]})

display(performance_df)

fig, ax = plt.subplots(1,2, figsize=(14,8))


#print("Confusion matrix for the train set")
#print(confusion_matrix(y_train,y_pred_train_dt).T)
plot_confusion_matrix(model,X_train,y_train,ax=ax[0], values_format = 'd')
ax[0].title.set_text("Train Set")

#print("Confusion matrix for the test set")

#print(confusion_matrix(y_test,y_pred_test_dt).T)
plot_confusion_matrix(model,X_test,y_test,ax=ax[1],values_format = 'd')
ax[1].title.set_text("Test Set")


# ### Features Importances

# In[500]:


plt.figure(figsize=(20,14)) 
plt.barh(x.columns,model.feature_importances_)


# In[501]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (34,20))

plot_tree(model,filled = True, rounded=True,feature_names=x.columns)
plt.show() 


# In[452]:


#if its biggerthan <=10.5 goes to the right = 69.75 -- in case the the client Churn is in binar 1 . el ultimo cuadrado azul significa que esos cllientes no salen


# In[ ]:


### apartit de aqui solo sirver con variables numericas, tengo que darle otro valor al target variable


# In[502]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def performance_model(y_train, y_test, y_pred_train, y_pred_test):

    # Model validation y_train - y_pred_train
    ME_train = np.mean(y_train-y_pred_train)
    ME_test  = np.mean(y_test-y_pred_test)

    MAE_train = mean_absolute_error(y_train,y_pred_train)
    MAE_test  = mean_absolute_error(y_test,y_pred_test)

    MSE_train = mean_squared_error(y_train,y_pred_train)
    MSE_test  = mean_squared_error(y_test,y_pred_test)

    RMSE_train = np.sqrt(MSE_train)
    RMSE_test  = np.sqrt(MSE_test)

    MAPE_train = np.mean((np.abs(y_train-y_pred_train) / y_train)* 100.)
    MAPE_test  = np.mean((np.abs(y_test-y_pred_test) / y_test)* 100.)

    R2_train = r2_score(y_train,y_pred_train)
    R2_test  = r2_score(y_test,y_pred_test)

    performance = pd.DataFrame({'Error_metric': ['Mean error','Mean absolute error','Mean squared error',
                                             'Root mean squared error','Mean absolute percentual error',
                                             'R2'],
                            'Train': [ME_train, MAE_train, MSE_train, RMSE_train, MAPE_train, R2_train],
                            'Test' : [ME_test, MAE_test , MSE_test, RMSE_test, MAPE_test, R2_test]})

    pd.options.display.float_format = '{:.2f}'.format


    df_train = pd.DataFrame({'Real_value': y_train, 'Predicted_value': y_pred_train})
    df_test  = pd.DataFrame({'Real_value': y_test,  'Predicted_value': y_pred_test})

    return performance, df_train, df_test


# In[503]:


performance, _ ,_ = performance_model(y_train, y_test, y_pred_train_dt, y_pred_test_dt )
performance


# ## Visualizing the decission tree

# In[504]:


from sklearn.tree import plot_tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,10))

plot_tree(model,filled = True, rounded=True)
plt.show()


# ## Round 3
# 
# apply K-fold cross validation on your models before and check the model score. Note: So far we have not balanced the data.

# ## Cross Validation
# ## Decision Trees Regression

# In[506]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


# In[520]:


numerical = pd.read_csv("numerical.csv")
categorical = pd.read_csv("categorical.csv")
targets = pd.read_csv("target.csv")


# In[521]:


numerical


# In[522]:


categorical


# In[523]:


targets


# In[ ]:





# # ROUND 4

# fit a Random forest Classifier on the data and compare the accuracy.
# tune the hyper paramters with gridsearch and check the results.

# In[525]:


#Import scikit-learn dataset library
from sklearn import datasets

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[526]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# 
# ## tune the hyper paramters with gridsearch and check the results.

# Step 2. – Training our random forest model

# In[534]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)


# Step 3. – Making predictions with our model

# In[535]:


predictions = rf_model.predict(X_test)
predictions


# In[536]:


y_test #see values


# In[537]:


rf_model.predict_proba(X_test) # the column of 0.44 belongs to the column-variable 0 (check in the below formula)
                             #   the column wiht the first 0.56 belongs to the 1, 


# In[538]:


rf_model.classes_


# If you’d like to know how important each feature is in predicting, that’s also possible with feature_importances_:

# In[579]:


rf_model.feature_importances_ # importance of every feature


# In[580]:


len(columns)


# In[581]:


importance


# ### el valor de cada variable q sale en importance, pertenece a estas varibales de X
# 
# x= churnData[["tenure","SeniorCitizen","MonthlyCharges","TotalCharges"]]

# In[ ]:





# Managing imbalance in the dataset
# 
# Check for the imbalance.
# Use the resampling strategies used in class for upsampling and downsampling to create a balance between the two classes.
# Each time fit the model and see how the accuracy of the model is.

# In[608]:


churnData["Churn"].value_counts()


# In[611]:


NEG_CLASS_CNT = 5163   #yes = 0 y no=1


# In[613]:


print("The majority class (negative cases) represents {:.2f}% of the data".format(NEG_CLASS_CNT/len(churnData['Churn'])*100))


# In[ ]:





# In[ ]:





# ## Cross Validation
# ## Decision Trees Regression

# In[582]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


# In[583]:


X, y = load_boston(return_X_y=True)
print("X has %d rows and %d columns"  %(X.shape[0],X.shape[1]))
print("y has %d rows"  %(y.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

print("X_train has %d rows and %d columns"  %(X_train.shape[0],X_train.shape[1]))
print("-----------------------------------")
print("The coefficient of determination for the test data is R2=%.2f"
      %(model.score(X_test, y_test)))
print("The coefficient of determination for the train data is R2=%.2f"
      %(model.score(X_train, y_train)))


# In[584]:


scores=cross_val_score(model, X_train, y_train, cv=5)
print("Cross validation scores: ", scores)
print("Score stats: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# # # Decision Trees Classification

# In[585]:


X, y = load_iris(return_X_y=True)
print("X has %d rows and %d columns"  %(X.shape[0],X.shape[1]))
print("y has %d rows"  %(y.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("The (mean) accuracy on the test set is %.2f" %(model.score(X_test, y_test)))
print("The (mean) accuracy on the train data is %.2f" %(model.score(X_train, y_train)))


# # Models Comparision (3 regression estimators)

# In[586]:


model1 = DecisionTreeRegressor()
model2 = LinearRegression()
model3 = KNeighborsRegressor()

model_pipeline = [model1, model2, model3]
model_names = ['Regression Tree', 'Linear Regression', 'KNN']
scores = {}
i=0
for model in model_pipeline:
    mean_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    scores[model_names[i]] = mean_score
    i = i+1
print(scores)


# In[588]:


print("Comparing the 3 regression scores we find \n")

pd.DataFrame([scores], index=["score"])


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[359]:


X_train_const_scaled = sm.add_constant(X_train_scaled) # adding a constant

model = sm.OLS(y_train, X_train_const_scaled).fit()
predictions_train = model.predict(X_train_const_scaled) 

X_test_const_scaled = sm.add_constant(X_test_scaled) # adding a constant
predictions_test = model.predict(X_test_const_scaled) 
print_model = model.summary()
print(print_model)


# In[ ]:





# In[432]:


for i in range(0,10):
    print(i)


# In[ ]:





# In[383]:



print(sum(list.values()))


# # HOW TO CREATEA A DATAFRAME WIHT VALUES I WANT  BY COLUMN AND THEN FILTER IT BY GROUP BY AN D CONDITION

# In[ ]:


# total sume of number A


# In[415]:


data = {'Company': ["A","B","A","B","A","B","A","B","A","B","A","B","A","B","A","B",], 
        'Value': [1,2,3,5,6,3,6,23,6,3,6,-5,-5,12,5,2,]
       }

# pass column names in the columns parameter 
df = pd.DataFrame(data)
df


# In[416]:


df.groupby("Company")['Value'].sum() ##las funciones van en parentesis 


# In[417]:


df.groupby("Company").filter(lambda x: df['Company'] == 'A') # error


# In[418]:



after = df.groupby('Company').sum()


# In[419]:


after


# In[420]:


df[df['Company'] == 'A'].groupby('Company').sum()


# In[ ]:




