
# coding: utf-8

# # Synopsys 

# ## Load Libraries

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import random
import mglearn
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf 




# Loading the data (569 30-dimensional entries)

# In[2]:


data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print ( "Sample counts per class: \n {}" . format ( { n : v for n , v in zip ( data.target_names , np.bincount(data.target ))}))
print ( "Feature names: \n {}".format(data.feature_names))

iterations = 20


# Splitting the data into training and test sets

# In[3]:


#random_seed = random.randint(10, 42)
random_seed = None
train, test,train_labels, test_labels = train_test_split(features,labels, test_size=0.33, random_state = random_seed)

print("before scaling ",train.shape)
print("random_seed ",random_seed)


# Creating scaler 
# see https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn

# ## Objectives of principal component analysis
# - PCA reduces attribute space from a larger number of variables to a smaller number of factors and as such is a "non-dependent" procedure (that is, it does not assume a independent variable is specified).
# - PCA is a dimensionality reduction or data compression method. The goal is dimension reduction and there is no guarantee that the dimensions are interpretable (a fact often not appreciated by (amateur) statisticians).
# - To select a subset of variables from a larger set, based on which original variables have the highest correlations with the principal component.
# 

# # Below is an unrelated 2D example only for illustration

# In[4]:


fig , axes = plt.subplots ( 2 , 1 , figsize = ( 10 , 20 ))
fig.clear()
mglearn.plots.plot_pca_illustration()
plt.plot()
plt.show()


# # Data Transformations - Scaling. 
# We use StandardScaler for preprocessing the data.

# In[5]:


df = pd.DataFrame(train, columns=feature_names)

# normalize data
data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train)
# Apply scale transform to both the training set and the test set.
train = scaler.transform(train)
test = scaler.transform(test)

# Make an instance of the PCA transformation
#pca = PCA(.95)
pca = PCA(n_components=5)
pca.fit(train)

#print(data_scaled.columns)

print pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5'])


#print("Explained Variance: %s") % pca.explained_variance_ratio_
#print("Explained Variance: %s") % pca.explained_variance_
#print("Explained Variance: %s") % pca.get_params()
#print("Explained Variance: %s") % pca.n_components_
#print("Explained Variance: %s") % pca.get_covariance()

#print(pca.components_)

var_exp = pca.explained_variance_ratio_
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 5))

    plt.bar(range(5), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(5), np.cumsum(var_exp), where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.plot()
    plt.show()



# In[6]:


#import mglearn
fig , axes = plt.subplots ( 10 , 3 , figsize = ( 10 , 20 ))
malignant = data.data[data.target==0 ] 
benign = data.data[data.target == 1 ] 
ax = axes.ravel()
for i in range ( 30 ): 
    _ , bins = np.histogram( data.data[:,i], bins = 50 )
    ax[i].hist(malignant [:, i ], bins = bins , color = mglearn.cm3(1), alpha =.7 )
    ax[i].hist ( benign [:, i ], bins = bins , color = mglearn.cm3(2), alpha =.7 )
    ax[i].set_title ( data.feature_names [i])
    ax[i].set_yticks (())
    ax[i].set_xlabel ( "Feature magnitude" ) 
    ax[i].set_ylabel ( "Frequency" ) 
    ax[i].legend ([ "malignant" , "benign" ], loc = "best" )    
plt.plot()
plt.show()


# # Perfom PCA

# In[7]:


train_non_pca = train
test_non_pca = test
train = pca.transform(train)
test = pca.transform(test)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train)

print(feature_columns)


# transform data onto the first two principal components 
#X_pca = pca.transform ( X_scaled )
print ( "Original shape before PCA: {}".format(str( train_non_pca.shape ))) 
print ( "Reduced shape after PCA: {}" . format ( str ( train.shape )))
#print ( "PCA component shape: {}" . format ( pca.components_.shape ))

plt . matshow ( pca . components_ , cmap = 'viridis' )
plt . yticks ([ 0 , 1, 2 , 3, 4 ], [ "First component" , "Second component","Third component","Fourth component","Fifth component" ]) 
plt.colorbar ()
plt . xticks ( range ( len ( data.feature_names )), data.feature_names , rotation = 60 , ha = 'left' )
plt . xlabel ( "Feature" )
plt . ylabel ( "Principal components" )
plt.plot()
plt.show()


# ## Visualization after PCI - Principal Components 1 and 2

# In[8]:


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA of the cancer data', fontsize = 20)
targets = ['benign', 'malignant']
colors = ['r', 'b']
# Loop and create train_malignant and train_benign


train_malignant = []
train_benign = []

print("==>train.shape", train.shape)
for index in range(train.shape[0]):
#    print np.array(train[index]), train_labels[index]
    
    if train_labels[index]==1:
        train_malignant.append(train[index])
    else:
        train_benign.append(train[index])

np_train_m = np.array(train_malignant)
np_train_b = np.array(train_benign)


ax.scatter(np_train_m[:,0],np_train_m[:,1],color='r')
ax.scatter(np_train_b[:,0],np_train_b[:,1],color='b')

ax.legend(targets)
ax.grid()
plt.plot()
plt.show()


# # Logistic regression on original (30 features) dataset

# In[9]:


def logistic_regression_30_features():
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify= cancer.target, random_state = None)

    # normalize data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    logreg = LogisticRegression().fit(X_train,y_train)

    print ("Test set score: {:.6f}".format(logreg.score(X_test,y_test)))
    return logreg.score(X_test,y_test)


# In[10]:



lr = []
for i in range(1,iterations):
    accuracy_logistic_regression = logistic_regression_30_features()
    lr.append(accuracy_logistic_regression)
print(lr)
    


# ## Logistic regression on PCA reduced set

# In[11]:


#svc = svm.SVC(C=10)
#svc.fit(train,train_labels)
#preds_svc_reduced = svc.predict(test)
#accuracy_svc_reduced = accuracy_score(test_labels,preds_svc_reduced)
#print(train.shape)
#print("accuracy_svm_reduced=%f "%accuracy_svc_reduced)

def logistic_regression_30_features_reduced():
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify= cancer.target, random_state = None)

    # normalize data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=5)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    
    logreg = LogisticRegression().fit(X_train,y_train)

    #print ("Train set score: {:.6f}".format(logreg.score(X_train,y_train)))
    #preds_train_lgr = logreg.predict(X_train)
    #accuracy_logistic_regression_train = accuracy_score(y_train,preds_train_lgr)
    #print ("Train set score predict: {:.6f}".format(accuracy_logistic_regression_train))
    print ("Test set score pca: {:.6f}".format(logreg.score(X_test,y_test)))
    #preds_test_lgr = logreg.predict(X_test)
    #accuracy_logistic_regression_test = accuracy_score(y_test,preds_test_lgr)
    #print ("Train set score predict: {:.6f}".format(accuracy_logistic_regression_test))
    return logreg.score(X_test,y_test)



# In[12]:


lrp_pca = []
for i in range(1,iterations):
    accuracy_logistic_regression_reduced = logistic_regression_30_features_reduced()
    #print("accuracy_logistic_regression_reduced=%f "%accuracy_logistic_regression_reduced)
    lrp_pca.append(accuracy_logistic_regression_reduced)
print(lrp_pca)


# # SVM on original (30 features) dataset

# In[13]:


#svc = svm.SVC(C=10)
#svc.fit(train_non_pca,train_labels)
#preds_svc = svc.predict(test_non_pca)
#accuracy_svc = accuracy_score(test_labels,preds_svc)
#print(train_non_pca.shape)
#print("accuracy_svm=%f "%accuracy_svc)


def svm_30_features():
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify = cancer.target, random_state = None)

    # normalize data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Build Linear Support Vector Machine
    linearSVC = LinearSVC(C=1)
    linearSVC.fit(X_train,y_train)

    #print ("SVM:Train set score: {:.6f}".format(linearSVC.score(X_train,y_train)))
    #preds_train_svc = linearSVC.predict(X_train)
    #accuracy_svc_train = accuracy_score(y_train,preds_train_svc)
    #print ("SVM:Train set score predict: {:.6f}".format(accuracy_svc_train))
   
    print ("SVM:Test set score: {:.6f}".format(linearSVC.score(X_test,y_test)))
    #preds_test_svc = linearSVC.predict(X_test)
    #accuracy_svc_test = accuracy_score(y_test,preds_test_svc)
    #print ("SVM:Test set score predict: {:.6f}".format(accuracy_svc_test))
    return linearSVC.score(X_test,y_test)


# In[14]:


svm = []
for i in range(1,iterations):
    accuracy_svc = svm_30_features()
    #print("accuracy_svm=%f "%accuracy_svm)
    svm.append(accuracy_svc)
print(svm)


# ## SVM regression on PCA reduced set

# In[15]:


#svc = svm.SVC(C=10)
#svc.fit(train,train_labels)
#preds_svc_reduced = svc.predict(test)
#accuracy_svc_reduced = accuracy_score(test_labels,preds_svc_reduced)
#print(train.shape)
#print("accuracy_svm_reduced=%f "%accuracy_svc_reduced)

def svm_30_features_reduced():
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify = cancer.target, random_state = None)
  
    # normalize data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=5)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    
    # Build Linear Support Vector Machine
    linearSVC = LinearSVC(C=1)
    linearSVC.fit(X_train,y_train)

    #print ("SVM:Train set score: {:.6f}".format(linearSVC.score(X_train,y_train)))
    #preds_train_svc = linearSVC.predict(X_train)
    #accuracy_svc_train = accuracy_score(y_train,preds_train_svc)
    #print ("SVM:Train set score predict: {:.6f}".format(accuracy_svc_train))
   
    print ("SVM:Test set score pca: {:.6f}".format(linearSVC.score(X_test,y_test)))
    #preds_test_svc = linearSVC.predict(X_test)
    #accuracy_svc_test = accuracy_score(y_test,preds_test_svc)
    #print ("SVM:Test set score predict: {:.6f}".format(accuracy_svc_test))
    return linearSVC.score(X_test,y_test)



# In[16]:


svm_pca = []
for i in range(1,iterations):
    accuracy_svc_reduced = svm_30_features_reduced()
    #print("accuracy_svm_reduced=%f "%accuracy_svc_reduced)
    svm_pca.append(accuracy_svc_reduced)
print(svm_pca)


# # NEURAL NETWORKS
# 

# In[17]:


#import graphviz
#mglearn.plots.plot_two_hidden_layer_graph()
#import graphviz
#display ( mglearn . plots . plot_single_hidden_layer_graph ())
#plt.plot()
#plt.show()


# ## Deep Neural Network with 30 features and Deep Neural Network with 5 features (PCA reduced)

# ## 1. Build Deep Neural Network with 10,20,10 hidden layers trained with PCA reduced data set

# In[18]:


import tensorflow as tf


# Define the deep neural network: 
# - 3 hidden layers with 10,20 and 10 units respectively. The last layer is neuron with output 0 or 1
# - It uses the reduced dimensionality training data

# In[19]:


def NeuralNetwork(train,test, train_labels, test_labels,hidden_units):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train)
    classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_units, n_classes=2)
    classifier_tf.fit(train, train_labels, steps=1000)

    predictions = list(classifier_tf.predict(test, as_iterable=True))

    print(test_labels.shape)
    print(np.array(predictions).shape)
    
    return accuracy_score(test_labels,np.array(predictions))


# In[20]:


##data_scaled.columns
##accuracy_10_20_10_reduced = NeuralNetwork(train,test,data_scaled.columns,train_labels,[10, 20, 10])
#accuracy_10_20_10_reduced = NeuralNetwork(train,test,train_labels,[10, 20, 10])
#print("Neural Network Accuracy Reduced: %f" % accuracy_10_20_10_reduced )


nn_10_20_10_pca = []
for i in range(1,iterations):
    # 1. Load Data
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify = cancer.target, random_state = None)

    df = pd.DataFrame(X_train, columns=cancer['feature_names'])

    # 2. normalize data
    data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # 3. PCA Reduction
    pca = PCA(n_components=5)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # 4. Train Neural Network
    accuracy_10_20_10_reduced = NeuralNetwork(X_train,X_test,y_train,y_test, [10, 20, 10])

    print("Neural Network Accuracy Reduced: %f" % accuracy_10_20_10_reduced )
    nn_10_20_10_pca.append(accuracy_10_20_10_reduced)
print(nn_10_20_10_pca)


# ## 2. Build Deep Neural Network with 10,20,10 hidden layers trained on 30 features dataset

# In[21]:


##### The commented out data is with 30 variables.
def NeuralNetworkNotReduced(hidden_units):
    cancer_data = load_breast_cancer()
    print("Number of instances in the dataset: %d" % len(cancer_data.target))

    in_train, in_test, out_train, out_test = train_test_split(cancer_data['data'],cancer_data['target'],test_size=0.33, random_state = None)

    data_scaler = StandardScaler()
    ##### Fit train data 
    data_scaler.fit(in_train)
    in_train = data_scaler.transform(in_train)
    in_test = data_scaler.transform(in_test)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(in_train)
    classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
     hidden_units=hidden_units, 
     n_classes=2)
    classifier_tf.fit(in_train, out_train, steps=1000)
    predictions = list(classifier_tf.predict(in_test, as_iterable=True))

    print(out_test)
    print(predictions)
    # accuracy_10_20_10 = accuracy_score(out_test,predictions)
    # print("Neural Network Accuracy: %f" % accuracy_10_20_10 )
    return accuracy_score(out_test,predictions)


# In[22]:


nn_10_20_10 = []
for i in range(1,iterations):
    accuracy_10_20_10 = NeuralNetworkNotReduced([10,20,10])
    print("Neural Network Accuracy: %f" % accuracy_10_20_10 )
    nn_10_20_10.append(accuracy_10_20_10)
print(nn_10_20_10)


# # Shallow Neural Network with 30 features and Shallow Neural Network with 5 features (PCA reduced)

# ## 3. Build Shallow Neural Network with 1 hidden layer, contianing 1 neurons trained with PCA reduced data set

# In[23]:



#accuracy_2_2_reduced = NeuralNetwork(train,test,data_scaled.columns,train_labels,[2])
#accuracy_2_2_reduced = NeuralNetwork(train,test,train_labels,[1])
#print("Neural Network Accuracy Reduced: %f" % accuracy_2_2_reduced )



##data_scaled.columns
##accuracy_10_20_10_reduced = NeuralNetwork(train,test,data_scaled.columns,train_labels,[10, 20, 10])
#accuracy_10_20_10_reduced = NeuralNetwork(train,test,train_labels,[10, 20, 10])
#print("Neural Network Accuracy Reduced: %f" % accuracy_10_20_10_reduced )

nn_2_pca = []
for i in range(1,iterations):
    # 1. Load Data
    cancer = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target,stratify = cancer.target, random_state = None)
  
    df = pd.DataFrame(X_train, columns=cancer['feature_names'])

    # 2. normalize data
    data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply scale transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # 3. PCA Reduction
    pca = PCA(n_components=5)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # 4. Train Neural Network
    accuracy_2_2_reduced = NeuralNetwork(X_train,X_test,y_train,y_test, [1])

    print("Neural Network Accuracy Reduced: %f" % accuracy_2_2_reduced )
    nn_2_pca.append(accuracy_2_2_reduced)
    
print(nn_2_pca)


# ## 4. Build Shallow Neural Network with 1 hidden layer with 2 neurons trained on 30 features dataset

# In[24]:


nn_2 = []
for i in range(1,iterations):
    accuracy_2_2 = NeuralNetworkNotReduced([1])
    print("Neural Network Accuracy: %f" % accuracy_2_2 )
    nn_2.append(accuracy_2_2)
print(nn_2)


# # SUMMARY

# In[29]:


print("training data size: ",train_non_pca.shape) 
print("training data size with pca reduced features: ",train.shape) 
print("---------------------------------------")
print("accuracy_logistic_regression=%f "%np.mean(lr))
print("accuracy_logistic_regression_pca=%f "% np.mean(lrp_pca))
print("---------------------------------------")
print("accuracy_svm=%f "%np.mean(svm))
print("accuracy_svm_pca=%f "%np.mean(svm_pca))
print("---------------------------------------")
print("Accuracy Neural Network [10,20,10]  : %f" % np.mean(nn_10_20_10) )
print("Accuracy Neural Network [10,20,10]  pca: %f" % np.mean(nn_10_20_10_pca) )
print("Accuracy Neural Network [1] : %f" % np.mean(nn_2) )
print("Accuracy Neural Network [1]  pca: %f" % np.mean(nn_2_pca) )



# # Conclusion

# - The dataset is lineary separable (see 2 component graphic)
# - That is the reason why the logistic regression and the neural network perform similarly
# - because of the linear separabiltiy of the dataset, the neural network performs very well with 1 hidden layer only.
# - logistic regression with PCA has slightly better performance, compared to the remaining

# In[26]:


#lr=[0.978723,0.957447,0.984043,0.973404, 0.984043]
#lrp_pca=[0.984043,0.962766,0.978723,0.973404,0.968085]
#svm=[0.984043, 0.962766, 0.957447, 0.978723,0.962766]
#svm_pca=[0.952128,0.946809,0.941489,0.936170, 0.946809]
#nn_10_20_10=[0.930070,0.965035,0.979021,0.986014,0.984043]
#nn_10_20_10_pca=[0.968085,0.984043,0.973404,0.973404,0.973404]
#nn_2=[0.978723,0.984043,0.973404,0.957447, 0.968085]
#nn_2_pca=[0.97202797202797198, 0.95804195804195802, 0.97902097902097907, 0.98601398601398604, 0.965034965034965, 0.965034965034965, 0.97202797202797198, 0.965034965034965, 0.97202797202797198, 0.97902097902097907, 0.97902097902097907, 0.965034965034965, 0.97202797202797198, 0.95804195804195802, 0.94405594405594406, 0.97202797202797198, 0.965034965034965, 0.95804195804195802, 0.99300699300699302]


# In[31]:


plt.hist(lr, bins = 30 , color = 'gray' )
plt.hist(lrp_pca, bins = 30 , color = 'black' )
plt.hist(svm, bins = 30 , color = 'green' )
plt.hist(svm_pca, bins = 30 , color = 'blue' )
plt . ylim ( 0 , 15 )
plt . xlim ( 0.95 , 1.0 )
plt . title ( "Linear Classifier and SVM" )
plt. legend(["linear regression","linear regression pca","svm","svm pca"])
plt.show()
plt.hist(nn_10_20_10, bins = 30 , color = 'red' )
plt.hist(nn_10_20_10_pca, bins = 30 , color = 'brown' )
plt.hist(nn_2, bins = 30 , color = 'orange' )
plt.hist(nn_2_pca, bins = 30 , color = 'magenta' )
plt . ylim ( 0 , 15 )
plt . xlim ( 0.95 , 1.0 )
plt . title ( "Deep and Shallow neural networks" )
plt. legend(["nn [10,20,10]","nn [10,20,10] with pca","nn [2]","nn [2] with pca"])
plt.show()

