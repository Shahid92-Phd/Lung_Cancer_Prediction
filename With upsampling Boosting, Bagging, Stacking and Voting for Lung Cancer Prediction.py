#!/usr/bin/env python
# coding: utf-8

# In[68]:


#pip install catboost


# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer , StandardScaler ,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score,ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance
from sklearn.linear_model import LogisticRegression , RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from numpy import absolute
from numpy import sqrt
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier , GradientBoostingClassifier , VotingClassifier , RandomForestClassifier


# ### Dataset Description

# In[70]:


data = pd.read_csv("Lung_Cancer_Dataset.csv")
data.head(10)


# In[71]:


data.tail()


# In[72]:


#Shape of Data
data.shape


# In[73]:


data.info()


# In[74]:


#lets describe the data
data.describe()


# In[75]:


data.count()


# In[76]:


data.isnull().sum()


# ## Exploarory Data Analysis

# In[77]:


# sns.pairplot(data)


# In[78]:


data.plot(color = 'green', kind='box', figsize=(25, 15), subplots=True, layout=(5,4))
plt.show()


# In[79]:


#histogram
data.hist(color='blue',bins=10,figsize=(17,12))
plt.show()


# In[80]:


#Boxplot of each column
data.plot( kind='density', figsize=(20,18), subplots=True, layout=(6,3),sharex=False)

plt.show()


# In[81]:


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(12,8))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[82]:


import plotly.graph_objects as go


# In[83]:


column_names = data.columns
no_of_boxes = len(column_names)
colors = [ 'hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, no_of_boxes)]

fig = go.Figure(data = [go.Box(y = data.loc[:, column_names[i]], marker_color = colors[i], name = column_names[i], boxmean = True, showlegend = True) for i in range(no_of_boxes)])

fig.update_layout(
    xaxis=dict(showgrid = True, zeroline = True, showticklabels = True),
    yaxis=dict(zeroline = True, gridcolor = 'white'),
    paper_bgcolor = 'rgb(233,233,233)',
    plot_bgcolor = 'rgb(233,233,233)')

fig.show()


# In[84]:


plt.figure(figsize=(8,8))
sns.heatmap(data.isnull())


# In[85]:


ob_series = pd.Series(data['LC'])
# Plot the pie chart
plt.figure(figsize=(15, 5))
data['LC'].value_counts().plot.pie(explode=None, autopct='%1.1f%%', shadow=True)
plt.title('Distribution of LC')
plt.show()


# In[86]:


#check the group by values
data['LC'].value_counts()


# ## FacetGrid for each Attribute

# In[87]:


data.columns


# In[88]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(data, hue="GD", aspect=4)
fig.map(sns.kdeplot, 'LC', shade=True,)
oldest = data['LC'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# ## Data Sampling

# In[89]:


data_boxcox = data.copy()
X = data_boxcox.drop(["LC"],axis=1)
Y = data_boxcox["LC"]


# In[90]:


#To keep BoxCox data as it is to use the same for later.
data_bal = data_boxcox.copy()

#Getting seperated data with 1 and 0 status.
df_majority = data_bal[data_bal.LC==0]
df_minority = data_bal[data_bal.LC==1]

#Here we are downsampling the Majority Class Data Points. 
#i.e. We will get equal amount of datapoint as Minority class from Majority class

df_manjority_downsampled = resample(df_majority,replace=True,n_samples=267,random_state=123)
df_downsampled = pd.concat([df_manjority_downsampled,df_minority])
print("Downsampled data:->\n",df_downsampled.LC.value_counts())

#Here we are upsampling the Minority Class Data Points. 
#i.e. We will get equal amount of datapoint as Majority class from Minority class
df_monority_upsampled = resample(df_minority,replace=True,n_samples=100,random_state=123)
df_upsampled = pd.concat([df_majority,df_monority_upsampled])
print("Upsampled data:->\n",df_upsampled.LC.value_counts())


# ## Data Splitting

# In[91]:


#classes = ["0", "1","2","3","4","5","6"]
X = df_downsampled.drop(['LC'],axis=1)
Y = df_downsampled['LC']
X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25,random_state=1)


# In[92]:


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import itertools
import xgboost as xgb
from sklearn.metrics import confusion_matrix


# ## Proposed Stacking and Voting

# In[93]:


import numpy as np
import pandas as pd
from pandas_summary import DataFrameSummary
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
#from sklearn.inspection import plot_partial_dependence  # newly learnt this time!

#from sklearn.metrics import classification_report, recall_score, plot_confusion_matrix, plot_precision_recall_curve, roc_curve
from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef

#import matplotlib as mlp
#from matplotlib import Artist
#from matplotlib.artist import Artist
#import seaborn as sns
import warnings
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm


plt.rcParams['axes.unicode_minus'] = False
plt.style.use('fivethirtyeight')
sns.set(font_scale = 1)  
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

print("Let's start!")


# In[94]:


display(df_downsampled.shape, df_downsampled.head())


# In[95]:


df_info = DataFrameSummary(df_downsampled)
df_info.summary().T


# In[96]:


# Or customize according to what you'd like to know.

df_info2 = pd.DataFrame(columns=['Name of Col', 'Num of Null', 'Dtype', 'N_unique'])

for i in range(0, len(df_downsampled.columns)):
    df_info2.loc[i] = [df_downsampled.columns[i],
                      df_downsampled[df_downsampled.columns[i]].isnull().sum(),
                      df_downsampled[df_downsampled.columns[i]].dtypes,
                      df_downsampled[df_downsampled.columns[i]].nunique()]
    
df_info2


# In[97]:


dtype = pd.DataFrame(df_info.summary().loc['types'] == 'numeric')
num_cols = dtype[dtype['types'] == True].index.to_list()
num_cols


# In[98]:


cat_cols = list(set(df_downsampled.columns) - set(num_cols))
cat_cols.remove('GD')

cat_cols


# In[99]:


df_downsampled.columns


# In[100]:


import pycaret
from pycaret.classification import *


# In[101]:


cat_cols = list(set(df_downsampled.columns) - set(num_cols))
cat_cols.remove('LC')

cat_cols


# In[102]:


setup(data = df_downsampled, 
      target = 'LC',
      session_id = 42,
      preprocess = True,
      index=False,
      numeric_features = cat_cols
     )
#       silent = True


# In[103]:


df_downsampled['LC'].value_counts()


# In[104]:


top5 = compare_models()


# In[105]:


dt = create_model('dt')


# In[106]:


evaluate_model(dt)


# In[107]:


et = create_model('et')


# In[108]:


evaluate_model(et)


# In[109]:


rf = create_model('rf')


# In[110]:


evaluate_model(rf)


# In[111]:


catboost = create_model('catboost')


# In[112]:


evaluate_model(catboost)


# In[113]:


lightgbm = create_model('lightgbm')


# In[114]:


evaluate_model(lightgbm)


# In[115]:


xgboost = create_model('xgboost')


# In[116]:


evaluate_model(xgboost)


# ## Tuning each Model

# In[117]:


tuned_dt = tune_model(dt, optimize = 'Accuracy')


# In[118]:


tuned_et = tune_model(et, optimize = 'Accuracy')


# In[119]:


tuned_rf = tune_model(rf, optimize = 'Accuracy')


# In[120]:


tuned_catboost= tune_model(catboost, optimize = 'Accuracy')


# In[121]:


tuned_lightgbm = tune_model(lightgbm, optimize = 'Accuracy')


# In[122]:


tuned_xgboost = tune_model(xgboost, optimize = 'Accuracy')


# In[123]:


evaluate_model(tuned_dt)


# In[124]:


evaluate_model(tuned_et)


# In[125]:


evaluate_model(tuned_rf)


# In[126]:


evaluate_model(tuned_catboost)


# In[127]:


evaluate_model(tuned_lightgbm)


# In[128]:


evaluate_model(tuned_xgboost)


# ## Voting Ensembles

# In[129]:


blend_soft = blend_models(estimator_list =[tuned_dt,tuned_et,tuned_rf,tuned_catboost,tuned_lightgbm],  method = 'soft')


# In[130]:


evaluate_model(blend_soft)


# In[131]:


blend_hard = blend_models(estimator_list =[tuned_dt,tuned_et,tuned_rf,tuned_catboost,tuned_lightgbm],  method = 'hard')


# In[132]:


evaluate_model(blend_hard)


# ## Stacking Ensembles

# In[133]:


stack_model = stack_models(estimator_list = [tuned_dt,tuned_et,tuned_rf,tuned_catboost,tuned_lightgbm], meta_model = tuned_dt, optimize = 'Accuracy')


# In[134]:


evaluate_model(stack_model)


# ## END

# ## Final Model

# In[135]:


final_model = finalize_model(stack_model)


# In[136]:


evaluate_model(final_model)


# ## Top Performers

# In[137]:


et	Extra Trees Classifier	0.9946	1.0000	0.9526	1.0000	0.9751	0.9721	0.9728	0.0480
rf	Random Forest Classifier	0.9928	1.0000	0.9368	1.0000	0.9668	0.9628	0.9638	0.0580
catboost	CatBoost Classifier	0.9916	0.9945	0.9263	1.0000	0.9611	0.9564	0.9577	0.7820
lightgbm	Light Gradient Boosting Machine	0.9910	0.9930	0.9211	1.0000	0.9576	0.9526	0.9544	0.0930
xgboost	Extreme Gradient Boosting	0.9898	0.9847	0.9105	1.0000	0.9522	0.9465	0.9484	0.0250
gbc	Gradient Boosting Classifier	0.9868	0.9846	0.8842	1.0000	0.9371	0.9298	0.9329	0.0410
lr	Logistic Regression	0.9802	0.9647	0.8251	1.0000	0.9031	0.8922	0.8980	0.6910
ada	Ada Boost Classifier	0.9778	0.9711	0.8041	1.0000	0.8900	0.8779	0.8851	0.0280
lda	Linear Discriminant Analysis	0.9743	0.9565	0.8096	0.9576	0.8759	0.8617	0.8664	0.0080
knn	K Neighbors Classifier	0.9671	0.9627	0.7196	0.9875	0.8286	0.8111	0.8257	0.0140
ridge	Ridge Classifier	0.9629	0.0000	0.7032	0.9587	0.8060	0.7863	0.8008	0.0080
qda	Quadratic Discriminant Analysis	0.9569	0.9813	0.9471	0.7521	0.8359	0.8117	0.8204	0.0090
svm	SVM - Linear Kernel	0.9263	0.0000	0.4673	0.8877	0.5505	0.5213	0.5733	0.0090
nb	Naive Bayes	0.9072	0.9335	0.7728	0.5938	0.6622	0.6110	0.6237	0.0090
dummy	Dummy Classifier	0.8868	0.5000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0130


# In[ ]:


lr = create_model('lr')


# In[75]:


ada = create_model('ada')


# In[76]:


lda = create_model('lda')


# In[81]:


nb = create_model('nb')


# In[78]:


blend_soft = blend_models(estimator_list =[lr, ada, lda],  method = 'soft')


# In[82]:


stack_model = stack_models(estimator_list = [ada, lda, nb], meta_model = dt, optimize = 'Accuracy')


# In[ ]:




