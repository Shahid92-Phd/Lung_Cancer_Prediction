#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


data = pd.read_csv("Lung_Cancer_Dataset.csv")
data.head(10)


# In[3]:


data.tail()


# In[4]:


#Shape of Data
data.shape


# In[5]:


data.info()


# In[6]:


#lets describe the data
data.describe().T


# In[7]:


data.count()


# In[8]:


data.isnull().sum()


# In[9]:


cols =data.columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))


# In[10]:


data['LC'].value_counts()


# ## Exploarory Data Analysis

# In[11]:


# sns.pairplot(data)


# In[12]:


data.plot(color = 'green', kind='box', figsize=(25, 15), subplots=True, layout=(5,4))
plt.show()


# In[13]:


#histogram
data.hist(color='blue',bins=10,figsize=(17,12))
plt.show()


# In[14]:


#Boxplot of each column
data.plot( kind='density', figsize=(20,18), subplots=True, layout=(6,3),sharex=False)

plt.show()


# In[15]:


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,12))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[16]:


import plotly.graph_objects as go


# In[17]:


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


# In[18]:


data.columns


# In[19]:


# Plotting boxplots to numeric features
num_var = ['GD', 'AG', 'SK', 'YF', 'AT', 'PP', 'CD', 'FT', 'AL', 'WZ', 'AC', 'CO',
       'SB', 'SD', 'CP', 'LC']
plt.figure(figsize=(20,10))
sns.boxplot(data=data[num_var],
                 palette="colorblind")
plt.title('Numerical features outlayers');


# In[20]:


#reading data from pandas
col_names = ['GD', 'AG', 'S1', 'Y0', 'AN', 'PP', 'CD', '0T', 'AL', 'WH', 'AC', 'C0','SB', 'SD', 'CP', 'LC']
#df = pd.read_csv("Obesity_Dataset.csv", header=1, names=col_names)


# In[21]:


plt.figure(figsize=(8,8))
sns.heatmap(data.isnull())


# In[22]:


ob_series = pd.Series(data['LC'])
# Plot the pie chart
plt.figure(figsize=(15, 5))
data['LC'].value_counts().plot.pie(explode=None, autopct='%1.1f%%', shadow=True)
plt.title('Distribution of LC')
plt.show()


# In[23]:


#check the group by values
data['LC'].value_counts()


# ## FacetGrid for each Attribute

# In[24]:


data.columns


# In[25]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(data, hue="LC", aspect=4)
fig.map(sns.kdeplot, 'GD', shade=True,)
oldest = data['GD'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# ## Data Splitting

# In[26]:


#classes = ["0", "1","2","3","4","5","6"]
X = data.drop(['LC'],axis=1)
Y = data['LC']
X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25,random_state=1)


# In[27]:


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import itertools
import xgboost as xgb
from sklearn.metrics import confusion_matrix


# ## Proposed Stacking and Voting

# In[28]:


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


# In[29]:


display(data.shape, data.head())


# In[30]:


data['LC'].value_counts()


# In[31]:


df_info = DataFrameSummary(data)
df_info.summary().T


# In[38]:


# Or customize according to what you'd like to know.

df_info2 = pd.DataFrame(columns=['Name of Col', 'Num of Null', 'Dtype', 'N_unique'])

for i in range(0, len(data.columns)):
    df_info2.loc[i] = [data.columns[i],
                      data[data.columns[i]].isnull().sum(),
                      data[data.columns[i]].dtypes,
                      data[data.columns[i]].nunique()]
    
df_info2


# In[44]:


dtype = pd.DataFrame(df_info.summary().loc['types'] == 'numeric')
num_cols = dtype[dtype['types'] == True].index.to_list()
num_cols


# In[45]:


cat_cols = list(set(data.columns) - set(num_cols))
cat_cols.remove('LC')

cat_cols


# In[48]:


cat_cols = list(set(data.columns) - set(num_cols))
cat_cols.remove('LC')
cat_cols


# In[47]:


import pycaret
from pycaret.classification import *


# In[43]:


setup(data = data, 
      target = 'LC',
      session_id = 42,
      preprocess = True,
      index=False,
      numeric_features = cat_cols
     )
#       silent = True


# In[112]:


models()


# In[113]:


top5 = compare_models()


# In[114]:


rf = create_model('rf')


# In[115]:


evaluate_model(rf)


# In[116]:


dt = create_model('dt')


# In[117]:


evaluate_model(dt)


# In[118]:


lda = create_model('lda')


# In[119]:


evaluate_model(lda)


# In[120]:


lr = create_model('lr')


# In[121]:


evaluate_model(lr)


# In[122]:


gbc = create_model('gbc')


# In[123]:


evaluate_model(gbc)


# In[124]:


lightgbm = create_model('lightgbm')


# In[125]:


evaluate_model(lightgbm)


# In[126]:


nb = create_model('nb')


# In[127]:


evaluate_model(nb)


# In[128]:


ridge = create_model('ridge')


# In[129]:


evaluate_model(ridge)


# ## Tuning each Model

# In[130]:


tuned_lda = tune_model(lda, optimize = 'Accuracy')


# In[131]:


tuned_lr = tune_model(lr, optimize = 'Accuracy')


# In[132]:


tuned_gbc = tune_model(gbc, optimize = 'Accuracy')


# In[133]:


tuned_lightgbm = tune_model(lightgbm, optimize = 'Accuracy')


# In[134]:


tuned_nb = tune_model(nb, optimize = 'Accuracy')


# In[135]:


tuned_ridge = tune_model(ridge, optimize = 'Accuracy')


# In[136]:


evaluate_model(tuned_lda)


# In[137]:


evaluate_model(tuned_lr)


# In[138]:


evaluate_model(tuned_gbc)


# In[139]:


evaluate_model(tuned_lightgbm)


# In[140]:


evaluate_model(tuned_nb)


# In[141]:


evaluate_model(tuned_ridge)


# ## Voting Ensembles

# In[142]:


blend_soft = blend_models(estimator_list =[tuned_lda,tuned_lr,tuned_gbc,tuned_lightgbm, tuned_nb],  method = 'soft')


# In[143]:


evaluate_model(blend_soft)


# In[144]:


blend_hard = blend_models(estimator_list =[tuned_lda,tuned_lr,tuned_gbc,tuned_lightgbm, tuned_nb],  method = 'hard')


# In[145]:


evaluate_model(blend_hard)


# ## Stacking Ensembles

# In[146]:


stack_model = stack_models(estimator_list = [tuned_lr,tuned_gbc,tuned_lightgbm], meta_model = tuned_lr, optimize = 'Accuracy')


# In[147]:


evaluate_model(stack_model)


# ## END

# ### Top Performers

# In[124]:


blend_soft = blend_models(estimator_list =[lr, lda, gbc,lightgbm, nb],  method = 'soft')


# In[125]:


stack_model = stack_models(estimator_list = [lr, lda,tuned_lr,gbc,lightgbm, nb], meta_model = lr, optimize = 'Accuracy')


# In[ ]:





# In[ ]:




