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


data = pd.read_csv("Lung_Cancer_Dataset1.csv")
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


# In[32]:


# Or customize according to what you'd like to know.

df_info2 = pd.DataFrame(columns=['Name of Col', 'Num of Null', 'Dtype', 'N_unique'])

for i in range(0, len(data.columns)):
    df_info2.loc[i] = [data.columns[i],
                      data[data.columns[i]].isnull().sum(),
                      data[data.columns[i]].dtypes,
                      data[data.columns[i]].nunique()]
    
df_info2


# In[33]:


dtype = pd.DataFrame(df_info.summary().loc['types'] == 'numeric')
num_cols = dtype[dtype['types'] == True].index.to_list()
num_cols


# In[34]:


cat_cols = list(set(data.columns) - set(num_cols))
cat_cols.remove('LC')

cat_cols


# In[35]:


cat_cols = list(set(data.columns) - set(num_cols))
cat_cols.remove('LC')
cat_cols


# In[36]:


import pycaret
from pycaret.classification import *


# In[37]:


setup(data = data, 
      target = 'LC',
      session_id = 42,
      preprocess = True,
      index=False,
      numeric_features = cat_cols
     )
#       silent = True


# In[38]:


top5 = compare_models()


# In[39]:


# check available models
#models()


# In[40]:


rf = create_model('rf')


# In[41]:


evaluate_model(rf)


# In[42]:


catboost = create_model('catboost')


# In[43]:


evaluate_model(catboost)


# In[44]:


gbc = create_model('gbc')


# In[45]:


evaluate_model(gbc)


# In[46]:


et = create_model('et')


# In[47]:


evaluate_model(et)


# In[48]:


xgboost = create_model('xgboost')


# In[49]:


evaluate_model(xgboost)


# In[50]:


lightgbm = create_model('lightgbm')


# In[51]:


evaluate_model(lightgbm)


# ## Tuning each Model

# In[52]:


tuned_rf = tune_model(rf, optimize = 'Accuracy')


# In[53]:


tuned_catboost = tune_model(catboost, optimize = 'Accuracy')


# In[54]:


tuned_gbc = tune_model(gbc, optimize = 'Accuracy')


# In[55]:


tuned_et = tune_model(et, optimize = 'Accuracy')


# In[56]:


tuned_xgboost = tune_model(xgboost, optimize = 'Accuracy')


# In[57]:


tuned_lightgbm = tune_model(lightgbm, optimize = 'Accuracy')


# In[58]:


evaluate_model(tuned_rf)


# In[59]:


evaluate_model(tuned_catboost)


# In[60]:


evaluate_model(tuned_gbc)


# In[61]:


evaluate_model(tuned_et)


# In[62]:


evaluate_model(tuned_xgboost)


# In[63]:


evaluate_model(tuned_lightgbm)


# ## Voting Ensembles

# In[64]:


blend_soft = blend_models(estimator_list =[tuned_rf,tuned_catboost,tuned_gbc,tuned_et, tuned_xgboost],  method = 'soft')


# In[65]:


evaluate_model(blend_soft)


# In[66]:


blend_hard = blend_models(estimator_list =[tuned_rf,tuned_catboost,tuned_gbc,tuned_et, tuned_xgboost],  method = 'hard')


# In[67]:


evaluate_model(blend_hard)


# ## Stacking Ensembles

# In[68]:


stack_model = stack_models(estimator_list = [tuned_rf,tuned_catboost,tuned_gbc,tuned_et, tuned_xgboost], meta_model = tuned_rf, optimize = 'recall')


# In[69]:


evaluate_model(stack_model)


# ## END

# In[71]:


# rf	Random Forest Classifier	0.9540	0.9918	0.9684	0.9475	0.9563	0.9078	0.9114	0.0610
# catboost	CatBoost Classifier	0.9540	0.9933	0.9632	0.9515	0.9560	0.9078	0.9107	0.6020
# gbc	Gradient Boosting Classifier	0.9486	0.9871	0.9474	0.9551	0.9495	0.8971	0.9006	0.0290
# et	Extra Trees Classifier	0.9459	0.9830	0.9579	0.9422	0.9482	0.8915	0.8956	0.0480
# xgboost	Extreme Gradient Boosting	0.9405	0.9874	0.9368	0.9499	0.9411	0.8809	0.8853	0.0180
# lightgbm	Light Gradient Boosting Machine	0.9378	0.9854	0.9421	0.9414	0.9398	0.8754	0.8794	0.0600
# dt	Decision Tree Classifier	0.9322	0.9326	0.9208	0.9480	0.9329	0.8644	0.8670	0.0090
# ada	Ada Boost Classifier	0.9297	0.9646	0.9263	0.9389	0.9312	0.8592	0.8620	0.0320
# qda	Quadratic Discriminant Analysis	0.9267	0.9396	0.9947	0.8805	0.9335	0.8529	0.8622	0.0130
# ridge	Ridge Classifier


# In[72]:


ada = create_model('ada')


# In[73]:


qda = create_model('qda')


# In[74]:


knn = create_model('knn')


# ## Top Performers

# In[75]:


blend_soft = blend_models(estimator_list =[gbc, et, xgboost, lightgbm, qda, ada],  method = 'soft')


# In[76]:


stack_model = stack_models(estimator_list = [lightgbm, qda, ada, knn], meta_model = rf, optimize = 'recall')


# ## SHAP Analysis

# In[77]:


import shap
shap.initjs()


# In[78]:


# Explain the model's predictions using SHAP
explainer = shap.Explainer(rf.predict, X_train)
shap_values = explainer(X_test)


# In[79]:


shap.plots.waterfall(shap_values[3], max_display=len(shap_values.feature_names))


# In[80]:


shap.plots.bar(shap_values, max_display=len(shap_values.feature_names))


# In[81]:


shap.plots.force(shap_values[3])


# In[82]:


shap.plots.force(shap_values[0:79])


# In[83]:


shap.plots.beeswarm(shap_values, max_display=len(shap_values.feature_names))


# In[84]:


# Explain the model's predictions using SHAP
explainer = shap.Explainer(et.predict, X_train)
shap_values = explainer(X_test)


# In[85]:


shap.plots.waterfall(shap_values[3], max_display=len(shap_values.feature_names))


# In[86]:


shap.plots.bar(shap_values, max_display=len(shap_values.feature_names))


# In[87]:


shap.plots.force(shap_values[3])


# In[88]:


shap.plots.force(shap_values[0:79])


# In[89]:


shap.plots.beeswarm(shap_values, max_display=len(shap_values.feature_names))


# ## LIME Analysis

# In[90]:


import lime
import lime.lime_tabular


# In[91]:


# Create a LimeTabularExplainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values,  # Use the NumPy array representation of X
    feature_names=X.columns.tolist(), # Feature names for interpretability
    class_names=['0','1'], # Class names
    mode='classification' # or 'regression' depending on the task
)


# In[92]:


# Choose an instance to explain
instance_index = 3 # Example: explain the first instance
instance = X.iloc[instance_index]


# In[93]:


# Explain the instance
explanation = explainer.explain_instance(
    instance.values,
    lambda x: rf.predict_proba(pd.DataFrame(x, columns=X.columns)), # Use a lambda function to wrap the prediction
    num_features=15  # Number of features to show in explanation
)


# In[94]:


# Show the explanation
explanation.show_in_notebook(show_table=True)


# In[95]:


explanation.as_list()


# In[96]:


# Explain the instance
explanation = explainer.explain_instance(
    instance.values,
    lambda x: et.predict_proba(pd.DataFrame(x, columns=X.columns)), # Use a lambda function to wrap the prediction
    num_features=15  # Number of features to show in explanation
)


# In[97]:


# Show the explanation
explanation.show_in_notebook(show_table=True)


# In[98]:


explanation.as_list()


# In[ ]:




