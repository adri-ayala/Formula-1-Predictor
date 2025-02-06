#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


races = pd.read_csv("Formula1_2024season_raceResults.csv")
races23 = pd.read_csv("Formula1_2023season_raceResults.csv")


# In[3]:


races.head()


# In[4]:


races.shape


# In[5]:


races23.shape


# In[6]:


races["Team"].value_counts()


# In[7]:


races23["Team"].value_counts()


# In[8]:


races.dtypes


# In[9]:


races["stl_code"]= races["Set Fastest Lap"].astype("category").cat.codes


# In[10]:


races["team_code"]= races["Team"].astype("category").cat.codes


# In[11]:


races["pos_code"]= races["Position"].astype("category").cat.codes


# In[12]:


races["target"]= (races["Position"]=="1").astype("int")


# In[13]:


races


# In[14]:


races23["stl_code"]= races23["Set Fastest Lap"].astype("category").cat.codes
races23["team_code"]= races23["Team"].astype("category").cat.codes
races23["pos_code"]= races["Position"].astype("category").cat.codes
races23["target"]= (races23["Position"]=="1").astype("int")
races23


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


feature_columns = ['Starting Grid', 'stl_code', 'team_code'] 
target_column = 'target'

# Prepare the train and test sets
train = races23[feature_columns + ['target']]  # Use races23 for training
test = races[feature_columns + ['target']]    # Use races for testing

# Split the features and the target for both datasets
X_train = train[feature_columns]  # Features for training
y_train = train[target_column] # Target for training

X_test = test[feature_columns]   # Features for testing
y_test = test[target_column]   # Target for testing

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)
combined = pd.DataFrame(dict(actual=y_test, predicted=y_pred), index=y_test.index)
  

# Evaluate the model 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# In[17]:


from sklearn.metrics import precision_score


# In[18]:


precision_score(y_test, y_pred)


# In[19]:


grouped_races= races.groupby(["Team","Driver"])


# In[20]:


grouped_races23= races23.groupby(["Team","Driver"])


# In[21]:


group = grouped_races.get_group(("Ferrari", "Carlos Sainz")).sort_index()


# In[22]:


group2 = grouped_races23.get_group(("Ferrari", "Carlos Sainz")).sort_index()


# In[23]:


group


# In[24]:


def rolling_averages(group,cols,new_cols):
    group=group.sort_index()
    rolling_stats=group[cols].rolling(3).mean()
    if rolling_stats.shape[1] == len(new_cols):
        group[new_cols] = rolling_stats
    else:
        raise ValueError(f"The number of columns in rolling_stats ({rolling_stats.shape[1]}) does not match the length of new_cols ({len(new_cols)})")
    #group[new_cols]=rolling_stats
    group=group.dropna(subset=new_cols)
    return group


# In[25]:


cols=["pos_code","Starting Grid","Points"]
new_cols = [f"{c}_rolling" for c in cols]


# In[26]:


rolling_averages(group,cols,new_cols)


# In[27]:


rolling_averages(group2, cols, new_cols)


# In[28]:


races_rolling = races.groupby(["Team", "Driver"]).apply(lambda x: rolling_averages(x, cols, new_cols), include_groups=False)


# In[29]:


races_rolling23= races23.groupby(["Team", "Driver"]).apply(lambda x: rolling_averages(x, cols, new_cols), include_groups=False)


# In[30]:


races_rolling


# In[31]:


races_rolling23


# In[32]:


races_rolling.to_csv("f124_team_driver_data.csv", index=False)


# In[33]:


races_rolling=races_rolling.droplevel(['Team','Driver'])


# In[34]:


races_rolling23=races_rolling23.droplevel(['Team','Driver'])


# In[35]:


races_rolling


# In[36]:


races_rolling23


# In[37]:


races_rolling.index=range(races_rolling.shape[0])


# In[38]:


races_rolling


# In[39]:


races_rolling23.index=range(races_rolling23.shape[0])


# In[40]:


def make_predictions(data1,data2,feature_columns):

    # Prepare the train and test sets
    train = data2[feature_columns + ['target']]  # Use races23 for training
    test = data1[feature_columns + ['target']]    # Use races for testing
    
    # Split the features and the target for both datasets
    X_train = train[feature_columns] # Features for training
    y_train = train[target_column] # Target for training
    
    X_test = test[feature_columns] # Features for testing
    y_test = test[target_column] # Target for testing
    
    # Initialize and train the model
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    rf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf.predict(X_test)
    combined = pd.DataFrame(dict(actual=y_test, predicted=y_pred), index=y_test.index)
      
    
    # Evaluate the model 
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    from sklearn.metrics import precision_score
    precision = precision_score(test["target"], y_pred)
    print(f"Model precision: {precision:.2f}")
    return combined, precision


# In[41]:


combined, error = make_predictions(races_rolling, races_rolling23,feature_columns + new_cols)


# In[42]:


combined


# In[43]:


combined = combined.merge(races_rolling[["Track", "No"]], left_index=True, right_index=True, suffixes=('_combined', '_races_rolling'))

combined


# In[ ]:




