#!/usr/bin/env python
# coding: utf-8

# In[351]:


#Import the libraries
import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from textblob import TextBlob
from sklearn import metrics
from textblob import TextBlob
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings


# In[352]:


#Ignore warnings
warnings.filterwarnings("ignore")


# In[353]:


#Web scraping the comments from Youtube videos
pd.set_option('display.max_colwidth', 1000)

with Chrome() as driver:
    wait = WebDriverWait(driver, 10)
    driver.get("https://www.youtube.com/watch?v=M3UuGKRmakI")

    for item in range(5): # by increasing the highest range you can get more content
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(3)

    comments = []
    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text"))):
        comments.append(comment.text)

    # Convert the comments list into a DataFrame
    df = pd.DataFrame({'comments': comments})


# In[354]:


# Print the DataFrame
df.head(5)


# In[355]:


df.shape


# In[356]:


#Convert the comments gathered into a CSV file
df.to_csv('comments8.csv', index=False)


# In[357]:


#Measure the degree to which the text expresses a positive or negative sentiment
df['polarity'] = df['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[358]:


#Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)


# In[359]:


df['pol_cat']  = 0


# In[360]:


#Continuous to categorical conversion
df['pol_cat'][df.polarity > 0] = 1
df['pol_cat'][df.polarity <= 0] = -1


# In[361]:


df.head()


# In[362]:


df['pol_cat'].value_counts()


# In[363]:


#Create separate dataframes for Negative,Positive & Neutral comments
data_pos = df[df['pol_cat'] == 1]
data_pos = data_pos.reset_index(drop = True)

data_neg = df[df['pol_cat'] == -1]
data_neg = data_neg.reset_index(drop = True)


# In[364]:


#Check the positive comments
data_pos.head()


# In[365]:


data_pos.shape


# In[366]:


#Check the positive comments
data_neg.head()


# In[367]:


data_neg.shape


# In[368]:


data_neg['comments'][40]


# In[369]:


#Distribution of negative and positive comments
df.pol_cat.value_counts().plot.bar()
df.pol_cat.value_counts()


# In[370]:


#Data pre-processing, convert comments to lowercase characters
df['comments'] = df['comments'].str.lower()


# In[371]:


#Remove unecessary spaces
df['comments'].str.strip()


# In[372]:


# Define a function to clean the English comments, remove other language comments
def clean_text(text):
    # Remove any non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove any non-alphanumeric characters and convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    text = text.lower()
    return text


# Use the langdetect library to detect the language of each comment
df['language'] = df['comments'].apply(lambda x: detect(x))

# Filter the dataframe to keep only the comments that are in English
df = df[df['language'] == 'en']

# Clean the English comments and store them in a new column called 'clean_comments'
df['clean_comments'] = df['comments'].apply(clean_text)

# Print the cleaned English comments
df['clean_comments'].head()


# In[373]:


#Compare original comments and cleaned comments
df.head()


# In[374]:


df.info()


# In[375]:


df.describe()


# In[376]:


df['clean_comments'].str.strip()


# In[377]:


#Download stopwords package
nltk.download("stopwords")


# In[378]:


#Download punkt package
nltk.download("punkt")


# In[379]:


print(stopwords.words('english'))


# In[380]:


stop_words = set(stopwords.words('english'))


# In[381]:


df['clean_comments'] = df['clean_comments'].str.strip()


# In[382]:


#View the transformed data
df.head(8)


# In[383]:


#Define function to remove stopwords
def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


# In[384]:


df['stop_comments'] = df['clean_comments'].apply(lambda x : remove_stopwords(x))


# In[385]:


df.head()


# In[386]:


#Split the dataset into training and testing data
X_train,X_test,y_train,y_test = train_test_split(df['stop_comments'],df['pol_cat'],test_size = 0.2,random_state = 324)


# In[387]:


X_train.shape


# In[388]:


X_test.shape


# In[389]:


df['pol_cat'].value_counts()


# In[390]:


#Use Count Vectorizer to transform the data
vect = CountVectorizer()
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)


# In[391]:


tf_train.shape


# In[392]:


#Print vocabulary
print(vect.vocabulary_)


# In[393]:


vocab = vect.vocabulary_


# In[394]:


#Build Logistic model
lr = LogisticRegression()
lr.fit(tf_train,y_train)


# In[395]:


#Accuracy score for train data 
lr.score(tf_train,y_train)


# In[396]:


#Accuracy score for test data 
lr.score(tf_test,y_test)


# In[397]:


predicted_lr = lr.predict(tf_test) #prediction for Logistic model


# In[398]:


#Build Multinomial model
model=MultinomialNB()


# In[399]:


#Fit the model
model.fit(tf_train, y_train)


# In[400]:


#Accuracy score for train data 
model.score(tf_train,y_train)


# In[401]:


#Accuracy score for test data 
model.score(tf_test,y_test)


# In[402]:


expected = y_test


# In[403]:


predicted_mnb = model.predict(tf_test) #prediction for MultinomialNB model


# In[404]:


#Prepare classification report for Multionmial Naive Baye's algorithm
cf_mnb=classification_report(expected, predicted_mnb)
print("Classification Report for Multionmial Naive Baye's algorithm:\n", cf_mnb)


# In[405]:


#Prepare classification report for Logistic Regression algorithm
cf_lr=classification_report(expected, predicted_lr)
print("Classification Report for Logistic Regression algorithm:\n", cf_lr)

