#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import seaborn as sns
import matplotlib.cm as cm
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


# finding all the rows where the sum of labels is zero(the comment is a Clean comment)
rowsums=df.iloc[:,2:].sum(axis=1)
df['clean']=(rowsums==0)
df['clean'].sum()


# In[10]:


# Total no.of toxic comments
len(df[df['toxic']==1])


# In[11]:


df.head(10)


# In[12]:


# Using seaborn and matplotlib to visualize the count of different categories of toxicity of comments

colors_list = ["red", "purple","maroon","yellow", "brown","black", "blue"]

palette= sns.xkcd_palette(colors_list)

x=df.iloc[:,2:].sum()

plt.figure(figsize=(10,6))
# x.index has all the toxicity labels and x.values has their respective count
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Class")
plt.xlabel('Toxicity Type', fontsize = 12)
plt.ylabel('Occurrences', fontsize=12)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, 
            ha='center', va='bottom')

plt.show()


# In[13]:


#The graph shows us that the dataset is highly imbalanced as more than 1.4lac comments are categorized as clean.
#Taking an insight of the length of the comments in the dataset.
comment = df['comment_text']
for i in range(5):
    print(i,"- " + comment[i] + "\n Length -" ,len(comment[i]))


# In[14]:


#The length of the comments looks to be quite large, so we'll visualize some more info about the comments.
## creating a numpy array of the length of each comment in the dataset.
x = np.array([len(comment[i]) for i in range(comment.shape[0])])


# In[15]:


print("""The maximum length of comment is:{} 
        \nThe minimum length of the comment is:{} 
        \nAnd the average length of a comment is: {}""".format(x.max(),x.min(),x.mean()))


# In[16]:


print('The average length of comment is : 394.073' )
bins = [1,200,400,600,800,1000,1200,1400]
plt.hist(x, bins=bins, color = 'Green')
plt.xlabel('Length of comments')
plt.ylabel('Number of comments')       
plt.axis([0, 1400, 0, 90000])
plt.grid(True)
plt.show()


# In[17]:


#It is visible that length of most of the comments(Approx 80,000) lies in the range of 0-200 and around 40,000 lie in between 200-400
#Now we will try to find the count of different toxicity of comments in each bin
label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
print(label.head(10))
label = label.values


# In[18]:


label.shape


# In[19]:


# Creating a zero matrix of shape (159571,6)
y = np.zeros(label.shape)
for i in range(label.shape[0]):
    l = len(comment[i])
    if label[i][0] :
        y[i][0] = l
    if label[i][1] :
        y[i][1] = l
    if label[i][2] :
        y[i][2] = l
    if label[i][3] :
        y[i][3] = l
    if label[i][4] :
        y[i][4] = l
    if label[i][5] :
        y[i][5] = l

label_plot = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
color = ['red','purple','maroon','yellow','black','green']    
plt.figure(figsize = (10,8))
plt.hist(y,bins = bins,label = label_plot,color = color)
plt.axis([0, 1400, 0, 12000])
plt.xlabel('Length of comments', fontsize = 14)
plt.ylabel('Number of comments', fontsize = 14) 
plt.legend()
plt.grid(True)
plt.show()


# In[20]:


#Removing excessive length comments
# creating a list of comments with less than 400 length of words.
trim_comments = [comment[i] for i in range(comment.shape[0]) if len(comment[i])<=400 ]

# creating corresponding labels for those comments
my_labels = np.array([label[i] for i in range(comment.shape[0]) if len(comment[i])<=400 ])


# In[21]:


my_labels[:10, :]


# In[22]:


print(len(trim_comments))
print(len(my_labels))
print("Thus number of removed comments = {}".format(159571-115910))


# In[23]:


#So now we are left with 115910 comments whose length is less than 400
print(len(trim_comments))
print(my_labels.shape)


# In[24]:


#Now we start the preprocessing of the comments.
# Punctuation removal

import string
print(string.punctuation)
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
print (punctuation_edit)
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)


# In[25]:


# Stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Adding alphabets to the set
for i in range(ord('a'),ord('z')+1):
    stop_words.add(chr(i))
print(stop_words)


# In[26]:


# Stemming and Lemmatizing
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# In[27]:


# Looping through all the comments and processing them through the functions defined above.

for i in range(len(trim_comments)):
    trim_comments[i] = trim_comments[i].lower().translate(trantab)
    word_list = []
    for word in trim_comments[i].split():
        if not word in stop_words:
            word_list.append(stemmer.stem(lemmatizer.lemmatize(word,pos="v")))
    trim_comments[i]  = " ".join(word_list)


# In[28]:


# Comments after stop words removal, stemming and lemmatizing.
for i in range(5):
    print(trim_comments[i],"\n")


# In[29]:


# Applying count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
 
#create object supplying our custom stop words
count_vector = CountVectorizer(stop_words=stop_words)
#fitting it to converts comments into bag of words format
tf = count_vector.fit_transform(trim_comments[:20000]).toarray()


# In[30]:


tf.shape


# In[31]:


"""Note: Due to hardware limitation the processing has stopped within 20,000 comments and 
the results and accuracy are obtained accordingly"""
#Splitting into training and testing
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = shuffle(tf, my_labels[:20000],3)

print(X_test.shape)
print(X_train.shape)


# In[32]:


from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def evaluate_score(Y_test,predict): 
    loss = hamming_loss(Y_test,predict)
    print("Hamming_loss : {}".format(loss*100))
    accuracy = accuracy_score(Y_test,predict)
    print("Accuracy : {}".format(accuracy*100))
    try : 
        loss = log_loss(Y_test,predict)
    except :
        loss = log_loss(Y_test,predict.toarray())
    print("Log_loss : {}".format(loss))


# In[33]:


from sklearn.naive_bayes import MultinomialNB


# In[34]:


#Binary Relevance (BR) Method with MultinomialNB classifiers
# clf will be the list of the classifiers for all the 6 labels
# each classifier is fit with the training data and corresponding classifier
clf = []
for i in range(6):
    clf.append(MultinomialNB())
    clf[i].fit(X_train,Y_train[:,i])


# In[35]:


# predict list contains the predictions, it is transposed later to get the proper shape
predict = []
for i in range(6):
    predict.append(clf[i].predict(X_test))

predict = np.asarray(np.transpose(predict))
print(predict.shape)


# In[36]:


#calculate scores
evaluate_score(Y_test,predict)


# In[37]:


#BR Method with SVM classifier (from scikit-multilearn)
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
classifier.fit(X_train, Y_train)


# In[38]:


#predictions
predictions = classifier.predict(X_test)


# In[39]:


#calculate scores
evaluate_score(Y_test,predictions)


# In[40]:


#BR Method with GaussianNB classifier.
from sklearn.naive_bayes import GaussianNB
#create and fit classifiers
clf = []
for i in range(6):
    clf.append(GaussianNB())
    clf[i].fit(X_train,Y_train[:,i])


# In[41]:


#predictions
predict = []
for ix in range(6):
    predict.append(clf[ix].predict(X_test))


# In[42]:


#calculate scores
predict = np.asarray(np.transpose(predict))
evaluate_score(Y_test,predict)


# In[43]:


#Visualizing the result 
x = ['BR-MultnomialNB','BR-SVC','BR-GaussianNB']
y = [3.65,4.36,20.74]
colors = itertools.cycle(['r', 'g', 'b'])
plt.figure(figsize= (8,6))
plt.ylabel('Hamming-Loss')
plt.xlabel('Model-details')
plt.xticks(rotation=90)
for i in range(len(y)):
    plt.bar(x[i], y[i], color=next(colors))
plt.show()


# In[44]:


#The hamming loss is maximum for BR-GaussianNB and minimum for BR-MultinomialNB
x = ['BR-MultNB','BR-SVC','BR-GausNB']
y = [1.97,0.46,1.422]
colors = itertools.cycle(['r', 'g', 'b'])
plt.figure(figsize=(8,6))
plt.ylabel('Log-Loss')
plt.xlabel('Model-details')
plt.xticks(rotation=90)
for i in range(len(y)):
    plt.bar(x[i], y[i], color=next(colors))
plt.show()


# In[ ]:




