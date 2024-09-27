## Spam Classification using a logistic regression classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import re


# Load the data
spam_ham = pd.read_csv("/Users/lucabutera/Downloads/spam_ham_dataset.csv")

print(spam_ham.columns)
print(spam_ham.head())

# Text data often contains a lot of newlines and carriage returns, which can be easily conflated with text in EDA
def clean(text):
    assert isinstance(text, pd.Series) and text.dtype == 'object'
    return text.str.replace('\n', ' ').str.replace('\r', ' ')

spam_ham['text'] = clean(spam_ham['text'])

# Extract the training and testing sets
train, test = train_test_split(spam_ham, test_size=0.3, random_state=42)

# Define function to extract most commonly appearing words
def simplesearch(textSeries, min_length = 4, max_length = 10):
    # This extracts the unique words of each email, then finds the most frequently appearing unique words
    tokenized = textSeries.str.lower().str.split(pat = r'[^A-Za-z]', regex = True).apply(set).apply(list)
    ser = pd.Series(tokenized.explode()).value_counts()
    return ser[(ser.index.str.len() >= min_length) & (ser.index.str.len() <= max_length)]

# Find the most common 100 words in each category
simplespam = simplesearch(train[train['label_num'] == 1]['text'])
simpleham = simplesearch(train[train['label_num'] == 0]['text'])

filteredspam = simplespam[simplespam >= 100]
filteredham = simpleham[simpleham >= 100]

# Just to see about how many words we are working with
print(f'Filtered Ham Length: {len(filteredham)}')
print(f'Filtered Spam Length: {len(filteredspam)}')

spam_words = simplespam.head(100).index
ham_words = simpleham.head(100).index

# Find words that are in spam but not in ham
spam_not_ham_words = spam_words.difference(ham_words, sort = False)
ham_not_spam_words = ham_words.difference(spam_words, sort = False)

# Filter simplespam based on spam_not_ham_words
spam_not_ham = simplespam.loc[spam_not_ham_words]
ham_not_spam = simpleham.loc[ham_not_spam_words]

# Working mostly on a spam classification, so words that are only in spam emails are likely more useful
print(spam_not_ham)
len(spam_not_ham)

# One-hot encode the existence of words in string-type Series
def words_in_texts(words, texts):
    return pd.DataFrame({word: texts.str.contains(word).astype(int) for word in words}).to_numpy()

wrds = spam_not_ham.index

X_train = words_in_texts(wrds, train['text'])
Y_train = np.array(train['label_num'])

X_train[:5], Y_train[:5]

my_model = LogisticRegression()
my_model.fit(X_train, Y_train)

training_accuracy = np.mean(my_model.predict(X_train) == Y_train)
print("Training Accuracy: ", training_accuracy)

X_test = words_in_texts(wrds, test['text'])
Y_test = np.array(test['label_num'])

test_accuracy = np.mean(my_model.predict(X_test) == Y_test)
print("Test Accuracy: ", test_accuracy) # In practice this has resulted between 84% and 94% accuracy, depending on the size and quality of the data