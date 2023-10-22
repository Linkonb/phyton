import codecs #a set of base classes which define the interface and can also be used to easily write your own codecs for use in Python
import collections  #provides different types of containers.
#import pyodbc # makes accessing ODBC databases simple.
import numpy as np #support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions 
import pandas as pd #data manipulation and analysis
import nltk #Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt #provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.


#import scipy
#import TensorFlow  

# phyton,numpy ,pandas ,matplotlib ,nltk

print("Buddha Dharma Shanga")


with codecs.open("content/text1.txt", "r", encoding="utf-8") as f:
  text1=f.read()
with codecs.open("content/text2.txt", "r", encoding="utf-8") as f:
  text2=f.read()

#function
def total_tokens(text):     
  n = WordPunctTokenizer().tokenize(text)
  return collections.Counter(n), len(n)


#function to create relative frequency and absolute frequency for most common words
def make_df(counter, size):
 #getting absolute and relative frequencies for each tokens in the counter list  
 absolute_frequency = np.array([el[1] for el in counter])
 relative_frequency = absolute_frequency / size
 
 #creating a data frame using obtained data above(absolute_frequency & relative_frequency)
 df = pd.DataFrame(data=np.array([absolute_frequency, relative_frequency]).T, index = [el[0] for el in counter], columns=["Absolute frequency", "Relative frequency"])
 df.index.name = "Most common words"
  
 return df

#for text1
text1_counter, text1_size = total_tokens(text1)
make_df(text1_counter.most_common(10), text1_size)


#for text2
text2_counter, text2_size = total_tokens(text2)
make_df(text2_counter.most_common(10), text2_size)

all_counter = text1_counter + text2_counter
all_df = make_df(all_counter.most_common(1000), 1)
x = all_df.index.values
 
#creating our new list for dataframe as df_data[] comprising of 
#text1 relative frequency as text1_c,   
#text2 relative frequency as text2_c, and   
#Relative frequency difference as difference for both text files
df_data = []
for word in x:
 #getting relative frequency for each word in text1 and text2 and loading the same into text1_C and text2_c respectively
 text1_c = text1_counter.get(word, 0) / text1_size
 text2_c = text2_counter.get(word, 0) / text2_size
  
 #calculating difference between text1_c and text2_c & getting mod for all(in case of negative difference value)
 difference = abs(text1_c - text2_c)
 
 #appending above three columns into the list
 df_data.append([text1_c, text2_c, difference])
 
#creating dataframe dist_df and loading above list into the same
dist_df = pd.DataFrame(data=df_data, index=x, columns=["text1 relative frequency", "text2 relative frequency","Relative frequency difference" ])
dist_df.index.name = "Most common words"
dist_df.sort_values("Relative frequency difference", ascending=False, inplace=True)
 
#printing our required result
dist_df.head(10)
dist_df.to_csv("output1.csv")

#creating our new list for dataframe as df_data[] comprising of 
#text1 relative frequency as text1_c,   
#text2 relative frequency as text2_c, and   
#Relative frequency difference as difference for both text files
df_data = []
for word in x:
 #getting relative frequency for each word in text1 and text2 and loading the same into text1_C and text2_c respectively
 text1_c = text1_counter.get(word, 0) / text1_size
 text2_c = text2_counter.get(word, 0) / text2_size
  
 #calculating difference between text1_c and text2_c & getting mod for all(in case of negative difference value)
 difference = abs(text1_c - text2_c)
 
 #appending above three columns into the list
 df_data.append([text1_c, text2_c, difference])
 
#creating dataframe dist_df and loading above list into the same
dist_df = pd.DataFrame(data=df_data, index=x, columns=["text1 relative frequency", "text2 relative frequency","Relative frequency difference" ])
dist_df.index.name = "Most common words"
dist_df.sort_values("Relative frequency difference", ascending=False, inplace=True)
 
#printing our required result
dist_df.head(10)
dist_df.to_csv("output2.csv")

#plot
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

#xpoints = text1_c
#ypoints = text2_c

plt.plot(xpoints, ypoints)
plt.show()
