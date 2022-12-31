# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer

# create an instance of class PorterStemmer
stemmer = PorterStemmer()

# importing json lib
import json
import pickle
import numpy as np

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

# open the JSON file, load data from it.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:

        # write stemming algorithm:
        if word not in ignore_words:
          w = stemmer.stem(word.lower())
          stem_words.append(w)      
    return stem_words

        
# creating a function to make corpus

for intent in data['intents']:

        # Add all patterns and tags to a list
    for pattern in intent['patterns']:  

            # tokenize the pattern          
            pattern_word = nltk.word_tokenize(pattern)

            # add the tokenized words to the words list        
            words.extend(pattern_word)  

            # add the 'tokenized word list' along with the 'tag' to pattern_word_tags_list
            pattern_word_tags_list.append((pattern_word, intent['tag']))
       
            
        # Add all tags to the classes list
    if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words) 

    print('stem_words list : ' , stem_words)


def create_bot_corpus(stem_words,classes):

        stem_words = sorted(list(set(stem_words)))
        classes = sorted(list(set(classes)))

        pickle.dump(stem_words,open('words.pkl','wb'))
        pickle.dump(stem_words,open('classes.pkl','wb'))

        return stem_words,classes

stem_words,classes = create_bot_corpus(stem_words,classes)

print(stem_words)

print(classes)

print(pattern_word_tags_list)