from datetime import date
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from bs4 import BeautifulSoup
import emoji
import string
import spacy
import nltk
tqdm.pandas()
nltk.download('stopwords')


def remove_stopwords(data, col):
    stop = stopwords.words('english')
    data[col] = data[col].apply(lambda x: " ".join([tok if tok not in stop else '' for tok in x.split()]))
    data[col] = data[col].apply(lambda x: " ".join(list(filter(lambda y: len(y) > 1, x.split()))))
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()
    return data[col]

def lemmatize(data, col):
    nlp = spacy.load('en_core_web_sm')
    data[col] = data[col].progress_apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
    return data[col]

def stem_preprocess(data, col):
    stemmer = SnowballStemmer(language='english')
    data[col] = data[col].progress_apply(lambda x: " ".join([stemmer.stem(w) for w in x.split()]))
    return data[col]

def preprocess_from_kaggle(data, col):
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    return data[col]

def preprocess_slang_sub(data, col):
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace(r"\'s", " ")
    return data[col]

def remove_duplicates(data, col):
    def unique_list(l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist

    data[col] = data[col].progress_apply(lambda x: ' '.join(unique_list(x.split())))
    return data[col]

def remove_punctuation(data, col):
    data[col] = data[col].str.translate(str.maketrans('', '', string.punctuation))
    return data[col]

def links_removing(data, col):
    def text_cleaning(text):
        template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
        text = template.sub(r'', text)
        
        soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
        only_text = soup.get_text()
        text = only_text
        
        text = emoji.demojize(text, delimiters=("", ""))
        
        text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
        text = re.sub(' +', ' ', text) #Remove Extra Spaces
        text = text.strip() # remove spaces at the beginning and at the end of string
        return text

    data[col] = data[col].progress_apply(lambda x: text_cleaning(x))
    return data[col]

def replaceMultiToxicWords(text):
    text = re.sub(r'(fuckfuck)','fuck fuck ',text)
    text = re.sub(r'(f+)( *)([u|*|_]+)( *)([c|*|_]+)( *)(k)+','fuck',text)
    text = re.sub(r'(h+)(a+)(h+)(a+)','ha ha ',text)
    text = re.sub(r'(s+ *h+ *[i|!]+ *t+)','shit',text)
    text = re.sub(r'\b(n+)(i+)(g+)(a+)\b','nigga',text)
    text = re.sub(r'\b(n+)([i|!]+)(g+)(e+)(r+)\b','nigger',text)
    text = re.sub(r'\b(d+)(o+)(u+)(c+)(h+)(e+)( *)(b+)(a+)(g+)\b','douchebag',text)
    text = re.sub(r'([a|@][$|s][s|$])','ass',text)
    text = re.sub(r'(\bfuk\b)','fuck',text)
    return text