import pickle
import pandas as pd
import numpy as np
import re, nltk, spacy, gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
from nltk.corpus import stopwords
from string import punctuation
nltk.download("stopwords")
nltk.download("wordnet")

def count_tag(data, ref_col, list_words): 
    ''' Count the number of occurrences and the average score for each tag
    
    Parameters:
    
        data (dataframe): dataframe to use 
        ref_col (serie): column of dataframe containing tags
        list_words (list): list of different tags
    '''
    
    keyword_count = dict()
    index = -1
    
    for s in list_words: 
        keyword_count[s] = []
        keyword_count[s].append(0)
        keyword_count[s].append(0)
        
    for list_keywords in data[ref_col].str.split('>'): 
        
        if type(list_keywords) == float and pd.isnull(list_keywords): 
            continue
        
        index += 1
            
        for s in [s for s in list_keywords if s in list_words]: 
            if pd.notnull(s):
                keyword_count[s][0] += 1
                    
    # conversion of our dictionary into a list
    keyword_occurences = []
    
    for tag, item in keyword_count.items():
        keyword_occurences.append([tag[1:], item[0], item[1]/item[0]])
        
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    
    return keyword_occurences

# Making a list of the different tags 
df = pickle.load(open('df.p', 'rb'))
set_tags = set()
scoring = list()

for list_keywords in df['Tags'].str.split('>').values:
    
    if isinstance(list_keywords, float): 
        continue 

    set_tags = set_tags.union(list_keywords)


keyword_occurences = count_tag(df, 'Tags', set_tags)
trunc_occurences = keyword_occurences[1:401]
top_tags = [i[0] for i in trunc_occurences]


def clean_text(text):
    ''' Lowering text and removing undesirable marks
    '''
    
    text = text.lower()
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text) # matches all whitespace characters
    text = text.strip(' ')
    return text

token = ToktokTokenizer()
punct = punctuation

def strip_list_noempty(mylist):
    
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']


def clean_punct(text): 
    
    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    
    for w in words:
        if w in top_tags:
            punctuation_filtered.append(w)
        else:
            w = re.sub('^[0-9]*', " ", w)
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

stop_words = set(stopwords.words("english"))

def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus

    Parameter:

    text: corpus to remove stop words from it
    '''

    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))


def lemmatization(texts, allowed_postags, stop_words=stop_words):
    ''' It keeps the lemma of the words (lemma is the uninflected form of a word),
    and deletes the underired POS tags
    
    Parameters:
    
    texts (list): text to lemmatize
    allowed_postags (list): list of allowed postags, like NOUN, ADL, VERB, ADV
    '''

    nlp = spacy.load('en', disable=['parser', 'ner'])
    lemma = wordnet.WordNetLemmatizer()       
    doc = nlp(texts) 
    texts_out = []
    
    for token in doc:
        
        if str(token) in top_tags:
            texts_out.append(str(token))
            
        elif token.pos_ in allowed_postags:
            
            if token.lemma_ not in ['-PRON-']:
                texts_out.append(token.lemma_)
                
            else:
                texts_out.append('')
     
    texts_out = ' '.join(texts_out)

    return texts_out


def Recommendation_system_tags(text):
    ''' Recomendation system for stackoverflow posts based on a OVR/linear SVC model, 
    it returns up to 5 tags.

    Parameters:

    text: the stackoverflow post of the user '''
    
    # Unserialization of dataframe

    # Binarizing the tags
    multilabel_binarizer = MultiLabelBinarizer()
    y_target = multilabel_binarizer.fit_transform(df['Tags'])

    # Sampling dataset
    vectorizer_X = TfidfVectorizer(analyzer='word', min_df=0.0, max_df = 1.0, 
                                    strip_accents = None, encoding = 'utf-8', 
                                    preprocessor=None, 
                                    token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                    max_features=1000)

    # TF-IDF matrix
    X_tfidf = vectorizer_X.fit_transform(df['Body'])

    text = clean_text(text)
    text = clean_punct(text)
    text = stopWordsRemove(text)
    text = lemmatization(text, ['NOUN', 'ADV'])
    text_tfidf = vectorizer_X.transform([text])

     # Unserialization of Linear SVC fitted
    svc = pickle.load(open('svc.p', 'rb'))
    y_pred_svc = svc.predict(text_tfidf)
    tags_svc = np.where(y_pred_svc==1)[1]
    tags = [multilabel_binarizer.classes_[tag] for tag in tags_svc]

    return tags[:5] #Maximum 5 tags


text = input('Ask a question: ')
tags = Recommendation_system_tags(text)
print('Recommended tags are:', tags)