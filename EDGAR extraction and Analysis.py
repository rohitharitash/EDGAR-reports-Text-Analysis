
# coding: utf-8

# # Data Extraction and Text Analysis
# 

# First function reading edar files.It will remove html tags and extract requried informtion

# In[13]:


# Requried imports
import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np


# In[14]:


# Text extraction patterns
mda_regex = r"item[^a-zA-Z\n]*\d\s*\.\s*management\'s discussion and analysis.*?^\s*item[^a-zA-Z\n]*\d\s*\.*"
qqd_regex = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Quantitative and Qualitative Disclosures about "             r"Market Risk.*?^\s*item\s*\d\s*"
riskfactor_regex = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Risk Factors.*?^\s*item\s*\d\s*"


# In[15]:


# Filepath locations
stopWordsFile = 'D:/data science/Blackcoffer project/StopWords_Generic.txt'
positiveWordsFile = 'D:/data science/Blackcoffer project/PositiveWords.txt'
nagitiveWordsFile = 'D:/data science/Blackcoffer project/NegativeWords.txt'
uncertainty_dictionaryFile = 'D:/data science/Blackcoffer project/uncertainty_dictionary.txt'
constraining_dictionaryFile = 'D:/data science/Blackcoffer project/constraining_dictionary.txt'


# In[16]:


# Function for extracting requried text
def rawdata_extract(path, cikListFile):
    html_regex = re.compile(r'<.*?>')
    extraxted_data=[]
    
    
    cikListFile = pd.read_csv(cikListFile)
    for index, row in cikListFile.iterrows():
        processingFile=row['SECFNAME'].split('/')
        inputFile = processingFile[3]
        cik=row['CIK']
        coname=row['CONAME']
        fyrmo=row['FYRMO']
        fdate = row['FDATE']
        form = row['FORM']
        secfname=row['SECFNAME']
        for fileName in os.listdir(path):
            filenameopen = os.path.join(path, fileName)
            dirFileName = filenameopen.split('\\')
            currentFile=dirFileName[1]

            if os.path.isfile(filenameopen) and currentFile == inputFile :
                resultdict = dict()
                resultdict['CIK'] = cik
                resultdict['CONAME'] = coname
                resultdict['FYRMO'] = fyrmo
                resultdict['FDATE'] = fdate
                resultdict['FORM'] = form
                resultdict['SECFNAME'] = secfname
                
                with open(filenameopen, 'r', encoding='utf-8', errors="replace") as in_file:
                    content = in_file.read()
                    content = re.sub(html_regex,'',content)
                    content = content.replace('&nbsp;','')
                    content = re.sub(r'&#\d+;', '', content)
                    matches_mda = re.findall(mda_regex, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if matches_mda:
                        result = max(matches_mda, key=len)
                        result = str(result).replace('\n', '')
                        resultdict['mda_extract'] = result
                    else:
                        resultdict['mda_extract'] = ""
                    match_qqd = re.findall(qqd_regex, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if match_qqd:
                        result_qqd = max(match_qqd, key=len)
                        result_qqd = str(result_qqd).replace('\n','')
                        resultdict['qqd_extract']= result_qqd
                    else:
                        resultdict['qqd_extract'] = ""
                    match_riskfactor = re.findall(riskfactor_regex, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if match_riskfactor:
                        result_riskfactor = max(match_riskfactor, key=len)
                        result_riskfactor = str(result_riskfactor).replace('\n', '')
                        resultdict['riskfactor_extract'] = result_riskfactor
                    else:
                        resultdict['riskfactor_extract'] = ""
                    extraxted_data.append(resultdict)

                in_file.close()

    return extraxted_data


# # Section 1.1: Positive score, negative score, polarity score

# Loading stop words dictionary for removing stop words

# In[17]:


with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []


# tokenizeing module and filtering tokens using stop words list, removing punctuations

# In[18]:


# Tokenizer
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words


# In[19]:


# Loading positive words
with open(positiveWordsFile,'r') as posfile:
    positivewords=posfile.read().lower()
positiveWordList=positivewords.split('\n')


# In[20]:


# Loading negative words
with open(nagitiveWordsFile ,'r') as negfile:
    negativeword=negfile.read().lower()
negativeWordList=negativeword.split('\n')


# In[21]:


# Calculating positive score 
def positive_score(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in positiveWordList:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos


# In[22]:


# Calculating Negative score
def negative_word(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in negativeWordList:
            numNegWords -=1
    sumNeg = numNegWords 
    sumNeg = sumNeg * -1
    return sumNeg


# In[23]:


# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
    return pol_score


# # Section 2 -Analysis of Readability -  Average Sentence Length, percentage of complex words, fog index

# In[24]:


# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
     
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)


# In[25]:


# Calculating percentage of complex word 
# It is calculated using Percentage of Complex words = the number of complex words / the number of words 

def percentage_complex_word(text):
    tokens = tokenizer(text)
    complexWord = 0
    complex_word_percentage = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    if len(tokens) != 0:
        complex_word_percentage = complexWord/len(tokens)
    
    return complex_word_percentage
                        


# In[26]:


# calculating Fog Index 
# Fog index is calculated using -- Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex


# # Section 4: Complex word count

# In[27]:


# Counting complex words
def complex_word_count(text):
    tokens = tokenizer(text)
    complexWord = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    return complexWord


# # Section 5: Word count

# In[28]:


#Counting total words

def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)


# In[29]:


# calculating uncertainty_score
with open(uncertainty_dictionaryFile ,'r') as uncertain_dict:
    uncertainDict=uncertain_dict.read().lower()
uncertainDictionary = uncertainDict.split('\n')

def uncertainty_score(text):
    uncertainWordnum =0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in uncertainDictionary:
            uncertainWordnum +=1
    sumUncertainityScore = uncertainWordnum 
    
    return sumUncertainityScore



# In[30]:


# calculating constraining score
with open(constraining_dictionaryFile ,'r') as constraining_dict:
    constrainDict=constraining_dict.read().lower()
constrainDictionary = constrainDict.split('\n')

def constraining_score(text):
    constrainWordnum =0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in constrainDictionary:
            constrainWordnum +=1
    sumConstrainScore = constrainWordnum 
    
    return sumConstrainScore



# In[31]:


# Calculating positive word proportion

def positive_word_prop(positiveScore,wordcount):
    positive_word_proportion = 0
    if wordcount !=0:
        positive_word_proportion = positiveScore / wordcount
        
    return positive_word_proportion


# In[32]:


# Calculating negative word proportion

def negative_word_prop(negativeScore,wordcount):
    negative_word_proportion = 0
    if wordcount !=0:
        negative_word_proportion = negativeScore / wordcount
        
    return negative_word_proportion


# In[33]:


# Calculating uncertain word proportion

def uncertain_word_prop(uncertainScore,wordcount):
    uncertain_word_proportion = 0
    if wordcount !=0:
        uncertain_word_proportion = uncertainScore / wordcount
        
    return uncertain_word_proportion


# In[34]:


# Calculating constraining word proportion

def constraining_word_prop(constrainingScore,wordcount):
    constraining_word_proportion = 0
    if wordcount !=0:
        constraining_word_proportion = constrainingScore / wordcount
        
    return constraining_word_proportion


# In[35]:


# calculating Constraining words for whole report

def constrain_word_whole(mdaText,qqdmrText,rfText):
    wholeDoc = mdaText + qqdmrText + rfText
    constrainWordnumWhole =0
    rawToken = tokenizer(wholeDoc)
    for word in rawToken:
        if word in constrainDictionary:
            constrainWordnumWhole +=1
    sumConstrainScoreWhole = constrainWordnumWhole 
    
    return sumConstrainScoreWhole


# In[36]:


inputDirectory = 'D:/data science/Blackcoffer project/test'
masterFile = 'D:/data science/Blackcoffer project/cik_list1.csv'
dataList = rawdata_extract( inputDirectory , masterFile )
df = pd.DataFrame(dataList)

df['mda_positive_score'] = df.mda_extract.apply(positive_score)
df['mda_negative_score'] = df.mda_extract.apply(negative_word)
df['mda_polarity_score'] = np.vectorize(polarity_score)(df['mda_positive_score'],df['mda_negative_score'])
df['mda_average_sentence_length'] = df.mda_extract.apply(average_sentence_length)
df['mda_percentage_of_complex_words'] = df.mda_extract.apply(percentage_complex_word)
df['mda_fog_index'] = np.vectorize(fog_index)(df['mda_average_sentence_length'],df['mda_percentage_of_complex_words'])
df['mda_complex_word_count']= df.mda_extract.apply(complex_word_count)
df['mda_word_count'] = df.mda_extract.apply(total_word_count)
df['mda_uncertainty_score']=df.mda_extract.apply(uncertainty_score)
df['mda_constraining_score'] = df.mda_extract.apply(constraining_score)
df['mda_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['mda_positive_score'],df['mda_word_count'])
df['mda_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['mda_negative_score'],df['mda_word_count'])
df['mda_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['mda_uncertainty_score'],df['mda_word_count'])
df['mda_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['mda_constraining_score'],df['mda_word_count'])

df['qqdmr_positive_score'] = df.qqd_extract.apply(positive_score)
df['qqdmr_negative_score'] = df.qqd_extract.apply(negative_word)
df['qqdmr_polarity_score'] = np.vectorize(polarity_score)(df['qqdmr_positive_score'],df['qqdmr_negative_score'])
df['qqdmr_average_sentence_length'] = df.qqd_extract.apply(average_sentence_length)
df['qqdmr_percentage_of_complex_words'] = df.qqd_extract.apply(percentage_complex_word)
df['qqdmr_fog_index'] = np.vectorize(fog_index)(df['qqdmr_average_sentence_length'],df['qqdmr_percentage_of_complex_words'])
df['qqdmr_complex_word_count']= df.qqd_extract.apply(complex_word_count)
df['qqdmr_word_count'] = df.qqd_extract.apply(total_word_count)
df['qqdmr_uncertainty_score']=df.qqd_extract.apply(uncertainty_score)
df['qqdmr_constraining_score'] = df.qqd_extract.apply(constraining_score)
df['qqdmr_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['qqdmr_positive_score'],df['qqdmr_word_count'])
df['qqdmr_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['qqdmr_negative_score'],df['qqdmr_word_count'])
df['qqdmr_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['qqdmr_uncertainty_score'],df['qqdmr_word_count'])
df['qqdmr_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['qqdmr_constraining_score'],df['qqdmr_word_count'])

df['rf_positive_score'] = df.riskfactor_extract.apply(positive_score)
df['rf_negative_score'] = df.riskfactor_extract.apply(negative_word)
df['rf_polarity_score'] = np.vectorize(polarity_score)(df['rf_positive_score'],df['rf_negative_score'])
df['rf_average_sentence_length'] = df.riskfactor_extract.apply(average_sentence_length)
df['rf_percentage_of_complex_words'] = df.riskfactor_extract.apply(percentage_complex_word)
df['rf_fog_index'] = np.vectorize(fog_index)(df['rf_average_sentence_length'],df['rf_percentage_of_complex_words'])
df['rf_complex_word_count']= df.riskfactor_extract.apply(complex_word_count)
df['rf_word_count'] = df.riskfactor_extract.apply(total_word_count)
df['rf_uncertainty_score']=df.riskfactor_extract.apply(uncertainty_score)
df['rf_constraining_score'] = df.riskfactor_extract.apply(constraining_score)
df['rf_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['rf_positive_score'],df['rf_word_count'])
df['rf_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['rf_negative_score'],df['rf_word_count'])
df['rf_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['rf_uncertainty_score'],df['rf_word_count'])
df['rf_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['rf_constraining_score'],df['rf_word_count'])

df['constraining_words_whole_report'] = np.vectorize(constrain_word_whole)(df['mda_extract'],df['qqd_extract'],df['riskfactor_extract'])




# In[37]:


df.shape


# # Final Output 

# In[38]:


inputTextCol = ['mda_extract','qqd_extract','riskfactor_extract']
finalOutput = df.drop(inputTextCol,1)

finalOutput.head(150)


# In[39]:


# Writing to csv file
finalOutput.to_csv('textAnalysisOutput.csv', sep=',', encoding='utf-8')

