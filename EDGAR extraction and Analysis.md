
# Data Extraction and Text Analysis


First function reading edar files.It will remove html tags and extract requried informtion


```python
# Requried imports
import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np
```


```python
# Text extraction patterns
mda_regex = r"item[^a-zA-Z\n]*\d\s*\.\s*management\'s discussion and analysis.*?^\s*item[^a-zA-Z\n]*\d\s*\.*"
qqd_regex = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Quantitative and Qualitative Disclosures about " \
            r"Market Risk.*?^\s*item\s*\d\s*"
riskfactor_regex = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Risk Factors.*?^\s*item\s*\d\s*"
```


```python
# Filepath locations
stopWordsFile = 'D:/data science/Blackcoffer project/StopWords_Generic.txt'
positiveWordsFile = 'D:/data science/Blackcoffer project/PositiveWords.txt'
nagitiveWordsFile = 'D:/data science/Blackcoffer project/NegativeWords.txt'
uncertainty_dictionaryFile = 'D:/data science/Blackcoffer project/uncertainty_dictionary.txt'
constraining_dictionaryFile = 'D:/data science/Blackcoffer project/constraining_dictionary.txt'

```


```python
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
```

# Section 1.1: Positive score, negative score, polarity score

Loading stop words dictionary for removing stop words


```python
with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []

```

tokenizeing module and filtering tokens using stop words list, removing punctuations


```python
# Tokenizer
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words

```


```python
# Loading positive words
with open(positiveWordsFile,'r') as posfile:
    positivewords=posfile.read().lower()
positiveWordList=positivewords.split('\n')

```


```python
# Loading negative words
with open(nagitiveWordsFile ,'r') as negfile:
    negativeword=negfile.read().lower()
negativeWordList=negativeword.split('\n')

```


```python
# Calculating positive score 
def positive_score(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in positiveWordList:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos
```


```python
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

```


```python
# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
    return pol_score

```

# Section 2 -Analysis of Readability -  Average Sentence Length, percentage of complex words, fog index


```python
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

```


```python
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
                        
```


```python
# calculating Fog Index 
# Fog index is calculated using -- Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex

```

# Section 4: Complex word count


```python
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
```

# Section 5: Word count


```python
#Counting total words

def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

```


```python
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


```


```python
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


```


```python
# Calculating positive word proportion

def positive_word_prop(positiveScore,wordcount):
    positive_word_proportion = 0
    if wordcount !=0:
        positive_word_proportion = positiveScore / wordcount
        
    return positive_word_proportion

```


```python
# Calculating negative word proportion

def negative_word_prop(negativeScore,wordcount):
    negative_word_proportion = 0
    if wordcount !=0:
        negative_word_proportion = negativeScore / wordcount
        
    return negative_word_proportion
```


```python
# Calculating uncertain word proportion

def uncertain_word_prop(uncertainScore,wordcount):
    uncertain_word_proportion = 0
    if wordcount !=0:
        uncertain_word_proportion = uncertainScore / wordcount
        
    return uncertain_word_proportion
```


```python
# Calculating constraining word proportion

def constraining_word_prop(constrainingScore,wordcount):
    constraining_word_proportion = 0
    if wordcount !=0:
        constraining_word_proportion = constrainingScore / wordcount
        
    return constraining_word_proportion
```


```python
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
```


```python
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



```


```python
df.shape
```




    (152, 52)



# Final Output 


```python
inputTextCol = ['mda_extract','qqd_extract','riskfactor_extract']
finalOutput = df.drop(inputTextCol,1)

finalOutput.head(150)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>CONAME</th>
      <th>FDATE</th>
      <th>FORM</th>
      <th>FYRMO</th>
      <th>SECFNAME</th>
      <th>mda_positive_score</th>
      <th>mda_negative_score</th>
      <th>mda_polarity_score</th>
      <th>mda_average_sentence_length</th>
      <th>...</th>
      <th>rf_fog_index</th>
      <th>rf_complex_word_count</th>
      <th>rf_word_count</th>
      <th>rf_uncertainty_score</th>
      <th>rf_constraining_score</th>
      <th>rf_positive_word_proportion</th>
      <th>rf_negative_word_proportion</th>
      <th>rf_uncertainty_word_proportion</th>
      <th>rf_constraining_word_proportion</th>
      <th>constraining_words_whole_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>3/6/1998</td>
      <td>10-K405</td>
      <td>199803</td>
      <td>edgar/data/3662/0000950170-98-000413.txt</td>
      <td>17</td>
      <td>61</td>
      <td>-0.564103</td>
      <td>24</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>5/15/1998</td>
      <td>10-Q</td>
      <td>199805</td>
      <td>edgar/data/3662/0000950170-98-001001.txt</td>
      <td>9</td>
      <td>46</td>
      <td>-0.672727</td>
      <td>30</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>8/13/1998</td>
      <td>NT 10-Q</td>
      <td>199808</td>
      <td>edgar/data/3662/0000950172-98-000783.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/12/1998</td>
      <td>10-K/A</td>
      <td>199811</td>
      <td>edgar/data/3662/0000950170-98-002145.txt</td>
      <td>41</td>
      <td>119</td>
      <td>-0.487500</td>
      <td>23</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/16/1998</td>
      <td>NT 10-Q</td>
      <td>199811</td>
      <td>edgar/data/3662/0000950172-98-001203.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/25/1998</td>
      <td>10-Q/A</td>
      <td>199811</td>
      <td>edgar/data/3662/0000950170-98-002278.txt</td>
      <td>19</td>
      <td>63</td>
      <td>-0.536585</td>
      <td>23</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>12/22/1998</td>
      <td>10-Q</td>
      <td>199812</td>
      <td>edgar/data/3662/0000950170-98-002401.txt</td>
      <td>40</td>
      <td>106</td>
      <td>-0.452055</td>
      <td>22</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>12/22/1998</td>
      <td>10-Q</td>
      <td>199812</td>
      <td>edgar/data/3662/0000950170-98-002402.txt</td>
      <td>38</td>
      <td>102</td>
      <td>-0.457143</td>
      <td>22</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>3/31/1999</td>
      <td>NT 10-K</td>
      <td>199903</td>
      <td>edgar/data/3662/0000950172-99-000362.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>5/11/1999</td>
      <td>10-K</td>
      <td>199905</td>
      <td>edgar/data/3662/0000950170-99-000775.txt</td>
      <td>71</td>
      <td>270</td>
      <td>-0.583578</td>
      <td>23</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>5/17/1999</td>
      <td>NT 10-Q</td>
      <td>199905</td>
      <td>edgar/data/3662/0000950172-99-000584.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>6/11/1999</td>
      <td>10-Q</td>
      <td>199906</td>
      <td>edgar/data/3662/0000950170-99-001005.txt</td>
      <td>31</td>
      <td>60</td>
      <td>-0.318681</td>
      <td>23</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>8/16/1999</td>
      <td>NT 10-Q</td>
      <td>199908</td>
      <td>edgar/data/3662/0000950172-99-001074.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>8/19/1999</td>
      <td>10-Q</td>
      <td>199908</td>
      <td>edgar/data/3662/0000950170-99-001361.txt</td>
      <td>67</td>
      <td>129</td>
      <td>-0.316327</td>
      <td>21</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/8/1999</td>
      <td>10-K/A</td>
      <td>199911</td>
      <td>edgar/data/3662/0000889812-99-003241.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>21</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/8/1999</td>
      <td>10-Q/A</td>
      <td>199911</td>
      <td>edgar/data/3662/0000950170-99-001639.txt</td>
      <td>69</td>
      <td>144</td>
      <td>-0.352113</td>
      <td>21</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/8/1999</td>
      <td>10-Q/A</td>
      <td>199911</td>
      <td>edgar/data/3662/0000950170-99-001640.txt</td>
      <td>40</td>
      <td>127</td>
      <td>-0.520958</td>
      <td>24</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/15/1999</td>
      <td>NT 10-Q</td>
      <td>199911</td>
      <td>edgar/data/3662/0000950172-99-001626.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3662</td>
      <td>SUNBEAM CORP/FL/</td>
      <td>11/19/1999</td>
      <td>10-Q</td>
      <td>199911</td>
      <td>edgar/data/3662/0000950170-99-001856.txt</td>
      <td>73</td>
      <td>154</td>
      <td>-0.356828</td>
      <td>23</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>3/22/2006</td>
      <td>10-K</td>
      <td>200603</td>
      <td>edgar/data/3982/0000950129-06-002926.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>5/1/2006</td>
      <td>10-K/A</td>
      <td>200605</td>
      <td>edgar/data/3982/0000950129-06-004690.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>5/10/2006</td>
      <td>10-Q</td>
      <td>200605</td>
      <td>edgar/data/3982/0000950129-06-005244.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>7/24/2006</td>
      <td>10-K/A</td>
      <td>200607</td>
      <td>edgar/data/3982/0000950129-06-007243.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>7/24/2006</td>
      <td>10-Q/A</td>
      <td>200607</td>
      <td>edgar/data/3982/0000950129-06-007244.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>8/14/2006</td>
      <td>10-Q</td>
      <td>200608</td>
      <td>edgar/data/3982/0000950129-06-007871.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>11/8/2006</td>
      <td>10-Q</td>
      <td>200611</td>
      <td>edgar/data/3982/0000950129-06-009522.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>11.387751</td>
      <td>958</td>
      <td>2041</td>
      <td>106</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>12/29/2006</td>
      <td>10-Q/A</td>
      <td>200612</td>
      <td>edgar/data/3982/0000950134-06-023819.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>3/15/2007</td>
      <td>10-K</td>
      <td>200703</td>
      <td>edgar/data/3982/0000950129-07-001381.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>7.350471</td>
      <td>2514</td>
      <td>6683</td>
      <td>79</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>5/10/2007</td>
      <td>10-Q</td>
      <td>200705</td>
      <td>edgar/data/3982/0000950129-07-002432.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>4.978723</td>
      <td>21</td>
      <td>47</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3982</td>
      <td>ALLIS CHALMERS ENERGY INC.</td>
      <td>8/9/2007</td>
      <td>10-Q</td>
      <td>200708</td>
      <td>edgar/data/3982/0000950129-07-003918.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>4.983333</td>
      <td>22</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>10/21/2009</td>
      <td>10-Q</td>
      <td>200910</td>
      <td>edgar/data/6201/0000006201-09-000038.txt</td>
      <td>43</td>
      <td>90</td>
      <td>-0.353383</td>
      <td>114</td>
      <td>...</td>
      <td>42.949902</td>
      <td>1719</td>
      <td>4587</td>
      <td>94</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>122</td>
    </tr>
    <tr>
      <th>121</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>11/6/2009</td>
      <td>10-Q/A</td>
      <td>200911</td>
      <td>edgar/data/6201/0000006201-09-000040.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>122</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>2/17/2010</td>
      <td>10-K</td>
      <td>201002</td>
      <td>edgar/data/6201/0000006201-10-000006.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>14.183350</td>
      <td>1817</td>
      <td>3964</td>
      <td>164</td>
      <td>88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88</td>
    </tr>
    <tr>
      <th>123</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>4/21/2010</td>
      <td>10-Q</td>
      <td>201004</td>
      <td>edgar/data/6201/0000006201-10-000013.txt</td>
      <td>52</td>
      <td>79</td>
      <td>-0.206107</td>
      <td>86</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>124</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>7/21/2010</td>
      <td>10-Q</td>
      <td>201007</td>
      <td>edgar/data/6201/0000950123-10-066894.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>125</th>
      <td>6201</td>
      <td>AMR CORP</td>
      <td>10/20/2010</td>
      <td>10-Q</td>
      <td>201010</td>
      <td>edgar/data/6201/0000950123-10-094605.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>126</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>2/14/1994</td>
      <td>10-Q</td>
      <td>199402</td>
      <td>edgar/data/6260/0000006260-94-000014.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>5</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>5/12/1994</td>
      <td>10-Q</td>
      <td>199405</td>
      <td>edgar/data/6260/0000006260-94-000016.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>5</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>12/29/1997</td>
      <td>10-K</td>
      <td>199712</td>
      <td>edgar/data/6260/0000006260-97-000011.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>16</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>129</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>2/17/1998</td>
      <td>10-Q</td>
      <td>199802</td>
      <td>edgar/data/6260/0000006260-98-000001.txt</td>
      <td>8</td>
      <td>10</td>
      <td>-0.111111</td>
      <td>29</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>130</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>5/13/1998</td>
      <td>10-Q</td>
      <td>199805</td>
      <td>edgar/data/6260/0000006260-98-000003.txt</td>
      <td>17</td>
      <td>21</td>
      <td>-0.105263</td>
      <td>27</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>131</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>8/14/1998</td>
      <td>10-Q</td>
      <td>199808</td>
      <td>edgar/data/6260/0000914121-98-000672.txt</td>
      <td>13</td>
      <td>30</td>
      <td>-0.395349</td>
      <td>28</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>132</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>12/29/1998</td>
      <td>10-K405</td>
      <td>199812</td>
      <td>edgar/data/6260/0001047469-98-045227.txt</td>
      <td>18</td>
      <td>47</td>
      <td>-0.446154</td>
      <td>20</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>133</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>2/11/1999</td>
      <td>10-Q</td>
      <td>199902</td>
      <td>edgar/data/6260/0000006260-99-000005.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>5/17/1999</td>
      <td>10-Q</td>
      <td>199905</td>
      <td>edgar/data/6260/0000006260-99-000007.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>6</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>135</th>
      <td>6260</td>
      <td>ANACOMP INC</td>
      <td>8/13/1999</td>
      <td>10-Q</td>
      <td>199908</td>
      <td>edgar/data/6260/0000006260-99-000010.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>6</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>11/12/1998</td>
      <td>10-Q</td>
      <td>199811</td>
      <td>edgar/data/11860/0000011860-98-000022.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>137</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>3/24/1999</td>
      <td>10-K405</td>
      <td>199903</td>
      <td>edgar/data/11860/0001021408-99-000543.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>7</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>5/14/1999</td>
      <td>10-Q</td>
      <td>199905</td>
      <td>edgar/data/11860/0000011860-99-000025.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>139</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>6/29/1999</td>
      <td>10-K/A</td>
      <td>199906</td>
      <td>edgar/data/11860/0000011860-99-000030.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>140</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>8/5/1999</td>
      <td>10-Q</td>
      <td>199908</td>
      <td>edgar/data/11860/0000011860-99-000035.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>11/2/1999</td>
      <td>10-Q</td>
      <td>199911</td>
      <td>edgar/data/11860/0000011860-99-000042.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>142</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>3/9/2000</td>
      <td>10-K</td>
      <td>200003</td>
      <td>edgar/data/11860/0000011860-00-000019.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>9</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>143</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>5/3/2000</td>
      <td>10-Q</td>
      <td>200005</td>
      <td>edgar/data/11860/0000011860-00-000022.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>144</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>6/28/2000</td>
      <td>10-K/A</td>
      <td>200006</td>
      <td>edgar/data/11860/0000011860-00-000025.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>7/26/2000</td>
      <td>10-Q</td>
      <td>200007</td>
      <td>edgar/data/11860/0000011860-00-000028.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>11860</td>
      <td>BETHLEHEM STEEL CORP /DE/</td>
      <td>10/25/2000</td>
      <td>10-Q</td>
      <td>200010</td>
      <td>edgar/data/11860/0000011860-00-000038.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>147</th>
      <td>12239</td>
      <td>SPHERIX INC</td>
      <td>4/2/2007</td>
      <td>10-K</td>
      <td>200704</td>
      <td>edgar/data/12239/0001104659-07-024804.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>12239</td>
      <td>SPHERIX INC</td>
      <td>5/16/2007</td>
      <td>NT 10-Q</td>
      <td>200705</td>
      <td>edgar/data/12239/0001104659-07-040463.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>12239</td>
      <td>SPHERIX INC</td>
      <td>5/18/2007</td>
      <td>10-Q</td>
      <td>200705</td>
      <td>edgar/data/12239/0001104659-07-041441.txt</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>8.525000</td>
      <td>20</td>
      <td>64</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 49 columns</p>
</div>




```python
# Writing to csv file
finalOutput.to_csv('textAnalysisOutput.csv', sep=',', encoding='utf-8')

```
