# EDGAR-reports-Text-Analysis
Data from EDGAR filling was extracted and text analysis was performed.

In this project, text data extraction and text analytics was performed on EDGAR filling. The analysis was on done on 10k and 10Q filling. It was performed using python.

## Input

The input files consist of different filling from EDGAR. The format was .txt. Total 152 files were processed. 

## Extraction and Analysis

A. Basic cleaning was performed and target sections were extracted using regex.

Target section were - 
1. Management's Discussion and Analysis

2. Quantitative and Qualitative Disclosures about Market Risk	

3. Risk Factors

B. Different parts of text analysis were performed which included - 

1. Sentiment Analysis
  
2. Analysis of Readability
  
3. complex word count
  
4. word count

### Sentiment Analysis

Sentiment Analysis was performed using lexical based approach.

**Positive Score**: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.

**Negative Score**: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. I multiply the score with -1 so that the score is a positive number.

Polarity Score: This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
**Polarity Score** = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
Range is from -1 to +1

All the required dictionaries were created using -  https://sraf.nd.edu/textual-analysis/resources/#LM%20Sentiment%20Word%20Lists

### Analysis of Readability

Average sentence length, Fog index, complex word count and total word count were calculated.
The following formulas were used -

**Average Sentence Length** = the number of words / the number of sentences

**Percentage of Complex words** = the number of complex words / the number of words 
where Complex words are words in the text that contain more than two syllables.

**Fog Index** = 0.4 * (Average Sentence Length + Percentage of Complex words)

•	Apart from these, 6 other metrics were calculated. 
•	They were - 
•	positive word proportion
•	Negative word proportion
•	uncertain word score and proportion
•	constraining word score and proportion

Instruction to execute the python note book and script are included in Execution instrictions.pdf
Financial reports can be downloaded from EDGAR server during offline hours.

All the required dictionaries are included in the git. 

