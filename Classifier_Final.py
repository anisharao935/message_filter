# Naive Bayes algorithm with stopwords removed

import csv
import string

# helper functions
def contains(inputList, element):
  '''uses binary search to find if a given list contains a certain element'''
  low = 0
  high = len(inputList)
  while low < high:
    mid = (low+high)//2
    #print(low, high, mid)
    #print(inputList[mid])
    if inputList[mid] == element:
      return True
    elif inputList[mid] < element:
      low = mid + 1
    else:
      high = mid
  return False
  
def get_words(fname):
    '''returns list of processed [no whitespace/capitalization] lines of text from given file'''
    f = open(fname)
    lines_original = f.readlines()
    lines_processed = []
    for line in lines_original:
        lines_processed.append(line.strip().lower())
    f.close()
    lines_processed.sort()
    return lines_processed

def tokenize(message):
    '''breaks message down into list of words without punctuation, whitespace, etc.'''
    message = message.strip()
    #print(message)
    for num in string.digits:
        message = message.replace(num, "")
    for punc in string.punctuation:
        if punc!="'": # preserve contractions
            message = message.replace(punc, "")
    processed = message.split()
    #print(processed)
    stopwords = get_words("/Users/anisha/Downloads/sms-messages/stopwords.txt")
    #print(stopwords)
    tokens = []
    for token in processed:
        if token!='' and not contains(stopwords, token.lower()):
            tokens.append(token.lower())
    #print(tokens)
    return tokens


# main function for classifying messages
def calculate_probability(message, isBusiness):
    '''calculates probability all words of a given message are business/personal'''
    global matrix
    global allTokens 
    global businessTokens 
    global personalTokens
    tokens = tokenize(message)
    #print(tokens)
    prob = 1
    tokenCount = 0
    if isBusiness:
        categoryID = 0
        category = businessTokens
    else:
        categoryID = 1
        category = personalTokens
    for token in tokens:
        if token in category:
            category_appearances = matrix[token][categoryID]
        else:
            category_appearances = 0
        prob *= (category_appearances + 1)/(len(category) + len(allTokens))
    return prob # average frequency of words

# uses index to return True (for business) or False (for personal)
def is_business(message):
    '''classifies given message as business or personal'''
    bus_prob = calculate_probability(message, True)
    pers_prob = calculate_probability(message, False)
    return bus_prob > pers_prob

# main code to check messages

fpath = "/Users/anisha/Downloads/sms-messages/personal-biz.csv"
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
lines_read = 0
matrix = {}
allTokens = set()
businessTokens = set()
personalTokens = set()
cutoff_num = 5000
# go through set number of messages
with open(fpath, newline='', encoding="latin1") as messages_file:
    lines = csv.DictReader(messages_file)
    for line in lines:
        lines_read += 1
        answer = line['v1']
        message = line['v2']
        if lines_read < 4000:
            # frequency train on first 4000
            tokens = tokenize(message)
            if answer=="business":
                use = 0
            else:
                use = 1
            for token in tokens:
                if token not in allTokens:
                    allTokens.add(token)
                    matrix[token] = [0, 0]
                matrix[token][use] += 1
                if use==0 and token not in businessTokens:
                    businessTokens.add(token)
                elif use==1 and token not in personalTokens:
                    personalTokens.add(token)
        else:
            # test on last 1000
            sender = is_business(message)
            #print(message, sender, answer)
            # update counters for true/false positive/negatives
            if sender==True:
                if answer=="business":
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if answer=="personal":
                    true_negative += 1
                else:
                    false_negative += 1
        if lines_read==cutoff_num:
            break
    # output results, calculate relevant measures
    accuracy = (true_positive + true_negative)/(true_positive + false_positive + true_negative + false_negative)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    print("{} true positives, {} false positives, {} true negatives, {} false negatives".format(true_positive, false_positive, true_negative, false_negative))
    print("accuracy {}, precision {}, recall {}".format(accuracy, precision, recall))


    
    
        






