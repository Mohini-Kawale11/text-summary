# >>> nltk.download('words')
# >>> nltk.download('maxent_ne_chunker')
# >>> nltk.download('averaged_perceptron_tagger')

import os



import re
import csv
import math
import spacy
import nltk
import pandas as pd
from sklearn import metrics
from nltk.tag import pos_tag 
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from threadpoolctl import threadpool_limits

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score


ps = PorterStemmer()
threadpool_limits(limits=1, user_api='openmp')
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import PyPDF2

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        if num_pages >= 100:
            num_pages = 100
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

nlp = spacy.load("en_core_web_sm")

def answer_question(text, question):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        
        question_doc = nlp(question)
        keywords = [token.lemma_ for token in question_doc if token.is_alpha and not token.is_stop]

        best_sentence = None
        best_count = 0

        for sentence in sentences:
            count = sum(1 for keyword in keywords if keyword in sentence.lemma_)
            if count > best_count:
                best_count = count
                best_sentence = sentence

        return best_sentence.text if best_sentence else "No relevant answer found."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in processing the question."


def evaluate_accuracy(pdf_text, questions_and_answers):
    y_true = []
    y_pred = []

    for qa in questions_and_answers:
        question = qa['question']
        true_answer = qa['answer']
        predicted_answer = answer_question(pdf_text, question)

        y_true.append(true_answer)
        y_pred.append(predicted_answer)

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    return precision, recall, f1

def main():
    with threadpool_limits(limits=1, user_api='blas'):

        pdf_file_path = input("Enter the path to the PDF file: ")
        word_limit = int(input("Enter the desired word limit for the summary: "))
        text = extract_text_from_pdf(pdf_file_path)
        accuracy = get_data(text, "", 0, word_limit)
        print("Summary:", accuracy)
        
        question = input("Enter your question based on the PDF content: ")
        answer = answer_question(text, question)
        print("Answer:", answer)

    # Collect user input for questions and answers
    # questions_and_answers = []
    # while True:
    #     question = input("Enter a question (or 'done' to finish): ")
    #     if question.lower() == 'done':
    #         break
    #     answer = input("Enter the correct answer: ")
    #     questions_and_answers.append({"question": question, "answer": answer})

    # # Evaluate accuracy
    # evaluate_accuracy(text, questions_and_answers)






caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

stop = set(stopwords.words('english'))

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    #if "," in text: text = text.replace(",\"","\",")

    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    #text = text.replace(",","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

#named entity recoginition
def ner(sample):
    sentences = nltk.sent_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    return len(entity_names)  

#Using Jaccard similarity to calculate if two sentences are similar
def similar(tokens_a, tokens_b) :
    ratio = len(set(tokens_a).intersection(tokens_b))/ float(len(set(tokens_a).union(tokens_b)))
    return ratio

# ......................part 1 (cue phrases).................
def cue_phrases(sent_tokens):
    QPhrases=["incidentally", "example", "anyway", "furthermore","according",
            "first", "second", "then", "now", "thus", "moreover", "therefore", "hence", "lastly", "finally", "summary"]

    cue_phrases={}
    for sentence in sent_tokens:
        cue_phrases[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for word in word_tokens:
            if word.lower() in QPhrases:
                cue_phrases[sentence] += 1
    maximum_frequency = max(cue_phrases.values())
    for k in cue_phrases.keys():
        try:
            cue_phrases[k] = cue_phrases[k] / maximum_frequency
            cue_phrases[k]=round(cue_phrases[k],3)
        except ZeroDivisionError:
            x=0
    return cue_phrases


# .......................part2 (numerical data)...................
def numeric_data(sent_tokens):
    numeric_data={}
    for sentence in sent_tokens:
        numeric_data[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isdigit():
                numeric_data[sentence] += 1
    maximum_frequency = max(numeric_data.values())
    for k in numeric_data.keys():
        try:
            numeric_data[k] = (numeric_data[k]/maximum_frequency)
            numeric_data[k] = round(numeric_data[k], 3)
        except ZeroDivisionError:
            x=0
    return numeric_data


#....................part3(sentence length)........................
def sent_len_score(sent_tokens):
    sent_len_score={}
    for sentence in sent_tokens:
        sent_len_score[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        if len(word_tokens) in range(0,10):
            sent_len_score[sentence]=1-0.02*(10-len(word_tokens))
        elif len(word_tokens) in range(10,20):
            sent_len_score[sentence]=1
        else:
            sent_len_score[sentence]=1-(0.02)*(len(word_tokens)-20)
    for k in sent_len_score.keys():
        sent_len_score[k]=round(sent_len_score[k],4)
    return sent_len_score


#....................part4(sentence position)........................
def sentence_position(sent_tokens):
    sentence_position={}
    d=1
    no_of_sent=len(sent_tokens)
    for i in range(no_of_sent):
        a=1/d
        b=1/(no_of_sent-d+1)
        sentence_position[sent_tokens[d-1]]=max(a,b)
        d=d+1
    for k in sentence_position.keys():
        sentence_position[k]=round(sentence_position[k],3)
    return sentence_position


#.........Create a frequency table to compute the frequency of each word........
def word_frequency(sent_tokens,word_tokens_refined):
    freqTable = {}
    for word in word_tokens_refined:    
        if word in freqTable:         
            freqTable[word] += 1    
        else:         
            freqTable[word] = 1
    for k in freqTable.keys():
        freqTable[k]= math.log10(1+freqTable[k])
#Compute word frequnecy score of each sentence
    word_frequency={}
    for sentence in sent_tokens:
        word_frequency[sentence]=0
        e=nltk.word_tokenize(sentence)
        f=[]
        for word in e:
            f.append(ps.stem(word))
        for word,freq in freqTable.items():
            if word in f:
                word_frequency[sentence]+=freq
    maximum=max(word_frequency.values())
    for key in word_frequency.keys():
        try:
            word_frequency[key]=word_frequency[key]/maximum
            word_frequency[key]=round(word_frequency[key],3)
        except ZeroDivisionError:
            x=0
    return word_frequency


#........................part 6 (upper cases).................................
def upper_case(sent_tokens):
    upper_case={}
    for sentence in sent_tokens:
        upper_case[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isupper():
                upper_case[sentence] += 1
    maximum_frequency = max(upper_case.values())
    for k in upper_case.keys():
        try:
            upper_case[k] = (upper_case[k]/maximum_frequency)
            upper_case[k] = round(upper_case[k], 3)
        except ZeroDivisionError:
            x=0
    return upper_case


#......................... part7 (number of proper noun)...................
def proper_noun(sent_tokens):
    proper_noun={}
    for sentence in sent_tokens:
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
        proper_noun[sentence]=len(propernouns)
    maximum_frequency = max(proper_noun.values())
    for k in proper_noun.keys():
        try:
            proper_noun[k] = (proper_noun[k]/maximum_frequency)
            proper_noun[k] = round(proper_noun[k], 3)
        except ZeroDivisionError:
            x=0
    return proper_noun


#.................. part 8 (word matches with heading) ...................
def head_match(sent_tokens):
    head_match={}
    heading=sent_tokens[0]
    stopWords = list(set(stopwords.words("english")))
    for sentence in sent_tokens:
        head_match[sentence]=0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k not in stopWords:
                k = ps.stem(k)
                if k in ps.stem(heading):
                    head_match[sentence] += 1
    maximum_frequency = max(head_match.values())
    for k in head_match.keys():
        try:
            head_match[k] = (head_match[k]/maximum_frequency)
            head_match[k] = round(head_match[k], 3)
        except ZeroDivisionError:
            x=0
    return head_match


#..................... part 9(Centrality)..........................
def centrality(sent_tokens,cv,word_tokens_refined):
    global corpus
    l=len(sent_tokens)
    centrality_value={}
    Tf_Idf = cv.fit_transform(corpus).toarray()
    
    # Cosine Similarity
    cosine_value={}
    for i in range(0,l):
        sentence=sent_tokens[i]
        cosine_value[sentence]=0
        for j in range (0,l):
            
            dot_product=0
            len_vec1=0
            len_vec2=0
            for k in range(len(word_tokens_refined)):
                dot_product+=Tf_Idf[i][k]*Tf_Idf[j][k]
                len_vec1+=Tf_Idf[i][k]*Tf_Idf[i][k]
                len_vec2+=Tf_Idf[j][k]*Tf_Idf[j][k]

            val=1
            if(len_vec1!=0 and len_vec2!=0):
                val=dot_product/(math.sqrt(len_vec1)*math.sqrt(len_vec2))
            cosine_value[sentence]+=val
    
    for i in range(0,l):
        sentence=sent_tokens[i]
        cosine_value[sentence]=cosine_value[sentence]/l
        
    cosine_value=sorted(cosine_value.items(), key=lambda x: x[1], reverse=True)
    
    #centrality
    cos_value={}
    factor=1/l
    for i in range(len(cosine_value)):
        k=cosine_value[i][0]
        cos_value[k]=1-i*factor
        cos_value[k]=round(cos_value[k],3)
        i+=1
    
    for i in range(0,l):
        sentence=sent_tokens[i]
        centrality_value[sentence]=cos_value[sentence]
    
    return centrality_value


#..................... part 10(thematic).........................
def thematicFeature(sent_tokens) :
    word_list = []
    for sentence in sent_tokens :
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                #print(word)
                word_list.append(word)
            except Exception as e:
                print("ERR")
    counts = Counter(word_list)
    number_of_words = len(counts)
    most_common = counts.most_common(10)
    thematic_words = []
    for data in most_common :
        thematic_words.append(data[0])
    scores = []
    for sentence in sent_tokens :
        score = 0
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                if word in thematic_words :
                    score = score + 1
                #print(word)
            except Exception as e:
                print("ERR")
        score = 1.0*score/(number_of_words)
        scores.append(score)
    max_value=max(scores)
    if(max_value!=0):
        for k in range(len(scores)):
            scores[k] = (scores[k]/max_value)
            scores[k] = round(scores[k], 3)
    return scores

#..................... part 11(Named Entity Recoginition).........................
def namedEntityRecog(sent_tokens) :
    counts = []
    for sentence in sent_tokens :
        count = ner(sentence)
        counts.append(count)
    max_value=max(counts)
    if(max_value!=0):
        for k in range(len(counts)):
            counts[k] = (counts[k]/max_value)
            counts[k] = round(counts[k], 3)
    return counts


#..................... part 12(Pos tagging).........................
def posTagger(tokenized_sentences) :
    tagged = []
    for sentence in tokenized_sentences :
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged

#..................... part 13(jaccards similarity).........................
def similarityScores(tokenized_sentences) :
    scores = []
    for sentence in tokenized_sentences :
        score = 0;
        for sen in tokenized_sentences :
            if sen != sentence :
                score += similar(sentence,sen)
        scores.append(score)
    max_value=max(scores)
    if(max_value!=0):
        for k in range(len(scores)):
            scores[k] = (scores[k]/max_value)
            scores[k] = round(scores[k], 3)
    return scores


corpus=[]

def get_data(text,text1,flag,word_limit=200):
    
    #sent_tokens = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])|\.(?=[^0-9])|\n', text)
    #sent_tokens = nltk.sent_tokenize(text)
    sent_tokens = split_into_sentences(text)
    word_tokens = nltk.word_tokenize(text)

    sent_tokens_temp=[]
    for sentence in sent_tokens:
        if(sentence==''):
            continue
        if(sentence in sent_tokens_temp):
            continue
        sent_tokens_temp.append(sentence)
    sent_tokens=sent_tokens_temp
    
    word_tokens_lower=[word.lower() for word in word_tokens]
    stopWords = list(set(stopwords.words("english")))
    word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]

    for sentence in sent_tokens:
        word_tokens=nltk.word_tokenize(sentence)
        word_tokens_lower=[word.lower() for word in word_tokens]
        stopWords = list(set(stopwords.words("english")))
        word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]
        word_tokens_refined = ' '.join(word_tokens_refined)
        corpus.append(word_tokens_refined)

    cv = TfidfVectorizer()
    g=cue_phrases(sent_tokens)
    z=list(g.keys())
    g=list(g.values())
    h=numeric_data(sent_tokens)
    h=list(h.values())
    i=sent_len_score(sent_tokens)
    i=list(i.values())
    j=sentence_position(sent_tokens)
    j=list(j.values())   
    p=upper_case(sent_tokens)
    p=list(p.values())
    l=head_match(sent_tokens)
    l=list(l.values())
    m=word_frequency(sent_tokens,word_tokens_refined)
    m=list(m.values())
    n=proper_noun(sent_tokens)
    n=list(n.values())
    c=centrality(sent_tokens,cv,word_tokens_refined)
    c=list(c.values())
    d=thematicFeature(sent_tokens)
    e=namedEntityRecog(sent_tokens)
    #q=posTagger(sent_tokens)
    r=similarityScores(sent_tokens)

    if flag == 0:
        total_score = {}
        sumValues = 0
        for k in range(len(sent_tokens)):
            score = g[k] + h[k] + i[k] + j[k] + p[k] + l[k] + m[k] + n[k] + c[k] + d[k] + e[k] + r[k]
            total_score[sent_tokens[k]] = score
            sumValues += score

        average = sumValues / len(sent_tokens)

        # Storing sentences into our summary.
        summary = ""
        total_words = 0

        # Loop over sentences and add them to summary until word limit is reached
        for sentence, score in sorted(total_score.items(), key=lambda x: x[1], reverse=True):
            words_in_sentence = len(sentence.split())
            if total_words + words_in_sentence <= word_limit:
                summary += " " + sentence
                total_words += words_in_sentence
            else:
                break

        return summary



    #sent_tokens1 = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])|\.(?=[^0-9])|\n', text1)
    #sent_tokens1 = nltk.sent_tokenize(text1)
    sent_tokens1=split_into_sentences(text1)
    word_tokens1 = nltk.word_tokenize(text1)

    sent_tokens1_temp=[]
    for sentence in sent_tokens1:
        if(sentence==''):
            continue
        if(sentence in sent_tokens1_temp):
            continue
        sent_tokens1_temp.append(sentence)
    sent_tokens1=sent_tokens1_temp

    label={}
    for sentence in sent_tokens:
        label[sentence]=0
    for sent in sent_tokens1:
        if sent in sent_tokens:
            label[sent]=1
                    
    o=list(label.values())

    df=pd.DataFrame(columns=['cue_phrase','numerical_data','sent_length','sent_position','word_freq','upper','proper_nouns','head_matching','centrality','thematic','ner','jaccard','key','label'])
    df = df.append(pd.DataFrame({'cue_phrase': g,
                                 'numerical_data': h,
                                 'sent_length': i,
                                 'sent_position': j,
                                 'upper': p,
                                 'head_matching': l,
                                 'word_freq': m,
                                 'proper_nouns': n,
                                 'centrality':c,
                                 'thematic':d,
                                 'ner':e,
                                 'jaccard':r,
                                 'key': z,
                                 'label': o}), ignore_index=True)

    df['label']=df['label'].astype(int)

    columns=['cue_phrase','numerical_data','sent_length','sent_position','word_freq','upper','proper_nouns','head_matching','centrality','thematic','ner','jaccard']
    training=df[columns]
    test=df.label


    X_train, X_test, y_train, y_test = train_test_split(training, test, test_size=0.3)

    clf2 = LogisticRegression()
    #Train the model using the training sets
    clf2.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf2.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    accuracy=metrics.accuracy_score(y_test, y_pred)*100

    return accuracy



if __name__ == "__main__":
    main()