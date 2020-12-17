#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################

# HOMEWORK 3 - DMT
#TAGLIERI ALESSANDROO
#LATO GUGLIELMO

#TASK 2


#IMPORT LIBRARIES
import json
from flair.data import Sentence
from flair.models import SequenceTagger
import string
from modules import build_model_by_name
from utils import print_sentence_predictions, load_vocab
import options as options
import evaluation_metrics as evaluation_metrics
import numpy as np
from matplotlib import pyplot as plt
from modules import build_model_by_name
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import json
from sklearn.metrics import accuracy_score

# read train set from json
train_data = []
for line in open('/Users/digitalfirst/Desktop/dmt_3hw/train.jsonl', 'r'):
    train_data.append(json.loads(line))
    

    
# read TEST set from json
official_data = []
for line in open('/Users/digitalfirst/Desktop/dmt_3hw/singletoken_test_fever_homework_NLP.jsonl', 'r'):
    official_data.append(json.loads(line))
    
# read bert vocab from json
with open('/Users/digitalfirst/Desktop/dmt_3hw/cased_L-12_H-768_A-12/vocab.txt') as f:
    content = f.readlines()
bert_vocab = [x.strip().lower() for x in content] 
    
    
#filter train and dev set witohout claim with label equal to "NOT ENOUGH INFO"
train_new = []
for current in train_data:
    if current['label'] != 'NOT ENOUGH INFO':
        train_new.append(current)   

#we put data from official data in data_point_official with right format (as indiucated in hw track)
data_point_official = []
for current in official_data:
    data_point = {}
    data_point["claim"] =  current['claim']
    data_point["id"] =  current['id']
    data_point["label"] =  ''
    data_point["entity"] =  [{'mention' : current['entity']['mention'], 'start_char' : current['entity']['start_character'], 'end_char' : current['entity']['end_character']}]
    data_point_official.append(data_point)
    
#initialize togger with ner option        
tagger = SequenceTagger.load('ner')



#iterate very claim in train set and put in data_point_train all claims that have requisiteise wirtten in the track        
data_point_train = []

for datapoint in train_new:

    new_train = {}
    new_train['id']=datapoint['id']
    new_train['label']=datapoint['label']
    new_train['claim']=datapoint['claim']

    sentence = Sentence(datapoint['claim'])
    
    tagger.predict(sentence)
    
    
    new_train['entity']=[]
    
    #iterate all entity in every claim
    for sentence in sentence.to_dict(tag_type='ner')['entities']:
       
        
        new_entities = {}
        #if entiy contains - or space (two or more words) skip this claim and we don't condier it as claim
        if ('-' in sentence['text']) or (' ' in sentence['text']):
            continue
        # we delete puntaction in the recognized entity and set it as lower word. After we check if this entity is present in bert vocabulary
        new_sentence = sentence['text'].lower().translate(str.maketrans('', '', string.punctuation))
        if new_sentence in bert_vocab:
        
            new_entities['mention']=new_sentence
            new_entities['start_char']=sentence['start_pos']
            #if word contains puntuaction at the end of the word we delete this puntuaction and change end podition of this entity
            if sentence['text'][-1]=='.':
                new_entities['end_char']=sentence['end_pos']-1
            else:
                new_entities['end_char']=sentence['end_pos']
            new_train['entity'].append(new_entities)
            
        else:
            continue
    #if the current claim has only one entity that respects all requisities, we put it in data_point_dev
    if len(new_train['entity'])==1:
        data_point_train.append(new_train)



# fucntion that from a single data_point, it give in output a claim with [MASK] inside the entity. It returns this entity into the var token
def getClaimwithMaks(data_point):
    sentence = data_point['claim']
    token = data_point['entity'][0]['mention']
    start = data_point['entity'][0]['start_char']
    end = data_point['entity'][0]['end_char']
    claim_withMask = sentence[:start] + "[MASK]" + sentence[end:]
    
    return claim_withMask, token




#this function is our evaluation.py . It takes in input args and models (the same in evaluation.py) and
# claimWithMAsk is a claim with [mask] inside of entity and token represents the current entity
def LAMA_results(args, claimWithMask, token, models):
    #result will contain label SUPPORT or REJECT as the result of predictions
    result=""
    
    
    args.text=claimWithMask
    if not args.text and not args.interactive:
        msg = "ERROR: either you start LAMA eval_generation with the "               "interactive option (--i) or you pass in input a piece of text (--t)"
        raise ValueError(msg)

    stopping_condition = True

    #print("Language Models: {}".format(args.models_names))
    
    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    while stopping_condition:
        if args.text:
            text = args.text
            stopping_condition = False
        else:
            text = input("insert text:")

        if args.split_sentence:
            import spacy
            # use spacy to tokenize input sentence
            nlp = spacy.load(args.spacy_model)
            tokens = nlp(text)
            print(tokens)
            sentences = []
            for s in tokens.sents:
                print(" - {}".format(s))
                sentences.append(s.text)
        else:
            sentences = [text]

        if len(sentences) > 2:
            print("WARNING: only the first two sentences in the text will be considered!")
            sentences = sentences[:2]
        
        for model_name, model in models.items():
            #print("\n{}:".format(model_name))
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list
            
            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                # we change evaluation_metrics function. In this function we return also a dataframe that has the same content of msg that is printed with top 1' predictions with their prob.
                # we store this dataframe in df_predictions
                a,b,c,d, df_predictions = evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list)
                
                list_predictions = df_predictions['prediction'].tolist()
                # we set lower all predictions becuase our entity is all in lower case. This to avoid confusion.
                list_predictions=[x.lower() for x in list_predictions]
                
                
                #we check if current entity is equal to the one of the top 10 predictions. If it's equal, we set SUPPORT as result, otherwise REJECT
                if token in list_predictions:
                    result="SUPPORT"
              
                    
                else:
                    result="REJECT"
                   

                return result

#This function is a reformulation of get_contextual_embeddings. It takes in input args as in the previous functions) models to specify bert model and other two var.
# sentence_mask is a sentence with [mask] nad sentence_noMask is a sentence wihout mask and with entity
def main(args, sentence_mask, sentence_noMask, models):
    sentences = [
        [sentence_mask],  # single-sentence instance
        [sentence_noMask],  # two-sentence
    ]

    #print("Language Models: {}".format(args.models_names))

    for model_name, model in models.items():
        #print("\n{}:".format(model_name))
        if args.cuda:
            model.try_cuda()
        contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
            sentences)

        
        #print(f'Number of layers: {len(contextual_embeddings)}')
        #for layer_id, layer in enumerate(contextual_embeddings):
            #print(f'Layer {layer_id} has shape: {layer.shape}')
        
        
        #print("sentence_lengths: {}".format(sentence_lengths))
        #print("tokenized_text_list: {}".format(tokenized_text_list))
        
        #get index of entity masked in tokenized list
        for index in range(0,len(tokenized_text_list[0])):
            #print(tokenized_text_list[index])
            if "[MASK]"==tokenized_text_list[0][index]:
                position = index +1
                break
        #first list is first vector about masked sentence
        first = contextual_embeddings[11][0][position]
        #second list is second vector about no-masked sentence
        second = contextual_embeddings[11][1][position]
        #return a concatenation sof these 2 list
        return np.concatenate((first, second))

parser = options.get_eval_generation_parser()
# we set bert option and other necessary options that we give to LAMA tool
parser.add_argument('--cuda', action='store_true', help='Try to run on GPU')
args = parser.parse_args(["--lm", "bert"])
# we store model that we give to LAMA evaluation tool
args.models_names = [x.strip().lower() for x in args.models.split(",")]
models = {}
for lm in args.models_names:
    models[lm] = build_model_by_name(lm, args)


#rappresentation_train is a matrix with all input vectors that we want give to svm classfier with their accuracy
rappresentation_train = []
#accuracy is a list that represents result of every vector: SUPPORT or REJECT
accuracy_train = []   
# for every data point in data_point_train we get rappresentation frm main function and accuracy 
for current in data_point_train:
    claimWithMask, token = getClaimwithMaks(current)
    rappresentation_train.append(main(args, claimWithMask, current['claim'] , models))
    accuracy_train.append(LAMA_results(args,claimWithMask,token, models))

    
#rappresentation_test is a matrix with all vectors that we have calcukated from test set
rappresentation_test = [] 
#for every data point in test set we get all vectors representation
for current in data_point_official:
    claimWithMask, token = getClaimwithMaks(current)
    rappresentation_test.append(main(args, claimWithMask, current['claim'] , models))

    
#We used GridSearch to find the optimal paramters for svm model. We comment the following code beacuse it need too much time to be executed. 

"""
param = {'kernel':('linear','rbf','poly','sigmoid'),
        'C':[1,52,5],
        'degree':[3,8],
        'coef0':[0.001,10,0.5],
        'gamma':('auto','scale')}

SVModel = svm.SVC()
Grids=GridSearchCV(SVModel, param, cv=5)

Grids.fit(rappresentation_train, accuracy_train)

print(Grids.best_params_)

The best parameters are the following:
{'C': 15, 'coef0': 0.001, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
"""
#Classification with SVM

clf = svm.SVC(kernel='rbf',C=15,degree=3,coef0=0.001,gamma='scale')
# fit train set
clf.fit(rappresentation_train, accuracy_train)
#predict with test set 
predicted = clf.predict(rappresentation_test)

#predicted will be a list of 1082 values of SUPPORT or REJECT that determines the accuracy of test set.
print(predicted)

#save predicted in json file. We use a dict where we write id of claim with its prediction
json_dict={}
for i in range(0,len(data_point_official[:20])):
  
    json_dict.update({data_point_official[i]['id']: predicted[i]})

with open('predicted.json', 'w') as fp:
    json.dump(json_dict, fp)

# we caluclate the true value for test set. In this way we are able to calcualte accuracy score 
y_true = []
for current in data_point_official:
    claimWithMask, token = getClaimwithMaks(current)
    y_true.append(LAMA_results(args,claimWithMask,token.lower(), models))
    
#calculate accuracy score
acc_score=accuracy_score(y_true, predicted)

print(acc_score)
#the accuracy is circa 0.80


