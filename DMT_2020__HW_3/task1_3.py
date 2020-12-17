

#########################################

# HOMEWORK 3 - DMT
#TAGLIERI ALESSANDROO
#LATO GUGLIELMO

#TASK 1.3


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


    
# read dev set from json file
dev_data = []
for line in open('/Users/digitalfirst/Desktop/dmt_3hw/paper_dev.jsonl', 'r'):
    dev_data.append(json.loads(line))

# read bert vocab from txt file
with open('/Users/digitalfirst/Desktop/dmt_3hw/cased_L-12_H-768_A-12/vocab.txt') as f:
    content = f.readlines()
bert_vocab = [x.strip().lower() for x in content] 
 
            
#filter train and dev set witohout claim with label equal to "NOT ENOUGH INFO"
dev_new = []
for current in dev_data:
    if current['label'] != 'NOT ENOUGH INFO':
        dev_new.append(current)
        
#initialize togger with ner option        
tagger = SequenceTagger.load('ner')

#iterate very claim in dev set and put in data_point_dev all claims that have requisiteise wirtten in the track
data_point_dev = []

for datapoint in dev_new:
    new_dev = {}
    new_dev['id']=datapoint['id']
    new_dev['label']=datapoint['label']
    new_dev['claim']=datapoint['claim']

    sentence = Sentence(datapoint['claim'])
    
    tagger.predict(sentence)
    new_dev['entity']=[]
    
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
            new_dev['entity'].append(new_entities)
            
        else:
            continue
    #if the current claim has only one entity that respects all requisities, we put it in data_point_dev
    if len(new_dev['entity'])==1:
        data_point_dev.append(new_dev)

# fucntion that from a single data_point, it give in output a claim with [MASK] inside the entity. It returns this entity into the var token
def getClaimwithMaks(data_point):
    sentence = data_point['claim']
    token = data_point['entity'][0]['mention']
    start = data_point['entity'][0]['start_char']
    end = data_point['entity'][0]['end_char']
    claim_withMask = sentence[:start] + "[MASK]" + sentence[end:]
    
    return claim_withMask, token




#this function is our evaluation.py . It takes in input args and models (the same in evaluation.py) and data_point that is the current data_point that we want check with LAMA
def LAMA_results(args, data_point, models):
    #result_task3 will contain label SUPPORT or REJECT as the result of predictions and if lama recongizes this entity, we set also its prob
    result_task3={}
    
    #we store in claimWithMask the claim with [MASK]. Token contains the current entity
    claimWithMask, token = getClaimwithMaks(data_point)
    args.text=claimWithMask
    if not args.text and not args.interactive:
        msg = "ERROR: either you start LAMA eval_generation with the "               "interactive option (--i) or you pass in input a piece of text (--t)"
        raise ValueError(msg)

    stopping_condition = True

    print("Language Models: {}".format(args.models_names))
    
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
            print("\n{}:".format(model_name))
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
                # we put in list_prob all prob of the top 10 predictions
                list_prob = df_predictions['prob'].tolist()
                #we check if current entity is equal to the one of the top 10 predictions. If it's equal, we set SUPPORT label and set its prob as result, otherwise REJECT
                if token in list_predictions:
                    
                    result_task3["label"]="SUPPORT"
                    result_task3["prob"]=list_prob[list_predictions.index(token)]
                    
                else:
                    
                    result_task3["label"]="REJECT"
                    result_task3["prob"]=0
                   

                return result_task3
                    

                

parser = options.get_eval_generation_parser()
# we set bert option that we give to LAMA evaluation tool
args = parser.parse_args(["--lm", "bert"])
args.models_names = [x.strip().lower() for x in args.models.split(",")]
#results_task1 will contain a list of SUPPORT or REJECT labels
results_task3 = []
# we store model that we give to LAMA evaluation tool
models = {}
for lm in args.models_names:
    models[lm] = build_model_by_name(lm, args)
    
#  we iterate every data_point in data_point_dev. For every of them, we call LAMA_results function and store its result in result list
for data_point in data_point_dev:
    result = LAMA_results(args, data_point, models)
    results_task3.append(result)
    
#var that will contain count of SUPPORT labels
tot_supp = 0
#we calculate accuracy with list of labels in result_task3
new_result_task3 = []
for current in results_task3:
    if current['label']=='SUPPORT':
        new_result_task3.append(current)

#this list will contain a list of threshold that we plot. It goes from -5 to 0. Min and max values that all predictions could have
array = np.arange(0, -5, -0.1).tolist()

#result_final is a list that will contain list of amount of entity recognized for every threshold
result_final=[]
#for every threshold we count how many entityì LAMA tool has predicted in the right way
for value in array:
    count=0
    for current in new_result_task3:
        if current['prob'] > value:
            count+=1
    
    result_final.append((count/len(results_task3))*100)
plt.title("TASK 1.3: Accuracy based on p threshold")
plt.xlabel("Accuracy")
plt.ylabel("Threshold - prob")
plt.grid()
plt.plot(result_final, array)
plt.savefig('task1.png')

