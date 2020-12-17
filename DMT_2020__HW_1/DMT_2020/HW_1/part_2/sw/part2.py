


#HOMEWORK 1 - DMT. PART-2 NEAR-DUPLICATES-DETECTION

#GROUP MEMBERS: Alessandro Taglieri, Guglielmo Lato




# IMPORT LIBRARIES


import pandas as pd
import string
import csv
import random
import math
import os
from pathlib import Path
import time


#set path

path=Path(os.getcwd())
path_start=str(path.parent.parent)+"/part_2"
data = pd.read_csv(path_start + '/dataset/250K_lyrics_from_MetroLyrics.csv', sep=",",index_col='ID',header=0)

# function that removes punctuation and converts to lowercase
def normalize(words): 
    new_words=words.translate(str.maketrans('', '', string.punctuation)).lower()

    return new_words


data.lyrics=data.lyrics.apply(lambda x:normalize(x))



def create_unique_shingles(empty_dict): 
    for i in range(0,len(data)):
        unique_shingles=set()# set because we dont want to take into account the same shingle in the same song
        try: # song number 158 doesnt exist
            tokens=data['lyrics'][i].split()
            if len(tokens) >=3: # there are some songs with fewer words
                for index in range(len(tokens) - 3 + 1):
                    # Construct the shingle text by combining k words together.
                    shingle = tokens[index:index + 3]
                    # Hash the shingle to a 32-bit integer.  
                    shingle = ' '.join(shingle)
                    if shingle not in unique_shingles:
                        unique_shingles.add(shingle)
                    else:
                        del shingle
                        index = index - 1
            
        except:
            continue
        empty_dict[i]=unique_shingles
    return(empty_dict)


results={}
create_unique_shingles(results)


d={}
id = 0
for k, v in results.items():# this function assings an id number for each unique shingle of a song. it is kinda the opposite of what we did in the previosu function, because now the key of this dictionary is the shingle, not the document id, and the value is an id for each shingle.
    for elem in v:
        if str(elem) not in d.keys():
            d[str(elem)]=id
            id += 1


numsh={}

for k, v in results.items(): # this dictionary is practically the tsv that we will create later. it has as key the id of each song and as as values the set of unique shingles of each song.
    lista=set()
    for elem in v:
        number=d.get(str(elem))
        lista.add(number)
    numsh[k]=list(lista)


# create the tsv file with the shingles
file=open(path_start + "/data/dict.tsv", "w",newline='') 
w = csv.writer(file, delimiter='\t')
w.writerow(['set_id','set_as_list_of_elements_id'])

for key, val in numsh.items():
    if len(val)>=1:
        #w.writerow([key, val])
        w.writerow(['id_'+str(key), val])
file.close()




# START TO CREATE HASH FUNCTIONS


num_hash_functions = 150 #num_hash_functions is our n. n must be b*r.
upper_bound_on_number_of_distinct_terms  = len(d.keys())
#upper_bound_on_number_of_distinct_terms =   138492
#upper_bound_on_number_of_distinct_terms =  3746518


### primality checker
def is_prime(number):
    for j in range(2, int(math.sqrt(number)+1)):
        if (number % j) == 0: 
            return False
    return True


file=open(path_start+"/hash_functions/hashtable.tsv", "w",newline='')
w = csv.writer(file, delimiter='\t')
w.writerow(['a', 'b', 'p', 'n'])
set_of_all_hash_functions = set()
while len(set_of_all_hash_functions) < num_hash_functions:
    a = random.randint(1, upper_bound_on_number_of_distinct_terms-1)
    b = random.randint(0, upper_bound_on_number_of_distinct_terms-1)
    p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    while is_prime(p) == False:
        p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    current_hash_function_id = tuple([a, b, p])
    set_of_all_hash_functions.add(current_hash_function_id)
    w.writerow([a, b, p, upper_bound_on_number_of_distinct_terms])  
file.close()

# FINISH CREATION OF HASH FUNCTIONS

#execution of java tool 'near.duplates-detector'. we execute it from cmd and we set:
# j: jaccard value = 0.89
# r = 5
# b = 30
# ./data/hashtable.tsv : csv that contains hash functions
# ./data/dict.tsv : dict of files that we've just created
# ./data/nearduplicate.tsv : it is the file that will contain our result of he execution of this cmd command


#we calculated the time of execution of java tool
start_time = time.time()
os.chdir(path_start)
os.system("java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.89 5 30 ./hash_functions/hashtable.tsv ./data/dict.tsv ./data/nearduplicate.tsv")
end_time = time.time() - start_time
print("execution time of 'java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.89 5 30' is:  ", end_time)


nearDuplicates = pd.read_csv(path_start + '/data/nearduplicate.tsv', sep=",",header=0)


nearDuplicates=pd.read_csv(path_start + "/data/nearduplicate.tsv",sep='\t', usecols=['estimated_jaccard','name_set_1','name_set_2'])


print("the number of candidate near_dupl:  "+str(nearDuplicates.shape[0]))



jaccard_similiarity=[0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]
for j in jaccard_similiarity:
    
    print ("the number of near dupl with j>= " +str(j) + " is: "+  str(len(nearDuplicates[(nearDuplicates['estimated_jaccard']>=j)])))


#small fucntion that calculates false negative and false positive
def calcFalsePositive(j, r, b):
    return 1-(1-(j**r))**b
def calcFalseNegative(j, r, b):
    return (1-(j**r))**b


jaccard_values_FP = [0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
r=5
b=30
for j in jaccard_values_FP:
    print("Porbability of false positives with jaccard value = " +str(j) + " is: "  +str(calcFalsePositive(j, r, b)))



jaccard_values_FN = [0.89,0.9,0.95,1]
r=5
b=30
for j in jaccard_values_FN:
    print("Porbability of false negatives with jaccard value = " +str(j) + " is: "  +str(calcFalseNegative(j, r, b)))






