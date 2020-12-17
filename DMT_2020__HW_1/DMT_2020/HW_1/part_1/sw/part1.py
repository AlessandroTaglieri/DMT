


#HOMEWORK 1 - DMT. PART-1 SEARCH ENGINE EVALUATION

#GROUP MEMBERS: Alessandro Taglieri, Guglielmo Lato



# IMPORT LIBRARIES


from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer,SimpleAnalyzer,StandardAnalyzer,RegexAnalyzer
from whoosh.analysis import FancyAnalyzer,NgramAnalyzer,KeywordAnalyzer,LanguageAnalyzer

from whoosh.writing import AsyncWriter
import os
from whoosh import index
from whoosh.qparser import *
from whoosh import scoring
import csv

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import collections
from matplotlib import colors as mcolors
import random
import pylab 




# SET PATHs FOR DIFFERENT DATASET


from pathlib import Path
path=Path(os.getcwd())
path_start=str(path.parent.parent)


se = 0

#ALLOW TO CHOOSE DATASET BETWEN 'CRANFIELD' AND 'TIME'
while se != 1 and se != 2:
    se = int(input("If you want execute this program on cranfield_DATASET write '1', else write '2' if you want execute it on time_DATASET:   "))


if se == 1:
    path_cranfield=path_start+"/part_1/Cranfield_DATASET"
    path_dir=path_cranfield
    quesries_cran=path_dir+"/cran_Queries.tsv"
    queries_path=quesries_cran
    gt_cranfield=path_dir+"/cran_Ground_Truth.tsv"
    gt_path=gt_cranfield
    
    mrrs_cran="mmrs_cran.csv"
    mrrs_name=mrrs_cran
    r_distr_table_name_cran="R-Precision distribution table_cran.csv"
    r_distr_table_name=r_distr_table_name_cran
    img_cran=path_start+"/part_1/pK@k_cran.png"
    img_path=img_cran
    img_path_2_cran=path_start+"/part_1/nDCG@K_cran.png"
    img_path_2=img_path_2_cran
    
else:
    path_time=path_start+"/part_1/Time_DATASET"
    path_dir=path_time
    queries_time=path_dir+"/time_Queries.tsv"
    queries_path=queries_time
    gt_time=path_dir+"/time_Ground_Truth.tsv"
    gt_path=gt_time
    mrrs_time="mmrs_time.csv"
    mrrs_name=mrrs_time
    r_distr_table_name_time="R-Precision distribution table_time.csv"
    r_distr_table_name=r_distr_table_name_time
    img_time=path_start+'/part_1/pK@k_time.png'
    img_path=img_time
    img_path_2_time=path_start+"/part_1/nDCG@K_time.png"
    img_path_2=img_path_2_time
    

#get number of documents, number of queries and number of queries in Gorund Truth

number_of_files = len([f for f in os.listdir(path_dir+"/DOCUMENTS/")if os.path.isfile(os.path.join(path_dir+"/DOCUMENTS/", f))])
with open(queries_path,"r") as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    row_count_queries = len(data)
with open(gt_path,"r") as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    row_count_gt = len(data)
    
#print requested info
print("Number of documents: ", number_of_files)
print("Number of queries: ", row_count_queries)
print("Number of queries in Ground Truth: ", row_count_gt)



"""
  Method that returns a dataframe with columns ['ID','title','body'] from content of html pages stored in directory
  input: path to the .html files
  output: dataframe ['ID','title','body'] from the content of the .html pages 
"""
def creating_csv_from_html(path):

    document=pd.DataFrame(columns=['ID','title','body']) #initialization of the dataframe 
    doc_ids=range(1,number_of_files) #number of .html files stored in the folder

    for i in doc_ids: #looping through each of the html files
        filename=path+'_'*6+str(i)+'.html' #storing file name
        with open(filename) as f:
            content = f.read() #reading content of the file

        soup = BeautifulSoup(content, 'html.parser') #taking html content with BeautifulSoup library and parsing 
        title=soup.title.string # storing title from parsed html content
        body=soup.body.string # storing body from parsed html content


    
        x=len(body.split(' ')) #check if there is document without body, with only one character
        if x<2:    
            continue #documents with ids :
        document=document.append({'ID':i,'title':title,'body':body},ignore_index=True) #adding new document to the dataframe 
        # each new row is each new content taken from new parsed .html file

    return document


files_text=creating_csv_from_html(path_dir+"/DOCUMENTS/") #calling the method and storing the dataframe into files_text variable
files_text.to_csv(path_dir+"/csv_test.csv") #saving the dataframe into Cranfield_DATASET/Time_DATASET folder


"""
   Method that creates schema and stores index file based on the retrieved 'csv_test.csv' file  
   input:  
       selected_analyzer - selected text analyzer from the whoosh library
       name_of_file - name of .csv file stored from dataframe variable 'files_text'
       scoring_function - selected scoring function from the whoosh library
       path - path where index files are stored
   """
def creating_searching_ranking(selected_analyzer, name_of_file,scoring_function,path):

    #creating Schema with fields id, title and content
    schema = Schema(id=ID(stored=True), title=TEXT(stored=False, analyzer=selected_analyzer),
                content=TEXT(stored=False, analyzer=selected_analyzer))
    directory_containing_the_index = path 
    ix = create_in(directory_containing_the_index, schema) #writing index based on schema in the directory where the 'path' is
    directory_containing_the_index = path
    ix = index.open_dir(directory_containing_the_index) #opening the index file 
    writer =  AsyncWriter(ix) #writer will be used to add content to the fields

    ALL_DOCUMENTS_file_name = name_of_file #path to the file 
    in_file = open(ALL_DOCUMENTS_file_name, "r", encoding='latin1')
    csv_reader = csv.reader(in_file, delimiter=',')  #reading the file
    csv_reader.__next__()# to skip the header: first line contains the name of each field.

    for record in csv_reader: #for each row in the 'csv_test' file 
        id = record[1] #read id
        title = record[2] #read title
        content = record[3] #read body
        writer.add_document(id=id, content=title+' '+content)


    writer.commit()
    in_file.close() #finish writing in the index file
    





'''
    Method that given the input query and given the specific SE configuration returns the results of the search
    input:
        selected_analyzer - selected text analyzer from the whoosh library
        scoring_function - selected scoring function from the whoosh library
        input_query - query that's being used for evaluation
        path - path where index files are stored
        max_number_of_results - maximal number of results that should be retrieved which are equal to the number of relevant documents related to that specific query (which we need for calculating R-precision)
    output: answer - dataframe with results of the given SE given the query; columns of dataframe: ["Rank" , "Doc_ID" , "Score"]
''' 
def exec_searching_ranking(selected_analyzer,scoring_function,input_query,path,max_number_of_results):
  
    
    directory_containing_the_index = path 
    ix = index.open_dir(directory_containing_the_index) #index file for the given SE
    qp = QueryParser("content", ix.schema)
    parsed_query = qp.parse(input_query)# parsing the INPUT query
    #print("Input Query : " + input_query)
    #print("Parsed Query: " + str(parsed_query))

    searcher = ix.searcher(weighting=scoring_function) #defining scoring_function for search engine

    results = searcher.search(parsed_query,limit=max_number_of_results) #saving results of the query and limiting max number of results

    #print("Rank" + "\t" + "DocID" + "\t" + "Score")
    answer=pd.DataFrame(columns=["Rank" , "Doc_ID" , "Score"])  # dataframe with results for the given SE given the query
    #row_answer=pd.DataFrame()
    for hit in results:
        new_row = {'Rank':str(hit.rank), 'Doc_ID':int(hit['id']), 'Score':str(hit.score)}
      #  print(str(hit.rank) + "\t" + hit['id'] + "\t" + str(hit.score))
        #row_answer=pd.DataFrame([str(hit.rank) , int(hit['id']), str(hit.score)]).T
        answer=answer.append(new_row, ignore_index=True)
   
    searcher.close()
    return answer





'''
    Method that returns number of relevant documents related to the specific input query
    input:  dq - dictionary, where key=Query_id, value=number of relevant documents related to that Query_id
            q_n - Query_id
    output: number of relevant documents
'''

def count_of_vals(dq,q_n):

    return dq[q_n]
    


'''
    Method that given the specific SE configuration(selected_analyzer,scoring_function)
    executes and returns the results for ALL the queries 
    input:
        selected_analyzer - selected text analyzer from the whoosh library
        scoring_function - selected scoring function from the whoosh library
    output: answer_q - dataframe with the results of the given SE for ALL the queries; columns of df: ["Rank" , "Doc_ID" , "Score"]
''' 
def exec_queries(selected_analyzer,scoring_function):


    answer_q=pd.DataFrame() #  dataframe with the results of the given SE for ALL the queries; 
    aa=pd.DataFrame() #tmp dataframe 
    
    #all the queries file
    Queries_file=queries_path
    Queries=pd.read_csv(Queries_file,sep='\t')
    gt=pd.read_csv(gt_path, sep='\t') #ground truth
    Q=list(gt['Query_id'].unique()) #list of unique Query ids
    
    dq=collections.defaultdict(int) #dictionary, where key=Query_id, value=number of relevant documents related to that Query_id
    for i in Q: # for each query)_id
        dq[i]=len(list(gt[gt['Query_id']==i]['Relevant_Doc_id']))

    
    name_of_file_1=path_dir+"/csv_test.csv"

    # calling the method that creates schema and stores index file based on the retrieved 'csv_test.csv' file  
    creating_searching_ranking(selected_analyzer,name_of_file_1,scoring_function,path_start+"/part_1/")
    for i in Q:

        max_number_of_results_1q=count_of_vals(dq,i)
        if max_number_of_results_1q==0:
            max_number_of_results_1q=1
            
        # calling the method that given the input query and given the specific SE configuration returns the results of the search
        aa=exec_searching_ranking(selected_analyzer,scoring_function,list(Queries[Queries['Query_ID']==i]['Query'])[0],path_start+"/part_1/",max_number_of_results_1q)
        aa['Query_id']=i
        answer_q=answer_q.append(aa)#[['Query_id',1]] APPEND dataframe for each query

    return answer_q
          




def mrr(gt,sr1):
    '''
    Method that calculates MRR: Mean Reciprocal Rank
    input: gt - ground truth
           sr1 - search engine results, dataframe with columns (Rank, Doc_ID, Score, Query_id)
    output: mrr value for that specific input 'sr1' (search engine configuration)
    '''
    
    mrr=0
    Q=set(gt['Query_id'].unique()) #number of unique queries in the ground truth
    dd=collections.defaultdict(int) #default dictionary, where (key=Query_id,value=list of document ids) from se result

    for i in Q:
        dd[i]=list(sr1[sr1['Query_id']==i]['Doc_ID'])

    dq=collections.defaultdict(int) #default dictionary, where (key=Query_id,value=list of relevant document ids) from ground truth
    for i in Q:
        dq[i]=list(gt[gt['Query_id']==i]['Relevant_Doc_id'])
   
    tq=list() #->temporary list- stores list of relevant doc_ids for every query id in the loop

    for q in Q: #for every query_id
        tq=dq[q]

        for i in range(len(dd[q])): #for every document_id in query_id q

            #if document id is in the list of the relevant doc ids 
            if dd[q][i] in tq: # dd[q]-list of document ids with query_id q -> [i] is index of list

                mrr=mrr+(1/(i+1))	#mrr value is sum on Reciprocal Ranks (+1 cause ranking ofc starts with 1)
                break #if it is break cause it found the first doc id from the relevant doc ids in the ground truth


    mrr=mrr/(len(Q)) #MEAN of the sum of reciprocal ranks
    return mrr

'''
    Method that calculates MRR: Mean Reciprocal Rank and saves a table with MRR evaluation for every search engine configuration 
'''
def exec_comp():
   
    #text analyzers
    selected_analyzers = [StemmingAnalyzer(),SimpleAnalyzer(),StandardAnalyzer(),RegexAnalyzer(),FancyAnalyzer(),NgramAnalyzer(5),KeywordAnalyzer(),LanguageAnalyzer('en')]#text analyzers
    sel_ana=['StemmingAnalyzer()','SimpleAnalyzer()','StandardAnalyzer()','RegexAnalyzer()','FancyAnalyzer()','NgramAnalyzer(5)','KeywordAnalyzer()','LanguageAnalyzer()']#text which will be used for graph and for mrr table

    i=0 #counter
    mrrs=[] #list where MRR values for each SE configuration will be stored

    #scoring functions
    scoring_functions = [scoring.TF_IDF(),scoring.Frequency(),scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)]
    scor_func=[' TF_IDF',' Frequency',' BM25F']

    #ground truth
    gt1=pd.read_csv(gt_path, sep='\t')
    
    #combinations for every chosen analyzer with every chosen scoring function
    for x in range(len(selected_analyzers)):
        for y in range(len(scoring_functions)):
            
            i=i+1
            sr_1=exec_queries(selected_analyzers[x],scoring_functions[y]) # execute queries for the chosen configuration combination
            sr_1.to_csv(path_start+"/part_1/"+str(i)+"__.csv",index=False) #save results of the search engine
            mrrs.append((sel_ana[x]+scor_func[y],mrr(gt1,sr_1))) #calculate MRR
    mrrs_saving=pd.DataFrame(mrrs)
    print(mrrs_saving)
    mrrs_saving.to_csv(path_start+"/part_1/"+mrrs_name, index=False) #store MRR table



#MAIN call

exec_comp() #execution of the file with all the methods -> main call



"""
    method that returns only the list of indices of the acceptable SE configurations
    input: path to the saved mrrs.csv file(MRR table where first column is SE configuration(text analyzer+scoring function) and second column is mrr value)
    output: list of indices in the MRR table file of acceptable SE configurations
"""
def read_mrr(path):
  
    mrrs=pd.read_csv(path+mrrs_name) # reading "mrrs_cran/time.csv" file

    idx=mrrs.sort_values('1').tail(5).index + 1 #we added 1 because SEs start from 2nd line in csv file 
    return list(idx)





"""
    method that returns the list of names of the acceptable SE configurations
    input: path to the saved mrrs.csv file(MRR table where first column is SE configuration(text analyzer+scoring function) and second column is mrr value)
    output: list of names of the acceptable SE configurations in MRR table
"""
def read_mrr_names(path):
   
    mrrs=pd.read_csv(path+mrrs_name) # reading MRR table
    
    a = mrrs.sort_values('1').tail(5) #take last 5 of list. They are 5 best SE because we've sorted list in ASC order
    
    return list(a['0']) 




path=path_start+"/part_1/"
list_of_ser=read_mrr(path) #list of indices of the acceptable SE configurations in MRR table
list_of_names=read_mrr_names(path) #list of names of the acceptable SE configurations in MRR table



"""
    method that returns dictionary with list of doc_ids in respect to the query id from the results of the SE
    input: path - path to the part_1 folder of the project
           ser - result of the search engine
    output: dd - dictionary, key=Query_id, value=list of doc_ids in respect to the query id
    """
def create_dict_of_ser(path,ser):
    
    ser_d=pd.read_csv(path+str(ser)+"__.csv")
    Q=list(ser_d['Query_id'].unique())
    dd={1:list(ser_d[ser_d['Query_id']==1]['Doc_ID'])}
    for i in Q:
        dd[i]=list(ser_d[ser_d['Query_id']==i]['Doc_ID'])
    return dd



gt=pd.read_csv(gt_path, sep='\t')

Q=list(gt['Query_id'].unique())
dq={1:list(gt[gt['Query_id']==1]['Relevant_Doc_id'])}
for i in Q:
    dq[i]=list(gt[gt['Query_id']==i]['Relevant_Doc_id']) #dq, dictionary, where key=query_id, value=list of Relevant_Doc_ids

ser1=create_dict_of_ser(path,list_of_ser[0])




'''
    Method that calculates R-Precision
'''
def r_precision(ser,dq0):
    
    r_p=[]
    val_sum=0
    for k in ser.keys(): #ser.keys() are Query_ids
        tq=ser[k] #list of Relevant_Doc_ids for that Query_id
        for x in tq: #for each doc id from Relevant_Doc_ids for that Query_id
            if x in dq0[k]: #if it is in the Relevant_Doc_ids
                val_sum+=1 # count how many of them is RELEVANT docs
        r_p.append(val_sum/len(dq0[k]))  #divide by the length of the RELEVANT docs from the Ground truth
        val_sum=0
    return pd.DataFrame(r_p, columns=['r_p'])





final=pd.DataFrame()
all_dicts=[]
for i in list_of_ser:
    ser1=create_dict_of_ser(path,i)
    r_p=r_precision(ser1,dq)
    nums=[i,r_p.r_p.mean(),r_p.r_p.min(),r_p.r_p.quantile(0.25),r_p.r_p.median(),r_p.r_p.quantile(0.75),r_p.r_p.max()]

    nums=pd.DataFrame(nums).T
    final=final.append(nums)
    all_dicts.append(ser1)
    
#Making the R-Precision distribution table    
final.columns=['ser_id','mean','min','1quartile','median','3quartile','max']
final=final.reset_index()
final['SE conf']=list_of_names
final=final[['ser_id','SE conf','mean','min','1quartile','median','3quartile','max']]
print(final)
final=final.round(3)
final.to_csv(path_dir+'/'+r_distr_table_name)




# p@k

def myPk(query_r,query_gt,k):
    count=0
    myPk=0
   
    for x in query_r[0:k]:
       
        if x in query_gt:
            count+=1

    myPk = count/min(k,len(query_gt))
    return myPk



# Calculation of p@K:


def myPK_for_ser(ser_list,all_dict,dq,k):
    s_d=pd.DataFrame()
    for i in range(len(ser_list)):
        dcg=[]
        w_dict=all_dict[i]
        for y in w_dict.keys():
            dcg.append(myPk(w_dict[y],dq[y],k))
        s_d_temp=pd.DataFrame(dcg).T
        s_d=s_d.append(s_d_temp)
    return s_d

#calculate avg values

def avg_s_dcg(DCG):
    avg_of_k=[]
    for i in range(DCG.shape[0]):

        avg_of_k.append(np.mean(list(list(DCG[i:(i+1)].values)[0])))
    return avg_of_k



#creating DataFrame with p@K values
pK_df=pd.DataFrame()
klist=[1,3,5,10]
for k in klist:
    pK=myPK_for_ser(list_of_ser,all_dicts,dq,k)

    avgs=avg_s_dcg(pK)
    pK_df[k]=avgs
pK_df=pK_df.T



# p@k plot for top 5 configurations


def pK_plot():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) #list of colors
    cols=random.sample(list(colors.keys()),len(list_of_names)) #take random color sample from colors list->length of sample is number of accepted configurations

    pylab.figure(figsize=(10,10))    
    for i in range(0,len(list_of_names)):
        y = pK_df[i]
        pylab.plot(pK_df.index, y,label=list_of_names[i],color=cols[i])
        pylab.legend(loc='upper left',prop={'size': 10})
        pylab.xlabel("K values")
        pylab.ylabel("average p@K")
        pylab.title("p@K plot")
        pylab.savefig(img_path)

    pylab.show()
    


# nDCG@K

def myNDCG(query_r,query_gt,k):
    s=0
    dsg=[]
    idsg=[]
    for x in range(len(query_r[0:k])):
        if query_r[x] in query_gt:
            s=1
        else:
            s=0


        dsg.append(s/np.log2(x+2)) # we added +1 to formula from lab because counting starts from 0 in python
        idsg.append(1/np.log2(x+2))
        
    dcg=sum(dsg)
    idcg=sum(idsg)
    return dcg/idcg



# Calculation of nDCG: normalized Discounted Cumulative Gain


def myDCG_for_ser(ser_list,all_dict,dq,k):
    s_d=pd.DataFrame()
    for i in range(len(ser_list)):
        dcg=[]
        w_dict=all_dict[i]
        for y in w_dict.keys():
            dcg.append(myNDCG(w_dict[y],dq[y],k))
        s_d_temp=pd.DataFrame(dcg).T
        s_d=s_d.append(s_d_temp)
    return s_d



#creating DataFrame with nDCG values
k_nDCG=pd.DataFrame()
klist=[1,3,5,10]
for k in klist:
    DCG=myDCG_for_ser(list_of_ser,all_dicts,dq,k)

    avgs=avg_s_dcg(DCG)
    k_nDCG[k]=avgs
k_nDCG=k_nDCG.T




# nDCG@k plot for top 5 configurations


def nDCG_plot():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) #list of colors
    cols=random.sample(list(colors.keys()),len(list_of_names)) #take random color sample from colors list->length of sample is number of accepted configurations

    pylab.figure(figsize=(10,10))    
    for i in range(0,len(list_of_names)):
        y = k_nDCG[i]#nDCG@k values for the accepted configurations
        pylab.plot(k_nDCG.index, y,label=list_of_names[i],color=cols[i])
        pylab.legend(loc='upper left',prop={'size': 10})
        pylab.xlabel("K values")
        pylab.ylabel("average nDCG")
        pylab.title("nDCG@K plot")
    pylab.savefig(img_path_2)


    pylab.show()



pK_plot()

nDCG_plot()





