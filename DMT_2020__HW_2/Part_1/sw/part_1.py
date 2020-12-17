#HOMEWORK 2 - DMT. PART-1 RECOMMENDATION SYSTEM EVALUATION

#GROUP MEMBERS: Alessandro Taglieri, Guglielmo Lato


# PART 1.1 - Recommendation-System





from pathlib import Path
#Matrix Factorization-based algorithms
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
#SlopeOne-collaborative filtering algorithm
from surprise import SlopeOne
#k-NN inspired algorithms
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import CoClustering

#Algorithm predicting the baseline estimate for given user and item.
from surprise import BaselineOnly
#Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal.
from surprise import NormalPredictor

from surprise import Reader
from surprise import Dataset


from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from surprise.model_selection import RandomizedSearchCV
from surprise.model_selection import GridSearchCV

from tabulate import tabulate
import time
import datetime
import numpy as np
import os
import multiprocessing


########################################
print("Number of CPU cores in my pc: ",multiprocessing.cpu_count())
#number of cpu cores (n_jobs in code) is equal to 12 (fro my pc). We'll use all cpu cores.
########################################
print("\n\n\n\n###############  PART 1.1 ###################\n\n")
alg_list=[SVD,SVDpp,NMF,SlopeOne,KNNBasic,KNNWithMeans,KNNBaseline,CoClustering,BaselineOnly,NormalPredictor]
alg_names_lst=['SVD','SVDpp','NMF','SlopeOne','KNNBasic','KNNWithMeans','KNNBaseline','CoClustering','BaselineOnly','NormalPredictor']

# path of dataset file
path=Path(os.getcwd())
file_path=str(path.parent.parent)+"/part_1/dataset/ratings.csv"


print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)


print("\n\nPerforming splits...")
kf = KFold(n_splits=5, random_state=0)




'''
Print a table of mean RMSE for all the algs
'''
table = []
for idx,klass in enumerate(alg_list):
    print(alg_names_lst[idx],klass())
    start = time.time()
    out = cross_validate(klass(), data, ['rmse'], kf,n_jobs=12,verbose=True)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    new_line = [alg_names_lst[idx], mean_rmse, cv_time]
    table.append(new_line)
    print('Finished.')
header = ['RMSE','Time']
print(tabulate(table, header, tablefmt="pipe"))



# PART 1.2 - Recommendation-System
#           KNNBaseline optimization 






print("\n\n\n\n###############  PART 1.2 - KNNBaseline Optimization ###################\n\n")

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)#skip header
data = Dataset.load_from_file(file_path, reader=reader)


kf = KFold(n_splits=5, random_state=0)

start=datetime.datetime.now()
print('Optimizing hyperparameters of the KNNBaseline')
print(start)

'''
Optimizing hyperparameters of the KNNBaseline
In this case we used RandomSearchCV with 30 iterations;
For KNN algorithm the most important parameters, the one that have the biggest 
impact are value of k, which is recommended to be odd value, and similarity 
function
'''
'''
At first we use these parameters just to see if recommendation to use 
pearson_baseline as a similarity function is valid.
similarity_options={
        'name':['cosine','msd','pearson','pearson_baseline'],
        'user_based': [True,False],
        'shrinkage':[1,10,250,100,500,1000,1240]+list(range(50,150))
}
parameters_distributions = {
   
'k': np.arange(1,60,2),
              'min_k':[1,2,3,4,5,6,7,8,9,10,11],
              'sim_options':similarity_options}
'''
#And these were the results
#0.8865


# =============================================================================
# 'Best found parameters for KNNBaseline in the first Iteration
alg=KNNBaseline(k= 45, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)
#54m18s was the TIME FOR EXECUTION 



'''
Then we tries just the pearson_baseline as a similarity function...
and tried to see the k value which is the best one.
'''


current_algo= KNNBaseline

similarity_options={
        'name':['pearson_baseline'], #it is recommended to use Pearson Baseline
        'user_based': [True,False]
        }
parameters_distributions = {
   
'k': np.arange(1,60,2),
              'min_k':[1,2,3,4,5,6,7,8,9,10,11],
              'sim_options':similarity_options}
searchCV = RandomizedSearchCV(current_algo,
                            parameters_distributions,
                            n_iter=30,
                            measures=['rmse'],
                            n_jobs=12,
                            cv=5)
searchCV.fit(data)
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(searchCV.best_params['rmse'])

#Second iteration 
#0.8864
#{'k': 37, 'min_k': 11, 'sim_options': {'name': 'pearson_baseline', 'user_based': False}}
alg=KNNBaseline(k= 37, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)
#9M45S



# 'Best found parameters for KNNBaseline 
alg=KNNBaseline(k= 37, min_k=11, sim_options= {'name':'pearson_baseline', 'user_based':False})
cross_validate(alg, data, measures=['RMSE'], cv=kf, verbose=True)






# PART 1.2 - Recommendation-System
#           SVD optimization 

'''
Part 1.2

        
In this scrypt we performed 2 Grid Search Cross Validations over 5 folds to try 
to find the best hyper parameters.
Since execution for first alg were slower we decided to choose parameters wiser.
In first GridSearchCV execution we choose init_mean, lr_all, reg_all
'''

'''
Since in the first part 
'''
print("\n\n\n\n###############  PART 1.2 - SVD Optimization ###################\n\n")

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)#skip header
data = Dataset.load_from_file(file_path, reader=reader)



'''
Optimizing hyperparameters of the SVD
'''
kf = KFold(n_splits=5, random_state=0) #making folds for cross validation


start=datetime.datetime.now()
print('Optimizing hyperparameters of the SVD')
print(start)

 
#FIRST OPTIMIZATION OF SVD

param_grid = {'init_mean':[0.1,0.15],
              'lr_all':[0.005,0.01,0.025], #0.025 default
              'reg_all':[0.02,0.005,0.1]} #0.1 default
grid_search = GridSearchCV(SVD,param_grid,measures=['rmse'],
                           cv=5,n_jobs=12)
grid_search.fit(data)

  
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(grid_search.best_params['rmse'])


# After first grid search  --> {'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}

'Best found paramteres for SVD'
# =============================================================================
# 0.8845
#{'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}
opt_svd_alg=SVD(init_mean=0.15,lr_all=0.025,reg_all=0.1)
cross_validate(opt_svd_alg,data,measures=['rmse'],cv=kf,n_jobs=12,verbose=True)
# =============================================================================



 
'''
Additional optimization of just number of factors with other chosen fixed hyperparameters
'''

start=datetime.datetime.now()
print('Optimizing hyperparameters of the SVD')

print(start)
current_algo = SVD

#SECOND and LAST OPTIMIZATION OF SVD

param_grid = {'n_factors': [50,100,125,150,200],
              'init_mean':[0.15],
              'lr_all':[0.025],
              'reg_all':[0.1]}
grid_search = GridSearchCV(SVD,param_grid,measures=['rmse'],
                           cv=5,n_jobs=12)
grid_search.fit(data)


  
end=datetime.datetime.now()
print(end-start,"\nEnd.....")
print(grid_search.best_params['rmse'])


opt_svd_alg=SVD(n_factors=150,lr_all=0.025,reg_all=0.1,init_mean=0.15)
cross_validate(opt_svd_alg,data,measures=['rmse'],cv=kf,n_jobs=12,verbose=True)

#{'n_factors': 150, 'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}
#0.8835
'''
Execution time(I GridSearch+ II GridSearch) ---> 9m40s
'''
##0:04:30 + #0:02:25=5m55s
#{'init_mean': 0.15, 'lr_all': 0.025, 'reg_all': 0.1}#(n_factors=150}





