#!/usr/bin/env python

### Module imports ###
import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm,preprocessing 
import CalctTFIdfs2

def CalculatePairWiseTrainData(features,scores):
  score_dictionary = {}
  feature_index = {}
  
  X = []
  y = []
  
  #normalization
  fTempList = map(lambda x: x[2],features)
  normalizedFeatureVectors = preprocessing.scale(fTempList)
  
  for i in range(0,len(features)):
    features[i] = (features[i][0],features[i][1], normalizedFeatureVectors[i])  
  
  #here i convert features list to dictionary    
  for i in range(0,len(features)):
    query = str(features[i][0])
    url = features[i][1]
    featureVec = features[i][2]
    if query not in feature_index:
      feature_index[query] = [(url, featureVec)] 
    else:
      feature_index[query].append((url,featureVec))
  
  
  #calculate feature vector differences and their scores.

  #here i convert scores list to dictionary
  for i in range(0,len(scores)):
    query = str(scores[i][0])
    url = str(scores[i][1])
    score = scores[i][2]
    if query not in score_dictionary:
      score_dictionary[query] = {url: score }
    else:
      score_dictionary[query][url] = score

  for featureQuery in feature_index.keys():
    urls = feature_index[featureQuery]
    for i in range(0,len(urls)):
      for b in range(i + 1 , len(urls)):
        #X.append(urls[i][0] + " - " + urls[b][0])
        url1 = urls[i][0]
        url2 = urls[b][0]
        
        firstUrlScore = score_dictionary[featureQuery][url1]
        secondUrlScore = score_dictionary[featureQuery][url2]
        
        if firstUrlScore == secondUrlScore:
          continue
        elif firstUrlScore > secondUrlScore:
          diff = np.array(urls[i][1]) - np.array(urls[b][1])
          X.append(list(diff))
          y.append(1)
        else:
          diff = np.array(urls[i][1]) - np.array(urls[b][1])
          X.append(list(diff))
          y.append(-1)
          
  return (X,y)
  
def SortTempDocumentVector(documentVecs):
  tempList = []
  for i in sorted(documentVecs.keys()):
    item  = ((i,documentVecs[i]))
    tempList.append(item)
  
  tempList  = sorted(tempList, key= lambda tup: tup[1],reverse=True)
  fTempList = map(lambda x: x[0],tempList)
  return fTempList

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
  # stub, you need to implement
  #scoresWithDescription is in form of tuples like this (query,url, score)
  scoresWithDescription =  CalctTFIdfs2.GetScores(train_rel_file)

  #featuresWithDescription is in form of tuples like this (query,url,featuresVector)
  featuresWithDescription =  CalctTFIdfs2.GetFeatures(train_data_file)

  y = map(lambda x: x[2],scoresWithDescription)
  X = map(lambda x: x[2],featuresWithDescription)
  
  return (X, y)

def pointwise_test_features(test_data_file):
  index_map = {}
  featuresWithDescription =  CalctTFIdfs2.GetFeatures(test_data_file)
  
  for i in range(0,len(featuresWithDescription)):
    query = featuresWithDescription[i][0]
    url = featuresWithDescription[i][1]
    
    if query not in index_map:
      index_map[query] = {url: i }
    else:
      index_map[query][url]= i
  
  X = map(lambda x: x[2],featuresWithDescription)
  queriesList = map(lambda x: x[0],featuresWithDescription)
  
  queries = set(queriesList)
  queries = list(queries)
  queries = sorted(queries)
  
  return (X, queries, index_map)
 
def pointwise_learning(X, y):
  # stub, you need to implement
  model = linear_model.LinearRegression()
  model.fit(X,y)
  return model

def pointwise_testing(X, model):
  y = model.predict(X)
  return y

##############################
##### Pair-wise approach #####
##############################
def pairwise_train_features(train_data_file, train_rel_file):
 
  #scoresWithDescription is in form of tuples like this (query,url, score)
  scoresWithDescription =  CalctTFIdfs2.GetScores(train_rel_file)

  #featuresWithDescription is in form of tuples like this (query,url,featuresVector)
  featuresWithDescription =  CalctTFIdfs2.GetFeatures(train_data_file)

  (X,y) = CalculatePairWiseTrainData(featuresWithDescription,scoresWithDescription)
  
  return (X, y)

def pairwise_test_features(test_data_file):
  
  queryUrl_feature_dic = {}
  featuresWithDescription =  CalctTFIdfs2.GetFeatures(test_data_file)
  
  for i in range(0,len(featuresWithDescription)):
    query = featuresWithDescription[i][0]
    url = featuresWithDescription[i][1]
    
    if query not in queryUrl_feature_dic:
      queryUrl_feature_dic[query] = {url: featuresWithDescription[i][2] }
    else:
      queryUrl_feature_dic[query][url]= featuresWithDescription[i][2]
  
  X = map(lambda x: x[2],featuresWithDescription)
  queriesList = map(lambda x: x[0],featuresWithDescription)
  
  queries = set(queriesList)
  queries = list(queries)
  queries = sorted(queries)
  
  return (X, queries, queryUrl_feature_dic)

def pairwise_learning(X, y):
  # stub, you need to implement
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X,y)
  #burda ft gibi bisey yapmam lazim.
  return model

def pairwise_testing(X, model,queryUrl_feature_dic):
  
  weights  = list(model.coef_[0])
  for query in queryUrl_feature_dic.keys():
    for url in queryUrl_feature_dic[query].keys():
      featureVector = queryUrl_feature_dic[query][url]
      score = np.dot(featureVector, weights)
      queryUrl_feature_dic[query][url] = score
  
  return queryUrl_feature_dic

####################
##### Task3 Learning #####
####################

def pairwise_train_features2(train_data_file, train_rel_file):
 
  #scoresWithDescription is in form of tuples like this (query,url, score)
  scoresWithDescription =  CalctTFIdfs2.GetScores(train_rel_file)

  #featuresWithDescription is in form of tuples like this (query,url,featuresVector)
  featuresWithDescription =  CalctTFIdfs2.GetFeaturesWithBinaryPdf(train_data_file)

  (X,y) = CalculatePairWiseTrainData(featuresWithDescription,scoresWithDescription)
  
  return (X, y)


def pairwise_learning2(X, y):
  # stub, you need to implement
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X,y)
  #burda ft gibi bisey yapmam lazim.
  return model

####################
##### Task3 Learning Ends #####
####################


####################
##### Task3 Testing #####
####################
def pairwise_test_features2(test_data_file):
  
  queryUrl_feature_dic = {}
  featuresWithDescription =  CalctTFIdfs2.GetFeaturesWithBinaryPdf(test_data_file)
  
  for i in range(0,len(featuresWithDescription)):
    query = featuresWithDescription[i][0]
    url = featuresWithDescription[i][1]
    
    if query not in queryUrl_feature_dic:
      queryUrl_feature_dic[query] = {url: featuresWithDescription[i][2] }
    else:
      queryUrl_feature_dic[query][url]= featuresWithDescription[i][2]
  
  X = map(lambda x: x[2],featuresWithDescription)
  queriesList = map(lambda x: x[0],featuresWithDescription)
  
  queries = set(queriesList)
  queries = list(queries)
  queries = sorted(queries)
  
  return (X, queries, queryUrl_feature_dic)

def pairwise_testing2(X, model,queryUrl_feature_dic):
  
  weights  = list(model.coef_[0])
  for query in queryUrl_feature_dic.keys():
    for url in queryUrl_feature_dic[query].keys():
      featureVector = queryUrl_feature_dic[query][url]
      score = np.dot(featureVector, weights)
      queryUrl_feature_dic[query][url] = score
  
  return queryUrl_feature_dic

####################
##### Task3 Testing Ends #####
####################

####################
##### Training #####
####################
def train(train_data_file, train_rel_file, task):
  sys.stderr.write('\n## Training with feature_file = %s, rel_file = %s ... \n' % (train_data_file, train_rel_file))
  
  if task == 1:
    print >> sys.stderr, "Training Task 1\n"
    # Step (1): construct your feature and label arrays here
    (X, y) = pointwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pointwise_learning(X, y)
  elif task == 2:
    print >> sys.stderr, "Training Task 2\n"
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pairwise_learning(X, y)
  elif task == 3:
    print >> sys.stderr, "Training Task 3\n"
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features2(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pairwise_learning2(X, y)

  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra Credit\n"

  else: 
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    model = linear_model.LinearRegression()
    model.fit(X, y)
  
  # some debug output
  weights = model.coef_
  print >> sys.stderr, "Weights:", str(weights)

  return model 

###################
##### Testing #####
###################
def test(test_data_file, model, task):
  sys.stderr.write('\n## Testing with feature_file = %s ... \n' % (test_data_file))

  if task == 1:
    print >> sys.stderr, "Testing Task 1\n"
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pointwise_testing(X, model)
    
    # some debug output
    #for query in queries:
    # for url in index_map[query]:
    #   print >> sys.stderr, "Query:", query, ", url:", url, ", value:", y[index_map[query][url]]
    
    #add scores to query , url s instead of index.
    for query in queries:
      for url in index_map[query]:
        score =  y[index_map[query][url]]
        index_map[query][url] = score
    
    for query in index_map.keys():
      documentListwithScores =  index_map[query]
      sortedDocuments = SortTempDocumentVector(documentListwithScores)
      index_map[query] = sortedDocuments
    
  elif task == 2:
    print >> sys.stderr, "Testing Task 2\n"
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    index_map = pairwise_testing(X, model,index_map)
    
    for query in index_map.keys():
      documentListwithScores =  index_map[query]
      sortedDocuments = SortTempDocumentVector(documentListwithScores)
      index_map[query] = sortedDocuments
    
  elif task == 3: 
    print >> sys.stderr, "Testing Task 3\n"
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features2(test_data_file)
    
    # Step (2): implement your prediction code here
    index_map = pairwise_testing2(X, model,index_map)
    
    for query in index_map.keys():
      documentListwithScores =  index_map[query]
      sortedDocuments = SortTempDocumentVector(documentListwithScores)
      index_map[query] = sortedDocuments

  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra credit\n"

  else:
    queries = ['query1', 'query2']
    index_map = {'query1' : {'url1':0}, 'query2': {'url2':1}}
    X = [[0.5, 0.5], [1.5, 1.5]]  
    y = model.predict(X)

  for query in index_map:
    print("query: " + query)
    for res in index_map[query]:
      print("  url: " + res)
  
  # Step (3): output your ranking result to stdout in the format that will be scored by the ndcg.py code

if __name__ == '__main__':
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  
  if len(sys.argv) != 5:
    print >> sys.stderr, "Usage:", sys.argv[0], "train_data_file train_rel_file test_data_file task"
    sys.exit(1)
  
  train_data_file = sys.argv[1]
  #train_data_file = "queryDocTrainData.train" 
  train_rel_file = sys.argv[2]
  #train_rel_file = "queryDocTrainRel.train"
  test_data_file = sys.argv[3]
  #test_data_file = "queryDocTrainData.dev"
  task = int(sys.argv[4])
  #task = 3
  print >> sys.stderr, "### Running task", task, "..."
 
  model = train(train_data_file, train_rel_file, task)
  
  test(test_data_file, model, task)
