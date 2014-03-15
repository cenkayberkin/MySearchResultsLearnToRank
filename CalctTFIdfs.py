import sys , math
import re , pickle
from math import log
from decimal import Decimal
import numpy as np

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair

#print >> sys.stderr, 'Task 1, Cosine Similarity'
termDic_f = open('TermFreq.dict','r')
print >> sys.stderr, 'loading Term Dictionary'
termDic = pickle.load(termDic_f)

def getFreq(text, key):
  count = 0
  tokens = text.lower().split()
  for i in tokens:
    if i == key:
      count += 1
  return count

def parseUrl(url):
  tokens = re.findall('[a-z0-9]+', url.lower())
  token = ""
  for i in tokens:
    token += i + " "
  return token

def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

def PrepareQueryVector(query):
    N = 98998
    qv = []
    
    for term in query:
      if term in termDic:
        #s = termDic[term]
        qv.append(log(Decimal(N + 1) / Decimal(int(termDic[term]) + 1)))
      else:
        qv.append(log((N + 1)/ 1))
    
    return qv

def PrepareQueryUrlVector(query,url,qv,dv):
  result = []
  
  tfdu = np.array(dv["url"])
  tfdt = np.array(dv["title"])
  tfdb = np.array(dv["body"])
  tfdh = np.array(dv["header"])
  tfda = np.array(dv["anchor"])
  
  tfdu = np.array(qv) * tfdu
  tfdt = np.array(qv) * tfdt
  tfdb = np.array(qv) * tfdb
  tfdh = np.array(qv) * tfdh
  tfda = np.array(qv) * tfda
  
  result.append(sum(tfdu))
  result.append(sum(tfdt))
  result.append(sum(tfdb))
  result.append(sum(tfdh))
  result.append(sum(tfda))
  
  return (query,url,result)

def UniteTempDocumentVectors(documentVecs):
  tempList = []
  for i in sorted(documentVecs.keys()):
    tempList.append(documentVecs[i]["quVector"])
 
  return tempList

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    totalFeaturesVectors = []
    
    for query in queries.keys():
      tempdocumentVectors = {}
      results = queries[query]
      
      queryTerms = sorted(set(query.split()))
      qv = PrepareQueryVector(queryTerms)
      
      for r in results:
        tempdocumentVectors[r] = {}
        #bodylength = features[query][r].setdefault('body_length', 0)
        
        """For document body """
        tempdocumentVectors[r]["body"] = []
        for key in queryTerms:
          if key in features[query][r].setdefault('body_hits',{}):
            keyFreq = len(features[query][r].setdefault('body_hits',{})[key])
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["body"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["body"].append(0)
        #tempdocumentVectors[r]["body"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["body"])
        """body done """
        
        """For document title """
        tempdocumentVectors[r]["title"] = []
        for key in queryTerms:
          title = features[query][r].setdefault('title',{})
          keyFreq = getFreq(title,key)
          if keyFreq > 0:  
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["title"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["title"].append(0)
        #tempdocumentVectors[r]["title"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["title"])
        """title done """
        
        """For document header """
        tempdocumentVectors[r]["header"] = []
        for key in queryTerms:
          headers = features[query][r].setdefault('header',{})
          
          header= ""
          for i in headers:
            header += i + " "
            
          keyFreq = getFreq(header,key)
          if keyFreq > 0:  
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["header"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["header"].append(0)
        #tempdocumentVectors[r]["header"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["header"])
        """header done """
        
        """For document anchor """
        tempdocumentVectors[r]["anchor"] = []
        for key in queryTerms:
          #f = features[query][r]
          anchors = features[query][r].setdefault('anchors',{})
          totalCount = 0
          for anchor in anchors:
            anchorCount = features[query][r]["anchors"][anchor]
            keyFreq = getFreq(anchor,key)  
            count = keyFreq * anchorCount
            totalCount += count
          
          if totalCount > 0:  
            subLinearScaled = 1 + math.log(totalCount)
            tempdocumentVectors[r]["anchor"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["anchor"].append(0)
        #tempdocumentVectors[r]["anchor"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["anchor"])
        """anchor done"""
        
        """For document url """
        tempdocumentVectors[r]["url"] = []
        for key in queryTerms:
          url = parseUrl(r)
          UrlkeyFreq = getFreq(url,key)
          if UrlkeyFreq > 0:
            subLinearScaled = 1 + math.log(UrlkeyFreq)
            tempdocumentVectors[r]["url"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["url"].append(0)
        #tempdocumentVectors[r]["url"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["url"])
        """url done"""
        #hesapla score u qv ile kendisi arasinda sonra buraya yaz.
        #tempdocumentVectors[r]["pagerank"] = features[query][r].setdefault('pagerank',1)
        tempdocumentVectors[r]["quVector"]= PrepareQueryUrlVector(query, r,qv,tempdocumentVectors[r])
      totalFeaturesVectors =  totalFeaturesVectors + (UniteTempDocumentVectors(tempdocumentVectors))

    return totalFeaturesVectors

#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)


#ExtractScores
def getQueries(rankingFile):
    pat = re.compile('((^|\n)query.*?($|\n))')
    rankings = open(rankingFile,'r')
    res = filter(lambda x: not(x is '' or x=='\n'), pat.split(rankings.read()))

    for item in res:
      #if (item.strip().strip('query:').strip().startswith('stanford cs stanford')):
      #  print "found it"
      if (item.strip().startswith('query:')):
        query = str(item).strip().lstrip('query:').strip()
      else:
        results = filter(lambda x: not(x=='' or x=='\n'), 
                         re.findall('url: .*', item.strip()))
        yield(query, results)
    rankings.close()
    
def ExtractScores(groundTruthFile):
    groundTruth = []
    numQueries = 0
    
    #populate map with ground truth for each query
    for (query, results) in getQueries(groundTruthFile):
      #groundTruth[query] = {}
      for res in results:
        temp = res.rsplit(' ', 1)
        url = temp[0].lstrip('url:').strip()
        rel = float(temp[1].strip())
        groundTruth.append((query,url,rel))
    return groundTruth

def SortFeaturesOrScores(tlist):
  tempList  = sorted(tlist, key= lambda tup: (tup[0],tup[1]))
  
  #extrcts tuples
  #fTempList = map(lambda x: x[0],tempList)
  return tempList

#inparams
#  featureFile: file containing query and url features
def GetScores(train_rel_file):
    #groundTruthFile = "./queryDocTrainRel.train"
    scores  = ExtractScores(train_rel_file)
    sortedScores = SortFeaturesOrScores(scores)
    return sortedScores

def GetFeatures(train_data_file):    
    #featureFile = "./queryDocTrainData.train"
    
    #populate map with features from file
    (queries, features) = extractFeatures(train_data_file)

    #calling baseline ranking system, replace with yours
    featuresVectors = baseline(queries, features)
    sortedFeatures = SortFeaturesOrScores(featuresVectors)
    return sortedFeatures

#scoresWithDescription =  GetScores("queryDocTrainRel.train")
  
