import sys , math
import re , pickle
from math import log
from decimal import Decimal
import numpy as np
import itertools


avlenb = 0
avlenu = 0
avlent = 0
avlenh = 0
avlena = 0

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair

#print >> sys.stderr, 'Task 1, Cosine Similarity'
termDic_f = open('TermFreq.dict','r')
print >> sys.stderr, 'loading Term Dictionary'
termDic = pickle.load(termDic_f)



def SortTempDocumentVector(documentVecs):
  tempList = []
  for i in sorted(documentVecs.keys()):
    item  = ((i,documentVecs[i]["score"]))
    tempList.append(item)
  
  tempList  = sorted(tempList, key= lambda tup: tup[1],reverse=True)
  fTempList = map(lambda x: x[0],tempList)
  return fTempList

def GetClosestNum(l):
  minNum = sys.maxint
  for i in itertools.product(*l):
    l =  sorted(i)
    #print l
    r = abs(l[0] - l[-1])
    if r < minNum:
      minNum = r
  return minNum

def CalculateSWindow(qterms, bodyHits):
  b = 3
  index = 0
  for term in qterms:
    if term in bodyHits:
      index = index +  1
  
  #Check if all qterms are in bodyHits, if there are then continue
  if len(qterms) != index:
    return 1
  
  l = bodyHits.values()
  minNum = GetClosestNum(l)
  #print minNum
  
  if minNum == 0:
    c1 = 0
  else:
    c1 = Decimal(len(qterms) - 1 ) / Decimal(minNum)
     
  windowScore = Decimal(b) * (1 + c1)
  
  return windowScore

def getFreq(text, key):
  count = 0
  tokens = str(text).lower().split()
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

def PrepareQueryUrlVectorWithExtraFeatures(query,url,qv,dv):
  cu = 3
  ct = 2
  cb = 0.5
  ch = 3
  ca = 10
  
  result = []
  
  
  #regular tfidf values
  tfduTf = np.array(dv["urlTf"])
  tfdtTf = np.array(dv["titleTf"])
  tfdbTf = np.array(dv["bodyTf"])
  tfdhTf = np.array(dv["headerTf"])
  tfdaTf = np.array(dv["anchorTf"])
  
  tfduTf = map(lambda x: float(x * cu), tfduTf)
  tfdtTf = map(lambda x: float(x * ct), tfdtTf)
  tfdbTf = map(lambda x: float(x * cb), tfdbTf)
  tfdhTf = map(lambda x: float(x * ch), tfdhTf)
  tfdaTf = map(lambda x: float(x * ca), tfdaTf)
  
  tfduTf = np.array(qv) * np.array(tfduTf)
  tfdtTf = np.array(qv) * np.array(tfdtTf)
  tfdbTf = np.array(qv) * np.array(tfdbTf)
  tfdhTf = np.array(qv) * np.array(tfdhTf)
  tfdaTf = np.array(qv) * np.array(tfdaTf)
  
  result.append(sum(tfduTf))
  result.append(sum(tfdtTf))
  result.append(sum(tfdbTf))
  result.append(sum(tfdhTf))
  result.append(sum(tfdaTf))
  
  """
  #BM25 values
  tfdu = np.array(dv["url"])
  tfdt = np.array(dv["title"])
  tfdb = np.array(dv["body"])
  tfdh = np.array(dv["header"])
  tfda = np.array(dv["anchor"])
  
  tfdu = map(lambda x: float(x) * float(cu), tfdu)
  tfdt = map(lambda x: float(x) * float(ct), tfdt)
  tfdb = map(lambda x: float(x) * float(cb), tfdb)
  tfdh = map(lambda x: float(x) * float(ch), tfdh)
  tfda = map(lambda x: float(x) * float(ca), tfda)
  
  tfdu = np.array(qv) * np.array(tfdu)
  tfdt = np.array(qv) * np.array(tfdt)
  tfdb = np.array(qv) * np.array(tfdb)
  tfdh = np.array(qv) * np.array(tfdh)
  tfda = np.array(qv) * np.array(tfda)
  
  result.append(sum(tfdu))
  result.append(sum(tfdt))
  result.append(sum(tfdb))
  result.append(sum(tfdh))
  result.append(sum(tfda))
  """
  
  #instead of bm25 values i tried using bm25 score
  #bm25Score = float(dv["score"])
  #result.append(bm25Score)
  
  #adding small window score
  smallWindowScore = dv["swindow"]
  #print >> sys.stderr, "Small win score " + str(smallWindowScore) 
  result.append(float(smallWindowScore))
  
  docPageRank = dv["pagerank"]
  if docPageRank == 0:
    docPageRank = 1
  #logarithm of pagerank for Vj function
  docPageRank = 1 + log(int(docPageRank))
  result.append(docPageRank)
  
  pdfMatch = re.search('(?:([^:/?#]+):)?(?://([^/?#]*))?([^?#]*\.(?:pdf))(?:\?([^#]*))?(?:#(.*))?', url)

  #Check if url ends with pdf
  if pdfMatch:
      result.append(1)
  else:
      result.append(0)
  """
  htmlMatch = re.search('(?:([^:/?#]+):)?(?://([^/?#]*))?([^?#]*\.(?:html))(?:\?([^#]*))?(?:#(.*))?', url)
  #Check if url ends with html
  if htmlMatch:
      result.append(1)
  else:
      result.append(0)
  """
  
  return (query,url,result)

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

def CalculateAvlens(queries,features):
  
  totalTrainDataBodylength = 0
  totalNumberOfBody = 0
  tempDocs = {}
  
  totalTrainDatatitleLen = 0
  totalNumberOfTitle = 0
  
  totalTrainDataHeadersLen = 0
  totalNumberOfHeader = 0
  
  totalTrainDataAnchorsLen = 0
  totalNumberOfAnchors = 0
  
  totalTrainDataUrlsLen = 0
  totalNumberOfUrls = 0
  
  for query in queries.keys():
      results = queries[query]
      for r in results:
        """extract query results"""
        if r not in tempDocs:
          tempDocs[r] = features[query][r]
  
  for i in tempDocs.keys():
    bodylength = tempDocs[i].setdefault('body_length', 0)
    totalNumberOfBody = totalNumberOfBody + 1
    totalTrainDataBodylength = totalTrainDataBodylength + bodylength
  #average lenght of document bodies
  global avlenb
  avlenb = totalTrainDataBodylength / totalNumberOfBody  
  
  for i in tempDocs.keys():
    title = tempDocs[i].setdefault('title', 0)
    totalNumberOfTitle = totalNumberOfTitle + 1
    totalTrainDatatitleLen = totalTrainDatatitleLen + len(title.split())
  #average lenght of document titles
  global avlent 
  avlent = Decimal(totalTrainDatatitleLen) / Decimal(totalNumberOfTitle)  
  
  for i in tempDocs.keys():
    headerList = tempDocs[i].setdefault('header', 0)
    if headerList != 0:
      headersLength =  sum([len(str(a).split()) for a in headerList])
      totalNumberOfHeader = totalNumberOfHeader + len(headerList) 
      totalTrainDataHeadersLen = totalTrainDataHeadersLen + headersLength
  global avlenh
  avlenh = Decimal(totalTrainDataHeadersLen) / Decimal(totalNumberOfHeader) 
  #average lenght of document headers
  
  for i in tempDocs.keys():
    anchorList = tempDocs[i].setdefault('anchors', 0)
    if anchorList != 0:
      anchorsLength =  sum([len(str(a).split()) * anchorList[a] for a in anchorList.keys()])
      anchors = sum([anchorList[a] for a in anchorList.keys()])
      totalNumberOfAnchors = totalNumberOfAnchors + anchors
      totalTrainDataAnchorsLen = totalTrainDataAnchorsLen + anchorsLength
    #average lenght of document anchors
  global avlena
  avlena = Decimal(totalTrainDataAnchorsLen) / Decimal(totalNumberOfAnchors) 
  
  for i in tempDocs.keys():
    url = parseUrl(i)
    urlLength = len(url.split())
    totalNumberOfUrls = totalNumberOfUrls + 1
    totalTrainDataUrlsLen = totalTrainDataUrlsLen + urlLength
    #average lenght of document headers
  global avlenu
  avlenu = Decimal(totalTrainDataUrlsLen) / Decimal(totalNumberOfUrls)
  
#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query


def CalculateBM25F(queryTerms, dv):
  
  #field weights
  Wb = 1
  Wu = 3
  Wt = 5
  Wa = 10
  Wh = 5
  
  K1= 2
  
  #field dependent normalized url frequency vector
  ftfdu = dv["url"]
  
  #field dependent normalized title frequency vector
  ftfdt = dv["title"]
  
  #field dependent normalized body frequency vector
  ftfdb = dv["body"]
  
  #field dependent normalized header frequency vector
  ftfdh = dv["header"]
  
  #field dependent normalized anchor frequency vector
  ftfda = dv["anchor"]

  #applying field weights
  ftfdu = map(lambda x: x * Wu, ftfdu)
  ftfdt = map(lambda x: x * Wt, ftfdt)
  ftfdb = map(lambda x: x * Wb, ftfdb)
  ftfda = map(lambda x: x * Wa, ftfda)
  ftfdh = map(lambda x: x * Wh, ftfdh)
  
  result1 = map(sum,zip(ftfdu,ftfdt))
  result2 = map(sum,zip(ftfda,ftfdb))
  res12 = map(sum,zip(result1,result2))
  finaldocVec = map(sum,zip(res12,ftfdh))
  
  docPageRank = dv["pagerank"]
  if docPageRank == 0:
    docPageRank = 1
  #logarithm of pagerank for Vj function
  docPageRank = 1 + log(int(docPageRank))

  #prepares idf of query terms
  qv = PrepareQueryVector(queryTerms)
  
  multlist = []
  for d, q in zip(finaldocVec ,qv):
    r = Decimal(d* Decimal(q)) / Decimal(K1) + Decimal(d)
    r = Decimal(r) + Decimal(docPageRank)
    multlist.append(r)
  
  result = sum(multlist)
  return result

def baselineBM25(queries, features):
    totalFeaturesVectors = []

    rankedQueries = {}
    for query in queries.keys():
      tempdocumentVectors = {}
      results = queries[query]
      queryTerms = sorted(set(query.split()))
      
      queryTerms = sorted(set(query.split()))
      qv = PrepareQueryVector(queryTerms)
      
      
      Bbody = 1
      Btitle = 3
      Bheader = 1 
      Burl = 5
      Banchor = 4
      
      for r in results:
        tempdocumentVectors[r] = {}
        
        """Calculating ftf document body term """
        tempdocumentVectors[r]["body"] = []
        for key in queryTerms:
          if key in features[query][r].setdefault('body_hits',{}):
            bodylenght = features[query][r].setdefault('body_length',{})
            keyFreqB = len(features[query][r].setdefault('body_hits',{})[key])
            lengthCalc = Decimal(bodylenght) / Decimal(avlenb - 1)
            ftfdbt =  Decimal(keyFreqB) / 1 + Decimal(Bbody * lengthCalc)
            tempdocumentVectors[r]["body"].append(ftfdbt)
          else:
            tempdocumentVectors[r]["body"].append(0)
        """body done """
        
        """For document body """
        tempdocumentVectors[r]["bodyTf"] = []
        for key in queryTerms:
          if key in features[query][r].setdefault('body_hits',{}):
            keyFreq = len(features[query][r].setdefault('body_hits',{})[key])
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["bodyTf"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["bodyTf"].append(0)
        #tempdocumentVectors[r]["body"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["body"])
        """body done """
        
        """For document title """
        tempdocumentVectors[r]["titleTf"] = []
        for key in queryTerms:
          title = features[query][r].setdefault('title',{})
          keyFreq = getFreq(title,key)
          if keyFreq > 0:  
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["titleTf"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["titleTf"].append(0)
        #tempdocumentVectors[r]["title"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["title"])
        """title done """
        
        """Calculating ftf document title term """
        tempdocumentVectors[r]["title"] = []
        for key in queryTerms:
          title = features[query][r].setdefault('title',{})
          titleLen = len(title)
          keyFreqT = getFreq(title,key)
          if keyFreqT > 0:
            lengthCalc = Decimal(titleLen) / Decimal(avlent -1)
            ftfdtt = keyFreqT / 1 + Btitle * lengthCalc
            tempdocumentVectors[r]["title"].append(ftfdtt)
          else:
            tempdocumentVectors[r]["title"].append(0)
        """title done """
        
        """For document header """
        tempdocumentVectors[r]["headerTf"] = []
        for key in queryTerms:
          headers = features[query][r].setdefault('header',{})
          
          header= ""
          for i in headers:
            header += i + " "
            
          keyFreq = getFreq(header,key)
          if keyFreq > 0:  
            subLinearScaled = 1 + math.log(keyFreq)
            tempdocumentVectors[r]["headerTf"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["headerTf"].append(0)
        #tempdocumentVectors[r]["header"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["header"])
        """header done """
        
        """Calculating ftf document header term """
        tempdocumentVectors[r]["header"] = []
        for key in queryTerms:
          headers = features[query][r].setdefault('header',{})
          #print headers
          header= ""
          for i in headers:
            header += i + " "
            
          headerLen = len(header.split())   
          keyFreqH = getFreq(header,key)
          if keyFreqH > 0:
            lengthCalc = Decimal(headerLen) / Decimal(avlenh -1)
            ftfdht = keyFreqH / 1 + Bheader * lengthCalc
            tempdocumentVectors[r]["header"].append(ftfdht)
          else:
            tempdocumentVectors[r]["header"].append(0)
        """header done """
        
        """Calculating ftf document Url term """
        tempdocumentVectors[r]["url"] = []
        for key in queryTerms:
          url = parseUrl(r)
          UrlkeyFreq = getFreq(url,key)
          if UrlkeyFreq > 0:
            UrlLength = len(url.split()) 
            lengthCalc = Decimal(UrlLength) / Decimal(avlenu -1)
            ftfdut = UrlkeyFreq / 1 + Burl * lengthCalc
            tempdocumentVectors[r]["url"].append(ftfdut)
          else:
            tempdocumentVectors[r]["url"].append(0)
        """url done"""
        
        """For document url """
        tempdocumentVectors[r]["urlTf"] = []
        for key in queryTerms:
          url = parseUrl(r)
          UrlkeyFreq = getFreq(url,key)
          if UrlkeyFreq > 0:
            subLinearScaled = 1 + math.log(UrlkeyFreq)
            tempdocumentVectors[r]["urlTf"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["urlTf"].append(0)
        
        """For document anchor """
        tempdocumentVectors[r]["anchorTf"] = []
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
            tempdocumentVectors[r]["anchorTf"].append(subLinearScaled)
          else:
            tempdocumentVectors[r]["anchorTf"].append(0)
        #tempdocumentVectors[r]["anchor"] = map(lambda x: x / (bodylength + 500) , tempdocumentVectors[r]["anchor"])
        """anchor done"""
        
        
        """Calculating ftf document Anchors term """
        tempdocumentVectors[r]["anchor"] = []
        for key in queryTerms:
          #f = features[query][r]
          anchors = features[query][r].setdefault('anchors', {})
          totalCount = 0
          anchorsLength = 0
          
          for anchor in anchors:
            #print anchor
            anchorCount = features[query][r]["anchors"][anchor]
            alen = len(anchor.split()) * anchorCount
            anchorsLength = anchorsLength + alen
            keyFreqA = getFreq(anchor,key)  
            count = keyFreqA * anchorCount
            totalCount += count
          
          if totalCount > 0:  
            lengthCalc = Decimal(anchorsLength) / Decimal(avlena - 1)
            ftfdat = totalCount / 1 + Banchor * lengthCalc
            tempdocumentVectors[r]["anchor"].append(ftfdat)
          else:
            tempdocumentVectors[r]["anchor"].append(0)
          """anchor done"""
        tempdocumentVectors[r]["swindow"]  = CalculateSWindow(queryTerms, features[query][r].setdefault('body_hits',{}))
        tempdocumentVectors[r]["pagerank"] = features[query][r].setdefault('pagerank',1)
        tempdocumentVectors[r]["score"]= CalculateBM25F(queryTerms,tempdocumentVectors[r])
        tempdocumentVectors[r]["quVector"]= PrepareQueryUrlVectorWithExtraFeatures(query, r,qv,tempdocumentVectors[r])

      totalFeaturesVectors =  totalFeaturesVectors + (UniteTempDocumentVectors(tempdocumentVectors))
      #rankedQueries[query] = SortTempDocumentVector(tempdocumentVectors)

    return totalFeaturesVectors

def baselineWithPdfFeature(queries, features):
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
        #tempdocumentVectors[r]["swindow"]  = CalculateSWindow(queryTerms, features[query][r].setdefault('body_hits',{}))
        tempdocumentVectors[r]["quVector"]= PrepareQueryUrlVectorWithExtraFeatures(query, r,qv,tempdocumentVectors[r])
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

def GetFeaturesWithBinaryPdf(train_data_file):    
    #featureFile = "./queryDocTrainData.train"
    
    #populate map with features from file
    (queries, features) = extractFeatures(train_data_file)

    #calling baseline ranking system, replace with yours
    featuresVectors = baselineBM25(queries, features)
    sortedFeatures = SortFeaturesOrScores(featuresVectors)
    return sortedFeatures


def GetFeatures(train_data_file):    
    #featureFile = "./queryDocTrainData.train"
    
    #populate map with features from file
    (queries, features) = extractFeatures(train_data_file)

    #calling baseline ranking system, replace with yours
    featuresVectors = baseline(queries, features)
    sortedFeatures = SortFeaturesOrScores(featuresVectors)
    return sortedFeatures

#scoresWithDescription =  GetScores("queryDocTrainRel.train")
#featuresWihtPdf = GetFeaturesWithBinaryPdf("queryDocTrainRel.train") 
