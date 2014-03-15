import CalctTFIdfs

#scores is in form of tuples like this (query,url, score)
scoresWithDescription =  CalctTFIdfs.GetScores()

#features is in form of tuples like this (query,url,featuresVector)
featuresWithDescription =  CalctTFIdfs.GetFeatures()

scores = map(lambda x: x[2],scoresWithDescription)
features = map(lambda x: x[2],featuresWithDescription)

print "cenk"