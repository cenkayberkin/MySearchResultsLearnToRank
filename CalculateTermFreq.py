import pickle
import sys

index_dir = "./OutDir"

doc_dict_f = open(index_dir + '/doc.dict', 'r')
word_dict_f = open(index_dir + '/word.dict', 'r')
file_pos_dict_f = open(index_dir + '/Pos.dict','r')
index_f = open(index_dir + '/corpus.index','r')

print >> sys.stderr, 'loading word dict'
word_dict = pickle.load(word_dict_f)
print >> sys.stderr, 'loading doc dict'
doc_id_dict = pickle.load(doc_dict_f)
print >> sys.stderr, 'loading index'
file_pos_dic = pickle.load(file_pos_dict_f)

def read_posting(termId):
    #termId = word_dict[term]
    termPosition = file_pos_dic[str(termId)]
    index_f.seek(termPosition)
    line = index_f.readline()
    parts =  line.split("|")
    termId = int(parts[0])
    df = int(parts[1])
    postings = parts[2].rstrip('\n').split(",")
    return (termId,df,postings)

termDictionary = {}

allQueryTermsFile = open('AllQueryTerms', 'r')

for i in allQueryTermsFile:
  t = i.strip()
  if t in word_dict:
    wordID  = word_dict[t]
    #print t + " " + str(wordID) + " " + str(read_posting(wordID)[1]) + "\n"
    termDictionary[t] = str(read_posting(wordID)[1])
    
termDic_f = open('TermFreq.dict','w')
pickle.dump(termDictionary,termDic_f)


  
    