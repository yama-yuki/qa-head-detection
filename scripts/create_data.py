import json
import pickle
import random
from tqdm import tqdm
from collections import OrderedDict

def make_paths():
    ##create a list of files to process
    data_dir = '../data/conllu/'
    file_names = ['A','B','C','D','E','F','G']
    in_ext = '.conllu'
    out_ext = '.pickle'
    path_list = [data_dir+file_name+in_ext for file_name in file_names]
    out_list = [data_dir+file_name+out_ext for file_name in file_names]
    return path_list, out_list

def read_conllu(path):
    ##.conllu to a list
    conll_list= []
    tmp = []
    with open(path, mode='r', encoding='utf-8') as d:
        lines = d.readlines()
        for line in tqdm(lines):
            if line[0] != '#' and line[0] != '\n':
                tmp.append(line.rstrip('\n').split())
            else:
                if tmp:
                    conll_list.append(tmp)
                    tmp = []
        if tmp:
            conll_list.append(tmp)
    return conll_list

def list_to_pickle(conll_list, out_path):
    with open(out_path, mode='wb') as o:
        pickle.dump(conll_list, o)
    print(str(out_path)+' SAVED')
    return

def stats(path_list):

    for path in path_list[:1]:
        total_sents = 0
        total_words = 0
        conll_list = read_conllu(path)
        for conll in conll_list:
            total_sents+=1
            total_words+=len(conll)
        print('DONE READING '+str(path))
        print('sents: '+str(total_sents))
        print('words: '+str(total_words))
    #with open('../tmp/conlls.pkl',mode='wb') as o:
        #pickle.dump(conlls, o)
    
    #with open('../tmp/conlls.pkl',mode='rb') as f:
        #conlls = pickle.load(f)
    

cnt = 0
def process(file_names):
    ## ['4', 'of', 'of', 'ADP', 'IN', '_', '7', 'case', '_', 'start_char=244|end_char=246']

    ## {"data": [{"title": "None", "paragraphs": 
    ## [{ "context": <context>, "qas": [
    ## {"answers": [{"answer_start": <id>, "text": <ans>}], "question": <question>, "id": <id>},

    data_dir = '../data/conllu/'

    new_data = {"data": [{"title": "None", "paragraphs":[]}]}
    
    file_names = file_names
    for pickle_name in file_names:
        pickle_path=data_dir+pickle_name+'.pickle'

        with open(pickle_path, mode='rb') as f:
            conll_list = pickle.load(f)
        #conll_list = conll_list[:100]

        global cnt
        for conll in tqdm(conll_list):
            entry = {"context":"","qas":[]}
            
            snt = [w[1] for w in conll]
            entry["context"] = ' '.join(snt)
            qas = []
            flag = False
            for i,word in enumerate(conll):
                if word[7] == 'advcl':
                    head_i = int(word[6])-1
                    
                    d = {"answers": [{"answer_start": "", "text": ""}], "question": "", "id": ""}
                    d["answers"][0]["text"] = conll[head_i][1] #head_word
                    d["answers"][0]["answer_start"] = len(' '.join(snt[:head_i]))+1
                    #d["question"] = "What is the head of "+word[1]+" ?"
                    d["question"] = word[1]
                    d["id"] = str(cnt)
                    qas.append(d)
                    flag = True
                    cnt+=1
            if flag == True:
                entry["qas"] = qas
                new_data["data"][0]["paragraphs"].append(entry)

        #print(new_data)

    out_dir = '../data/wiki/'
    out_path=out_dir+'advcl_tok'+'.json'
    with open(out_path, 'w') as o:
        json.dump(new_data, o)

    return

def dev_test(original_data):
    random.seed(0)

    dev_data = {"data": [{"title": "None", "paragraphs":[]}]}
    test_data = {"data": [{"title": "None", "paragraphs":[]}]}

    with open('../data/wiki/'+original_data+'.json') as f:
        data = json.load(f)
        print(len(data["data"][0]["paragraphs"]))

    
    num_list = random.sample(range(len(data["data"][0]["paragraphs"])-10000), k=10000)

    for num in num_list:
        dev_data["data"][0]["paragraphs"].append(data["data"][0]["paragraphs"].pop(num))

    num_list = random.sample(range(len(data["data"][0]["paragraphs"])-10000), k=10000)
    for num in num_list:
        test_data["data"][0]["paragraphs"].append(data["data"][0]["paragraphs"].pop(num))

    print('train: '+str(len(data["data"][0]["paragraphs"])))
    print('dev: '+str(len(dev_data["data"][0]["paragraphs"])))
    print('test: '+str(len(test_data["data"][0]["paragraphs"])))


    data_list = [data, dev_data, test_data]
    out_name = ['train', 'dev', 'test']

    for d,o in zip(data_list,out_name):
        out_dir = '../data/wiki/'
        out_path=out_dir+original_data+'_'+o+'.json'
        with open(out_path, 'w') as o:
            json.dump(d, o)
    
    return

if __name__ == '__main__':
    '''
    path_list, out_list = make_paths()

    for i in range(len(path_list)):
        path = path_list[i]
        out_path = out_list[i]

        conll_list = read_conllu(path)
        list_to_pickle(conll_list, out_path)
    '''

    #file_names = ['A','B','C','D','E','F','G']
    #process(file_names)

    dev_test('advcl_tok')
        


    #stats(path_list)
    #for file_path, out_path in zip(path_list, out_list):
        #process(file_path, out_path)

