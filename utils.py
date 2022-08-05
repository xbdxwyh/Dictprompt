import datasets as hfd
import pandas as pd
import jsonlines

def get_data(data_path):
    data = []
    with open(data_path,'r+') as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


def get_label_map_dict(map_reverse):
    if map_reverse:
        label_dict = {True:0,False:1}
    else:
        label_dict = {True:1,False:0}
    return label_dict

def read_superglue_train_set(data_path):
    data_train = get_data(data_path=data_path+"/train.jsonl")
    data_list = [[i for i in j.values()] for j in data_train]
    train_set = pd.DataFrame(data_list,columns=data_train[0].keys())
    return train_set

def read_superglue_val_set(data_path):
    data_val = get_data(data_path=data_path+"/val.jsonl")
    data_list = [[i for i in j.values()] for j in data_val]
    val_set = pd.DataFrame(data_list,columns=data_val[0].keys())
    return val_set

def read_superglue_test_set(data_path):
    data_test = get_data(data_path=data_path+"/test.jsonl")
    data_list = [[i for i in j.values()] for j in data_test]
    test_set = pd.DataFrame(data_list,columns=data_test[0].keys())
    return test_set

def read_superglue_dataset(data_path,load_test=False,return_hfd=True,map_reverse=False):
    label_dict = get_label_map_dict(map_reverse=map_reverse)
    
    train_set = read_superglue_train_set(data_path=data_path)
    val_set = read_superglue_val_set(data_path=data_path)

    train_set.label = train_set.label.map(label_dict)
    val_set.label = val_set.label.map(label_dict)

    if return_hfd:
        train_set = hfd.Dataset.from_pandas(train_set)
        val_set = hfd.Dataset.from_pandas(val_set)
    
    if load_test:
        test_set = read_superglue_test_set(data_path=data_path)
        if return_hfd:
            test_set = hfd.Dataset.from_pandas(test_set)
        return train_set,val_set,test_set
    else:
        return train_set,val_set


def read_sense_making_taskA_data(text_path,label_path):
    #text_path = data_path + "subtaskA_data_all.csv"
    #label_path = data_path + "subtaskA_answers_all.csv"
    
    text = pd.read_csv(text_path)
    label = pd.read_csv(label_path,header=None)
    label.columns = ["idx","label"]

    label.label = label.label.map(lambda x:int(x))

    data = pd.concat([text,label],axis=1)[['sent0','sent1','label']]
    return data


def read_sense_making_taskA_train(train_data_path):
    text_path = train_data_path + "/subtaskA_data_all.csv"
    label_path = train_data_path + "/subtaskA_answers_all.csv"
    train_data = read_sense_making_taskA_data(text_path,label_path)
    return train_data

def read_sense_making_taskA_test(test_data_path):
    text_path = test_data_path + "/subtaskA_test_data.csv"
    label_path = test_data_path + "/subtaskA_gold_answers.csv"
    test_data = read_sense_making_taskA_data(text_path,label_path)
    return test_data

def read_sense_making_taskA(dataset_path,return_hfd=True):
    train_data_path = dataset_path+"/Train"
    test_data_path = dataset_path+"/Test"

    train_set = read_sense_making_taskA_train(train_data_path)
    val_set = read_sense_making_taskA_test(test_data_path)

    if return_hfd:
        train_set = hfd.Dataset.from_pandas(train_set)
        val_set = hfd.Dataset.from_pandas(val_set)
    
    return train_set,val_set


def read_sense_making_taskB_data(text_path,label_path):
    text = pd.read_csv(text_path)
    label = pd.read_csv(label_path,header=None)
    label.columns = ["idx","label"]
    dict_map = {"A":0,"B":1,"C":2}
    label.label = label.label.map(dict_map)

    data = pd.concat([text,label],axis=1)[['FalseSent','OptionA','OptionB','OptionC',"label"]]
    return data

def read_sense_making_taskB(dataset_path,return_hfd=True):
    train_data_path = dataset_path+"/Train"
    test_data_path = dataset_path+"/Test"

    train_set = read_sense_making_taskB_data(
        text_path=train_data_path+"/subtaskB_data_all.csv",
        label_path=train_data_path+"/subtaskB_answers_all.csv"
    )

    val_set = read_sense_making_taskB_data(
        text_path=test_data_path+"/subtaskB_test_data.csv",
        label_path=test_data_path+"/subtaskB_gold_answers.csv"
    )

    if return_hfd:
        train_set = hfd.Dataset.from_pandas(train_set)
        val_set = hfd.Dataset.from_pandas(val_set)
    
    return train_set,val_set


def read_HellaSwag_dataset(dataset_path,load_test=False):
    train_data_path = dataset_path+"/hellaswag_train.jsonl"
    dev_data_path = dataset_path+"/hellaswag_val.jsonl"

    train_set = hfd.Dataset.from_json(train_data_path)
    dev_set = hfd.Dataset.from_json(dev_data_path)
    if load_test:
        test_data_path = dataset_path+"/hellaswag_test.jsonl"
        test_set = hfd.Dataset.from_json(test_data_path)
        return train_set,dev_set,test_set
    else:
        return train_set,dev_set


    
def tokenize_wsc_function(examples,tokenizer,max_length=256):
    # 先获取句子
    text = examples['text']
    #word1,word2 = examples['target']['span1_text'],examples['target']['span2_text']
    # 获取两个词语对应的文本
    word1_text = " ".join(text.split(' ')[0:examples['target']['span1_index']+1])
    word2_text = " ".join(text.split(' ')[0:examples['target']['span2_index']+1])
    # 获取词语的位置
    word1_loc = len(tokenizer.encode(word1_text,add_special_tokens=False))
    word2_loc = len(tokenizer.encode(word2_text,add_special_tokens=False))
    # 进行tokenized 的组合
    tokenized_sentence = tokenizer(text,padding='max_length',max_length = max_length,truncation = True)
    tokenized_sentence['word1_locs'] = word1_loc
    tokenized_sentence['word2_locs'] = word2_loc
    return tokenized_sentence#,word1_loc,word2_loc


def tokenize_wic_function(examples,tokenizer,max_length=512):
    # 获取词语对应的位置 以及对应的第二句偏移量
    word1_loc = len(tokenizer.encode(examples['sentence1'][0:examples['end1']],add_special_tokens=False))
    word2_loc = len(tokenizer.encode(examples['sentence2'][0:examples['end2']],add_special_tokens=False))
    offset = len(tokenizer.encode(examples['sentence1']))
    # tokenizer
    tokenized_sentence = tokenizer(examples['sentence1'],examples['sentence2'],padding='max_length',max_length = max_length,truncation = True)
    # 获取矩阵
    loc_word1 = [[0] * max_length]
    loc_word2 = [[0] * max_length]
    # 将单词对应的位置置为1
    loc_word1[0][word1_loc] = 1
    loc_word2[0][offset+word2_loc] = 1
    # 修改字典
    tokenized_sentence['word1_locs'] = loc_word1
    tokenized_sentence['word2_locs'] = loc_word2
    #print(tokenized_sentence['input_ids'][word1_loc],tokenized_sentence['input_ids'][offset+word2_loc])
    return tokenized_sentence#,word1_loc,word2_loc