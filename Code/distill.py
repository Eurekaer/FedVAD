import os
import json
import os
import csv

from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine

# Initialize the tokenizer and model from the pre-trained BERT base model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Ensure the model is in evaluation mode

# Define a function to get the BERT embeddings for a given text
def get_bert_embedding(text):
    # Tokenize the text and get the required input tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # No need to track gradients for embeddings
        # Get the outputs from the BERT model
        outputs = model(**inputs)
    # Use the embedding of the [CLS] token (first token) as the sentence representation
    return outputs.last_hidden_state[:, 0, :].squeeze().detach()

def get_public_train_list(tag):
    total_object_number = {}
    public_train_list = []
    with open('Object target distribution json file sent to each client on the server side.','r') as coc:
        data = json.load(coc)
    print(data)
    for i in tag:
        numbers = data['client' + str(int(i) + 1)]
        for n in numbers:
            if total_object_number.get(n):
                total_object_number[n] += int(numbers[n])
            else:
                total_object_number[n] = int(numbers[n])
    print(total_object_number)
    # 按值对字典进行排序
    sorted_items = sorted(total_object_number.items(), key=lambda x: x[1], reverse=True)
    print(sorted_items)
    # 计算前70%的元素数量
    num_items = int(len(sorted_items) * 0.7)

    # 创建新的字典，只包含前70%的元素
    sorted_dict = dict(sorted_items[:num_items])
    print(sorted_dict)
    long_string = ' '.join(sorted_dict.keys())

    # 打印结果
    print(long_string)
    embedding1 = get_bert_embedding(long_string)
    print(embedding1.shape)
    print(type(embedding1)) 
    embedding1_np = embedding1.numpy()
    import numpy as np
    path = 'Path to the object target distribution feature folder corresponding to each video file on the server side.'
    names = os.listdir(path)
    for name in names:
        embedding2 = np.load(path + name)
        embedding2_np = embedding2
        # Calculate the cosine similarity between the embeddings
        cosine_similarity = 1 - cosine(embedding1_np, embedding2_np)
        print(cosine_similarity)        
        if float(cosine_similarity) > 0.70:
            public_train_list.append(name)
    return public_train_list


def get_public_aug_train_list(tags):
    public_train_list = get_public_train_list(tags)
    tmp_path = 'Temporary storage of public dataset file paths.'
    with open(tmp_path,'w') as t:
        for train in public_train_list:
            name = train.split('.npy')[0]
            mid = name.split('_')[1]
            if len(mid) == 3:
                flags = ' 0.0'
            else:
                flags = ' 1.0'
            for i in range(0,10):
                n_name = 'train/' + name + '_' + str(i) + '.npy' + flags
                t.writelines(n_name + '\n')
    return tmp_path


get_public_aug_train_list([1,2,3])
