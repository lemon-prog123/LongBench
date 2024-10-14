from openai import OpenAI
import json
import argparse
import os
import numpy as np
from pred import seed_everything,load_model_and_tokenizer,post_process,build_chat
from datasets import load_dataset
from tqdm import tqdm
answer_dict={}
records=[]
path='pred/llama3.1-8B-Instruct'
try:
    with open(f"{path}/record.json", "r", encoding="utf-8") as f:
        records=json.load(f)
except:
    print("Record None!")
    records=None

print(os.getpid())
def judge(data):
    cnt=0
    correct=0
    client = OpenAI(
    base_url="http://localhost:3003/v1",
    api_key="122",
    )
    for json_obj in tqdm(data):
        #print(json_obj.keys())
        flag=False
        for gt in answers[cnt]:
            json_obj['prediction']=predictions[cnt]
            json_obj['answer']=gt
            answer_dict[gt]=cnt
            prompt = prompt_format.format(**json_obj)
            completion = client.chat.completions.create(
            model="/mnt/qwen2.5-72B/qwen/Qwen2___5-72B-Instruct",
            messages=[
            {"role": "user", "content": prompt}
            ])
            print(completion.choices[0].message.content)
            if completion.choices[0].message.content=='yes':
                flag=True
        if flag:
            correct+=1
        cnt=cnt+1
        #print(json_obj.keys())
    print(cnt)
    print(correct/cnt)

def judged(data):
    cnt=0
    correct=0
    client = OpenAI(
    base_url="http://localhost:3003/v1",
    api_key="122",
    )
    while(cnt<200):
        flag=False
        print('\n'+str(cnt)+'\n')
        json_obj=data[cnt]
        print(json_obj['input'])
        pred=json_obj['output']
        answers=json_obj['answers']
        try:
            pred_post=pred.split("Reasoning")[0].split("Answer:")[1]
        except:
            try:
                pred_post=pred.split("Reasoning")[0]
            except:
                pred_post=pred
        pred=pred_post
        print(json_obj['score'])
        print('\n')
        for gt in answers:
            json_obj['prediction']=pred
            json_obj['answer']=gt
            prompt = prompt_format.format(**json_obj)
            completion = client.chat.completions.create(
            model="/mnt/qwen2.5-72B/qwen/Qwen2___5-72B-Instruct",
            messages=[
            {"role": "user", "content": prompt}
            ])
            print(completion.choices[0].message.content)
            if 'yes' in completion.choices[0].message.content or 'Yes' in completion.choices[0].message.content:
                flag=True
        if flag:
            correct+=1
            data[cnt]['judged']=1
        else:
            data[cnt]['judged']=0
        cnt=cnt+1
    with open(f'pred/llama3.1-8B-Instruct/qasper_judged.json', "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(correct/cnt)

path=f"pred/llama3.1-8B-Instruct/"

all_files = os.listdir(path)
print("Evaluating on:", all_files)
predictions, answers, lengths = [], [], []
for filename in all_files:
    if not filename.endswith("jsonl"):
        continue
    dataset = filename.split('.')[0]
    with open(f"{path}{filename}", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]

dataset="qasper" #test with qasper
judgeprompt = json.load(open("config/judgeprompt.json", "r"))
prompt_format = judgeprompt[dataset]
'''
data = load_dataset('THUDM/LongBench', dataset, split='test')
world_size=1
data_all = [data_sample for data_sample in data]
data_subsets = [data_all[i::world_size] for i in range(world_size)]
data=data_subsets[0]
'''
with open(f"dataset/qasper_dataset.json", "r", encoding="utf-8") as f:
    data=json.load(f)
judged(data)


'''
client = OpenAI(
    base_url="http://localhost:3003/v1",
    api_key="122",
)

completion = client.chat.completions.create(
    model="/mnt/qwen2.5-72B/qwen/Qwen2___5-72B-Instruct",
    messages=[
    {"role": "user", "content": text_string}
    ]
)
'''