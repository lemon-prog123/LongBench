import os
import json
import argparse
import numpy as np
from pred import post_process

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--t', action='store_true', help="Evaluate on LongBench-t")
    parser.add_argument('--sample', action='store_true', help="Evaluate on test mode")
    parser.add_argument('--test', action='store_true', help="Evaluate on test mode")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            #prediction2=prediction.split(".")[0].split("Answer:")[1]
            #print(prediction2)
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def find_num(data,length):
    num_list=[]
    for i,json_obj in enumerate(data):
        lg=json_obj['length']
        if lg==length:
            num_list.append(i)
    return num_list
def scored(dataset, predictions, answers, lengths,all_classes,args):
    total_score = 0.
    records=[]
    preds=[]
    #with open(f"config/answer2dict.json", "r", encoding="utf-8") as f:
    #    answer_dict=json.load(f)
    from datasets import load_dataset
    dataset="qasper" #test with qasper
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    world_size=1
    data_all = [data_sample for data_sample in data]
    data_subsets = [data_all[i::world_size] for i in range(world_size)]
    data=data_subsets[0]
    nums=[]
    for (preds, ground_truths,length) in zip(predictions, answers,lengths):
        score = 0.
        max_score=-1
        cnt=0
        record=0
        max_pred=None
        for pred in preds:
            pred_post=post_process(pred, "llama3.1-8B-Instruct",args)
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](pred_post, ground_truth, all_classes=all_classes))
                if score>max_score:
                    max_score=score
                    record=cnt
                    #num=int(answer_dict[ground_truth])
                    max_pred=pred
            cnt+=1
        num_list=find_num(data,length)
        flag=False
        for i in num_list:
            if i not in nums:
                nums.append(i)
                num=i
                flag=True
                break
        if flag==False:
            print(num_list)
        data[num]['output']=max_pred
        data[num]['score']=(max_score*1000)/1000
        preds.append(max_pred)
        records.append(record)
        total_score += score
    
    with open(f'pred/{args.model}/record.json', "a", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    
    with open(f'pred/{args.model}/qasper_dataset.json', "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    elif args.t:
        path=f"pred_t/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        elif args.sample:
            score = scored(dataset, predictions, answers, lengths,all_classes,args)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    elif args.t:
        out_path=f"pred_t/{args.model}/result.json"
    else:
        out_path = f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
