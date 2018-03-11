""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def load():
    with open('/data/xuwenshen/workspace/squad/data/dev-v1.1.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    return dataset
    
def evaluate(predictions):
    
    dataset = load()
    out_file = open('/data/xuwenshen/workspace/squad/code/multi_task/dev_out.txt', 'w')
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            out_file.write(paragraph["context"] + "\n\n")
            for qa in paragraph['qas']:
                total += 1
                out_file.write(qa["question"] + "\n\n")
                if qa['id'] not in predictions:
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                out_file.write(ground_truths[0] + "  |  " + prediction)
                out_file.write("\n---------------------------\n\n")
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    out_file.close()
    return {'exact_match': exact_match, 'f1': f1}
