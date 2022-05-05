
import os
import collections

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.special import softmax

import random
import pandas as pd
import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset


def encode_data(tokenizer, sents, starts, ends, sym='[TGT]'):
    """
    Transform input sentences into tokenized input for the model.

    Input
    =====
    - tokenizer : transformers tokenizer
    - sents : list of strings
    - starts : list of ints pointing at the index at which the target word starts
    - ends : list of ints pointing at the index at which the target word ends
    - sym : string, symbol to use to signalize what the target word is (make sure
        you add it to the tokenizer vocabulary if not already there)

    Output
    ======
    - output_sents : list of strings
    - spans : list of tuples (start, end), where `start` is the index of
        the first subtoken corresponding to the target word, `end` the index
        of the last one.
    """
    output_sents, spans = [], []
    for sent, char_start, char_end in zip(sents, starts, ends):
        # insert target symbols
        if sym is not None:
            sent = sent[:char_start] + '{} '.format(sym) + \
                sent[char_start:char_end] + ' {}'.format(sym) + sent[char_end:]
        output_sents.append(sent)

        sent = tokenizer.encode_plus(sent, return_offsets_mapping=True)
        # transform character indices to subtoken indices
        target_start = target_end = None
        if sym is not None:
            char_start += len(sym) + 1
            char_end += len(sym) + 1
        for idx, (token_start, token_end) in enumerate(sent['offset_mapping']):
            if token_start == char_start:
                target_start = idx
            if token_end == char_end:
                target_end = idx
        if target_start is None or target_end is None:
            raise ValueError
        spans.append((target_start, target_end + 1))

    return output_sents, spans


def read_data(tokenizer, lhs, targets, rhs):

    sents, starts, ends = [], [], []
    for (lhs, target, rhs) in zip(lhs, targets, rhs):
        sents.append((lhs + ' ' + target + ' ' + rhs).strip())
        start = len(lhs) + 1 if lhs else 0
        starts.append(start)
        ends.append(start + len(target))

    return encode_data(tokenizer, sents, starts, ends)


def get_metrics(trues, preds):
    return {'accuracy': metrics.accuracy_score(trues, preds),
            'f1-micro': metrics.f1_score(trues, preds, average='micro'),
            'f1-macro': metrics.f1_score(trues, preds, average='macro')}


def sample_up_to_n(g, n):
    """
    Sample `n` items from a group, if less than `n` are available sample all.
    """
    if len(g) <= n:
        return g
    return g.sample(n=n)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='emanjavacas/MacBERTh')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--data-file', default='./eval_set_ing.csv')
    parser.add_argument('--test-files', nargs='+')
    parser.add_argument('--output-dir', default='/tmp/results')
    parser.add_argument('--max-per-class', type=int, default=np.inf)
    parser.add_argument('--results-path', default='results.parquet')
    parser.add_argument('--level', default='level-3')
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        do_eval=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True)

    # prepare data
    # df = pd.read_csv('./eval_set_ing.csv')
    df = pd.read_csv(args.data_file)
    lhs, targets, rhs = df['lhs-clean'].fillna(''), df['ing'].fillna(''), df['rhs-clean'].fillna('')
    labels = df[args.level].values
    label_mapping = {key: idx for idx, key in enumerate(sorted(set(labels)))}
    mapped_labels = list(map(label_mapping.get, labels))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    sents, spans = read_data(tokenizer, lhs, targets, rhs)


    def get_dataset(sents, spans, labels=None):
        dataset = {'text': sents, 'spans': spans}
        # for the test set we don't have labels
        if labels is not None:
            dataset['label'] = labels
        dataset = Dataset.from_dict(dataset)

        return dataset.map(
            lambda examples: tokenizer(examples['text'], truncation=True, max_length=512),
            batched=True
        ).remove_columns('text')

    # these are datasets for which we want to output predictions
    test_datasets = {}
    for path in args.test_files:
        test = pd.read_csv(path)
        test_sents, test_spans = read_data(
            tokenizer, 
            test['lhs-clean'].fillna(''), 
            test['ing'].fillna(''),
            test['rhs-clean'].fillna(''))
        test_datasets[path] = get_dataset(
            np.array(test_sents), np.array(test_spans))

    folds = []
    test_preds = collections.defaultdict(list)

    for fold, (train, dev) in enumerate(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=1001
            ).split(sents, labels)):

        # this relates to the experiments where we limit the amount of data per label
        if args.max_per_class < np.inf:
            train = pd.DataFrame(
                {'labels': np.array(mapped_labels)[train], 'index': train}
            ).groupby('labels').apply(
                lambda g: sample_up_to_n(g, args.max_per_class)
            ).reset_index(drop=True)['index'].values

        train_dataset = get_dataset(
            np.array(sents)[train], 
            np.array(spans)[train], 
            np.array(mapped_labels)[train])

        dev_dataset = get_dataset(
            np.array(sents)[dev], 
            np.array(spans)[dev], 
            np.array(mapped_labels)[dev])

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(set(labels)))
        # this is needed, since we have expanded the tokenizer to incorporate
        # the target special token [TGT]
        model.resize_token_embeddings(len(tokenizer))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            # early stopping
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

        trainer.train()
        # these are actually the logits
        preds, _, _ = trainer.predict(dev_dataset)

        scores = np.max(softmax(preds, axis=1), axis=1)
        preds = np.argmax(preds, axis=1)
        folds.append({'index': dev, 'preds': preds, 'fold': fold, 'scores': scores})

        for test in test_datasets:
            preds, _, _ = trainer.predict(test_datasets[test])
            test_preds[test].append(
                {'preds': np.argmax(preds, axis=1),
                 'scores': np.max(softmax(preds, axis=1), axis=1),
                 'fold': fold})

    if len(test_datasets) != 0:
        for test, preds in test_preds.items():
            infix = '.'.join(os.path.basename(test).split('.')[:-1])
            output_path = '.'.join(args.results_path.split('.')[:-1]) + \
                ".test={}.parquet".format(infix)
            pd.concat([pd.DataFrame(fold) for fold in preds]).to_parquet(output_path)

    pd.concat([pd.DataFrame(fold) for fold in folds]).to_parquet(args.results_path)

