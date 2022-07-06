import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments, Trainer
import torch
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
transformers.set_seed(1)

emotionids = [7974, 2755, 2490, 17437, 5823, 30883, 6378]
id2label = {7974: 0, 2755: 1, 2490: 2, 17437: 3, 5823: 4, 30883: 5, 6378: 6}
emotion2label = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
label2emotion = {}
for key in emotion2label.keys():
    label2emotion[emotion2label[key]] = key
label2id = {emotionids[i]: i for i in range(len(emotionids))}
############################################################################
emotionids_bert=[8699,  4474,  3571, 12039,  6569, 12721,  4963]
id2label_bert={emotionids_bert[i]:i for i in range(7)}
emotion2label_bert = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
label2emotion_bert = {}
for key in emotion2label_bert.keys():
    label2emotion_bert[emotion2label_bert[key]] = key
label2id_bertb = {emotionids_bert[i]: i for i in range(len(emotionids_bert))}
############################################################################
"""emotionids_iem = [27333, 5074, 7974, 6378, 2283, 8164]
id2label_iem = {27333: 0, 5074: 1, 7974: 2, 6378: 3, 2283: 4, 8164: 5}
emotion2label_iem = {'happy': 0, 'sad': 1, 'neutral': 2, 'anger': 3, 'excited': 4, 'frustrated': 5}"""
emotionids_iem = [5823, 17437, 7974, 6378, 2283, 8164]
id2label_iem = {5823: 0, 17437: 1, 7974: 2, 6378: 3, 2283: 4, 8164: 5}
emotion2label_iem = {'joy': 0, 'sadness': 1, 'neutral': 2, 'anger': 3, 'excited': 4, 'frustrated': 5}
label2emotion_iem = {}
for key in emotion2label_iem.keys():
    label2emotion_iem[emotion2label_iem[key]] = key
label2id_iem = {emotionids_iem[i]: i for i in range(len(emotionids_iem))}
############################################################################
emotionids_erm_bert=[6569, 5506, 9379, 8699, 12039,  3928,  3571]
id2label_erm_bert={emotionids_erm_bert[i]:i for i in range(7)}
emotion2label_erm_bert = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
label2emotion_erm_bert = {}
for key in emotion2label_erm_bert.keys():
    label2emotion_erm_bert[emotion2label_erm_bert[key]] = key
label2id_erm_bertb = {emotionids_erm_bert[i]: i for i in range(len(emotionids_erm_bert))}
############################################################################
"""emotionids_erm = [20768, 2650, 7758, 7053, 7974, 5074, 2247, 8265]
id2label_erm = {20768: 0, 2650: 1, 7758: 2, 7053: 3, 7974: 4, 5074: 5, 2247: 6, 8265: 6}
emotion2label_erm = {'joyful': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sad': 4, 'powerful': 5, 'scared': 6}"""
emotionids_erm = [5823, 7758, 7053, 7974, 17437, 2247, 2490]
id2label_erm = {5823: 0, 7758: 1, 7053: 2, 7974: 3, 17437: 4, 2247: 5,  2490: 6}
emotion2label_erm = {'joy': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sadness': 4, 'powerful': 5, 'fear': 6}
label2emotion_erm = {}
for key in emotion2label_erm.keys():
    label2emotion_erm[emotion2label_erm[key]] = key
label2id_erm = {emotionids_erm[i]: i for i in range(len(emotionids_erm))}

prompt1 = " The previous person feel [label] and I feel <mask>."
prompt2 = " The emotion of the last speaker [label] and my emotion is <mask>."
prompt = " My emotion is <mask>."
com_template = {
    'Desires': ' I desire: ',
    'ReceivesAction': ' Can receive or be affected by the action: ',
    'xAttr': ' I am seen as: ',
    'xEffect': ' As a result, I will: ',
    'xIntent': ' Because I wanted: ',
    'xWant': ' Therefore, I want: ',
}

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_proper_loaders(path, batch_size=16, valid=0, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path, n_classes=7, train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = MELDDataset(path, n_classes=7, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_proper_loaders_iem(path, batch_size=64,  num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path, 'train')
    testset = IEMOCAPDataset(path,  'test')
    #validset = IEMOCAPDataset(path, 'valid')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              #sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, None, test_loader


def get_proper_loaders_erm(path, batch_size=64,  num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset(path, 'train')
    testset = EmoryNLPRobertaCometDataset(path,  'test')
    #validset = EmoryNLPRobertaCometDataset(path, 'valid')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              #sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, None, test_loader


def get_loaders(path, batch_size=32, valid=0, num_workers=0, pin_memory=False):
    trainset = MELDDataset1(path, n_classes=7, train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset1(path, n_classes=7, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class MELDDataset1:
    def __init__(self, path, n_classes=7, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
            self.testVid = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class MELDDataset:
    def __init__(self, path, n_classes=7, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
            self.testVid = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [list(dat[i]) for i in dat]


class IEMOCAPDataset(Dataset):
    def __init__(self, path, split):
        self.videoSpeakers, self.videoLabels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.videoSentence, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        if split == 'train':
            self.keys = [x for x in self.trainIds]+[x for x in self.validIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [list(dat[i]) for i in dat]


class EmoryNLPRobertaCometDataset(Dataset):
    def __init__(self, path, split):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.videoSpeakers, self.videoLabels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.videoSentence, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]+[x for x in self.validIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [list(dat[i]) for i in dat]


class MELDComet:
    def __init__(self, path):
        self.com = pickle.load(open(path, 'rb'))
        """self.x1, self.x2, self.x3, self.x4, self.x5, self.x6,\
        self.o1, self.o2, self.o3 = pickle.load(open(path, 'rb'))"""
        '''
        ['xIntent', 'xAttr', 'xNeed', 'xWant', 'xEffect', 'xReact', 'oWant', 'oEffect', 'oReact']
        '''







