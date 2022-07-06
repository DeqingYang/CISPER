import prt_utils
from prt_utils import *
from prt_model import *
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments, Trainer, \
    BertTokenizer, BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
import torch
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader
import time
import warnings
import sys
warnings.filterwarnings("ignore")



def train_or_eval_dprt_Complete_Comet_Final_erm(epoch, emotion2label, label2emotion, id2label, dataset, comet, tokenizer, model,
                      optimizer, dataloader,  prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, train=True):
    if train:

        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                cur_emotions = dataset.videoLabels[vid]
                #cur_speakers = dataset.videoSpeakers[vid]
                cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                cur_text4 = torch.FloatTensor(dataset.roberta4[vid])
                U = (cur_text1 + cur_text2 + cur_text3 + cur_text4)/4
                #cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                #cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                #U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                    #print(cur_com[:6, ].shape)
                    com_x = cur_com[:6, ].reshape(1, -1, 6*768).cuda()  # (batch, seq,6*768)
                    com_r = cur_com[6:, ].reshape(1, -1, 3*768).cuda()
                    #print('com:',  com_r.shape)
                    x = ' <mask>'*(prp_l_e+prp_l_p+1)+' '+sent+' <mask>'*(prp_r_e+prp_r_p)

                    y = ' <mask>'*(prp_l_e+prp_l_p)+' '+cur_emotion+' '+sent+' <mask>'*(prp_r_e+prp_r_p)


                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_l_e+prp_l_p+1)] = -100
                    label[:, (-prp_r_e-prp_r_p-1):-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, prp_l_e+prp_l_p+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(id2label[pred] if pred in id2label else emotion2label['neutral'])
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                    cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                    cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                    cur_text4 = torch.FloatTensor(dataset.roberta4[vid])
                    U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4
                    # cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    # cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    # print(type(cur_text))
                    # U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.cuda()
                    U = U.unsqueeze(0)
                    conv = dataset.videoSentence[vid]
                    length = len(conv)
                    # print(f'Conversation{vid}')
                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                        # print(cur_com[:6, ].shape)
                        com_x = cur_com[:6, ].reshape(1, -1, 6 * 768).cuda()  # (batch, seq,6*768)
                        com_r = cur_com[6:, ].reshape(1, -1, 3 * 768).cuda()
                        x = ' <mask>' * (prp_l_e + prp_l_p + 1) + ' ' + sent + ' <mask>' * (prp_r_e + prp_r_p)
                        # print(x)
                        y = ' <mask>' * (prp_l_e + prp_l_p) + ' ' + cur_emotion + ' ' + sent + ' <mask>' * (
                                    prp_r_e + prp_r_p)
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_l_e + prp_l_p + 1)] = -100
                        label[:, (-prp_r_e - prp_r_p - 1):-1] = -100
                        # print(label)
                        ground_truth.append(cur_emotions[i])
                        label = label.cuda()
                        _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask,
                                                labels=label)
                        logits = logits[:, prp_l_e + prp_l_p + 1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(id2label[pred] if pred in id2label else emotion2label['neutral'])
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_prev_next(epoch, prompt, label2emotion, id2label, dataset,
                                   tokenizer, model, optimizer, dataloader, prev_len=1, after_len=1, train=True):
    if train:
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                conv = dataset.videoSentence[vid]
                length = len(conv)
                #cur_com = com[vid]
                for i in range(length):
                    sent = conv[i].replace('x92', "'")

                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = sent + prompt
                    for p in range(1, prev_len+1):
                        if i-p >=0:
                            x = conv[i-p].replace('x92', "'") + x
                    for a in range(1, after_len+1):
                        if i+a < length:
                            x += conv[i+a].replace('x92', "'")
                    y = x.replace('<mask>', cur_emotion)
                    input = tokenizer(x, return_tensors="pt")
                    loc = input['input_ids'].argmax()
                    input_ids = input['input_ids'].cuda()
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    ground_truth.append(int(label.squeeze()[loc]))
                    label = label.cuda()
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    loss = output.loss
                    logits = output.logits
                    loss_val.append(loss.item())
                    loss.backward()
                    logits = logits.data.cpu()
                    preds.append(int(logits[:, loc].argmax(-1)))
            optimizer.step()
            optimizer.zero_grad()
        losses.append(np.mean(loss_val))
        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    length = len(conv)

                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        x = sent + prompt
                        for p in range(1, prev_len + 1):
                            if i - p >= 0:
                                x = conv[i - p].replace('x92', "'") + x
                        for a in range(1, after_len + 1):
                            if i + a < length:
                                x += conv[i + a].replace('x92', "'")

                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        y = x.replace('<mask>', cur_emotion)
                        input = tokenizer(x, return_tensors="pt")
                        loc = input['input_ids'].argmax()

                        input_ids = input['input_ids'].cuda()
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        ground_truth.append(int(label.squeeze()[loc]))
                        output = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = output.logits
                        logits = logits.data.cpu()
                        preds.append(int(logits[:, loc].argmax(-1)))
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_mix(epoch, prompt, label2emotion, id2label, dataset, tokenizer, model,
                      optimizer, dataloader,  left_prp=3, right_prp=3, prev_len=1, after_len=1, train=True):
    if train:
        loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                conv = dataset.videoSentence[vid]
                length = len(conv)
                cur_com = com[vid]
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*(left_prp+1)+' '+sent+' <mask>'*(right_prp)
                    #print(x)
                    y = ' <mask>'*(left_prp)+' '+cur_emotion+' '+sent+' <mask>'*(right_prp)
                    input = tokenizer(x, return_tensors="pt")
                    loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:left_prp+1] = -100
                    label[:, -right_prp-1:-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    #loss = output.loss
                    #logits = output.logits[:, loc][:, emotionids]
                    loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    preds.append(int(logits.argmax(-1)))
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    length = len(conv)
                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * (left_prp + 1) + sent + ' <mask>' * (right_prp)
                        y = ' <mask>' * (left_prp) + ' '+cur_emotion + ' ' + sent + ' <mask>' * (right_prp)
                        input = tokenizer(x, return_tensors="pt")
                        loc = left_prp + 1
                        input_ids = input['input_ids'].cuda()
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        ground_truth.append(cur_emotions[i])
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits.data.cpu()
                        preds.append(int(logits.argmax(-1)))
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_mix1(epoch, prompt, label2emotion, label2id, dataset, tokenizer, model,
                      optimizer, dataloader,  left_prp=3, right_prp=3, prev_len=1, after_len=1, train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                conv = dataset.videoSentence[vid]
                length = len(conv)
                cur_com = com[vid]
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*(left_prp+1)+' '+sent+' <mask>'*(right_prp)
                    #print(x)
                    y = ' <mask>'*(left_prp)+' '+cur_emotion+' '+sent+' <mask>'*(right_prp)
                    input = tokenizer(x, return_tensors="pt")
                    loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:left_prp+1] = -100
                    label[:, -right_prp-1:-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, loc]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    length = len(conv)

                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * (left_prp + 1) + sent + ' <mask>' * (right_prp)
                        y = ' <mask>' * (left_prp) + ' '+cur_emotion + ' ' + sent + ' <mask>' * (right_prp)

                        input = tokenizer(x, return_tensors="pt")
                        loc = left_prp + 1
                        input_ids = input['input_ids'].cuda()
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        ground_truth.append(cur_emotions[i])
                        _, _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits[:, loc]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_dprt(epoch, prompt, label2emotion, label2id, dataset, tokenizer, model,
                      optimizer, dataloader,  prp_e=2, prp_p=3,  train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                cur_text = torch.FloatTensor(dataset.videoText[vid])
                cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                cur_com = com[vid]
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*((prp_e+prp_p)*(i+1)+1)+' '+sent+' <mask>'*((prp_e+prp_p)*(length-i-1))
                    #print(x)
                    y = ' <mask>'*((prp_e+prp_p)*(i+1))+' '+cur_emotion+' '+sent+' <mask>'*((prp_e+prp_p)*(length-i-1))
                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_e+prp_p)*i+1] = -100
                    label[:, -(prp_e+prp_p)*(length-i-1)-1:-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, (prp_e+prp_p)*(i+1)+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    cur_text = torch.FloatTensor(dataset.videoText[vid])
                    cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.unsqueeze(0)
                    length = len(conv)

                    for i in range(length):

                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * ((prp_e + prp_p) * (i+1) + 1) + ' ' + sent + ' <mask>' * (
                                    (prp_e + prp_p) * (length - i - 1))
                        # print(x)
                        y = ' <mask>' * ((prp_e + prp_p) * (i+1)) + ' ' + cur_emotion + ' ' + sent + ' <mask>' * (
                                    (prp_e + prp_p) * (length - i - 1))
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_e + prp_p) * (i+1) + 1] = -100
                        label[:, -(prp_e + prp_p) * (length - i - 1) - 1:-1] = -100
                        ground_truth.append(cur_emotions[i])
                        _, _, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits[:, (prp_e+prp_p)*(i+1)+1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_dprt_Simple(epoch, prompt, label2emotion, label2id, dataset, tokenizer, model,
                      optimizer, dataloader,  prp_l=4, prp_r=4,  train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t % 20 == 0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                cur_text = torch.FloatTensor(dataset.videoText[vid])
                cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                cur_com = com[vid]
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*(prp_l+1)+' '+sent+' <mask>'*(prp_r)
                    #print(x)
                    y = ' <mask>'*(prp_l)+' '+cur_emotion+' '+sent+' <mask>'*(prp_r)
                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_l+1)] = -100
                    label[:, (-prp_r-1):-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, prp_l+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    cur_text = torch.FloatTensor(dataset.videoText[vid])
                    cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.unsqueeze(0)
                    length = len(conv)

                    for i in range(length):

                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * (prp_l + 1) + ' ' + sent + ' <mask>' * (prp_r)
                        # print(x)
                        y = ' <mask>' * prp_l + ' ' + cur_emotion + ' ' + sent + ' <mask>' * prp_r
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_l + 1)] = -100
                        label[:, (-prp_r - 1):-1] = -100
                        ground_truth.append(cur_emotions[i])
                        _, _, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits[:, prp_l+1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_dprt_Complete(epoch, prompt, label2emotion, label2id, dataset, tokenizer, model,
                      optimizer, dataloader,  prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                cur_text = torch.FloatTensor(dataset.videoText[vid])
                cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                cur_com = com[vid]
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*(prp_l_e+prp_l_p+1)+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    #print(x)
                    y = ' <mask>'*(prp_l_e+prp_l_p)+' '+cur_emotion+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_l_e+prp_l_p+1)] = -100
                    label[:, (-prp_r_e-prp_r_p-1):-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, prp_l_e+prp_l_p+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    cur_text = torch.FloatTensor(dataset.videoText[vid])
                    cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.unsqueeze(0)
                    length = len(conv)

                    for i in range(length):

                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * (prp_l_e + prp_l_p + 1) + ' ' + sent + ' <mask>' * (prp_r_e + prp_r_p)
                        # print(x)
                        y = ' <mask>' * (prp_l_e + prp_l_p) + ' ' + cur_emotion + ' ' + sent + ' <mask>' * (
                                    prp_r_e + prp_r_p)
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_l_e + prp_l_p + 1)] = -100
                        label[:, (-prp_r_e - prp_r_p - 1):-1] = -100
                        # print(label)
                        ground_truth.append(cur_emotions[i])
                        label = label.cuda()
                        _, loss, logits = model(U, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                        logits = logits[:, prp_l_e + prp_l_p + 1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_dprt_Complete_Comet(epoch, prompt, label2emotion, label2id, dataset, comet, tokenizer, model,
                      optimizer, dataloader,  prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                cur_text = torch.FloatTensor(dataset.videoText[vid])
                cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    cur_com = torch.tensor([cur_comet[j] for j in range(9)])

                    #print(cur_com[:6, ].shape)
                    com_x = cur_com[:6, ].reshape(1, -1, 6*768).cuda()  # (batch, seq,6*768)
                    com_r = cur_com[6:, i].reshape(1, -1, 3*768).cuda()
                    #print('com:',  com_r.shape)
                    x = ' <mask>'*(prp_l_e+prp_l_p+1)+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    #print(x)
                    y = ' <mask>'*(prp_l_e+prp_l_p)+' '+cur_emotion+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_l_e+prp_l_p+1)] = -100
                    label[:, (-prp_r_e-prp_r_p-1):-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, prp_l_e+prp_l_p+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    cur_text = torch.FloatTensor(dataset.videoText[vid])
                    cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    # print(type(cur_text))
                    U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.unsqueeze(0)
                    conv = dataset.videoSentence[vid]
                    length = len(conv)
                    # print(f'Conversation{vid}')
                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        cur_com = torch.tensor([cur_comet[j] for j in range(9)])

                        # print(cur_com[:6, ].shape)
                        com_x = cur_com[:6, ].reshape(1, -1, 6 * 768).cuda()  # (batch, seq,6*768)
                        com_r = cur_com[6:, i].reshape(1, -1, 3 * 768).cuda()
                        x = ' <mask>' * (prp_l_e + prp_l_p + 1) + ' ' + sent + ' <mask>' * (prp_r_e + prp_r_p)
                        # print(x)
                        y = ' <mask>' * (prp_l_e + prp_l_p) + ' ' + cur_emotion + ' ' + sent + ' <mask>' * (
                                    prp_r_e + prp_r_p)
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_l_e + prp_l_p + 1)] = -100
                        label[:, (-prp_r_e - prp_r_p - 1):-1] = -100
                        # print(label)
                        ground_truth.append(cur_emotions[i])
                        label = label.cuda()
                        _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask,
                                                labels=label)
                        logits = logits[:, prp_l_e + prp_l_p + 1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_dprt_Complete_Comet_Final(epoch, prompt, label2emotion, label2id, dataset, comet, tokenizer, model,
                      optimizer, dataloader,  prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                cur_text = torch.FloatTensor(dataset.videoText[vid])
                cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                #print(type(cur_text))
                U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                #print(f'Conversation{vid}')
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    cur_com = torch.tensor([cur_comet[j] for j in range(9)])

                    #print(cur_com[:6, ].shape)
                    com_x = cur_com[:6, ].reshape(1, -1, 6*768).cuda()  # (batch, seq,6*768)
                    com_r = cur_com[6:, ].reshape(1, -1, 3*768).cuda()
                    #print('com:',  com_r.shape)
                    x = ' <mask>'*(prp_l_e+prp_l_p+1)+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    #print(x)
                    y = ' <mask>'*(prp_l_e+prp_l_p)+' '+cur_emotion+' '+sent+' <mask>'*(prp_r_e+prp_r_p)
                    input = tokenizer(x, return_tensors="pt")
                    #loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(prp_l_e+prp_l_p+1)] = -100
                    label[:, (-prp_r_e-prp_r_p-1):-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, prp_l_e+prp_l_p+1]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    cur_text = torch.FloatTensor(dataset.videoText[vid])
                    cur_visual = torch.FloatTensor(dataset.videoVisual[vid])
                    cur_audio = torch.FloatTensor(dataset.videoAudio[vid])
                    # print(type(cur_text))
                    U = torch.cat((cur_text, cur_visual, cur_audio), dim=-1).cuda()
                    U = U.unsqueeze(0)
                    conv = dataset.videoSentence[vid]
                    length = len(conv)
                    # print(f'Conversation{vid}')
                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        cur_com = torch.tensor([cur_comet[j] for j in range(9)])

                        # print(cur_com[:6, ].shape)
                        com_x = cur_com[:6, ].reshape(1, -1, 6 * 768).cuda()  # (batch, seq,6*768)
                        com_r = cur_com[6:, ].reshape(1, -1, 3 * 768).cuda()
                        x = ' <mask>' * (prp_l_e + prp_l_p + 1) + ' ' + sent + ' <mask>' * (prp_r_e + prp_r_p)
                        # print(x)
                        y = ' <mask>' * (prp_l_e + prp_l_p) + ' ' + cur_emotion + ' ' + sent + ' <mask>' * (
                                    prp_r_e + prp_r_p)
                        input = tokenizer(x, return_tensors="pt")
                        # loc = left_prp+1
                        input_ids = input['input_ids'].cuda()
                        # print(input_ids)
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        label[:, 1:(prp_l_e + prp_l_p + 1)] = -100
                        label[:, (-prp_r_e - prp_r_p - 1):-1] = -100
                        # print(label)
                        ground_truth.append(cur_emotions[i])
                        label = label.cuda()
                        _, loss, logits = model(U, com_x, com_r, i, input_ids=input_ids, attention_mask=attention_mask,
                                                labels=label)
                        logits = logits[:, prp_l_e + prp_l_p + 1]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def train_or_eval_mix_erm(epoch, prompt, label2emotion, label2id, dataset, tokenizer, model,
                      optimizer, dataloader,  left_prp=3, right_prp=3, prev_len=1, after_len=1, train=True):
    if train:
        #loss_func = nn.CrossEntropyLoss()
        print(f"Training epoch: {epoch}!!!")
        model.train()
        results = []
        losses = []
        start_time = time.time()
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        for data in dataloader:
            loss_val = []
            if t%20==0:
                print(f"cur: {t}")
                print(f"totalbatch: {total_batch}")
            t += 1
            vids = data[-1]
            for vid in vids:
                cur_emotions = dataset.videoLabels[vid]
                cur_speakers = dataset.videoSpeakers[vid]
                conv = dataset.videoSentence[vid]
                length = len(conv)
                #cur_com = com[vid]
                for i in range(length):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i]
                    cur_emotion = label2emotion[cur_emotion]
                    x = ' <mask>'*(left_prp+1)+' '+sent+' <mask>'*(right_prp)
                    #print(x)
                    y = ' <mask>'*(left_prp)+' '+cur_emotion+' '+sent+' <mask>'*(right_prp)
                    input = tokenizer(x, return_tensors="pt")
                    loc = left_prp+1
                    input_ids = input['input_ids'].cuda()
                    #print(input_ids)
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:left_prp+1] = -100
                    label[:, -right_prp-1:-1] = -100
                    #print(label)
                    ground_truth.append(cur_emotions[i])
                    label = label.cuda()
                    _, loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                    logits = logits[:, loc]
                    #loss = loss_func(logits, torch.LongTensor([cur_emotions[i]]).cuda())
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()

        results.append(precision_recall_fscore_support(ground_truth, preds))
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)

        end_time = time.time()
        print(f"epoch: {epoch} avg_loss: {np.mean(loss_val)} cost time: {end_time - start_time}")
        #print(classification_report(ground_truth, preds, digits=4))
    else:
        print("Testing!!!")
        model.eval()
        results = []
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            for data in dataloader:

                if t % 20 == 0:
                    print(f"cur: {t}")
                    print(f"totalbatch: {total_batch}")
                t += 1
                vids = data[-1]
                for vid in vids:
                    cur_emotions = dataset.videoLabels[vid]
                    cur_speakers = dataset.videoSpeakers[vid]
                    conv = dataset.videoSentence[vid]
                    length = len(conv)

                    for i in range(length):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i]
                        cur_emotion = label2emotion[cur_emotion]
                        x = ' <mask>' * (left_prp + 1) + sent + ' <mask>' * (right_prp)
                        y = ' <mask>' * (left_prp) + ' '+cur_emotion + ' ' + sent + ' <mask>' * (right_prp)

                        input = tokenizer(x, return_tensors="pt")
                        loc = left_prp + 1
                        input_ids = input['input_ids'].cuda()
                        attention_mask = input['attention_mask'].cuda()
                        label = tokenizer(y, return_tensors="pt")["input_ids"]
                        ground_truth.append(cur_emotions[i])
                        _, _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits[:, loc]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses) if train else (results, None)


def start_prompt_com(path, prompt, label2emotion, id2label, com, com_template, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = RobertaForMaskedLM.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses =  train_or_eval_com(e, prompt, label2emotion, id2label, dataset, com, com_template,
                                   tokenizer, model, optimizer, train_loader, train=True)
        test_results, test_losses = train_or_eval_com(e, prompt, label2emotion, id2label, dataset, com, com_template,
                                   tokenizer, model, optimizer, test_loader, train=False)


def start_prompt_prev_next_erm(path, prompt, label2emotion, id2label, epoch=20, batch_size=32, lr=1e-5, l2=1e-2,
                               prev_len=0, after_len=0):
    print(f'prev_len={prev_len}, after_len={after_len}')
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = RobertaForMaskedLM.from_pretrained('./roberta_large_mlm')
    dataset = EmoryNLPRobertaCometDataset(path=path, split='train')
    train_loader, valid_loader, test_loader = get_proper_loaders_erm(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_prev_next(e, prompt, label2emotion, id2label, dataset,
                                   tokenizer, model, optimizer, train_loader, prev_len=0, after_len=0, train=True)
        test_results, test_losses = train_or_eval_prev_next(e, prompt, label2emotion, id2label, dataset,
                                   tokenizer, model, optimizer, test_loader, prev_len=0, after_len=0, train=False)


def start_prompt_mix(path, prompt, label2emotion, id2label, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = MIXPrt.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_mix(e, prompt, label2emotion, id2label, dataset,
                                   tokenizer, model, optimizer, train_loader, left_prp=3, right_prp=3, train=True)
        test_results, test_losses = train_or_eval_mix(e, prompt, label2emotion, id2label, dataset,
                                   tokenizer, model, optimizer, test_loader, left_prp=3, right_prp=3, train=False)


def start_prompt_mix1(path, prompt, label2emotion, label2id, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = MIXPrt.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_mix1(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, train_loader, left_prp=3, right_prp=3, train=True)
        test_results, test_losses = train_or_eval_mix1(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, test_loader, left_prp=3, right_prp=3, train=False)


def start_prompt_dprt(path, prompt, label2emotion, label2id, prp_e=2, prp_p=2, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = DPrtNaive.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_dprt(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, train_loader, prp_e=prp_e, prp_p=prp_p, train=True)
        test_results, test_losses = train_or_eval_dprt(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, test_loader, prp_e=prp_e, prp_p=prp_p, train=False)


def start_prompt_dprt_Simple(path, prompt, label2emotion, label2id, prp_l=4, prp_r=4, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = DPrtNaiveSimple.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_dprt_Simple(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, train_loader, prp_l=prp_l, prp_r=prp_r, train=True)
        test_results, test_losses = train_or_eval_dprt_Simple(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, test_loader, prp_l=prp_l, prp_r=prp_r, train=False)


def start_prompt_dprt_Complete(path, prompt, label2emotion, label2id, prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = DPrtCompleteSimple.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_dprt_Complete(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, train_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=True)
        test_results, test_losses = train_or_eval_dprt_Complete(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, test_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=False)


def start_prompt_mix_erm(path, prompt, label2emotion, label2id, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    print('p-tuning')
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = MIXPrt.from_pretrained('./roberta_large_mlm')
    dataset = EmoryNLPRobertaCometDataset(path=path, split='train')
    train_loader, valid_loader, test_loader = get_proper_loaders_erm(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_mix_erm(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, train_loader, left_prp=3, right_prp=3, train=True)
        test_results, test_losses = train_or_eval_mix_erm(e, prompt, label2emotion, label2id, dataset,
                                   tokenizer, model, optimizer, test_loader, left_prp=3, right_prp=3, train=False)


def start_prompt_dprt_Complete_Comet(path, prompt, label2emotion, label2id, prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, epoch=20, batch_size=32, lr=1e-5, l2=1e-2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    model = DPrtCompleteComet.from_pretrained('./roberta_large_mlm')
    dataset = MELDDataset(path=path, n_classes=7)
    comet = MELDComet('meld_features_comet.pkl')
    train_loader, valid_loader, test_loader = get_proper_loaders(path, batch_size=batch_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_dprt_Complete_Comet(e, prompt, label2emotion, label2id, dataset, comet,
                                   tokenizer, model, optimizer, train_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=True)
        test_results, test_losses = train_or_eval_dprt_Complete_Comet(e, prompt, label2emotion, label2id, dataset, comet,
                                   tokenizer, model, optimizer, test_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=False)


def start_prompt_dprt_Complete_Comet_Final_erm(path, emotion2label, label2emotion, id2label, prp_l_e=3, prp_l_p=3,
                                               prp_r_e=3, prp_r_p=3, epoch=20, batch_size=32, lr=1e-5, l2=1e-2, ablition='norm'):
    print('#############################################################################')
    print(f'prp_l_e={prp_l_e}, prp_l_p={prp_l_p}, prp_r_e={prp_r_e}, prp_r_p={prp_r_p}')
    print(ablition)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dataset = EmoryNLPRobertaCometDataset(path=path, split='train')
    comet = MELDComet('emorynlp_features_comet.pkl')  # 名字不影响
    train_loader, valid_loader, test_loader = get_proper_loaders_erm(path, batch_size=batch_size)

    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer')
    if ablition == 'norm':
        model = DPrtCompleteCometFinalIEM.from_pretrained('./roberta_large_mlm')
    elif ablition == 'wocontext':
        model = DPrtCompleteCometFinalIEMWOContext.from_pretrained('./roberta_large_mlm')
    elif ablition == 'wocom':
        model = DPrtCompleteCometFinalIEMWOCom.from_pretrained('./roberta_large_mlm')
    elif ablition == 'wo':
        model = DPrtCompleteCometFinalIEMWO.from_pretrained('./roberta_large_mlm')

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for e in range(epoch):
        train_results, train_losses = train_or_eval_dprt_Complete_Comet_Final_erm(e, emotion2label, label2emotion, id2label, dataset, comet,
                                   tokenizer, model, optimizer, train_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=True)
        test_results, test_losses = train_or_eval_dprt_Complete_Comet_Final_erm(e, emotion2label, label2emotion, id2label, dataset, comet,
                                   tokenizer, model, optimizer, test_loader, prp_l_e=prp_l_e, prp_l_p=prp_l_p, prp_r_e=prp_r_e,
                                                                  prp_r_p=prp_r_p, train=False)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
path = 'emorynlp_features_roberta.pkl'
com = pickle.load(open('meld_com.pkl', 'rb'))[0]

start_prompt_dprt_Complete_Comet_Final_erm(path, emotion2label_erm, label2emotion_erm, id2label_erm,  prp_l_e=3, prp_l_p=3, prp_r_e=3, prp_r_p=3, epoch=60, batch_size=64, lr=1e-5, l2=1e-2, ablition='norm')