import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments,  BertForMaskedLM
import torch
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import numpy as np
import pickle
import os
import logging
import sys
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput
transformers.set_seed(1)
"""
    MODELS ON MELD & EmoryNLP
    MELD:
        DPrtCompleteCometFinal(Complete Dprt Model)
        
"""


class DPrtCompleteCometFinalCLS(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    Use [cls] for classification
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3, dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512

        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        #self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        self.tf_e = nn.LSTM(self.dim*2, self.dim, 4, bidirectional=True, batch_first=True)
        # model for p_p
        #self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.tf_p = nn.LSTM(self.dim * 2, self.dim, 4, bidirectional=True, batch_first=True)
        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)
        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()

        self.dense = nn.Linear(1024, 1024)
        self.smx = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(1024)
        self.decoder = nn.Linear(1024, 7, bias=False)  # classification
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)[0]  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)[0]
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # batch, seq_len, dim
        cls = sequence_output[:, 0, :]  # batch, dim
        cls = self.layer_norm(self.dense(cls))
        cls = self.gelu(cls)
        prediction_scores = self.smx(self.decoder(cls))
        #prediction_scores = self.lm_head(sequence_output)
        loss = None
        #masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, 7), labels.view(-1))

        return loss, prediction_scores


class DPrtNaiveSimple(RobertaForMaskedLM):
    """
    Use less fake tokens
    """
    def __init__(self, config, prp_l=4, prp_r=4,  dropout=0.1):
        super().__init__(config)
        self.prp_l = prp_l
        self.prp_r = prp_r  # [cls] sentence(t-1) [e] [e] [p] [p] <mask> sentence(t) [p] [p] [e] [e] sentence(t+1) [sep]
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)
        self.lineu = nn.Linear(1024 + 342 + 300, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l+self.prp_r))
        self.bi_lstm_u = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv
        p = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l + 1)] +
                                       [i for i in range(input_ids.shape[1] - self.prp_r - 1,
                                                         input_ids.shape[1]-1)]]
                                      * input_ids.shape[0]).cuda())  # (batch, prp_l+prp_r, self.dim)
        pos = [i for i in range(1, self.prp_l + 1)] + [i for i in range(input_ids.shape[1]-self.prp_r-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        p_E = self.bi_lstm_u(U)[0]  # 生成经过Bi-LSTM后的话语向量  (batch, seq_len, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        # print(input_ids.shape)
        # print(pos_e)
        # print(p.shape)
        # sys.exit(0)
        G = self.linee(G).reshape(p.shape)  # (batch, prp_l+prp_r, self.dim)
        p = torch.cat((p, G), dim=-1)

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]  # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


### MELD ###
class DPrtCompleteCometFinalRNN(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        #self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        self.tf_e = nn.LSTM(self.dim*2, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_p
        #self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.tf_p = nn.LSTM(self.dim * 2, self.dim, 2, bidirectional=True, batch_first=True)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)[0]  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)[0]
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalLeft(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    Left only
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        #self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        #self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)

        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*self.prp_l_e)
        self.linep = nn.Linear(1024, self.dim * self.prp_l_p)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        #p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        #p_e_r = self.emb(torch.LongTensor(
        #    [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
        #    input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = p_e_l
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e, 2*self.dim)
        # modeling [p]
        p_p = p_p_l
        #print(p_p.shape, r.shape)
        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        p = torch.cat((G, p_p), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]
        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)
        inputs_embeds[:, pos, :] = p
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinal(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalWOContext(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv
        X = torch.randn(X.shape).cuda()
        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalWOCom(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv
        #X = torch.randn(X.shape).cuda()
        x = torch.randn(x.shape).cuda()
        r = torch.randn(r.shape).cuda()
        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalWO(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024+342+300, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv
        X = torch.randn(X.shape).cuda()
        x = torch.randn(x.shape).cuda()
        r = torch.randn(r.shape).cuda()
        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


### EmoryNLP ###
class DPrtCompleteCometFinalIEMRNN(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        # self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        # self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        self.tf_e = nn.LSTM(self.dim * 2, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_p
        # self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        # self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.tf_p = nn.LSTM(self.dim * 2, self.dim, 2, bidirectional=True, batch_first=True)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)[0]  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)[0]
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [27333, 5074, 7974, 6378, 2283, 8164]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalIEM(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [27333, 5074, 7974, 6378, 2283, 8164]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalIEMWOContext(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv
        X = torch.randn(X.shape).cuda()

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [27333, 5074, 7974, 6378, 2283, 8164]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalIEMWOCom(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """

    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3, dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        # self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)

        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6 * 768, 512)
        self.line_cr = nn.Linear(3 * 768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim * (self.prp_l_e + self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        # L = int(X.shape[1])  # length of conv
        X = torch.randn(X.shape).cuda()
        #x = torch.randn(x.shape).cuda()
        #r = torch.randn(r.shape).cuda()

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e + 1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor(
            [[i for i in range(self.prp_l_e + 1, self.prp_l_p + self.prp_l_e + 1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1,
                                                             input_ids.shape[1] - self.prp_r_e - 1)]] * input_ids.shape[
                                              0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1] - 1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1] - 1)]
        pos_p = [i for i in range(self.prp_l_e + 1, self.prp_l_e + self.prp_l_p + 1)] + \
                [i for i in
                 range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1, input_ids.shape[1] - self.prp_r_e - 1)]
        pos = [i for i in range(1, self.prp_l_e + self.prp_l_p + 1)] + \
              [i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1, input_ids.shape[1] - 1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        # print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e + self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [27333, 5074, 7974, 6378, 2283, 8164]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteCometFinalIEMWO(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """

    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3, dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        # self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # model for p_p
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)

        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6 * 768, 512)
        self.line_cr = nn.Linear(3 * 768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim * (self.prp_l_e + self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        # L = int(X.shape[1])  # length of conv
        #X = torch.randn(X.shape).cuda()
        #x = torch.randn(x.shape).cuda()
        #r = torch.randn(r.shape).cuda()

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e + 1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor(
            [[i for i in range(self.prp_l_e + 1, self.prp_l_p + self.prp_l_e + 1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1,
                                                             input_ids.shape[1] - self.prp_r_e - 1)]] * input_ids.shape[
                                              0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1] - 1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1] - 1)]
        pos_p = [i for i in range(self.prp_l_e + 1, self.prp_l_e + self.prp_l_p + 1)] + \
                [i for i in
                 range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1, input_ids.shape[1] - self.prp_r_e - 1)]
        pos = [i for i in range(1, self.prp_l_e + self.prp_l_p + 1)] + \
              [i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p - 1, input_ids.shape[1] - 1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        # print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e + self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [27333, 5074, 7974, 6378, 2283, 8164]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits
###############################################################


class DPrtCompleteComet(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=8)
        self.tf = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.bi_lstm_u = nn.LSTM(self.dim, self.dim//2, 2, bidirectional=True, batch_first=True)
        self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, self.dim*(self.prp_l_e+self.prp_r_e))
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(1024 + 342 + 300, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, 1, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        U = self.tf(U)
        p_E = self.bi_lstm_u(U)[0]  # 生成经过Bi-LSTM后的话语向量  (batch, seq_len, dim)
        #print(self.line_cx(x).shape, p_E.shape)
        p_E = torch.cat((p_E, self.line_cx(x)), dim=-1) # (batch, seq, 2*dim)

        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), dim)
        G = p_E[:, loc, :]  # (batch, 1, dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)

        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r).reshape(p_p.shape)
        p_p = torch.cat((p_p, p_cp), dim=-1)
        p_p = self.line_crp(p_p)
        p_p = self.tf_p(p_p)  # (batch, prp_l_p+prp_r_p, self.dim)

        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, self.dim)
        p = self.bi_lstm(p)[0]
        # print(input_ids.shape)
        # print(pos_e)
        # print(p.shape)
        # sys.exit(0)

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrtCompleteSimple(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation
    """
    def __init__(self, config, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=8)
        self.tf = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.bi_lstm_u = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(1024 + 342 + 300, 512)

        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        U = self.tf(U)
        p_E = self.bi_lstm_u(U)[0]  # 生成经过Bi-LSTM后的话语向量  (batch, seq_len, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)

        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        p_p = self.tf_p(p_p)  # (batch, prp_l_p+prp_r_p, self.dim)

        p_e = torch.cat((p_e_l, p_e_r), dim=1)

        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, self.dim)
        p = self.bi_lstm(p)[0]
        # print(input_ids.shape)
        # print(pos_e)
        # print(p.shape)
        # sys.exit(0)

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPrt(RobertaForMaskedLM):
    def __init__(self, config, prp_e=2, prp_p=2, dropout=0.1):
        super().__init__(config)
        self.prp_e = prp_e
        self.prp_p = prp_p  # [cls] sentence(t-1) [e] [e] [p] [p] <mask> sentence(t) [p] [p] [e] [e] sentence(t+1) [sep]
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)
        # model for [e]
        self.linee = nn.Linear(1024+342+300, 1024)
        self.lstm_e = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024) , the features of conversation at each time step
            index_mask,  # location that indicating current sentence in the conversation defaults to (batch,seq)
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        loc = index_mask.argmax(-1).squeeze(0)
        L = loc.shape[1]  # length of conv
        p = self.emb(torch.LongTensor([[i for i in range(1, 2*loc*self.prp_l + 1)] +
                                       [i for i in range(input_ids.shape[1] - 2*(L-loc-1)*self.prp_r, input_ids.shape[1])]]
                                      * input_ids.shape[0]).cuda())
        # 初始化embedding成(batch,prp_len,dim)形状的向量
        p = self.bi_lstm(p)[0]  # 生成经过Bi-LSTM后的向量
        p = self.gelu(self.line1(p))
        p = self.gelu(self.line2(p))
        p = self.gelu(self.line3(p))  # (batch, prp_len, dim)
        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)
        inputs_embeds[:, 1:self.prp_l + 1, :] = p[:, :self.prp_l]  # 修改赋值
        inputs_embeds[:, input_ids.shape[1] - self.prp_r - 1:input_ids.shape[1] - 1, :] = p[:, self.prp_l:]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        #print(return_dict)
        #if not return_dict:
            #output = (prediction_scores,) + outputs[2:]
            #return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        logit = mlm_out.logits[:, self.prp_l+1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]  # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class MIXPrt(RobertaForMaskedLM):
    def __init__(self, config, prp_l=3, prp_r=3):
        super().__init__(config)
        self.prp_l = prp_l
        self.prp_r = prp_r  # [cls] e e e <mask> sentence e e e [sep]
        self.dim = 512
        self.emb = nn.Embedding(512, self.dim)  # convert dictonary size to maximum
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        self.b_emb = self.get_input_embeddings()
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)

    def forward(
            self,
            input_ids=None,  # [CLS] e(p) e(p) [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):

        p = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l + 1)] +
                                       [i for i in range(input_ids.shape[1] - self.prp_r, input_ids.shape[1])]]
                                      * input_ids.shape[0]).cuda())
        # 初始化embedding成(batch,prp_len,dim)形状的向量
        p = self.bi_lstm(p)[0]  # 生成经过Bi-LSTM后的向量
        p = self.gelu(self.line1(p))
        p = self.gelu(self.line2(p))
        p = self.gelu(self.line3(p))  # (batch, prp_len, dim)
        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)
        inputs_embeds[:, 1:self.prp_l + 1, :] = p[:, :self.prp_l]  # 修改赋值
        inputs_embeds[:, input_ids.shape[1] - self.prp_r - 1:input_ids.shape[1] - 1, :] = p[:, self.prp_l:]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        #print(return_dict)
        #if not return_dict:
            #output = (prediction_scores,) + outputs[2:]
            #return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        logit = mlm_out.logits[:, self.prp_l+1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]  # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


### Different PLM ###
class DPFPLM(RobertaForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        #self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        self.tf_e = nn.LSTM(self.dim*2, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_p
        #self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.tf_p = nn.LSTM(self.dim * 2, self.dim, 2, bidirectional=True, batch_first=True)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)[0]  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)[0]
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits


class DPTBert(BertForMaskedLM):
    """
    Use individual fake tokens for prompt and conversation, and using Comet features to initialize Prompt tokens
    """
    def __init__(self, config, u_dim=1024, prp_l_e=3, prp_r_e=3, prp_l_p=3, prp_r_p=3,  dropout=0.1):
        super().__init__(config)
        self.prp_l_e = prp_l_e
        self.prp_r_e = prp_r_e
        self.prp_l_p = prp_l_p
        self.prp_r_p = prp_r_p
        self.dim = 512
        self.b_emb = self.get_input_embeddings()  # transfer input_ids to embeddings
        self.emb = nn.Embedding(512, self.dim)  # convert dictionary size to maximum
        #self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_e
        #self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        self.tf_e = nn.LSTM(self.dim*2, self.dim, 2, bidirectional=True, batch_first=True)
        # model for p_p
        #self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2*self.dim, nhead=8)
        #self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        self.tf_p = nn.LSTM(self.dim * 2, self.dim, 2, bidirectional=True, batch_first=True)


        self.bi_lstm = nn.LSTM(2*self.dim, self.dim, 2, bidirectional=True, batch_first=True)
        #self.bi_lstm_p = nn.LSTM(self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        self.line_cx = nn.Linear(6*768, 512)
        self.line_cr = nn.Linear(3*768, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.linee = nn.Linear(1024, self.dim*(self.prp_l_e+self.prp_r_e))
        self.linep = nn.Linear(1024, self.dim * (self.prp_l_p + self.prp_r_p))
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            X,  # shape of (batch, seq, dim=1024+342+300) , the features of conversation at each time step
            x,  # Common for speaker, use to update [e]  (batch, seq, 6*768)
            r,  # Common for listener, use to update [p]  (batch, seq, 3*768)
            loc,
            input_ids=None,  # [CLS] [e][e][p][p] [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):
        # loc = index_mask.argmax(-1).squeeze(0)  # cur location in the conv
        #L = int(X.shape[1])  # length of conv

        p_e_l = self.emb(torch.LongTensor([[i for i in range(1, self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_l = self.emb(torch.LongTensor([[i for i in range(self.prp_l_e+1, self.prp_l_p+self.prp_l_e+1)]] * input_ids.shape[0]).cuda())
        p_p_r = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.prp_r_e - self.prp_r_p-1, input_ids.shape[1] - self.prp_r_e -1)]] * input_ids.shape[0]).cuda())
        p_e_r = self.emb(torch.LongTensor(
            [[i for i in range(input_ids.shape[1] - self.prp_r_e-1, input_ids.shape[1]-1)]] *
            input_ids.shape[0]).cuda())
        pos_e = [i for i in range(1, self.prp_l_e + 1)] + \
                [i for i in range(input_ids.shape[1] - self.prp_r_e - 1, input_ids.shape[1]-1)]
        pos_p = [i for i in range(self.prp_l_e+1, self.prp_l_e+self.prp_l_p+1)] + \
                [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-self.prp_r_e-1)]
        pos = [i for i in range(1, self.prp_l_e+self.prp_l_p+1)] + \
              [i for i in range(input_ids.shape[1]-self.prp_r_e-self.prp_r_p-1, input_ids.shape[1]-1)]
        U = self.lineu(X)
        # modeling [e]
        p_E = torch.cat((U, self.line_cx(x)), dim=-1)  # (batch, seq, 2*dim)
        p_E = self.tf_e(p_E)[0]  # (batch, seq, 2*dim)
        p_E = self.dropout(self.gelu(self.line1(p_E)))
        p_E = self.gelu(self.line2(p_E))
        p_E = self.dropout(self.gelu(self.line3(p_E)))  # (batch, (seq_len), 2*dim)
        G = p_E[:, loc, :]  # (batch, 1, 2*dim)
        p_e = torch.cat((p_e_l, p_e_r), dim=1)
        G = self.linee(G).reshape(p_e.shape)  # (batch, prp_l_e+prp_r_e, self.dim)
        G = torch.cat((G, p_e), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)
        # modeling [p]
        p_p = torch.cat((p_p_l, p_p_r), dim=1)
        #print(p_p.shape, r.shape)

        p_cp = self.line_cr(r)  # (batch, seq, dim)
        p_P = torch.cat((U, p_cp), dim=-1)
        p_P = self.tf_p(p_P)[0]
        P = p_P[:, loc, :]
        P = self.linep(P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)  # (batch, prp_l_e+prp_r_e, 2*self.dim)

        # modeling for both
        E_l = G[:, 0:self.prp_l_e, :]
        E_r = G[:, self.prp_l_e:(self.prp_l_e+self.prp_r_e), :]
        p = torch.cat((E_l, p_p, E_r), dim=1)  # (batch, prp_l_e+prp_l_p+prp_r_p+prp_r_e, 2*self.dim)
        p = self.bi_lstm(p)[0]

        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)

        inputs_embeds[:, pos, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        print(prediction_scores.shape)
        print(labels.shape)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # print(return_dict)
        # if not return_dict:
        # output = (prediction_scores,) + outputs[2:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # print(mlm_out.logits.shape)
        # print(loc)
        logit = mlm_out.logits[:, self.prp_l_e + self.prp_l_p + 1][:, [8699,  4474,  3571, 12039,  6569, 12721,  4963]]
        # (batch, n_class)
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits

class mlp_classifier(nn.Module):
    def __init__(self, input_dim=1024+300+342, hidden_dim=512, n_classes=7, dropout=0.1):
        super(mlp_classifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, U_l, U_a, U_v, qmask, umask, att2=True):
        """
        :param U: batch sentences(batch, seq_len, input_dim)
        :return: log_prob: (batch, seq, 7)
        """
        U = torch.cat((U_l, U_a, U_v), dim=-1)
        U = F.gelu(self.mlp1(U))
        U = self.dropout(U)
        U = self.mlp2(U)
        log_prob = F.log_softmax(F.gelu(U), dim=-1)
        return log_prob


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_a, orig_d_v, d_l=64, d_a=64, d_v=64, vonly=True, aonly=True, lonly=True,
                 num_heads=4, layers=2, attn_dropout=0.1, dropout=0.05, attn_mask=False, output_dim=32):
        super(MULTModel, self).__init__()
        """
        Construct a MulT model.
        """

        self.orig_d_l, self.orig_d_a, self.orig_d_v = orig_d_l, orig_d_a, orig_d_v
        self.d_l, self.d_a, self.d_v = d_l, d_a, d_v
        self.vonly, self.aonly, self.lonly = vonly, aonly, lonly
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.attn_dropout_a = attn_dropout
        self.attn_dropout_v = attn_dropout
        self.relu_dropout = dropout
        self.res_dropout = dropout
        self.out_dropout = dropout
        self.embed_dropout = dropout
        self.dropout = dropout
        self.attn_mask = attn_mask
        #combined_dim = self.d_l + self.d_a + self.d_v
        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = orig_d_l + orig_d_a + orig_d_v  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        x_l = F.dropout(x_l.transpose(1, 2), p=self.dropout)
        x_a = x_a.transpose(1, 2)  # (seq, batch, dim)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # (dim, seq, batch)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction
            last_h_l = last_hs = h_ls # Take the last output for prediction
        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            last_h_a = last_hs = h_as
        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            last_h_v = last_hs = h_vs
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=2)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        return output # (batch, seq, output_dim)


class Main_net(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, orig_d_l=1024, orig_d_a=300, orig_d_v=342, n_classes=7, dropout=0.1):
        """

        :param input_dim: Actually the output dimension of MULTModel
        :param hidden_dim:
        :param orig_d_l:
        :param orig_d_a:
        :param orig_d_v:
        :param n_classes:
        :param dropout:
        """
        super(Main_net, self).__init__()
        self.MULTModel = MULTModel(orig_d_l=orig_d_l, orig_d_a=orig_d_a, orig_d_v=orig_d_v)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(2*input_dim, hidden_dim)
        #self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, U_l, U_a, U_v):
        """
        3 modal input of shape (batch, seq, dim)
        """
        MULTOutput = self.MULTModel(U_l, U_a, U_v)  # (batch, seq, dim_v+dim_l+dim_a)
        U = torch.cat((U_l, U_a, U_v), dim=-1)  # (batch, seq, dim_v+dim_l+dim_a)
        U = torch.cat((U, MULTOutput.permute(1, 0, 2)), dim=-1) # (batch, seq, 2*(dim_v+dim_l+dim_a))
        #U = MULTOutput.permute(1, 0, 2)
        U = self.mlp2(F.gelu(self.mlp1(U)))
        U = self.dropout(U)
        log_prob = F.log_softmax(F.gelu(U), dim=-1)
        return log_prob


class Main_net1(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, D_g=144, D_p=144, D_e=64, D_h=64, D_a=64,
                 orig_d_l=1024, orig_d_a=300, orig_d_v=342, n_classes=7, dropout=0.1):
        super(Main_net1, self).__init__()
        self.MULTModel = MULTModel(orig_d_l=orig_d_l, orig_d_a=orig_d_a, orig_d_v=orig_d_v)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        self.D_m = D_m = orig_d_l + orig_d_a +orig_d_v
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn_f = DialogueRNN(2 * D_m, D_g, D_p, D_e, D_a=D_a, dropout=dropout+0.1)
        self.dialog_rnn_r = DialogueRNN(2 * D_m, D_g, D_p, D_e, D_a=D_a, dropout=dropout+0.1)
        self.linear = nn.Linear(2 * D_e, 2 * D_h)
        self.smax_fc = nn.Linear(2 * D_h, n_classes)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)  # batch, seq, dim
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U_l, U_a, U_v, qmask, umask, att2=True):
        """
        3 modal input of shape (batch, seq, dim)
        """
        MULTOutput = self.MULTModel(U_l, U_a, U_v)  # (batch, seq, dim_v+dim_l+dim_a)
        U = torch.cat((U_l, U_a, U_v), dim=-1)  # (batch, seq, dim_v+dim_l+dim_a)
        U = torch.cat((U, MULTOutput.permute(1, 0, 2)), dim=-1)  # (batch, seq, 2*(dim_v+dim_l+dim_a))
        U = U.permute(1, 0, 2)  # (seq, batch, 2*(dim_v+dim_l+dim_a))
        #print(U.shape)
        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)

        rev_U = self._reverse_seq(U, umask)

        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)

        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)

        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), dim=-1)
        #print(log_prob.shape)
        return log_prob


class PrtAttentionBlock(nn.Module):
    def __init__(self, dim_in_e=1024, dim_in_p=1024, dim_k=1024, dim_v=1024, num_heads=16, ):
        super(PrtAttentionBlock, self).__init__()
        pass


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_in_e=1024, dim_in_p=1024, dim_k=1024, dim_v=1024, num_heads=16):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in_e = dim_in_e
        self.dim_in_p = dim_in_p
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in_e, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in_p, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in_p, dim_v, bias=False)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(dim_k // num_heads))

    def forward(self, X_e, X_p):
        """
        Cross Attention from X_p to X_e
        :param X_e: (batch, seq, dim)
        :param X_p:
        :return:
        """
        batche, ne, dim_in_e = X_e.shape
        batchp, np, dim_in_p = X_p.shape
        dk = self.dim_k // self.num_heads  # dim_k of each head
        dv = self.dim_v // self.num_heads  # dim_v of each head

        q = self.linear_q(X_e).reshape(batche, ne, self.num_heads, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(X_p).reshape(batchp, np, self.num_heads, dk).transpose(1, 2)
        v = self.linear_v(X_p).reshape(batchp, np, self.num_heads, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, n_heads, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n_heads, n, n
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        attout = att.transpose(1, 2).reshape(batchp, np, self.dim_v)  # batch, n, dim_v
        return attout


class Attr(nn.Module):
    def __init__(self, dim=1024, hidden=64):
        super(Attr, self).__init__()
        self.Wq = nn.Linear(dim, hidden)
        self.Wk = nn.Linear(dim, hidden)
        self.Wv = nn.Linear(dim, hidden)
        self.hidden = hidden
        self.sfm = nn.Softmax(dim=-1)

    def forward(self, X_e, X_p):
        """
        Attention from X_p to X_e
        :param X_e: (seq, dim)
        :param X_p: (seq, dim)
        :return:
        """
        Query = self.Wq(X_e)
        Key = self.Wk(X_p)
        Value = self.Wv(X_p)
        return torch.mm(self.sfm(torch.mm(Query, torch.t(Key))/torch.sqrt(torch.tensor(self.hidden, dtype=torch.float32))), Value)