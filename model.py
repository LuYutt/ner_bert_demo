# coding=utf-8
"""
    @project: bert_share
    @Author：LuYutang
    @file： model.py
    @date：2024/1/29 11:44
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torchcrf import CRF
from os.path import exists

# 超参数
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001

# 加载BERT模型和tokenizer
MODEL_PATH = 'D:/transformer_file/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert_model = BertModel.from_pretrained(MODEL_PATH)


class EntityModel(nn.Module):
    def __init__(self, bert_model, hidden_size, num_tags):
        super(EntityModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(bidirectional=True, input_size=hidden_size, hidden_size=hidden_size // 2,
                              batch_first=True)
        # 忘加层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output = self.layer_norm(lstm_output)

        logits = self.fc(lstm_output)
        if labels is not None:
            # print("Attention mask before CRF:", attention_mask.byte())
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            return loss
        else:
            tags = self.crf.decode(logits, mask=attention_mask.byte())
            return tags
