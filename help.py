# coding=utf-8
"""
    @project: bert_share
    @Author：LuYutang
    @file： help.py
    @date：2024/1/29 11:52
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

MODEL_PATH = 'D:/transformer_file/bert-base-chinese'
# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

label2id = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    'O': 3,
    'B_T': 4,
    'I_T': 5,
    'B_LOC': 6,
    'I_LOC': 7,
    'B_ORG': 8,
    'I_ORG': 9,
    'B_PER': 10,
    'I_PER': 11
}

id2label = {
    0: '[PAD]',
    1: '[CLS]',
    2: '[SEP]',
    3: 'O',
    4: 'B_T',
    5: 'I_T',
    6: 'B_LOC',
    7: 'I_LOC',
    8: 'B_ORG',
    9: 'I_ORG',
    10: 'B_PER',
    11: 'I_PER'
}

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


def tokenize_and_preserve_labels(text, labels):
    tokenized_text = []
    token_labels = []
    for word, label in zip(text, labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_text.extend(tokenized_word)
        token_labels.extend([label] * n_subwords)

    return tokenized_text, token_labels


def pad_sequences(sequences, max_len, padding_value=0):
    padded_sequences = torch.zeros((len(sequences), max_len)).long()
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len <= max_len:
            padded_sequences[i, :seq_len] = torch.tensor(seq)
        else:
            padded_sequences[i, :] = torch.tensor(seq[:max_len])
    return padded_sequences


def train(model, optimizer, train_dataloader):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        #print(type(batch))
        #print(batch)
        #print(batch[0])
        #print(batch[1])
        #print(batch[2])

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        loss = model(input_ids, attention_mask, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss


def evaluate(model, eval_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader)):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

    avg_eval_loss = total_loss / len(eval_dataloader)
    return avg_eval_loss


def predict(model, text):
    model.eval()
    tokenized_text = tokenizer.tokenize(text)
    #print("token",tokenized_text)
    tokenized_text_with_labels = [(token, 'O') for token in tokenized_text]
    #print("tokenized_text_with_labels",tokenized_text_with_labels)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    attention_mask = torch.ones_like(input_ids)
    #print("attention_mask",attention_mask)

    with torch.no_grad():
        #print("device",device)
        tags = model(input_ids.to(device), attention_mask.to(device))
    print("tags0",tags[0])

    tag_labels = [id2label[tag] for tag in tags[0]]
    return list(zip(tokenized_text, tag_labels))
