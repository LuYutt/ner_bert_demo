# coding=utf-8
"""
    @project: bert_share
    @Author：LuYutang
    @file： train.py
    @date：2024/1/29 11:58
"""

from model import *
from help import *
from tqdm import tqdm
import torch.utils.data

# 加载训练数据集
train_data = []

with open('./mydata/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    text_lines = lines[50000:80000]
with open('./mydata/train_TAG.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels_lines = lines[50000:80000]
for i in tqdm(range(len(text_lines))):
    text_line = text_lines[i].strip()
    text_label = labels_lines[i].strip()
    text_line = text_line.split()
    # text_line.insert(0, "[CLS]")
    # text_line.append("[SEP]")

    # labels
    text_label = text_label.split()
    # text_label.insert(0, "[CLS]")
    # text_label.append("[SEP]")
    train_data.append((text_line, text_label))

# 将数据集转换为模型所需的格式
train_input_ids = []
train_attention_masks = []
train_labels = []
print(train_data[0])

for words, labels in train_data:
    tokenized_text, token_labels = tokenize_and_preserve_labels(words, labels)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    attention_mask = [1] * len(input_ids)

    train_input_ids.append(input_ids)
    train_attention_masks.append(attention_mask)
    train_labels.append([label2id[label] for label in token_labels])

train_input_ids = pad_sequences(train_input_ids, MAX_LEN)
train_attention_masks = pad_sequences(train_attention_masks, MAX_LEN)
train_labels = pad_sequences(train_labels, MAX_LEN, padding_value=-1)

train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 同样地，还需要加载验证集和测试集，并将它们转换为模型所需的格式
# 加载测试数据集

eval_data = []

with open('./mydata/dev.txt', 'r', encoding='utf-8') as f:
    text_lines = f.readlines()[:1000]
with open('./mydata/dev_TAG.txt', 'r', encoding='utf-8') as f:
    labels_lines = f.readlines()[:1000]
for i in tqdm(range(len(text_lines))):
    text_line = text_lines[i].strip()
    text_label = labels_lines[i].strip()
    # print(text_line)
    text_line = text_line.split()
    # text_line.insert(0,"")[CLS]
    # text_line.append("[SEP]")

    text_label = text_label.split()
    # text_label.insert(0, "[CLS]")
    # text_label.append("[SEP]")
    eval_data.append((text_line, text_label))

# 将数据集转换为模型所需的格式
eval_input_ids = []
eval_attention_masks = []
eval_labels = []

for words, labels in eval_data:
    tokenized_text, token_labels = tokenize_and_preserve_labels(words, labels)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    attention_mask = [1] * len(input_ids)

    eval_input_ids.append(input_ids)
    eval_attention_masks.append(attention_mask)
    eval_labels.append([label2id[label] for label in token_labels])

eval_input_ids = pad_sequences(eval_input_ids, MAX_LEN)
eval_attention_masks = pad_sequences(eval_attention_masks, MAX_LEN)
eval_labels = pad_sequences(eval_labels, MAX_LEN, padding_value=-1)

eval_dataset = torch.utils.data.TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 训练模型
model = EntityModel(bert_model, hidden_size=768, num_tags=len(label2id))
model_path = 'checkpoint/Epochs-10.pt'
if exists(model_path):
    m_state_dict = torch.load(model_path)
    model.load_state_dict(m_state_dict)
    print("加载成功！")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    avg_train_loss = train(model, optimizer, train_dataloader)
    avg_eval_loss = evaluate(model, eval_dataloader)
    log = f'2-Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}\n'
    print(log)
    with open("./log.txt", 'a', encoding="utf-8") as f:
        f.write(log)
    print("正在保存模型的参数！")
    path = './checkpoint/' + '2-Epochs-' + str(epoch + 1) + '.pt'
    torch.save(model.state_dict(), path)
    print('保存完毕！')

print("测试开始！")
# 测试模型
test_sentences = [
    '中 国 人 民 银 行 将 在 2 0 2 4 年 1 月 1 日 开 始 陆 续 在 北 京 、上 海 等 地 发 行 龙 年 纪 念 钞 。 记 者 王 麻 子 报 道 。',
    '我喜欢去巴黎旅游', '巴 黎 是 一 座 美 丽 的 城 市']
for sentence in test_sentences:
    tags = predict(model, sentence)
    print(tags)
