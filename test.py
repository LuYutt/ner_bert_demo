# coding=utf-8
"""
    @project: bert_share
    @Author：LuYutang
    @file： test.py
    @date：2024/1/29 22:03
"""
from model import *
from help import *
from os.path import exists
my_model = EntityModel(bert_model, hidden_size=768, num_tags=len(label2id))
model_path = 'checkpoint/Epochs-10.pt'
if exists(model_path):
    m_state_dict = torch.load(model_path)
    my_model.load_state_dict(m_state_dict)
    print("加载成功！")
print("测试开始！")
if torch.cuda.is_available():
    model = my_model.cuda()
else:
    model = my_model.cpu()
# 测试模型
test_sentences = ['中 国 人 民 银 行 将 在 2 0 2 4 年 1 月 1 日 开 始 陆 续 在 北 京 、上 海 等 地 发 行 龙 年 纪 念 钞 。 记 者 王 麻 子 报 道 。',  '巴 黎 是 一 座 美 丽 的 城 市']
for sentence in test_sentences:
    tags = predict(my_model, sentence)
    print(tags)
