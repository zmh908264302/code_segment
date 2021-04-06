"""
本例以莫烦自然语言系列之Skip-Gram（https://mofanpy.com/tutorials/machine-learning/nlp/skip-gram/）为模板
实现同样采样莫凡中简化构造思想：本来skip-gram是一个中心词预测上下文，在是现实时可以是一个词预测多次，达到同等效果。

最后预测的实例无法实现与CBOW同等的效果，因为输入确定，输出即确定。我们这里真正要学习的是词向量，而不是预测结果。更多参见讨论。
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp.tools.utils import process_w2v_data

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


class SkipGram(nn.Module):
    def __init__(self, vocab_num, emb_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_num, emb_dim)

    def forward(self, x):
        outputs = self.embeddings(x)
        return outputs


def train(model, dataset, loss_func, optimizer, bsz=8):
    data_iter = DataLoader(dataset, batch_size=bsz)
    losses = []
    epoches = 10
    for epoch in range(epoches):
        total_loss = 0
        step = 0
        model.train()
        for x, y in tqdm(data_iter):
            step += 1
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss.item())
        losses.append(total_loss / step)
    plt.plot(range(1, epoches + 1), losses)
    plt.show()
    return model


if __name__ == '__main__':
    dataset, w2i, i2w = process_w2v_data(corpus, skip_window=2, method="skip_gram")
    model = SkipGram(len(dataset), 32)
    loss_func = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model = train(model, dataset, loss_func, optimizer)

    # 5 2 4 8 6
    # e h 9 u f
    model.eval()
    for _ in range(4):
        logits = model(torch.tensor([w2i["9"]]))
        out = torch.max(logits, 1)
        print(i2w[out[1].item()])
