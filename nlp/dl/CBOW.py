"""
本例以莫烦自然语言系列之Continuous Bag of Words (CBOW)（https://mofanpy.com/tutorials/machine-learning/nlp/cbow/）和
word2vec的PyTorch实现（https://samaelchen.github.io/word2vec_pytorch/）之toy 版本为蓝本
"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = "cuda" if torch.cuda.is_available() else "cpu"


class CBOW(nn.Module):
    def __init__(self, vocab_num, emb_dim, skip_window, hidden_size=128):
        super(CBOW, self).__init__()
        self.vocab_num = vocab_num
        self.embeddings = nn.Embedding(vocab_num, emb_dim)
        # 上下文，因此需要乘2
        self.linear1 = nn.Linear(2 * skip_window * emb_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_num)

        self.embeddings.weight.data.uniform_(-0.5 / vocab_num, 0.5 / vocab_num)

    def forward(self, x):
        # [bsz, seq_len, emb_dim] -> [bsz, seq_len * emb_dim]
        outputs = self.embeddings(x).view(x.shape[0], -1)  # batch size: x.shape[0]
        out = F.relu(self.linear1(outputs))
        out = self.linear2(out)
        logits = F.log_softmax(out, dim=1)
        return logits


def train(model, dataset, loss_func, optimizer, bsz=4):
    data_iter = DataLoader(dataset, bsz)  # , drop_last=True)
    losses = []
    epoches = 20
    for epoch in range(epoches):
        total_loss = 0
        step = 0
        model.train()
        for x, y in tqdm(data_iter):
            step += 1
            model.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss.item())
        losses.append(total_loss / step)
    plt.plot(range(1, epoches + 1), losses)
    plt.show()


if __name__ == '__main__':
    dataset, w2i, i2w = process_w2v_data(corpus, skip_window=2, method="cbow")
    model = CBOW(len(dataset), 32, 2)
    loss_func = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train(model, dataset, loss_func, optimizer)
    # 5 2 4 8 6
    # d g 9 s a
    logits = model(torch.tensor([[w2i["5"], w2i["2"], w2i["8"], w2i["6"]]]))
    logits = model(torch.tensor([[w2i["d"], w2i["g"], w2i["s"], w2i["a"]]]))
    out = torch.max(logits, 1)
    print(i2w[out[1].item()])
