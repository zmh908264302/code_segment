import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nlp.tools.utils import DateData


class CNNTranslation(torch.nn.Module):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units
        self.dec_v_dim = dec_v_dim

        # encoder
        self.enc_embeddings = torch.nn.Embedding(enc_v_dim, emb_dim)
        # self.enc_embeddings.weight.data.normal_(0, 0.1)
        self.conv2ds = [torch.nn.Conv2d(1, 16, kernel_size=(n, emb_dim), padding=0) for n in range(2, 5)]
        self.max_pools = [torch.nn.MaxPool2d((n, 1)) for n in [7, 6, 5]]
        self.encoder = torch.nn.Linear(16 * 3, units)

        # decoder
        self.dec_embeddings = torch.nn.Embedding(dec_v_dim, emb_dim)
        # self.dec_embeddings.weight.data.normal_(0, 0.1)
        self.decoder_cell = torch.nn.LSTMCell(emb_dim, units)
        self.decoder_dense = torch.nn.Linear(units, dec_v_dim)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        embedded = self.enc_embeddings(x)  # [n, step, emb]
        o = torch.unsqueeze(embedded, dim=1)  # [n, 1, step=8, emb=16]
        co = [F.relu(conv2d(o)) for conv2d in self.conv2ds]
        co = [self.max_pools[i](co[i]) for i in range(len(co))]
        co = [torch.squeeze(torch.squeeze(c, dim=3), dim=2) for c in co]
        o = torch.cat(co, dim=1)
        h = self.encoder(o)
        return [h, h]

    def inference(self, x):
        self.eval()
        hx, cx = self.encode(x)
        start = torch.ones(x.shape[0], 1)
        start[:, 0] = torch.tensor(self.start_token)
        start = start.type(torch.LongTensor)
        dec_emb_in = self.dec_embeddings(start)  # [n, step, emb]
        dec_emb_in = dec_emb_in.permute(1, 0, 2)  # [step, n, emb]
        dec_in = dec_emb_in[0]  # The first word use for decoding
        output = []
        for i in range(self.max_pred_len):
            hx, cx = self.decoder_cell(dec_in, (hx, cx))
            o = self.decoder_dense(hx)
            o = o.argmax(dim=1).view(-1, 1)
            dec_in = self.dec_embeddings(o).permute(1, 0, 2)[0]
            output.append(o)
        output = torch.stack(output, dim=0)  # [self.max_pred_len, n, 1]
        self.train()

        return output.permute(1, 0, 2).view(-1, self.max_pred_len)  # [n, self.max_pred_len]

    def train_logit(self, x, y):
        hx, cx = self.encode(x)  # [n, units]
        dec_in = y[:, :-1]
        dec_emb_in = self.dec_embeddings(dec_in)
        dec_emb_in = dec_emb_in.permute(1, 0, 2)
        output = []
        for i in range(dec_emb_in.shape[0]):
            hx, cx = self.decoder_cell(dec_emb_in[i], (hx, cx))
            o = self.decoder_dense(hx)
            output.append(o)
        output = torch.stack(output, dim=0)
        return output.permute(1, 0, 2)

    def step(self, x, y):
        self.opt.zero_grad()
        logit = self.train_logit(x, y)
        dec_out = y[:, 1:]

        loss = F.cross_entropy(logit.reshape(-1, self.dec_v_dim), dec_out.reshape(-1).long())
        loss.backward()
        self.opt.step()
        return loss.detach().numpy()


def train():
    dataset = DateData(4000)
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3], "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
          f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CNNTranslation(dataset.num_word, dataset.num_word, emb_dim=16, units=32, max_pred_len=11,
                           start_token=dataset.start_token, end_token=dataset.end_token)

    for i in range(100):
        for batch_idx, batch in enumerate(loader):
            bx, by, decoder_len = batch
            loss = model.step(bx, by)
            if batch_idx % 70 == 0:
                target = dataset.idx2str(by[0, 1:-1].data.numpy())
                pred = model.inference(bx[0:1])
                res = dataset.idx2str(pred[0].data.numpy())
                src = dataset.idx2str(bx[0].data.numpy())
                print(
                    "Epoch: ", i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                    "| input: ", src,
                    "| target: ", target,
                    "| inference: ", res,
                )


if __name__ == "__main__":
    train()
