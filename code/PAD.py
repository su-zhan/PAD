import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F


class PAD(nn.Module):
    def __init__(self, dropout = 0.1):
        super(PAD, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=512, batch_first = True)
        self.GloEnc = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.SelEnc = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.InterAct = nn.TransformerEncoder(encoder_layer, num_layers=3)
        enc_layer =nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=512, batch_first = True)
        self.DocEnc = nn.TransformerEncoder(enc_layer, num_layers = 1)
        
        self.fc1 = nn.Linear(768 * 4, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)

        self.rfc1 = nn.Linear(18,18)
        self.rfc2 = nn.Linear(18,8)
        self.rfc3 = nn.Linear(8,1)

        init.xavier_normal_(self.rfc1.weight)
        init.xavier_normal_(self.rfc2.weight)
        init.xavier_normal_(self.rfc3.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, PAS_NUM_PER_DOC, E, segment, positions, attn_mask, Ri, context_mask, candidate_mask, pos_mask = None, neg_mask = None, train_flag = False):
        bs = E.shape[0]
        df = E.shape[2]

        Eq = E[:,0,:].unsqueeze(1)
        
        Ex = E[:,1:,:]
        Px = Eq * Ex
        Px = Px.reshape(bs * 50, PAS_NUM_PER_DOC + 1, df)
        Ex = Ex.reshape(bs * 50, PAS_NUM_PER_DOC + 1, df)

        add_mask = ~attn_mask[:,1:]
        add_mask = add_mask.reshape(bs * 50, PAS_NUM_PER_DOC + 1, 1)
        Pi = torch.sum(Px * add_mask, dim = 1)
        Pi = Pi.reshape(bs, 50, df)
        Xi = torch.sum(Ex * add_mask, dim = 1)
        Xi = Xi.reshape(bs, 50, df)


        E = E + segment + positions
        X = self.GloEnc(E, src_key_padding_mask=attn_mask)
        context_y = self.SelEnc(X, src_key_padding_mask=context_mask)
        Xs = context_y[:,0,:].unsqueeze(1)
        X_cross = Xs * X[:,1:,:]
        candidate_y = self.InterAct(X_cross, src_key_padding_mask=candidate_mask[:,1:])

        Xs = Xs.repeat(1, 50, 1)

        selected_mask = attn_mask[:,1:]
        selected_mask = selected_mask.reshape(bs * 50, PAS_NUM_PER_DOC+1)
        Y = candidate_y.reshape(bs * 50, PAS_NUM_PER_DOC+1, df)
        Zx = self.DocEnc(Y, src_key_padding_mask = selected_mask)

        Zi = Zx[:,0,:]
        Zi = Zi.reshape(bs, 50, df)
        Hi = torch.cat([Xs, Zi, Xi, Pi], dim = 2).float()
        Hi = Hi.reshape(-1, Hi.shape[2])

        Hi = self.dropout(Hi)

        s = F.relu(self.fc1(Hi))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)

        Ri = Ri.reshape(Ri.shape[0] * Ri.shape[1], 18)
        sr = F.relu(self.rfc1(Ri))
        sr = F.relu(self.rfc2(sr))
        sr = self.rfc3(sr)


        s = s.view(s.shape[0] * s.shape[1])
        sr = sr.view(sr.shape[0] * sr.shape[1])
        score = s + sr
        
        if train_flag:
            pos_mask = pos_mask.view(pos_mask.shape[0] * pos_mask.shape[1])
            score1 = torch.masked_select(score, pos_mask)
            neg_mask = neg_mask.view(neg_mask.shape[0] * neg_mask.shape[1])
            score2 = torch.masked_select(score, neg_mask)
            return score1, score2
        else:
            return score
