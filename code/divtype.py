from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, train_list):
        self.data = train_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        weight = self.data[idx][0]
        pos_mask = self.data[idx][1].clone().detach().bool()
        neg_mask = self.data[idx][2].clone().detach().bool()
        context_mask = self.data[idx][3].clone().detach().bool()
        candidate_mask = self.data[idx][4].clone().detach().bool()

        X = self.data[idx][5].clone().detach().float()
        segment = self.data[idx][6].clone().detach().float()
        positions = self.data[idx][7].clone().detach().float()
        attn_mask = self.data[idx][8].clone().detach().bool()
        rel_feat = self.data[idx][9].clone().detach().float()
        
        pos_mask.requires_grad = False
        neg_mask.requires_grad = False
        context_mask.requires_grad = False
        candidate_mask.requires_grad = False
        X.requires_grad = False
        segment.requires_grad = False
        positions.requires_grad = False
        attn_mask.requires_grad = False
        rel_feat.requires_grad = False
        
        return X, segment, positions, attn_mask, rel_feat, context_mask, candidate_mask, pos_mask, neg_mask, weight


class TestDataset(Dataset):
    def __init__(self, test_list):
        self.data = test_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][0].clone().detach().float()
        segment = self.data[idx][1].clone().detach().float()
        positions = self.data[idx][2].clone().detach().float()
        attn_mask = self.data[idx][3].clone().detach().bool()
        rel_feat = self.data[idx][4].clone().detach().float()
        context_mask = self.data[idx][5].clone().detach().bool()
        candidate_mask = self.data[idx][6].clone().detach().bool()

        X.requires_grad = False
        segment.requires_grad = False
        positions.requires_grad = False
        attn_mask.requires_grad = False
        rel_feat.requires_grad = False
        context_mask.requires_grad = False
        candidate_mask.requires_grad = False

        return X, segment, positions, attn_mask, rel_feat, context_mask, candidate_mask
