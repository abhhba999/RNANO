import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        att_scores = F.softmax(self.fc(x), dim=1)
        x = x * att_scores
        return x

class Model(nn.Module):
    def __init__(self, kmer_nums):
        super(Model, self).__init__()
        
        self.embed = nn.Embedding(kmer_nums, 2)
           
        self.layers1 = nn.Sequential(
                    nn.Linear(in_features=30+10, out_features=150, bias=True),
                    nn.BatchNorm1d(150),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p=0.3),
                    AttentionLayer(150, 150)
                )

        self.layers2 = nn.Sequential(
                    nn.Linear(in_features=150, out_features=32, bias=True),
                    nn.BatchNorm1d(32),
                    nn.ELU(),
                    nn.Dropout(p=0.3)
                )
       
        self.probability_layer = nn.Sequential(
                nn.Linear(in_features=32, out_features=1, bias=True),
                nn.Sigmoid()
            )
        self.read_level_pred = None

        
    def forward(self, x):
        kmer = x["kmer"].view(-1, 1)
        x = x["X"].view(-1, 30)
        kmer = self.embed(kmer.long()).reshape(-1, 10)
        x = torch.cat((x, kmer), 1)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.probability_layer(x)
        x = x.view(-1, 20)
        self.read_level_pred = x
        x = 1 - torch.prod(1 - x, axis=1)
        return x
