"""Lexical, frozen dense and trainable dense retrieval with explicit rankings."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rank(scores,topk):
    return np.argsort(-np.asarray(scores),kind='stable')[:topk]


def reciprocal_rank_fusion(rankings,k=60):
    scores={}
    for ranking in rankings:
        for position,index in enumerate(ranking):
            scores[int(index)]=scores.get(int(index),0.)+1/(k+position+1)
    return sorted(scores,key=lambda i:(-scores[i],i))


class Projection(nn.Module):
    def __init__(self,input_dim=384,output_dim=128):
        super().__init__();self.linear=nn.Linear(input_dim,output_dim,bias=False)

    def forward(self,values):return F.normalize(self.linear(values),dim=-1)


def contrastive_loss(queries,documents,document_ids,temperature=.07):
    """Every in-batch copy of the correct document is a positive, never a negative."""
    logits=queries@documents.T/temperature
    positives=document_ids[:,None]==document_ids[None,:]
    return (torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~positives,-torch.inf),dim=1)).mean()


def gold_label(claim):
    labels={e['label'] for values in claim['evidence'].values() for e in values}
    if len(labels)>1:raise ValueError('Mixed support/refute annotations require document-level evaluation')
    return {'SUPPORT':'supported','CONTRADICT':'contradicted'}.get(next(iter(labels),None),'insufficient')
