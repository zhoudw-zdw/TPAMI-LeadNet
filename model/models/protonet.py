import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.use_euclidean = False
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet(dropblock_size=args.dropblock_size)            
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')    

    def split_instances(self, data):
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
      
    def forward(self, x, concept_label_set=None, selected_concepts=None):
        x = x.squeeze(0)
        support_idx, query_idx = self.split_instances(x)
        instance_embs = self.encoder(x)
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

        logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature

        if self.training:
            return logits, None
        else:
            return logits, None
