from __future__ import division, absolute_import
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from .cross_entropy_loss import CrossEntropyLoss

class ArcFaceLoss1(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, use_gpu=True, label_smooth=True, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.label_smooth = label_smooth
        self.use_gpu = use_gpu
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        # self.ce = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=label_smooth)
        if self.use_gpu:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size).cuda())
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        print(f"Using ArcFace Loss weight nc_em: {num_classes}x{embed_size}")
        
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, targets):
        return self.forward2(inputs, targets)
        
    def forward1(self, embedding: torch.Tensor, ground_truth):
        if self.use_gpu:
            ground_truth = ground_truth.cuda()
        """
        This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
        54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # print(f"ground_truth shape: {ground_truth.size()}")
        # print(f"embedding shape: {embedding.size()}")
        # print(f"weight shape: {self.weight.size()}")
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale

        loss = self.ce(output, ground_truth)
        res = {'loss': loss, 'probs': output}
        return res

    def forward2(self, embedding: torch.Tensor, ground_truth):
        if self.use_gpu:
            ground_truth = ground_truth.cuda()
        """
        This Implementation is from https://github.com/deepinsight/insightface, which takes
        66.45489303627983 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        Please noted that, different with forward1&3, this implementation ignore the samples that
        caused \theta + m > \pi to happen if easy_margin is False. And if easy_margin is True,
        it will do nothing even if \theta + m > \pi.
        """
        embedding = F.normalize(embedding)
        w = F.normalize(self.weight)
        cos_theta = F.linear(embedding, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.easy_margin:
            mask = torch.ones_like(ground_truth)
        else:
            mask = torch.gather(cos_theta, 1, ground_truth.view(-1, 1)).view(-1)
            mask = torch.where(mask.acos_() + self.margin > math.pi, 0, 1)
        mask = torch.where(mask != 0)[0]
        m_hot = torch.zeros(mask.shape[0], cos_theta.shape[1], device=cos_theta.device)
        m_hot.scatter_(1, ground_truth[mask, None], self.margin)
        theta = cos_theta.acos()
        output = (theta + m_hot).cos()
        output.mul_(self.scale)
        loss = self.ce(output[mask], ground_truth[mask])
        return loss

    def forward3(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is modified from forward1, which takes
        52.49644996365532 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        """
        if self.use_gpu:
            ground_truth = ground_truth.cuda()
            
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        # output = cos_theta + one_hot
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss

""" NEW IMPLEMENTATION """

def box_and_whisker_algorithm(similarities, wisk):        
    l = similarities.size(0)
    sorted_x = torch.sort(input=similarities, descending=False)[0]
    
    lower_quartile = sorted_x[int(0.25 * l)]
    upper_quartile = sorted_x[int(0.75 * l)]
            
    IQR = (upper_quartile - lower_quartile)        
    minimum = lower_quartile - wisk * IQR        
    maximum = upper_quartile + wisk * IQR
    mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
    sn_prime = sorted_x[mask]
    return sn_prime

def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return (similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])


class ArcFaceLoss2(nn.Module):
    def __init__(self, embed_size, num_classes, use_gpu=True):
        super(ArcFaceLoss2, self).__init__()
        
        opt = {"s" : 64, "m" : 0.50}
        aux_opt = {"wisk" : 1.5}
        
        self.head = ArcFace(embed_size=embed_size, num_classes=num_classes, s=opt['s'], m=opt['m'])  
                                 
        self.aux = UNPG(s=opt['s'], wisk=aux_opt['wisk'])                   
        self.wisk = aux_opt['wisk']

        self.use_gpu = use_gpu
        self.num_gpu = torch.cuda.device_count()
                           
    def forward(self, deep_features, labels, rank=-1): 
        
        if self.use_gpu:
            labels = labels.cuda()
            
        #if rank != -1: # We found performance degradation during work distributed training for metric losses. We will fix it later.
        #    deep_features, labels = gather(deep_features, labels)

        # print("type(deep_features): ", deep_features.device.type)
        # print("type(labels): ", labels.device.type)
        # print("deep_features.size(): ", deep_features.size())
        # print("labels.size(): ", labels.size())
        
        loss, opt_loss, regular_g = 0, 0, 0
        cosine = self.head(deep_features, labels) 
                    
        norm_x = F.normalize(deep_features)
        sp, sn = convert_label_to_similarity(norm_x, labels)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        aux_sn = []
        sn_prime = box_and_whisker_algorithm(sn, wisk=self.wisk)
        aux_sn.append(sn_prime)
        aux_sn = torch.cat(aux_sn, dim=0)
        aux_sn = torch.stack([aux_sn] * self.num_gpu, dim=0)
        loss += self.aux(cosine, aux_sn, labels)
        
        return loss
    
class ArcFace(nn.Module):    
    def __init__(self, embed_size, num_classes, s=32.0, m=0.50, use_gpu=True):
        super(ArcFace, self).__init__()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size).cuda())
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        # cos(theta)             
        x, weight = F.normalize(x), F.normalize(self.weight)
        cosine = F.linear(x, weight)        
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
                
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_()
                
        return cosine

class UNPG(nn.Module):
    def __init__(self, s, wisk=1.0):
        super(UNPG, self).__init__()
        self.s = s
        self.wisk = wisk
        self.cross_entropy = nn.CrossEntropyLoss()
                         
    def forward(self, cosine, aux_sn, labels):                  
        # aux_sn = aux_sn.unsqueeze(0)                        
        one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
        aux_sn = one * aux_sn        
        cosine = torch.cat([cosine, aux_sn], dim=1)                      
        loss = self.cross_entropy(self.s * cosine, labels)
        return loss