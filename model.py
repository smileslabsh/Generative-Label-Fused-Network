# -----------------------------------------------------------
# Generative Label Fused Network implementation based on
# Position Focused Attention Network (PFAN) and Stacked Cross Attention Network (SCAN)
# the code of PFAN: https://github.com/HaoYang0123/Position-Focused-Attention-Network
# the code of SCAN: https://github.com/kuanghuei/SCAN
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict

import codecs


SPLIT_SIZE = 16
BOX_LEN = 15


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(nlp_bert, data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            nlp_bert, img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, nlp_bert, img_dim, embed_size, no_imgnorm=False, split_size=SPLIT_SIZE, position_embed_size=200):
        super(EncoderImagePrecomp, self).__init__()
        self.nlp_bert = nlp_bert
        self.embed_size = embed_size
        self.split_size = split_size
        self.position_size = split_size * split_size
        self.position_embed_size = position_embed_size
        self.no_imgnorm = no_imgnorm
        self.image_tag_graph_out = 64
        self.fc = nn.Linear(img_dim + self.position_embed_size + 768, embed_size)
        self.position_embedding = nn.Embedding(self.position_size + 1, self.position_embed_size)
        self.posi_attn_matrix_add_tag = nn.Linear(2048 + 768, self.position_embed_size, bias=False)

        self.init_weights()


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.posi_attn_matrix_add_tag.in_features +
                                  self.posi_attn_matrix_add_tag.out_features)
        self.posi_attn_matrix_add_tag.weight.data.uniform_(-r, r)

    def forward(self, images, images_tag, boxes):  # boxes
        new_boxes = boxes.view(boxes.size(0) * boxes.size(1), boxes.size(2))
        new_boxes_index = new_boxes[:, :int(new_boxes.size(1) / 2)].type(torch.LongTensor).cuda()
        new_boxes_weight = new_boxes[:, int(new_boxes.size(1) / 2):].cuda()

        image_add_tag = torch.cat((images, images_tag), dim=2)
        box_emb_features = self.position_embedding(new_boxes_index)
        box_emb_features = box_emb_features.view(boxes.size(0), boxes.size(1), boxes.size(2) // 2, -1)
        box_features = self.posi_attn(image_add_tag, box_emb_features, new_boxes_weight, self.posi_attn_matrix_add_tag)

        image_position = torch.cat((images, box_features, images_tag), dim=2)
        features = self.fc(image_position)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)

    def posi_attn(self, images, box_features, box_weights, posi_attn_matrix):
        size = images.size()
        posi_attn_embed = torch.zeros(size[0], size[1], box_features.size(3))
        box_weights = box_weights.view(size[0], size[1], -1)
        for i in range(size[0]):
            image_fea = images[i]
            box_fea = box_features[i]
            box_w = box_weights[i]

            # we next preject the regions of the i-th image
            region_temp = posi_attn_matrix(image_fea)
            region_temp = region_temp.unsqueeze(2).transpose(2, 1)
            attn_mat = torch.bmm(region_temp, box_fea.transpose(2, 1)).squeeze()
            attn_mat = nn.Tanh()(attn_mat)
            attn_mat = nn.Softmax()(attn_mat)
            attn_mat = attn_mat * box_w
            norm_mat = torch.sum(attn_mat, 1).repeat(BOX_LEN, 1).transpose(1, 0) + 1e-6
            attn_mat = attn_mat / norm_mat
            attn_mat = torch.bmm(attn_mat.unsqueeze(1), box_fea).squeeze()

            posi_attn_embed[i] = attn_mat.data
        posi_attn_embed = Variable(posi_attn_embed).cuda()
        return posi_attn_embed


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, bert_to_gru_size, drop_out,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.use_bi_gru = use_bi_gru
        self.postion_embedding = 200

        self.fc = nn.Linear(768, bert_to_gru_size)
        self.rnn = nn.GRU(bert_to_gru_size, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru,
                          dropout=drop_out)

        self.init_weights()

    def init_weights(self):
        # self.embed.weight.data.uniform_(-0.1, 0.1)

        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, x, lengths):
        """Handles variable size captions
        """

        x = self.fc(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(packed)

        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    queryT = torch.transpose(query, 1, 2)

    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        # attn = l1norm_d(attn, 2)
        attn = l1norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = l1norm_d(attn, 2)
        attn = l1norm(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    attn = attn.view(batch_size, queryL, sourceL)

    attnT = torch.transpose(attn, 1, 2).contiguous()
    max_attnT = torch.max(attnT, dim=1)[0]

    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    del attn
    return weightedContext, attnT, max_attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        weiContext, attn, max_attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)

        max_attn2 = max_attn.unsqueeze(0)
        max_attn2 = torch.transpose(max_attn2, 0, 1)
        max_attn2 = max_attn2.expand(attn.size()[0], attn.size()[1], attn.size()[2])
        diff_attn = (max_attn2 - attn)  # * max_attn2
        diff_score = diff_attn.mean(dim=1, keepdim=True).mean(dim=2, keepdim=False)

        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
            max_attn.mul_(opt.lambda_lse).exp_()
            max_attn = max_attn.sum(dim=1, keepdim=True)
            max_attn = torch.log(max_attn) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
            max_attn = max_attn.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
            max_attn = max_attn.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
            max_attn = max_attn.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        # print("Row sim(2)", row_sim+0.5*max_attn+0.5*diff_score)
        similarities.append(row_sim + 0.5 * max_attn)  # +0.5*torch.log(1+max_attn/diff_score))

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        weiContext, attn, max_attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, opt):
        nlp_bert = ''
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(nlp_bert, opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        print('Text Encoder vocab size:', opt.vocab_size)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, opt.bert_to_gru_size, opt.drop_out,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, images_tag, boxes, captions, lengths, volatile=False):  # boxes
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        images_tag = Variable(images_tag, volatile=volatile)
        boxes = Variable(boxes, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            images_tag = images_tag.cuda()
            # ndata.cuda()
            # edata.cuda()
            boxes = boxes.cuda()
            captions = captions.cuda()

        img_emb = self.img_enc(images, images_tag, boxes)  # boxes

        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, images_tag, boxes, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        boxes = boxes[:, :, 0:BOX_LEN * 2]
        img_emb, cap_emb, cap_lens = self.forward_emb(images, images_tag, boxes, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
