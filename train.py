# -----------------------------------------------------------
# Generative Label Fused Network implementation based on
# Position Focused Attention Network (PFAN) and Stacked Cross Attention Network (SCAN)
# the code of PFAN: https://github.com/HaoYang0123/Position-Focused-Attention-Network
# the code of SCAN: https://github.com/kuanghuei/SCAN
# ---------------------------------------------------------------
"""Training script"""
import nltk
import tensorboard_logger as tb_logger

import os
import time
import shutil

import torch
import numpy

import data_wyx as data
from vocab import Vocabulary, deserialize_vocab
from model import SCAN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i, shard_xattn_i2t

import logging

import argparse
import random
import numpy as np
import evaluation


SPLIT_SIZE_STR = '16X15'
save_post_str = 'POS'

mode_name = 'i2t'


def random_seed_setting(seed):
    # set the ranodm seed to make the result repeatable
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_path(opt):
    save_parameter_path = '/yourpath/save_parameter'
    dataset_name = opt.data_name
    t2iori2t = opt.cross_attn
    agg_fun = opt.agg_func
    embed_size = opt.embed_size
    lr_update = opt.lr_update
    margin = opt.margin
    batch_size = opt.batch_size
    bert_to_gru_size = opt.bert_to_gru_size
    drop_out = opt.drop_out

    folder_name = dataset_name + '_' + t2iori2t + '_' + str(bert_to_gru_size) + "_" + str(
        drop_out) + '_' + agg_fun + '_' + str(embed_size) + '_' + str(lr_update) + '_' + str(margin) + '_' + str(
        batch_size)
    path = os.path.join(save_parameter_path, folder_name)
    return path, folder_name


def main():
    # Hyper Parameters
    print("Start for main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--vocab_name', default='f30k_precomp_vocab.json',
                        help='vocab file name')
    parser.add_argument('--nltk_data_path', default='',
                        help='nltk_data_path')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=2000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--bert_to_gru_size', default=300, type=int, help='bert_to_gru_size')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out')

    opt = parser.parse_args()

    current_path, folder_name = save_path(opt)
    if not os.path.exists(current_path):
        os.mkdir(current_path)

    sh_path = '/yourpath'
    sh_name = 'start.sh'
    source_file = os.path.join(sh_path, sh_name)
    copy_sh_name = folder_name + '.sh'
    target_file = os.path.join(current_path, copy_sh_name)
    shutil.copyfile(source_file, target_file)

    opt.current_path = current_path
    opt.model_name = opt.current_path
    opt.logger_name = opt.current_path
    # opt.logger_name = os.path.join(opt.current_path, 'log')
    print(opt)

    global mode_name
    mode_name = opt.cross_attn
    nltk.data.path.append(opt.nltk_data_path)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # random seed setting
    random_seed_setting(1)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, opt.vocab_name))
    opt.vocab_size = len(vocab)
    print('vocab size:', opt.vocab_size)
    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt)

    model_file_name = 'model_best_' + mode_name + '_posiAttn_16X15.pth.tar'
    model_path = os.path.join(opt.current_path, model_file_name)
    if os.path.exists(model_path):
        # checkpoint = torch.load(model_path)
        opt.resume = model_path

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        print(is_best, rsum, best_rsum)
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, opt,
            filename='checkpoint_{}_posiAttn_' + SPLIT_SIZE_STR + '_{}_' + save_post_str + '.pth.tar'.format(mode_name,
                                                                                                             epoch),
            prefix=opt.model_name + '/')

        finally_test(opt)


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    # global max_len
    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        '''image, whole, box, caption, length, temp = train_data
        if max_len <caption.size(1):
            max_len = caption.size(1)
            print(max_len)
        '''
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        '''if model.Eiters % opt.val_step == 0:
            print("Start for validate")
            validate(opt, val_loader, model)'''


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(
        model, val_loader, opt.log_step, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    print("Img shape in validate:", img_embs.shape)

    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims, opt)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims, opt)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                # shutil.copyfile(prefix + filename, prefix + 'model_best_i2t_posiAttn_'+SPLIT_SIZE_STR+'_AU.pth.tar')
                current_path = opt.current_path
                file_name = prefix + 'model_best_{}_posiAttn_'.format(mode_name) + SPLIT_SIZE_STR + '.pth.tar'
                torch.save(state, os.path.join(current_path, file_name))
                print('Best Model parameters Update!')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k fparam_group['weight_decay']or the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
