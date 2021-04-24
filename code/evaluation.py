import os
import json
import torch
import numpy as np
import datetime
import dataloader as data
from model import QG
import argparse
import time
from utils import AverageMeter, LogCollector, to_np, trim_seqs
import logging
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import shutil
from torch.autograd import Variable
from nltk.translate.nist_score import sentence_nist, corpus_nist
from nltk.translate.meteor_score import meteor_score
import nltk
from rouge_score import rouge_scorer

nltk.download('wordnet')
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


with open('../glove/word2idx.json') as inf:
    vocab = json.load(inf)

vocab_rev={}
vocab_inv={}
for key,value in vocab.items():
    vocab_rev[value]=key
    vocab_inv[str(value)]=key
    
def nist(batch_targets, batch_outputs): 
    score={}
    score_nist=sentence_nist(batch_targets, batch_outputs, n=4)
    return score

def meteor(batch_targets, batch_outputs): 
    score={}
    score_meteor=meteor_score(batch_targets, batch_outputs)
    return score

torch.manual_seed(195)

def train(args, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
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
        if model.Eiters % args.logging_step == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))


def validate(args, val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()
    model.logger = val_logger
    end = time.time()
    max_length=50
    for i, val_data in enumerate(val_loader):

        decoder_outputs, sampled_idxs, mean, logvar, z = model.forward_emb(*val_data)

        if torch.cuda.is_available():
            val_data[1]=val_data[1].cuda()

        batch_size = val_data[1].size(0)
        max_length=50
        flattened_outputs = decoder_outputs.view(batch_size * max_length, -1)

        batch_outputs = trim_seqs(sampled_idxs)
        
        np_targets=trim_seqs(val_data[1].unsqueeze(-1))
            
        batch_targets = [[seq] for seq in np_targets]

        corpus_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)
        model.logger.update('C-BLEU', corpus_bleu_score, batch_size)
        
        corpus_nist_score = corpus_nist(batch_targets, batch_outputs, n=4)

        model.logger.update('C-NIST', corpus_nist_score, batch_size)

        corpus_meteor_score=0
        rouge_scores=0
        for j in range(batch_size):
            reference=[]
            for tid in range(len(batch_targets[j][0])):
                tok=batch_targets[j][0][tid]
                reference.append(vocab_inv[str(tok)])
            ref=[str(' '.join(reference))]
            
            hypothesis=[]
            for tid in range(len(batch_outputs[j])):
                tok=batch_outputs[j][tid]
                hypothesis.append(vocab_inv[str(tok)])
            hypo=str(' '.join(hypothesis))
            corpus_meteor_score+=meteor_score(ref, hypo)
            rouge_scores += rouge.score(ref[0],hypo)['rougeL'][2]
        
        rouge_scores=rouge_scores/batch_size
        model.logger.update('ROUGH-L', rouge_scores, batch_size)
        corpus_meteor_score=corpus_meteor_score/batch_size
        model.logger.update('C-METEOR', corpus_meteor_score, batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        
        model.Eiters += 1
        if model.Eiters % args.logging_step == 0:
            print('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(val_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
            
    print('Test: [{0}/{1}]\t'
            '{e_log}\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            .format(
                i, len(val_loader), batch_time=batch_time,
                e_log=str(model.logger))) 
    
    return 0

def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding for SQLNet(requires pretrained model).')
    parser.add_argument('--resume', default=None,
            help='resume from pretrained model.')
    parser.add_argument('--epoch', type=int, default=20,
            help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=2,
            help='batch size')
    parser.add_argument('--logging_step', type=int, default=50,
            help='logging step')
    parser.add_argument('--lr_update', type=int, default=10,
            help='lr update')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
            help='lr update')
    parser.add_argument('--prefix', type=str, default='bS2_',
            help='prefix of saved model')
    parser.add_argument('--withtab', type=int, default=1,
            help='sample from content vector')
    
    parser.add_argument('--teacher_forcing_fraction', type=float, default=1.0,
                        help='fraction of batches that will use teacher forcing during training')
    parser.add_argument('--scheduled_teacher_forcing', action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epocs')
    
    args = parser.parse_args()

    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/args.epoch)
    else:
        schedule = np.ones(args.epoch) * args.teacher_forcing_fraction
    
    train_loader, val_loader = data.get_loaders(args.batch_size, 8)
    
    test_loader=data.get_test_loader('test', args.batch_size, 8)
    
    if args.withtab:
        model = QG()
    else:
        model = QG()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
    
            model.teach_forcing=0.0
            model.mask_tf=True

            print('dev set')
            validate(args, val_loader, model)  
            
            print('test set')
            validate(args, test_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

if __name__ == '__main__':
    main()
