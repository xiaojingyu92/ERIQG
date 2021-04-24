"""Data provider"""

import torch
import torch.utils.data as data
import os
import numpy as np
import json

class PrecompDataset(data.Dataset):

    def __init__(self, data_split):

        with open('../glove/word2idx.json','rb') as inf:
            self.vocab = json.load(inf)

        self.seg=[]
        self.content=[]
        self.template=[]

        self.sql_data = []
        self.split = data_split
        with open('../data/'+'%s_tok.jsonl' % data_split, 'rb') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                self.sql_data.append(sql)

        self.length = len(self.sql_data)

        with open('../data/'+'%s_seg.json' % data_split,'rb') as inf:
            self.seg=json.load(inf) #1:template,0:context

    def __getitem__(self, index):

        pair=self.sql_data[index]
        question_tok=' '.join(pair['question_tok']).lower()
        question_tok=question_tok.replace('   ',' ')
        question_tok=question_tok.replace('  ',' ')
        question_tok=question_tok.split(' ')
        sql_query=pair['query_tok']


        sql = []
        sql.append(self.vocab['<BEG>'])
        for token in sql_query:
            if self.vocab.has_key(token.lower()):
                sql.append(self.vocab[token.lower()])
            elif token=='EQL':
                sql.append(self.vocab['='])
            elif token=='LT':
                sql.append(self.vocab['<'])
            elif token=='GT':
                sql.append(self.vocab['>'])
            else:
                sql.append(self.vocab['<UNK>'])

            if len(sql)>=49:
                break                 
                
        sql.append(self.vocab['<END>'])
        sql = torch.Tensor(sql)
        
        #1:template(replace), 0:context(output)               
        
        flag=0
        question=[]
        question.append(self.vocab['<BEG>'])
        mask=[]
        mask.append(1)
        template_tok=[]
        
        fg=(np.random.random_sample()<0.5)
        
        seg_rand=self.seg[index][:]
        
        fill=False
        flip=1
        for i in range(len(seg_rand)-1):
            if seg_rand[i+1]==0 and flip==1: 
                flip=0
                fill= (np.random.random_sample()<0.5)
            if seg_rand[i]==0 and seg_rand[i+1]==1:#content,template
                if fill:
                    seg_rand[i+1]=0
                flip=1

        for i in range(min(50,len(question_tok))):            

            if self.split == 'train':
                s= seg_rand[i]
            else:
                
                s= self.seg[index][i]

            token=question_tok[i]   
            if s == 0: #content
                if flag==0:
                    mask.append(1)  #indicate start content
                    question.append(self.vocab['<SOS>'])
                    template_tok.append('<SOS>')
                flag=1
                mask.append(0)

            else: #template
                if flag!=0:
                    mask.append(1) #indicate start template
                    question.append(self.vocab['<EOS>'])
                    flag=0
                mask.append(1)
                template_tok.append(token)

            if self.vocab.has_key(token.lower()):
                question.append(self.vocab[token.lower()])
            else:
                question.append(self.vocab['<UNK>'])

            if len(mask)>=49 or len(question)>=49:
                break 
        
        if len(question)>49 or len(mask)>49:
            question=question[:49]
            mask=mask[:49]
            
        if len(question)!=len(mask):
            print("Alert!!!")
        question.append(self.vocab['<END>'])
        mask.append(1)       
        question = torch.Tensor(question)
        mask = torch.Tensor(mask).byte()        
        
        
        template=[]
        template.append(self.vocab['<BEG>'])
        for token in template_tok:
            if token=='<SOS>':
                template.append(self.vocab['<SOS>'])
            elif self.vocab.has_key(token.lower()):
                template.append(self.vocab[token.lower()])
            else:
                template.append(self.vocab['<UNK>'])
                
            if len(template)>=49:
                break 

        if len(template)>49:
            template=template[:49]
            
        template.append(self.vocab['<END>'])
        template = torch.Tensor(template)
        
        
        return sql, question, mask, template, index

    def __len__(self):
        return self.length

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sqls, questions, masks, templates, ids = zip(*data)

    sqls_lens=[]
    sources=[]
    sqls_lens = [len(cap) for cap in sqls]
    sources = torch.zeros(len(sqls), max(sqls_lens)).long()
    for i, cap in enumerate(sqls):
        end = sqls_lens[i]
        sources[i, :end] = cap[:end]

    question_lens = [len(question) for question in questions]
    qs_tar = torch.zeros(len(questions), 50).long()
    for i, question in enumerate(questions):
        end = question_lens[i]
        qs_tar[i, :end] = question[:end]

    mask_lens = [len(mask) for mask in masks]
    mask_tar = torch.ones(len(masks), 50).byte()
    for i, mask in enumerate(masks):
        end = mask_lens[i]
        mask_tar[i, :end] = mask[:end].byte()

    template_lens = [len(template) for template in templates]
    temp_tar = torch.zeros(len(templates), 50).long()
    for i, template in enumerate(templates):
        end = template_lens[i]
        temp_tar[i, :end] = template[:end]

    return sources, qs_tar, mask_tar, temp_tar, sqls_lens, question_lens, mask_lens, template_lens, ids

def get_precomp_loader(data_split, batch_size=100,
                       shuffle=True, num_workers=2):

    dset = PrecompDataset(data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader

def get_loaders(batch_size, workers):
    train_loader = get_precomp_loader('train', batch_size, True, workers)
    val_loader = get_precomp_loader('dev', batch_size, False, workers)
    return train_loader, val_loader

def get_test_loader(split_name, batch_size, workers):
    test_loader = get_precomp_loader(split_name, batch_size, False, workers)
    return test_loader
