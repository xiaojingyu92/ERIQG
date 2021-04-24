import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import json
from utils import to_np, trim_seqs

with open('../glove/word2idx.json') as inf:
    vocab = json.load(inf)

vocab_rev={}
vocab_inv={}
for key,value in vocab.items():
    vocab_rev[value]=key
    vocab_inv[str(value)]=key

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """

    y_tensor = y.data if isinstance(y, Variable) else y

    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1

    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)

    y_one_hot = y_one_hot.view(y.size(0),y.size(1), -1)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


class EncoderRNN(nn.Module):
    def __init__(self, vocab, word_emb, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = 300
        self.vocab = vocab
        self.max_len=50
        self.embedding = nn.Embedding(len(self.vocab), 300)
        self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb.astype(np.float32)))
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, hidden_init, lengths):

        #add sort part
        lengths = torch.Tensor(lengths)
        _, idx_sort = torch.sort(torch.Tensor(lengths), dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        input_sort = inputs.index_select(0, idx_sort.cuda())

        lengths_sort = list(lengths[idx_sort])

        embedded_sort = self.embedding(input_sort)
        packed_embedded_sort = torch.nn.utils.rnn.pack_padded_sequence(embedded_sort, lengths_sort, batch_first=True)
        self.gru.flatten_parameters()
        output_sort, hidden_sort = self.gru(packed_embedded_sort, hidden_init)
        output_sort, output_lengths_sort = torch.nn.utils.rnn.pad_packed_sequence(output_sort, batch_first=True)

        #add unsort part
        output = output_sort.index_select(0, idx_unsort.cuda())

        hidden = hidden_sort.index_select(1, idx_unsort.cuda())
        hidden=torch.cat([hidden[0],hidden[1]],dim=-1)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

class Sample(nn.Module):
    def __init__(self, hidden_size, z_dim):
        super(Sample, self).__init__()
        self.z_dim=z_dim

        self.context_to_mean=nn.Linear(hidden_size, z_dim)

        self.context_to_logvar=nn.Linear(hidden_size, z_dim)

    def forward(self, t_hidden, c_hidden): # q: batch_sz x hid_sz

        batch_size=t_hidden.size()[0]

        template=torch.cat([t_hidden,c_hidden],-1) #(1, 1024)


        mean=self.context_to_mean(template)
        logvar = self.context_to_logvar(template) #batch_sz x z_hid_size
        std = torch.exp(0.5 * logvar)

        z = Variable(torch.randn([t_hidden.size()[0], self.z_dim])) #first dim = bi-direction

        z = z.cuda() if torch.cuda.is_available() else z
        z = z * std + mean   # [batch_sz x z_hid_sz]

        return z, mean, logvar

class DecoderRNN(nn.Module):
    def __init__(self, vocab, word_emb, hidden_size, max_length):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = 300
        self.vocab = vocab
        self.max_length = 50
        self.embedding = nn.Embedding(len(self.vocab), 300)
        self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb.astype(np.float32)))
        self.gru = nn.GRU(3 * self.hidden_size + self.embedding.embedding_dim, self.hidden_size, batch_first=True)  # input = (context + selective read size + embedding)
        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2*self.hidden_size, len(self.vocab))

    def forward(self, encoder_outputs, temp_outputs, z, templates, inputs, questions, targets=None, teachforce=0.0, mask_tf=False):

        seq_length = encoder_outputs.data.shape[1]
        batch_size=inputs.size()[0]
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))

        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        hidden = z.unsqueeze(0)

        if len(hidden.shape)<3:
            hidden=hidden.unsqueeze(0)

        # every decoder output seq starts with <BEG>
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings + seq_length)))
        sampled_idx = Variable(self.vocab['<BEG>']*torch.ones((batch_size, 1)).long())

        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()

        sos_output[:, 1] = 1.0  # index 1 is the <BEG> token, one-hot encoding

        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        one_hot_input_seq = to_one_hot(inputs, len(self.vocab) + seq_length)

        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_input_seq = one_hot_input_seq.cuda()


        flag = Variable((torch.ones(batch_size).long()))
        index = Variable((torch.zeros(batch_size).long()))
        c_len = Variable((torch.zeros(batch_size).long()))
        one=torch.ones(batch_size).long()
        if next(self.parameters()).is_cuda:
            flag = flag.cuda()
            index = index.cuda()
            c_len = c_len.cuda()
            one = one.cuda()

        pre_sampled_idx=sampled_idx.clone()

        eos_sampled_idx=pre_sampled_idx.clone()

        pre_seg=templates[torch.arange(templates.size(0)),index]

        pre_hidden=hidden.clone()

        for step_idx in range(1, self.max_length):

            eos_sampled_idx_clone=eos_sampled_idx.clone()
            eos_sampled_idx=pre_sampled_idx.clone()
            tmp_set=(eos_sampled_idx.unsqueeze(-1) == 6).nonzero().squeeze(-1)
            if tmp_set.size(0)>0:
                for r in range(tmp_set.size(0)):
                    rid=tmp_set[r,0]
                    eos_sampled_idx[rid]=eos_sampled_idx_clone[rid].clone()

            if mask_tf == True:

                flag = 0*flag #[0,0,0,0,0]

                flag1=(pre_seg!=5*one)#[1,0,0,0,1] #previous template token (i.e. in last iteration) not <SOS> [1,5,5,5,9]

                flag2=(pre_sampled_idx.squeeze(-1)==6*one)#previous input sampled_idx is <EOS>:[7,3,2,1,6],->[0,0,0,0,1]

                flag3=(c_len>20*one)#previous content length greater than 10 [1,2,3,10,1]->[0,0,0,1,0]

                flag=flag1|flag2|flag3 #[1,0,0,1,1]

                d_index=flag    #|pre_flag  #[1,0,0,1,1] #0: stay at <PH>/no teacher forcing 1:move to next template token
                index=index+d_index.long() #index of template token at *this* iteration(decide teacher foring or not as input)

                c_len=c_len+(~d_index).long() #incremental length of current content

                seg = templates[torch.arange(templates.size(0)),index]

                if torch.max(index) < temp_outputs.size(1):
                    t_hidden = temp_outputs[torch.arange(temp_outputs.size(0)),index].unsqueeze(0)
                else:
                    t_hidden = t_hidden

                index_set=(d_index.unsqueeze(-1) == 1).nonzero().squeeze(-1)
                if index_set.size(0)>0:

                    subset=(seg.unsqueeze(-1))[index_set[:,0], index_set[:,1]]

                    sampled_idx = sampled_idx.masked_scatter(d_index.unsqueeze(-1), subset)

                pre_sampled_idx=sampled_idx.clone()
                pre_seg=seg.clone()

            elif targets is not None and teachforce > 0.0 and step_idx < questions.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teachforce), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()

                index_set=(teacher_forcing_mask == 1).nonzero().squeeze(-1)
                if index_set.size(0)>0:
                    subset=questions[:,step_idx-1:step_idx][index_set[:,0], index_set[:,1]]
                    sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, subset)

#                #for training, the input samplae_idx <SOS>,...,<EOS> ----> hidden state of the Template token *<SOS> as output* )
                flag = 0*flag
                flag1=(pre_seg!=5*one)
                flag2=(pre_sampled_idx.squeeze(-1)==6*one)
                flag=flag1|flag2
                d_index=flag
                index=index+d_index.long()

                seg = templates[torch.arange(templates.size(0)),index]

                if torch.max(index) < temp_outputs.size(1):
                    t_hidden = temp_outputs[torch.arange(temp_outputs.size(0)),index].unsqueeze(0)
                else:
                    t_hidden = t_hidden

                pre_seg=seg.clone()
                pre_sampled_idx=sampled_idx.clone()

                
            sos_set=(sampled_idx.unsqueeze(-1) == 5).nonzero().squeeze(-1)
            eos_set=(sampled_idx.unsqueeze(-1) == 6).nonzero().squeeze(-1)
            pre_hidden_clone=pre_hidden.clone()
            pre_hidden=hidden.clone()
            if sos_set.size(0)>0:
                for r in range(sos_set.size(0)):
                    rid=sos_set[r,0]
                    pre_hidden[0,rid]=pre_hidden_clone[0,rid].clone()
            if eos_set.size(0)>0:
                for r in range(eos_set.size(0)):
                    rid=eos_set[r,0]
                    pre_hidden[0,rid]=pre_hidden_clone[0,rid].clone()
                    sampled_idx[rid]=eos_sampled_idx[rid]

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, pre_hidden, t_hidden, encoder_outputs, selective_read, one_hot_input_seq)


            decoder_outputs.append(output)

            if mask_tf == True:
                sampled_idxs.append(pre_sampled_idx)
            else:
                sampled_idxs.append(sampled_idx)

            if len(sampled_idxs)==(self.max_length):#last step
                break

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs#, hiddens_atts

    def step(self, prev_idx, prev_hidden, t_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq):

        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = len(self.vocab)

        # Call the RNN
        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)
        rnn_input = torch.cat((prev_hidden.transpose(0,1), t_hidden.transpose(0,1), prev_selective_read, embedded), dim=2)
        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]


        # Attention mechanism
        transformed_hidden = self.attn_W(hidden).view(batch_size, self.hidden_size, 1) # bS, hidden_size, 1
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)
        hidden_att = torch.cat((context.squeeze(1), hidden.squeeze(0)), dim=-1)


        # Generate mechanism
        gen_scores = self.out(hidden_att)
        gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too

        # Copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.

        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
        missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)



        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        gen_probs = probs[:, :vocab_size]

        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        copy_probs = probs[:, vocab_size:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10**-10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)


        # Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read



    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

class QG(object):

    def __init__(self):

        if torch.cuda.is_available():
            cudnn.benchmark = True

        hidden_size=512
        max_length=50
        self.teach_forcing=1.0
        self.mask_tf=False

        with open('../glove/word2idx.json') as inf:
            self.vocab = json.load(inf)
        with open('../glove/usedwordemb.npy') as inf:
            self.word_emb = np.load(inf)

        self.question_encoder = EncoderRNN(self.vocab, self.word_emb, hidden_size)

        self.sample = Sample(4*hidden_size, 64)

        decoder_hidden_size = 2 * hidden_size

        self.decoder = DecoderRNN(self.vocab, self.word_emb, decoder_hidden_size, max_length)

        self.hiddenz = nn.Linear(decoder_hidden_size+64, decoder_hidden_size)

        if torch.cuda.is_available():
            self.question_encoder.cuda()
            self.sample.cuda()
            self.decoder.cuda()
            self.hiddenz.cuda()

        self.loss_function = torch.nn.NLLLoss(ignore_index=0)

        params = list(self.question_encoder.parameters())
        params += list(self.sample.parameters())
        params += list(self.decoder.parameters())
        params += list(self.hiddenz.parameters())
        self.params = params

        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(params,lr=self.learning_rate, weight_decay = 0)
        self.Eiters=0

    def state_dict(self):
        state_dict = [self.question_encoder.state_dict(),self.sample.state_dict(), self.decoder.state_dict(), self.hiddenz.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.question_encoder.load_state_dict(state_dict[0])
        self.sample.load_state_dict(state_dict[1])
        self.decoder.load_state_dict(state_dict[2])
        self.hiddenz.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.question_encoder.train()
        self.sample.train()
        self.decoder.train()
        self.hiddenz.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.question_encoder.eval()
        self.sample.eval()
        self.decoder.eval()
        self.hiddenz.eval()

    def forward_emb(self, sources, qs_tar, mask_tar, temp_tar, sqls_lens, question_lens, mask_lens, template_lens, ids, volatile=False):

        sources = Variable(sources)
        qs_tar = Variable(qs_tar)
        mask_tar = Variable(mask_tar)
        temp_tar = Variable(temp_tar)
        if torch.cuda.is_available():
            sources=sources.cuda()
            temp_tar=temp_tar.cuda()
            mask_tar=mask_tar.cuda()
            qs_tar=qs_tar.cuda()

        batch_size = sources.data.shape[0]

        t_hidden_init = self.question_encoder.init_hidden(batch_size)

        sql_encoder_outputs, sql_hidden = self.question_encoder(sources, t_hidden_init, sqls_lens)

        temp_encoder_outputs, temp_hidden = self.question_encoder(temp_tar, t_hidden_init, template_lens)

        z, mean, logvar = self.sample(sql_hidden, temp_hidden)

        hidden_z = torch.cat((sql_hidden, z), dim=-1)

        hidden_z=self.hiddenz(hidden_z).squeeze(0)

        decoder_outputs, sampled_idxs = self.decoder(sql_encoder_outputs, temp_encoder_outputs, hidden_z, temp_tar, sources, qs_tar, targets=True, teachforce=self.teach_forcing, mask_tf=self.mask_tf)

        return decoder_outputs, sampled_idxs, mean, logvar, z

    def train_emb(self, sources, qs_tar, mask_tar, temp_tar, sqls_lens, question_lens, mask_lens, template_lens, ids, *args):

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        sources = Variable(sources)
        qs_tar = Variable(qs_tar)
        mask_tar = Variable(mask_tar)
        temp_tar = Variable(temp_tar)
        if torch.cuda.is_available():
            sources=sources.cuda()
            temp_tar=temp_tar.cuda()
            mask_tar=mask_tar.cuda()
            qs_tar=qs_tar.cuda()

        data = [sources, qs_tar, mask_tar, temp_tar, sqls_lens, question_lens, mask_lens, template_lens, ids]
        # compute the embeddings
        decoder_outputs, sampled_idxs, mean, logvar, z = self.forward_emb(*data)

        self.optimizer.zero_grad()

        batch_size = sources.shape[0]
        max_length=50

        flattened_outputs = decoder_outputs.view(batch_size * max_length, -1)

        sos_tar=qs_tar.clone()
        sos_tar[sos_tar==5] = 0

        loss = self.loss_function(flattened_outputs, sos_tar.contiguous().view(-1))

        kl_loss = (-0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

        self.logger.update('KL Loss', kl_loss.item(), 1)
        self.logger.update('CE Loss', loss.item(), 1)

        loss = loss + kl_loss

        loss.backward()

        self.optimizer.step()

        batch_outputs = trim_seqs(sampled_idxs)

        np_targets=trim_seqs(qs_tar.unsqueeze(-1))
        batch_targets = [[seq] for seq in np_targets]

        corpus_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)

        self.logger.update('C-BLEU', corpus_bleu_score, qs_tar.size(0))
