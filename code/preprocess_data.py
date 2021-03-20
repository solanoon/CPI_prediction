import pickle
import sys
import timeit
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import errno
import collections
from torch.autograd import Variable
from typing import Optional
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


class FocalLoss(nn.Module):
    def __init__(self, num_class=2, alpha=0.4, gamma=2, balance_index=0, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class 
        self.alpha = alpha 
        self.gamma = gamma 
        self.smooth = smooth 
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
    
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss 
'''

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
'''

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        C, P, inputs, correct_interaction = data[0], data[1], data[2:-1], data[-1]
        predicted_interaction = self.forward(inputs)
#         all__ = len(correct_interaction)
#         pos__ = sum(correct_interaction)
#         neg__ = all__ - pos__
#         weight = [pos__/all__, neg__/all__]
#         print(weight)

        if train:
#             loss = FocalLoss(gamma=2)(predicted_interaction, correct_interaction)
#             loss = F.cross_entropy(predicted_interaction, correct_interaction, weight=torch.FloatTensor(weight).cuda())
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return C, P, correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
#         print('when training, length of data: ',N)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        np.random.shuffle(dataset)
        C, P, T, Y, S = [], [], [], [], []
        for data in dataset:
            (comids, proids, correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)

            C.append(comids)
            P.append(proids)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
#         result_dic = pd.DataFrame(np.array([C,P,T,Y]).reshape(-1,4),columns = ['com_id','pro_id','label','predicted'])
        result_dic = pd.DataFrame(list(zip(C,P,T,Y)),columns = ['com_id','pro_id','label','predicted'])
        #print('True label: ',T[0])#print('protein: \n',P)
        #print('compound: \n',C)
        # print('sum of correct_labels and predicted labels: ', collections.Counter(T), collections.Counter(Y))
        # AUC = roc_auc_score(T, S) if len(T) != sum(T) else 0
        AUC = 0
        accuracy = accuracy_score(T, Y)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        #return AUC, precision, recall, result_dic
        #too many true label, delete AUC
        return AUC, precision, recall, accuracy, result_dic

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    
    def save_result(self,result_dic, filename):
        result_dic.to_csv(filename, index=False)


    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle = True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    pos_dataset = []
    neg_dataset = []
    for i_ in dataset:
        if i_[5] == 1:
            pos_dataset.append(i_)
        elif i_[5] == 0:
            neg_dataset.append(i_)
    print('length of positive: ', len(pos_dataset))
    print('length of negative: ', len(neg_dataset))
    n1 = int(ratio * len(pos_dataset))
    n2 = int(ratio * len(neg_dataset))
    print('train positive: %i, train negative: %i' % (n1, n2))
    pos_dataset_1, pos_dataset_2 = pos_dataset[:n1], pos_dataset[n1:]
    neg_dataset_1, neg_dataset_2 = neg_dataset[:n2], neg_dataset[n2:]
    dataset_1 = pos_dataset_1 + neg_dataset_1
    dataset_2 = pos_dataset_2 + neg_dataset_2
    print('length of train: %i, of validation: %i.'%(len(dataset_1),len(dataset_2)))
    return dataset_1, dataset_2

def load_preprocess_data(data_type, DATASET, radius, ngram):
    dir_input = ('../dataset/' + DATASET + '/' + str(i_) + '/input/' + data_type + '_radius' + radius + '_ngram' + ngram + '/')
    comid = np.load(dir_input + 'com_id.npy')
    proid = np.load(dir_input + 'pro_id.npy')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    print('In the %s set, length of data: %i, number of positives: %i' % (data_type, len(interactions), sum(interactions)))
    
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    
    dataset = list(zip(comid, proid, compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    return fingerprint_dict, word_dict, dataset

def update_dict(Adict, Bdict):
    final_dict = dict()
    final_dict.update(Adict)
    final_dict.update(Bdict)
    return final_dict


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration, fold, lossandepoch,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration, fold) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration, fold])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    for i_ in range(0, fold):
        
        tra_fp_dict, tra_word_dict, tra_dataset = load_preprocess_data('tra', DATASET, radius, ngram)
        tes_fp_dict, tes_word_dict, tes_dataset = load_preprocess_data('tes', DATASET, radius, ngram)

        fingerprint_dict = update_dict(tra_fp_dict, tes_fp_dict)
        word_dict = update_dict(tra_word_dict, tes_word_dict)

        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        dataset_train, dataset_dev = split_dataset(tra_dataset, 0.9)
        dataset_test = tes_dataset
        
#         tra_dir_input = ('../dataset/' + DATASET + '/' + str(i_) + '/input/'
#                      'tra_radius' + radius + '_ngram' + ngram + '/')
#         tra_comid = np.load(tra_dir_input + 'com_id.npy')
#         tra_proid = np.load(tra_dir_input + 'pro_id.npy')
#         tra_compounds = load_tensor(tra_dir_input + 'compounds', torch.LongTensor)
#         tra_adjacencies = load_tensor(tra_dir_input + 'adjacencies', torch.FloatTensor)
#         tra_proteins = load_tensor(tra_dir_input + 'proteins', torch.LongTensor)
#         tra_interactions = load_tensor(tra_dir_input + 'interactions', torch.LongTensor)
#         print('length of data: %i, number of positives: %i' % (len(tra_interactions), sum(tra_interactions)))
#         weight = [sum(tra_interactions)/len(tra_interactions), 1-(sum(tra_interactions)/len(tra_interactions))]
#         print('weight: ', weight)
#         tra_fingerprint_dict = load_pickle(tra_dir_input + 'fingerprint_dict.pickle')
#         tra_word_dict = load_pickle(tra_dir_input + 'word_dict.pickle')

#         tes_dir_input = ('../dataset/' + DATASET + '/' + str(i_) + '/input/'
#                      'tes_radius' + radius + '_ngram' + ngram + '/')
#         tes_comid = np.load(tes_dir_input + 'com_id.npy')
#         tes_proid = np.load(tes_dir_input + 'pro_id.npy')
#         tes_compounds = load_tensor(tes_dir_input + 'compounds', torch.LongTensor)
#         tes_adjacencies = load_tensor(tes_dir_input + 'adjacencies', torch.FloatTensor)
#         tes_proteins = load_tensor(tes_dir_input + 'proteins', torch.LongTensor)
#         tes_interactions = load_tensor(tes_dir_input + 'interactions', torch.LongTensor)
#         tes_fingerprint_dict = load_pickle(tes_dir_input + 'fingerprint_dict.pickle')
#         tes_word_dict = load_pickle(tes_dir_input + 'word_dict.pickle')

#         fingerprint_dict = dict()
#         fingerprint_dict.update(tra_fingerprint_dict)
#         fingerprint_dict.update(tes_fingerprint_dict)

#         word_dict = dict()
#         word_dict.update(tra_word_dict)
#         word_dict.update(tes_word_dict)

        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        """Create a dataset and split it into train/dev/test."""
#         tra_dataset = list(zip(tra_comid, tra_proid, tra_compounds, tra_adjacencies, tra_proteins, tra_interactions))
#         tes_dataset = list(zip(tes_comid, tes_proid, tes_compounds, tes_adjacencies, tes_proteins, tes_interactions))
#         dataset = shuffle_dataset(dataset, 1234)
#         tra_dataset = shuffle_dataset(tra_dataset, 1234)
#         dataset_train, dataset_dev = train_test_split(tra_dataset, stratify=tra_dataset[5])
        dataset_train, dataset_dev = split_dataset(tra_dataset, 0.8)
        # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
        dataset_test = tes_dataset
#         print('There are %i data in testset, among them %i are positives.' %(len(dataset_test), sum(tes_interactions)))

        """Set a model."""
        torch.manual_seed(1234)
        model = CompoundProteinInteractionPrediction().to(device)
        trainer = Trainer(model)
        tester = Tester(model)

        """Output files."""
        file_AUCs = '../output/result/AUCs--' + DATASET + '/' + lossandepoch + '_' + str(i_) + '.txt'
        file_result = '../output/result/' + DATASET + '/' + lossandepoch + '_' + str(i_) + '.csv'
        file_model = '../output/model/' + DATASET + '/' + lossandepoch + '_' + str(i_)

        AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
                'AUC_test\tPrecision_test\tRecall_test')
        perfor_ = ('Epoch\tTime(sec)\tLoss_train\tVal_pre\tVal_recall\tVal_acc\tTest_pre\tTest_recall\tTest_acc')
        files = [file_AUCs, file_result, file_model]
        for filename in files:
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        """Start training."""
        print('Training...')
#         print(AUCs)
        print(perfor_)
        start = timeit.default_timer()

        for epoch in range(0, iteration):

            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss_train = trainer.train(dataset_train)
            pre_dev, recall_dev, acc_dev = tester.test(dataset_dev)[1:4]
#             print('Epoch %i Validation Performance:\n Precision: %.4f, Recall: %.4f, Acc: %.4f' % (epoch, pre_dev, recall_dev, acc_dev))
            #AUC_test,  precision_test, recall_test, result_dic = tester.test(dataset_test)
            AUC_test, precision_test, recall_test, accuracy_test, result_dic = tester.test(dataset_test)
#             print('Epoch %i Test: Precision:\n %.2f. Recall: %.2f. Accuracy: %.2f.' % (epoch, precision_test, recall_test, accuracy_test))
            

            end = timeit.default_timer()
            time = end - start

            #AUCs = [epoch, time, loss_train, AUC_dev,
            #        AUC_test, precision_test, recall_test]
            #tester.save_AUCs(AUCs, file_AUCs)
            perf_ = [epoch, round(time,1), round(loss_train,3), round(pre_dev,3), round(recall_dev,3), round(acc_dev,3), round(precision_test,3), round(recall_test,3), round(accuracy_test,3)]
            tester.save_result(result_dic, file_result)
            tester.save_model(model, file_model)
            
            print('\t'.join(map(str, perf_)))

            #print('\t'.join(map(str, AUCs)))
