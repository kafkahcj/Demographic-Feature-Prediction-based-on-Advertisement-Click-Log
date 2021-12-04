import os
import random
import time
import heapq
import copy
import gc
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import argparse



################### Hyperparameters set ###################
parser = argparse.ArgumentParser(description='BiLSTM')
parser.add_argument('--global_seed', type=int, default=2021,
					help='the global seed')
## batch data number
parser.add_argument('--batch', type=int, default=1024,
					help='trian/test/validation batch size')
## embedding
parser.add_argument('--window', type=int, default=20,
					help='embedding window')
parser.add_argument('--size', type=int, default=300,
					help='embedding size')
## hyperparameter in model
parser.add_argument('--model_name', type=str, default='',
					help='the name suffix of the model,default is none')
parser.add_argument('--kfold', type=int, default=5,
					help='the number of fold')
parser.add_argument('--weight1', type=float, default=0.5,
					help='the weight of the No.1 best model in a fold')
parser.add_argument('--weight2', type=float, default=0.3,
					help='the weight of the No.2 best model in a fold')
parser.add_argument('--weight3', type=float, default=0.2,
					help='the weight of the No.3 best model in a fold')
parser.add_argument('--lstm_size', type=int, default=1024,
					help='the lstm size')
parser.add_argument('--fc1', type=int, default=1024,
					help='the number of neurons in fc1 in the model')
parser.add_argument('--fc2', type=int, default=512,
					help='the number of neurons in fc2 in the model')
parser.add_argument('--num_layers', type=int, default=2,
					help='the number of layers in lstm')
parser.add_argument('--rnn_dropout', type=float, default=0.0,
					help='dropout proportion in rnn layer')
parser.add_argument('--fc_dropout', type=float, default=0.0,
					help='dropout proportion in fc layer')
parser.add_argument('--embedding_dropout', type=float, default=0.0,
					help='dropout proportion in embedding layer')
parser.add_argument('--learning_rate', type=float, default=1e-3,
					help='the learning rate of the model')
parser.add_argument('--epochs', type=int, default=8,
					help='the number of epochs')

args = parser.parse_args()
################### data processing functions ###################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    setup_seed(GLOBAL_SEED)
    
    
class CustomDataset(Dataset):
    def __init__(self, seqs, labels, input_num, shuffle=False):
        self.seqs = seqs
        self.labels = labels
        self.input_num = input_num
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        length = int(self.seqs[idx].shape[0]/self.input_num)
        seq_list = list(torch.LongTensor(self.seqs[idx]).split(length, dim=0))          
        label = torch.LongTensor(self.labels[idx])
        # Randomly shuffle the data
        if self.shuffle and torch.rand(1) < 0.5:
            random_pos = torch.randperm(length)
            for i in range(len(seq_list)):
                seq_list[i] = seq_list[i][random_pos]
        return seq_list + [length, label]

def pad_truncate(Batch):
    *seqs, lengths, labels = list(zip(*Batch))
    # The length is cut to 99% of the size
    trun_len = torch.topk(torch.tensor(lengths), max(int(0.01*len(lengths)), 1))[0][-1]
    # To be safe, set a maximum length
    max_len = min(trun_len, 100)
    seq_list = list(pad_sequence(seq, batch_first=True)[:, :max_len] for seq in seqs)
    return seq_list, torch.tensor(lengths).clamp_max(max_len), torch.stack(labels)

############### model class and the loss ##################
class BiLSTM(nn.Module):
    def __init__(self, embedding_list, embedding_freeze, lstm_size, fc1, fc2, num_layers=1, rnn_dropout=0.2, embedding_dropout=0.2, fc_dropout=0.2):
        super().__init__()
        #embedding layers
        self.embedding_layers = nn.ModuleList([nn.Embedding.from_pretrained(torch.HalfTensor(embedding).cuda(), freeze=freeze) for embedding, freeze in zip(embedding_list, embedding_freeze)])
        
        self.input_dim = int(np.sum([embedding.shape[1] for embedding in embedding_list]))
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                                      hidden_size = lstm_size, 
                                      num_layers = num_layers,
                                      bidirectional = True, 
                                      batch_first = True, 
                                      dropout = rnn_dropout) 
                                                  
        
        
        self.fc1 = nn.Linear(2*lstm_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 12)
        
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
    
    def forward(self, seq_list, lengths):
        batch_size, total_length= seq_list[0].size()
        lstm_outputs = []
        click_time = seq_list[-1]
        embeddings = []
        
        for idx, seq in enumerate(seq_list[:-1]):
            #print(self.embedding_layers.num_embeddings)
            embedding = self.embedding_layers[idx](seq).to(torch.float32)
            embedding = self.embedding_dropout(embedding)
            embeddings.append(embedding)
        packed = pack_padded_sequence(torch.cat(embeddings, dim=-1), lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed)
        
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length, padding_value=-float('inf'))
        lstm_output = self.rnn_dropout(lstm_output)
        # lstm_output shape: (batchsize, total_length, 2*lstm_size)
        max_output = F.max_pool2d(lstm_output, (total_length, 1), stride=(1, 1)).squeeze()
        # output shape: (batchsize, 2*lstm_size)
        fc_out = F.relu(self.fc1(max_output))
        fc_out = self.fc_dropout(fc_out)
        fc_out = F.relu(self.fc2(fc_out))
        pred = self.fc3(fc_out)
        
        age_pred = pred[:, :10]
        gender_pred = pred[:, -2:]
        return age_pred, gender_pred

def criterion(age_output, gender_output, labels):
    age_loss = nn.CrossEntropyLoss()(age_output, labels[:, 0])
    gender_loss = nn.CrossEntropyLoss()(gender_output, labels[:, 1])
    return age_loss*0.6 + gender_loss*0.4
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(self.dim, 1)
                       
    def forward(self, X, attn_mask=None):
        score = self.fc1(X).squeeze(2)
        if attn_mask is not None:
            score.masked_fill_(attn_mask, -float('inf'))
        attn_weights = F.softmax(score, 1)
        context = torch.bmm(X.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return context


class AttentionBiLSTM(nn.Module):
    def __init__(self, embedding_list, embedding_freeze, lstm_size, fc1, fc2, num_layers=1, rnn_dropout=0.2, embedding_dropout=0.2, fc_dropout=0.2):
        super().__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding.from_pretrained(torch.HalfTensor(embedding).cuda(), freeze=freeze) for embedding, freeze in zip(embedding_list, embedding_freeze)])
        self.input_dim = np.sum([embedding.shape[1] for embedding in embedding_list])
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                              hidden_size = lstm_size, 
                              num_layers = num_layers,
                              bidirectional = True, 
                              batch_first = True, 
                              dropout = rnn_dropout) 
        
        self.attention = Attention(lstm_size*2)
        self.fc1 = nn.Linear(2*lstm_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 12)
        
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        
    
    def forward(self, seq_list, lengths):
        batch_size, total_length= seq_list[0].size()
        lstm_outputs = []
        click_time = seq_list[-1]

        embeddings = []
        for idx, seq in enumerate(seq_list[:-1]):
            embedding = self.embedding_layers[idx](seq).to(torch.float32)
            embedding = self.embedding_dropout(embedding)
            embeddings.append(embedding)
        packed = pack_padded_sequence(torch.cat(embeddings, dim=-1), lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length, padding_value=0)
        lstm_output = self.rnn_dropout(lstm_output)
        # lstm_output shape: (batchsize, total_length, 2*lstm_size)
        attention_output = self.attention(lstm_output, attn_mask=(seq_list[0]==0))
        # output shape: (batchsize, 2*lstm_size)
        fc_out = F.relu(self.fc1(attention_output))
        fc_out = self.fc_dropout(fc_out)
        fc_out = F.relu(self.fc2(fc_out))
        pred = self.fc3(fc_out)
        age_pred = pred[:, :10]
        gender_pred = pred[:, -2:]
        return age_pred, gender_pred

############### train,test,validate functions ##################
def validate(model, val_dataloader, criterion, history, n_iters):
    global writer
    global best_acc, best_model, validate_history
    model.eval()
    costs = []
    age_accs = []
    gender_accs = []
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            seq_list, lengths, labels = batch
            seq_list_device = [seq.cuda() for seq in seq_list]
            lengths_device = lengths.cuda()  
            labels = labels.cuda() 
            
            age_output, gender_output = model(seq_list_device, lengths_device.cpu()) 
            #loss
            loss = criterion(age_output, gender_output, labels)
            costs.append(loss.item())
            #get label
            _, age_preds = torch.max(age_output, 1)
            _, gender_preds = torch.max(gender_output, 1)
            #accuracy
            age_accs.append((age_preds == labels[:, 0]).float().mean().item())
            gender_accs.append((gender_preds == labels[:, 1]).float().mean().item())
            torch.cuda.empty_cache()
            
    mean_accs = np.mean(age_accs) + np.mean(gender_accs)  #accuracy of validation
    mean_costs = np.mean(costs)
    #plot the validate process
    writer.add_scalar('gender/validate_accuracy', np.mean(gender_accs), n_iters)
    writer.add_scalar('gender/validate_loss', mean_costs, n_iters)
    writer.add_scalar('age/validate_accuracy',np.mean(age_accs), n_iters)
    writer.add_scalar('age/validate_loss', mean_costs, n_iters)
    #save model
    if mean_accs > history['best_model'][0][0]:  
        save_dict = copy.deepcopy(model.state_dict())
        embedding_keys = []
        for key in save_dict.keys():
            if key.startswith('embedding'):
                embedding_keys.append(key)
        for key in embedding_keys:
            save_dict.pop(key)
        heapq.heapify(history['best_model'])
        checkpoint_pth = history['best_model'][0][1]
        heapq.heappushpop(history['best_model'], (mean_accs, checkpoint_pth))
        torch.save(save_dict, checkpoint_pth)
        del save_dict
        gc.collect()
        torch.cuda.empty_cache()
    return mean_costs, mean_accs

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epoch, history, validate_points, scheduler, step=True):
    global writer
    model.train()
    costs = []
    age_accs = []
    gender_accs = []
    val_loss, val_acc = 0, 0
    #use tqdm to show the training progress bar
    with tqdm(total=len(train_dataloader.dataset), desc='Epoch {}'.format(epoch)) as pbar:
        for idx, batch in enumerate(train_dataloader):
            #load data from the CustomDataset
            seq_list, lengths, labels = batch
            #put the data into cuda
            seq_list_device = [seq.cuda() for seq in seq_list]
            lengths_device = lengths.cuda()  #.cuda()
            labels = labels.cuda()
            #put the data into the model
            age_output, gender_output = model(seq_list_device, lengths_device.cpu())    
            #get loss
            loss = criterion(age_output, gender_output, labels)
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #adjuster the learning rate
            if step:
                scheduler.step()
            
            with torch.no_grad():
                #append the loss 
                costs.append(loss.item())
                #get the predicted label
                _, age_preds = torch.max(age_output, 1)
                _, gender_preds = torch.max(gender_output, 1)
                #get the accuracy
                age_accs.append((age_preds == labels[:, 0]).float().mean().item())
                gender_accs.append((gender_preds == labels[:, 1]).float().mean().item())
                #update the progress bar every batch
                pbar.update(labels.size(0))
            #
            
            n_iters = idx + len(train_dataloader)*(epoch-1)
            
            
            if idx in validate_points:
                val_loss, val_acc = validate(model, val_dataloader, criterion, history, n_iters)
                model.train()
            if n_iters % 5 ==0:
                writer.add_scalar('gender/train_accuracy', gender_accs[-1], n_iters)
                writer.add_scalar('gender/train_loss', costs[-1], n_iters)
                writer.add_scalar('age/train_accuracy', age_accs[-1], n_iters)
                writer.add_scalar('age/train_loss', costs[-1], n_iters)
                writer.add_scalar('model/learning_rate', scheduler.get_lr()[0], n_iters)
                pbar.set_postfix_str('loss:{:.4f}, acc:{:.4f}, val-loss:{:.4f}, val-acc:{:.4f}'.format(np.mean(costs[-10:]), np.mean(age_accs[-10:])+np.mean(gender_accs[-10:]), val_loss, val_acc))
            torch.cuda.empty_cache()

def test(oof_train_test, model, test_dataloader, val_dataloader, valid_index, weight=1):
#When testing here, the verification set is also calculated to facilitate subsequent model fusion and search weight improvement.
    model.eval()
    y_val = []
    age_pred = []
    gender_pred = []
    age_pred_val = []
    gender_pred_val = []
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            seq_list, lengths, labels = batch
            seq_list_device = [seq.cuda() for seq in seq_list]
            lengths_device = lengths.cuda()
            age_output, gender_output = model(seq_list_device, lengths_device.cpu())    
            age_pred.append(age_output.cpu())
            gender_pred.append(gender_output.cpu())
            torch.cuda.empty_cache()
            
        for idx, batch in enumerate(val_dataloader):
            seq_list, lengths, labels = batch
            seq_list_device = [seq.cuda() for seq in seq_list]
            lengths_device = lengths.cuda()
            age_output, gender_output = model(seq_list_device, lengths_device.cpu())
            age_pred_val.append(age_output.cpu())
            gender_pred_val.append(gender_output.cpu())
            y_val.append(labels)
            torch.cuda.empty_cache()

    # Add the predictions of the three best-performing models based on the weights 
    oof_train_test[valid_index, :10] += F.softmax(torch.cat(age_pred_val)).numpy() * weight
    oof_train_test[valid_index, 10:12] += F.softmax(torch.cat(gender_pred_val)).numpy() * weight
    oof_train_test[valid_index, 12:] = torch.cat(y_val).numpy()
    oof_train_test[80000:, :10] += F.softmax(torch.cat(age_pred)).numpy() * (1/args.kfold) * weight
    oof_train_test[80000:, 10:12] += F.softmax(torch.cat(gender_pred)).numpy() * (1/args.kfold) * weight



########################### set the random seeds to ensure the results identical #########################
GLOBAL_SEED = args.global_seed
setup_seed(GLOBAL_SEED)

########################### Load data #########################

data_path = './processed_data/'
model_save = './model_save/'
embedding_path = './embedding/word2vec'
res_path = './result/'
if not os.path.exists(model_save):
    os.makedirs(model_save)
if not os.path.exists(res_path):
    os.makedirs(res_path)

########################### read and process the data #########################
df = pd.read_pickle(os.path.join(data_path, 'processed_data_numerical.pkl'))
df['age'] = df['age'] - 1
df['gender'] = df['gender'] - 1

########################### read the embedding data #########################
# window=20, size=300
window = args.window
size = args.size

embedding = np.load(os.path.join(embedding_path, 'embedding_w2v_sg1_hs0_win{}_size{}.npz'.format(window,size)))

creative = embedding['creative_w2v']
ad= embedding['ad_w2v']
advertiser = embedding['advertiser_w2v']
product = embedding['product_w2v']
industry = embedding['industry_w2v']
product_cate = embedding['product_cate_w2v']
del embedding
gc.collect()#clean up the memory

########################### set the model name #########################
model_name = 'attention_bilstm_win{}_size{}{}'.format(window,size,args.model_name)  #change here every experiment

########################### get the embedding features we need  #########################
#get the sequence of users' click
data_seq = df[['creative_id', 'ad_id', 'advertiser_id', 'product_id', 'industry', 'click_times']]
data_seq = data_seq.progress_apply(lambda s: np.hstack(s.values), axis=1).values
#get the word embedding list
embedding_list = [creative, ad, advertiser, product, industry]

########################### construct the pytorch dataset and dataloader ###  #########################
input_num = 6 #sequence number
#set batch size
BATCH_SIZE_TRAIN = args.batch
BATCH_SIZE_VAL = args.batch
BATCH_SIZE_TEST = args.batch
#Kfold
K = args.kfold
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
data_folds = []
valid_indexs = [] # The validation predicted  the results in the final results following the index of the validated data

for idx, (train_index, valid_index) in enumerate(kf.split(X=df.iloc[:80000], y=df.iloc[:80000]['age'])):
    
    valid_indexs.append(valid_index)
    X_train, X_val = data_seq[train_index], data_seq[valid_index]
    X_test = data_seq[80000:] 
    y_train, y_val =  np.array(df.iloc[train_index, -2:]), np.array(df.iloc[valid_index, -2:])
    y_test = np.array(df.iloc[80000:,-2:])
    
    
    train_dataset = CustomDataset(X_train, y_train, input_num, shuffle=True)
    val_dataset = CustomDataset(X_val, y_val, input_num, shuffle=False)
    test_dataset = CustomDataset(X_test, y_test, input_num, shuffle=False)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE_TRAIN, 
                                  shuffle=True, 
                                  collate_fn=pad_truncate, 
                                  num_workers=0, 
                                  worker_init_fn=worker_init_fn)
    
    valid_dataloader = DataLoader(val_dataset, 
                                  batch_size=BATCH_SIZE_VAL, 
                                  sampler=SequentialSampler(val_dataset), 
                                  shuffle=False, 
                                  collate_fn=pad_truncate, 
                                  num_workers=0, 
                                  worker_init_fn=worker_init_fn)
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=BATCH_SIZE_TEST, 
                                 sampler=SequentialSampler(test_dataset), 
                                 shuffle=False, 
                                 collate_fn=pad_truncate, 
                                 num_workers=0, 
                                 worker_init_fn=worker_init_fn)
    data_folds.append((train_dataloader, valid_dataloader, test_dataloader))

del data_seq, creative, ad, advertiser, product, industry, product_cate
gc.collect()

################################## train the model ################################
#set the parameters 
#Columns 0 to 9 store the predicted probability distribution of age, 
#columns 10 to 11 store the predicted probability distribution of gender
#,and columns 12 and 13 store the real labels of age and gender, respectively  

#a list which save the validation and prediction results
oof_train_test = np.zeros((100000, 14))
#save accuracy for each fold
acc_folds = []
#get the best 3 model, to do the fusion model in the prediction 
best_checkpoint_num = 3
#set model hyperparameters
lstm_size = args.lstm_size
fc1 = args.fc1
fc2 = args.fc2
num_layers = args.num_layers
rnn_dropout =args.rnn_dropout
fc_dropout =args.fc_dropout
embedding_dropout =args.embedding_dropout
learning_rate = args.learning_rate
epochs = args.epochs

for idx, (train_dataloader, val_dataloader, test_dataloader) in enumerate(data_folds):
    history = {'best_model': []}
    #save model
    for i in range(best_checkpoint_num):
        history['best_model'].append((0, os.path.join(model_save, '{}_checkpoint_{}.pth'.format(model_name, i))))
    
    #embedding: creative_w2v, ad_w2v, advertiser_w2v, product_w2v, industry_w2v
    embedding_freeze = [True, True, True, True, True]
    
    #Return evenly spaced numbers over a specified interval. 
    validate_points = list(np.linspace(0, len(train_dataloader)-1, 5).astype(int))[1:]
    
    #set model
    model = AttentionBiLSTM(embedding_list, embedding_freeze, lstm_size=lstm_size, fc1=fc1, fc2=fc2,  num_layers=num_layers, rnn_dropout=rnn_dropout, fc_dropout=fc_dropout, embedding_dropout=embedding_dropout)    
    model = model.cuda() #cuda
    
    #set optimizer and epochs
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    
    
    #set learning rate optimization strategy
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=2e-3, step_size_up=int(len(train_dataloader)/2), cycle_momentum=False, mode='triangular')
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_dataloader), pct_start=0.2, anneal_strategy='linear', div_factor=30, final_div_factor=1e4)
    
    #start training
    for epoch in range(1, epochs+1):
        writer = SummaryWriter(log_dir='./runs/{}_fold_{}'.format(model_name, idx))
        train(model, train_dataloader, val_dataloader, criterion, optimizer, epoch, history, validate_points, scheduler, step=True)
        gc.collect()
        
    for (acc, checkpoint_pth), weight in zip(sorted(history['best_model'], reverse=True), [args.weight1, args.weight2, args.weight3]):
        #load the model
        model.load_state_dict(torch.load(checkpoint_pth, map_location=torch.device('cpu')), strict=False)
        #test model
        test(oof_train_test, model, test_dataloader, val_dataloader, valid_indexs[idx], weight=weight)
    #get the best model accuracy in the acc_folds
    acc_folds.append(sorted(history['best_model'], reverse=True)[0][0])
    np.save(os.path.join(model_save, "{}_fold_{}.npy".format(model_name, idx)), oof_train_test)
    del model, history
    gc.collect()
    torch.cuda.empty_cache()
    
    
    
#save results
np.save(os.path.join(res_path, "{}_{}folds_{:.4f}.npy".format(model_name,args.kfold, np.mean(acc_folds))), oof_train_test)


    
