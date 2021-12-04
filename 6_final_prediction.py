import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = './processed_data/'
res_path = './result/'
save_path = './final_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
##### load original data #####
df = pd.read_pickle(os.path.join(data_path, 'processed_data_numerical.pkl'))
df['age'] = df['age'] - 1
df['gender'] = df['gender'] - 1

print(os.listdir(res_path))

single_model_results=['attention_bilstm_win10_size512_5folds_1.1049.npy',
                      'attention_bilstm_win30_size300_5folds_1.1045.npy',
                      'attention_bilstm_win50_size300_5folds_1.1052.npy'
                      'bilstm_win100_size300_5folds_1.1054.npy', 
                      'bilstm_win10_size300_5folds_1.1048.npy',
                      'bilstm_win10_size512_5folds_1.1070.npy',
                      'bilstm_win128_size128_5folds_1.1067.npy',
                      'bilstm_win20_size300_5folds_1.1068.npy', 
                      'bilstm_win30_size300_5folds_1.1080.npy',
                      'bilstm_win40_size300_5folds_1.1065.npy',
                      'bilstm_win50_size300_10fold_10folds_1.1109.npy',
                      'bilstm_win50_size300_5folds_1.1081.npy']
def load_res(name):
    res = np.load(os.path.join(res_path, name))
    X_train = res[:80000, :12]
    y_train = res[:80000, 12:]
    X_test = res[80000:, :12]
    return X_train, y_train, X_test

X_train_list = []
y_train_list = []
X_test_list = []

for name in single_model_results:
    X_train, y_train, X_test = load_res(name)
    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)

X_train = np.stack(X_train_list)
y_train = y_train_list[0]
X_test = np.stack(X_test_list)

y_pred_age = X_test.mean(axis=0)[:, :10].argmax(axis=1)
y_pred_gender = X_test.mean(axis=0)[:, 10:].argmax(axis=1)


y_test_age =np.array(df.iloc[80000:,-2:]['age'])
y_test_gender=np.array(df.iloc[80000:,-2:]['gender'])

age_acc = (y_pred_age==y_test_age).mean()
gender_acc = (y_pred_gender==y_test_gender).mean()
all_acc = age_acc+gender_acc

print('age accuracy: {}  gender accuracy:{}  total accuracy:{} '.format(age_acc,gender_acc,all_acc))