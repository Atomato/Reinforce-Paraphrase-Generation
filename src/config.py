import os

# root_dir is "src/.."
root_dir = os.path.join(os.getcwd(),'..')

print_interval = 100
save_model_iter = 100

data_path = os.path.join(root_dir, "data/kor_double")

train_data_path = os.path.join(data_path, "chunked/train_*")
eval_data_path = os.path.join(data_path, "chunked/val_*")
decode_data_path = os.path.join(data_path, "chunked/test_*")
vocab_path = os.path.join(data_path, "vocab")
emb_v_path = os.path.join(data_path, "word_emb.txt")
emb_list_path = os.path.join(data_path, "word_list.txt")
post_process_rule_path = os.path.join(data_path, "post_process_rules.txt")

log_root = os.path.join(root_dir, "log/MLE_DOUBLE")

# Hyperparameters
mode = "MLE"   # other options: RL/GTI/SO/SIO/DAGGER/DAGGER*
alpha = 1.0
beta = 1.0
k1 = 0.9999
k2 = 3000.
hidden_dim= 256
emb_dim= 128
batch_size= 8
sample_size= 4

max_enc_steps= 50 # 20
max_dec_steps= 50 # 20

beam_size= 8
min_dec_steps= 5
vocab_size= 1700 # 5000

max_iterations = 5000000
lr = 1e-5
pointer_gen = True
is_coverage = False
lr_coverage = 0.15
cov_loss_wt = 1.0
max_grad_norm = 2.0
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
eps = 1e-12
use_gpu = True