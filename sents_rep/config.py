# Config
import  tensorflow as tf
import numpy as np

MODE_fw=tf.constant(np.array([1,0]),tf.int32)
MODE_bw=tf.constant(np.array([0,1]),tf.int32)

#
model_path='/tmp/nmt_transfer_en_zh_model1'
batch_size=8
sen_len=50
learning_step=10000000
nstep=1000
nepochs=10
batches=1420913
dev_batches=157700
langs=2
unk_index=0
beam_search=10

session_conf = tf.ConfigProto(
  device_count = {'CPU': 4, 'GPU':0}, 
  allow_soft_placement=True,
  log_device_placement=False,)

# data 
tran_ckpt_path1 = '/tmp/nmt_model_en_zh_good/inference_model'
tran_ckpt_path2 = '/tmp/nmt_model_zh_en_good/inference_model'
vocab_path_zh = '/tmp/nmt_data/vocab2.zh'
vocab_path_en = '/tmp/nmt_data/vocab2.en'
sentiment_dir = '/opt/xiahong/data/sentiment_corpus/data/'
tag_vocab_path = sentiment_dir + 'vocab.tag' # 0=neu, -1=neg, 1=pos

train_path_en = sentiment_dir + 'train.en'
train_path_zh = sentiment_dir + 'train.zh'

dev_path_en = sentiment_dir + 'dev.en'
dev_path_zh = sentiment_dir + 'dev.zh'
sampling_ratio = [1,0] 

# lstm cell
unit_type='lstm' 
num_units=128 
num_layers=2
num_residual_layers=0
forget_bias=0 
dropout=0.2
mode=tf.contrib.learn.ModeKeys.INFER
num_gpus=0
base_gpu=0
