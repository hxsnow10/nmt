# encoding=utf-8

from contact_bi_model import BiContactRepr, ReprClf, LstmclfModel
from data_utils import Dataset, MultiDataset, SampleDataset, sequence_label_line_processing
import config
from work_utils import train as train_func


# line_processing_function
en_process = sequence_label_line_processing( config.vocab_path_en, config.tag_vocab_path)
zh_process = sequence_label_line_processing( config.vocab_path_zh, config.tag_vocab_path)

# data
train_en = Dataset(config.train_path_en, en_process, len=config.batches)
train_zh = Dataset(config.train_path_zh, zh_process, len=0)
train = SampleDataset([train_en, train_zh], config.sampling_ratio)
# return train_en_sequence, en_sequence_lenth, tag_en, 
#        train_zh_sequence, zh_sequence_lenth, tag_zh,
#        mask([0,0],[0,1],[1,0],[1,1])

dev_en = Dataset(config.dev_path_en, en_process, len=config.dev_batches)
dev_zh = Dataset(config.dev_path_zh, zh_process, len=0)
dev = SampleDataset([dev_en, dev_zh], config.sampling_ratio)
k=0
#raw_input('wtf')
'''
for k,input_data in enumerate(train.epoch_data()):
    print k
    print k,'-'*30
    print input_data[2]
    raw_input('TARGET')
    for item in input_data:
        print type(item)
        #continue
        if item is not None:    
            print item.shape
        else:
            print item
    print '-'*30
print 'finished'
quit()
'''
# model
reprmodel = BiContactRepr (config.tran_ckpt_path1, config.tran_ckpt_path2)
model = ReprClf( reprmodel, LstmclfModel )
raw_input('wtf')
train_func(model,train,dev)
#evualate(model,dev)
