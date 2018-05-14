#encoding=utf-8
'''
generate source and target tensor from data
'''
import numpy as np
import config
import Queue
import random
from functools import wraps

class Vocab():

    def __init__(self, vocab_path):
        self.vocab, self.reverse_vocab=self.get_vocab(vocab_path)

    def get_vocab(self, vocab_path):
        ii=open(vocab_path, 'r')
        vocab,reverse_vocab={},{}
        for line in ii:
            word=line.strip()
            k=len(vocab)
            vocab[word]=k
            reverse_vocab[k]=word
        return vocab, reverse_vocab

    def __getitem__(self, x):
        return self.vocab[x]

    def __contains__(self, x):
        return x in self.vocab

def returnNone():
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            try:
                rval=func(*args, **kwargs)
            except Exception,e:
                rval=None
            return rval
        return wrap
    return decorator

class sequence_line_processing():
    
    def __init__(self, vocab_path, max_len=100, split=' ', return_length=True):
        self.vocab= Vocab(vocab_path)
        self.max_len =max_len
        self.split=split
        self.return_length=return_length

    def __call__(self, line):
        words=line.strip().split(self.split)
        indexs=[]
        for word in words:
            if word in self.vocab:
                indexs.append(self.vocab[word])
            else:
                indexs.append(self.vocab['<unk>'])
            if len(indexs)>=config.sen_len:
                break
        l=len(indexs)
        if len(indexs)<self.max_len:
            indexs=indexs+(self.max_len-len(indexs))*[self.vocab['</s>'],]
        if self.max_len!=1:
            item=np.array(indexs, dtype=np.int32)
        else:
            item=indexs[0]
        if not self.return_length:
            return item
        return [item, l]

class sequence_label_line_processing():

    def __init__(self, vocab_path, tag_vocab_path):
        self.sq_process = sequence_line_processing( vocab_path , max_len = config.sen_len )
        self.tag_process = sequence_line_processing( tag_vocab_path , return_length=False, max_len=1)
        self.size=3

    @returnNone()
    def __call__(self, line):
        words = line.strip().split('\t')
        tags=words[0]
        seq=' '.join(words[1:])
        rval = self.sq_process(seq)+[self.tag_process(tags)]
        return rval

class Dataset():

    def __init__(self, data_path, line_processing=None, queue_size=100000, save_all=False, use_length=True,
            len=len):
        self.file_reader = None
        self.data_path = data_path
        self.queue_size=queue_size
        self.line_processing=line_processing
        self.size=line_processing.size
        self.save_all=False
        self.len=len
        if save_all:
            self.all_data=list(self.epoch_data())
            self.save_all=save_all
            self.len=len(self.all_data)
            print 'len===============',self.len
    
    def __len__(self):
        if self.len:
            return self.len
        else:
            self.len=len(list(self.epoch_data()))

    def epoch_data(self):
        if self.save_all:
            for d in self.all_data:
                yield d
            return
        self.queue = Queue.Queue()
        print self.data_path
        self.file_reader = open(self.data_path, 'r')
        while True:
            if False:
            # if not self.queue.empty():
                #yield self.queue.get()
                pass
            else:
                batch=[[],]*self.size
                k=0
                for line in self.file_reader:
                    k+=1
                    items=self.line_processing(line)
                    if not items or len(items)!=self.size:
                        #print items
                        print 'Not correct parsed:\t',line,len(line)
                        #raw_input('xxxxxxxxxxxxx')
                        continue
                    for k,item in enumerate(items):
                        batch[k]=batch[k]+[item]
                    if len(batch[0])==config.batch_size:
                        batch=[np.array(x) for x in batch]
                        #self.queue.put( batch )
                        #print 'put'
                        yield batch
                        batch=[[],]*self.size
                    #if self.queue.qsize()==self.queue_size:
                    #    break
                if k==0:break
class MultiDataset():

    def __init__(self, datasets):
        self.datasets=datasets
        self.size=sum(d.size for d in datasets)

    def epoch_data(self):
        self.iterators=[x.epoch_data() for x in self.datasets]
        while True:
            batch_item = []
            for it in self.iterators:
                item=it.next()
                if type(item)==type([1,2]):
                    batch_item+=item
                else:
                    batch_item.append(item)
            if not batch_item:
                yield bctch_item
            else:
                break

    def __len__(self):
        return len(self.datasets[0])

class SampleDataset():

    def __init__(self, datasets, ratios):
        self.datasets=datasets
        self.ratios=[sum(ratios[:i+1])*1.0/sum(ratios) for i in range(len(ratios))]
        self.size=sum(d.size for d in datasets)

    def select(self,r):
        for i in range(len(self.ratios)):
            if self.ratios[i]>=r:
                return i
        
    def epoch_data(self):
        iterators=[x.epoch_data() for x in self.datasets]
        finished=[0,]*len(self.ratios)
        finished[1]=1
        # TODO 这里手动让第二数据集直接结束
        while True:
            r=random.random()
            k=self.select(r)
            batch_item=[None,]*self.size
            item=None
            try:
                item=iterators[k].next()
                mask=[0,]*len(self.datasets)
                mask[k]=1
                if not isinstance(item, list): item=[item]
                size=sum(d.size for d in self.datasets[:k])
                batch_item[size:size+len(item)]=item
                batch_item[size+len(item):size+2*len(item)]=item
                # 上面一行指导了无效数据
                batch_item.append(np.array(mask))
                yield batch_item
            except Exception,e:
                print e
                finished[k]=1
                if sum(finished)==len(finished):
                    break
    
    def __len__(self):
        self.len=sum(len(d) for d in  self.datasets)
        return self.len
