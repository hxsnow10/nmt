# encoding=utf-8
import sys
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from nmt.model import Model as TranslateModel
from nmt import model_helper
import config
from tensorflow.python.ops import lookup_ops
#reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
#    tgt_vocab_file, default_value=vocab_utils.UNK)


class Inputs():
    en_sequence = tf.placeholder(tf.int32,[config.batch_size, config.sen_len], name='sequence1')
    en_sequence_length = tf.placeholder(tf.int32, [config.batch_size], name='sequence_length1')
    tag_en = tf.placeholder(tf.int32, [config.batch_size])
    zh_sequence = tf.placeholder(tf.int32,[config.batch_size, config.sen_len], name='sequence2')
    zh_sequence_length = tf.placeholder(tf.int32, [config.batch_size], name='sequence_lenth2')
    tag_zh = tf.placeholder(tf.int32, [config.batch_size])

inputs=Inputs()

def equal(x,y):
    e = tf.equal(x,y)
    return tf.equal(tf.reduce_sum(tf.cast(e, tf.float32)),config.langs)

class BiContactRepr():

    def __init__(self, ckpt_path1, ckpt_path2):
        # TODO TRANSLATION MODEL do reverse sent 
        dir(tf.contrib)
        def load(ckpt_path):
            old_var_d={}
            with tf.Graph().as_default() as g:
                with tf.Session(config=config.session_conf) as sess: 
                    new_saver = tf.train.import_meta_graph('{}.meta'.format( ckpt_path ),clear_devices=True)
                    #sess.run(init)
                    all_vars=tf.global_variables()
                    new_saver.restore(sess, ckpt_path)
                    init = tf.global_variables_initializer()
                    table_init = tf.tables_initializer()
                    print init.name
                    print table_init.name
                    # raw_input('xxxxxxxxxxxx')
                    for v in all_vars:
                        w= sess.run(v)
                        #v._init_from_args(w,trainable=False)
                        #v=v.initial_value
                        old_var_d[v.name]=w
                        #print v.initial_value
                        #print(v_) 
                    #all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
                    #for name in all_names:
                    #    if 'table' in name or 'Next' in name:
                    #        print name
                    #raw_input('xxxxxxxxxxxxxxxxxxasdksasjkhdgsa')
                # meta_graph store all global variables
                print 'Original Vars=', all_vars
                #'''
                #with tf.Session(config=session_conf) as sess:
                #'''
            gdef = g.as_graph_def()

            return gdef, old_var_d
         
        gdef_1, var_d_1 = self.gdef_1=load(ckpt_path1)
        gdef_2, var_d_2 = self.gdef_2=load(ckpt_path2)
        #with tf.Graph().as_default() as g_combined:
        infer_inputs = self.infer_inputs=['IteratorGetNext:0', 'IteratorGetNext:1']
        res = self.res=['dynamic_seq2seq/encoder/rnn/TensorArrayStack/TensorArrayGatherV3:0',
            'dynamic_seq2seq/decoder/decoder/GatherTree:0','dynamic_seq2seq/decoder/decoder/while/Exit_20:0']
            #'dynamic_seq2seq/decoder/decoder/TensorArrayStack_1/TensorArrayGatherV3:0'] 
        # /usr/lib/python2.7/site-packages/tensorflow/python/framework/ops.py
        # 2400-2401 raise error, so i add # before them
        tsl = target_sequence_length = tf.constant(config.sen_len, dtype= tf.int32, shape=(config.batch_size,))
        # infact decoder return sequence_length
        self.inputs1 = inputs1 =  [inputs.en_sequence, inputs.en_sequence_length]
        self.inputs2 = inputs2 =  [inputs.zh_sequence, inputs.zh_sequence_length]
        def repr_reshape(x):
            return tf.transpose(x,[1, 0, 2])
        def sent_reshape(x):
            if config.beam_search:
                x=x[:,:,0]
            #x = tf.transpose(x)[:,:config.sen_len]
            x=tf.transpose(x)
            pad_num=config.sen_len-tf.shape(x)[1]
            pad=tf.ones([config.batch_size, pad_num],dtype=tf.int32) * config.unk_index
            x=tf.concat([x,pad],1)
            return x
        
        self.repr1, self.sent1_2, self.sent1_2_length=\
            tf.import_graph_def(gdef_1, input_map=dict(zip(infer_inputs, inputs1)), 
            return_elements=res, name='fw')
        print(self.sent1_2.get_shape())
        self.sent1_2=sent_reshape(self.sent1_2)
        self.repr1=repr_reshape(self.repr1)
        
        self.repr1_2_1, self.sent1_2_1, _=\
            tf.import_graph_def(gdef_2, input_map=dict(zip(infer_inputs,[self.sent1_2, self.sent1_2_length])), 
            return_elements=res, name='fwbw')
        self.repr1_2_1=repr_reshape(self.repr1_2_1)

        self.repr2, self.sent2_1, self.sent2_1_length=\
            tf.import_graph_def(gdef_2, input_map=dict(zip(infer_inputs, inputs2)), 
            return_elements=res, name='bw')
        self.sent2_1=sent_reshape(self.sent2_1)
        self.repr2=repr_reshape(self.repr2)
        
        self.repr2_1_2, self.sent2_1_2, _=\
            tf.import_graph_def(gdef_1, input_map=dict(zip(infer_inputs,[self.sent2_1, self.sent2_1_length])), 
            return_elements=res, name='bwfw')
        self.repr2_1_2=repr_reshape(self.repr2_1_2)

        self.build()
        all_vars = tf.global_variables()
        old_vars = [tf.get_default_graph().get_tensor_by_name('fw/'+name) for name in var_d_1]
        all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        self.old_inits = ops =\
            [o for o in tf.get_default_graph().get_operations() 
                if 'init' in o.name.split('/')[-1] or 'init_all_tables_1' in o.name.split('/')[-1]]
        print '='*20+'Old Variables fw'+'='*2
        with tf.Session(config=config.session_conf) as sess:
            sess.run(ops)
            for var in old_vars:
                print var.name,sess.run(var).shape

    def build(self):

        self.fw_repr = tf.concat([self.repr1, self.repr1_2_1], 2)
        self.bw_repr = tf.concat([self.repr2_1_2, self.repr2], 2)
        self.middle_repr = tf.concat([self.repr1, self.repr2], 2)
        self.mode=tf.placeholder(shape=[None],dtype=tf.int32)
        self.sent_repr=tf.case(
                [(equal(self.mode, config.MODE_fw),lambda :self.fw_repr),
                (equal(self.mode, config.MODE_bw),lambda :self.bw_repr)],
                default=lambda :self.middle_repr)
        self.sent_repr=self.fw_repr
        self.sent_repr.set_shape((config.batch_size,config.sen_len,2*1024))
        #TODO   I should make sure this shape and input shape compatible
        #TODO   I should make sure what mode this model of nmt before svaed
        #TODO   dropout of nmt model and new model
        self.inputs=self.inputs1 + self.inputs2 + [self.mode]

class ReprClf():
    
    def __init__(self, repr_model, clf_net_model, *args, **kwargs):
        self.sent_repr_model=repr_model
        self.sent1_2=self.sent_repr_model.sent1_2
        self.old_inits=self.sent_repr_model.old_inits
        self.clf_model=clf_net_model(inputs=self.sent_repr_model.sent_repr)
        self.target1=inputs.tag_en
        self.target2=inputs.tag_zh

        self.target=tf.case(
                [(equal(self.sent_repr_model.mode, config.MODE_fw),lambda :self.target1)],
                default=lambda: self.target2)

        self.loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.target, logits=self.clf_model.logits))
        
        self.inputs=self.sent_repr_model.inputs[:2]+[self.target1]+\
                self.sent_repr_model.inputs[2:4]+[self.target2]+\
                [self.sent_repr_model.inputs[-1]]

        self.outputs=tf.cast(tf.argmax(self.clf_model.logits, axis=1), tf.int32)
        with tf.Session() as sess:
            for var in tf.global_variables():
                print var.name, var.get_shape().as_list()
        #TODO evaluataons
        # self.accuracy,_=tf.metrics.accuracy(predictions=self.outputs, labels=self.target)
        # self.recall,_=tf.metrics.recall(predictions=self.outputs, labels=self.target)
        self.matches = tf.reduce_sum(tf.cast(tf.equal(self.outputs, self.target), tf.float32))
        self.size = tf.cast(tf.size(self.target),tf.float32)
        self.accuracy=self.matches/self.size
        self.evaluate_vars=[self.accuracy, self.loss]
        self.evaluate_names=['accuracy', 'loss']

        

    def preidicts():
        pass

class LstmclfModel():
    ''' stack lstm + dense ''' 
    def __init__(self, inputs=None, layers=1, hidden_sizes=[]):
        self.build(inputs)
        
    def lstm(self, inputs, layer_num=None):
        layer_num = layer_num  or config.layer_num
    def create_rnn_cell(self):
        cell=model_helper.create_rnn_cell(
                config.unit_type, config.num_units, config.num_layers, config.num_residual_layers,
                config.forget_bias, config.dropout, config.mode, config.num_gpus, config.base_gpu)
        return cell

    def build(self, inputs):
        with tf.variable_scope('clf'):
            fw_cell=self.create_rnn_cell()
            bw_cell=self.create_rnn_cell()
            outputs, outputs_states=tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
            out=tf.concat([outputs[0][:,-1],outputs[1][:,-1]],1)
            self.logits=tf.layers.dense(out, 2, activation=tf.sigmoid)
             
if __name__=='__main__':
   reprmodel = BiContactRepr (
           '/tmp/nmt_model_en_zh1/translate.ckpt-10000', 
           '/tmp/nmt_model_zh_en1/translate.ckpt-1000')
   model = ReprClf( reprmodel, LstmclfModel )
