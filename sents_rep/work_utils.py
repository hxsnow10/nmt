# encoding=utf-8
import config
import os
import tensorflow as tf
import traceback

def train_epoch(sess, model, train, dev, epoch):
    """
    Performs one complete pass over the train set and evaluate on dev
    Args:
        sess: tensorflow session
        train: dataset that yields tuple of sentences, tags
        dev: dataset
        tags: {tag: index} dictionary
        epoch: (int) number of the epoch
    """
    nbatches = config.batches
    # prog = Progbar(target=nbatches)
    for i, input_data in enumerate(train.epoch_data()):
        #for x in input_data:
        #    print x.shape
        step = epoch*nbatches + i
        print step,'started'
        fd= dict(x for x in zip(model.inputs, input_data) if x[1] is not None)
        try:
            #vs=sess.run(model.sent_repr_model.ooo, feed_dict=fd)
            #for v in vs:
            #    print v.shape
            sent2_length=sess.run(model.sent_repr_model.sent1_2_length,feed_dict=fd)
            print sent2_length
            sent2=sess.run(model.sent1_2, feed_dict=fd)
            fd[model.sent1_2]=sent2 #without doesn't work
            #print sent2.shape()
            _, train_loss= sess.run([model.train_op, model.loss], feed_dict=fd)
        except Exception,e:
            traceback.print_exc()
            print 'error',step,e
            continue
        print step,'ended', 'loss=',train_loss

        # prog.update(i + 1, [("train loss", train_loss)])

        # tensorboard
        if  step % 10 == 0:
            print 'step=',step,'train_loss=',train_loss
            #model.file_writer.add_summary(summary, step)
        if  step % config.nstep ==0 and False:
            results, score = evaluate(model, sess, dev)
            print results
            if score >= model.best_score:
                step_no_imprv = 0
                if not os.path.exists(config.model_output):
                    os.makedirs(config.model_output)
                saver.save(sess, config.model_output)
                best_score=score
                #logger.info("- new best score!{}".format(score))
            '''
            else:
                step_no_imprv += 1
                if step_no_imprv >= config.step_no_imprv:
                    logger.info("- early stopping {} epochs without improvement".format(
                                    nepoch_no_imprv))
                    break
            '''
    return 

def train(model, train, dev):
    best_score = 0
    saver = tf.train.Saver()
    model.train_op=tf.train.AdamOptimizer(0.0001).minimize(model.loss)
    #TODO model.merged
    model.best_score=0
    nepoch_no_imprv = 0
    with tf.Session(config=config.session_conf) as sess:
        init = tf.global_variables_initializer()
        table_init = tf.tables_initializer()
        #tf.get_default_graph().finalize()
        #tf.train.start_queue_runners(sess)
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        sess.run(init)
        sess.run(model.old_inits)
        sess.run(table_init)
        for epoch in range(config.nepochs):
            #logger.info("Epoch {:} out of {:}".format(epoch + 1, model.config.nepochs))
            print epoch
            train_epoch(sess, model, train, dev, epoch)

            # decay learning rate
            # config.lr *= config.lr_decay

def evaluate(model, sess,  dev):
    result={}
    names=model.evaluate_names
    def dict_add(x,y):
        for name in x:
            y[name]+=x[name]
        return y
    for i,input_data  in enumerate(dev.epoch_data()):
        #for x in input_data:
        #    print x.shape
        fd=dict(x for x in zip(model.inputs, input_data) if x[1] is not None)
        sent2=sess.run(model.sent1_2, feed_dict=fd)
        fd[model.sent1_2]=sent2 #without doesn't work
        #print 'SENT2 SHAPE', sent2.shape
        #print sess.run(model.target, feed_dict=fd)
        evaluate_r= sess.run(model.evaluate_vars, feed_dict=fd)
        evaluate_r=dict(zip(names, evaluate_r))
        result= dict_add(result, evaluate_r)
    return result, result['accuracy']

def predict(model, sess, source):
    fd, _ = model.get_feed_dict(source, None, None)
    return sess.run([model.sample_words],fd)
