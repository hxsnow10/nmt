import tensorflow as tf
f=g=lambda x:x+x
with tf.Graph().as_default() as g_1:
  input = tf.placeholder(tf.float32, name="input")
  weight = tf.Variable(10, dtype=tf.float32)
  y = input+weight
  # NOTE: using identity to get a known name for the output tensor.
  output = tf.identity(y, name="output")
  init = tf.global_variables_initializer()

gdef_1 = g_1.as_graph_def()

with tf.Graph().as_default() as g_2:  # NOTE: g_2 not g_1       
  input = tf.placeholder(tf.float32, name="input")
  weight = tf.Variable(20, dtype=tf.float32)
  z = input+weight
  output = tf.identity(z, name="output")
  init = tf.global_variables_initializer()
  print init.name
  
gdef_2 = g_2.as_graph_def()
print tf.global_variables()

with tf.Graph().as_default() as g_combined:
  x = tf.placeholder(tf.float32, name="x")

  # Import gdef_1, which performs f(x).
  # "input:0" and "output:0" are the names of tensors in gdef_1.
  y, = tf.import_graph_def(gdef_1, input_map={"input:0": x},
                           return_elements=["output:0"])

  # Import gdef_2, which performs g(y)
  z, = tf.import_graph_def(gdef_2, input_map={"input:0": y},
                           return_elements=["output:0"])

  print tf.global_variables()
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    ops = [o for o in tf.get_default_graph().get_operations() if 'init' in o.name.split('/')[-1]]
    sess.run(ops)
    print sess.run(z, feed_dict={x:100})
