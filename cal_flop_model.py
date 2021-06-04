import tensorflow as tf
import keras.backend as K
from model_zoo import *


def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

def get_flops():
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model_name = 'new_b0_ver0'
            model = build_new_efficient_net_b0(224,224,3, 2)


            opt_adam = keras.optimizers.Adam(lr=1e-4)
            opt_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

            model.compile(loss="categorical_crossentropy", optimizer=opt_adam, metrics=['categorical_accuracy'])
            # model = keras.applications.mobilenet.MobileNet(
            #         alpha=1, weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops


# .... Define your model here ....
# You need to have compiled your model before calling this.
# model_name = 'new_b0_ver0'
# model = build_new_efficient_net_b0(224,224,3, 2)


# opt_adam = keras.optimizers.Adam(lr=1e-4)
# opt_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# model.compile(loss="categorical_crossentropy", optimizer=opt_adam, metrics=['categorical_accuracy'])
print(get_flops())