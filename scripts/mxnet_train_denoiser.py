import mxnet as mx
import argparse
import progressbar
from AudioProcessing.DataUtils import load_wav_files
import numpy as np
from AudioProcessing.DataUtils import load_wav_files




def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a Speech Enhancement Neural Network')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--model_dir',
                        help='Path to write persistent training data (aka checkpoints and metadata)',
                        required=True,
                        type=str)

    rgroup.add_argument('--input_sample_dir',
                        help='Top level directory to augmented wave samples (must have sub dirs \'clean\' and \'noise\')',
                        required=True,
                        type=str)

    rgroup.add_argument('--target_sample_dir',
                        help='Top level directory to augmented wave samples (must have sub dirs \'clean\' and \'noise\')',
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ =='__main__':
    args = parse_args()

    batch_size = 32
    learning_rate = (('learning_rate', 0.01),)
    stats_per_batch = []

    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

    input_samples = load_wav_files(args.input_sample_dir)
    target_samples = load_wav_files(args.target_sample_dir)

    sample_rate = input_samples[0].sample_rate
    input_length = len(input_samples[0].data)

    input_np_data = np.asarray([x.data for x in input_samples])
    input_np_data = np.resize(input_np_data, (len(input_np_data), 1, 32000))
    target_np_data = np.asarray([x.data for x in target_samples])
    target_np_data = np.resize(target_np_data, (len(target_np_data), 1, 32000))


    print(input_np_data.shape)
    #target_data = mx.io.NDArrayIter(target_np_data, label=None, shuffle=True, batch_size=batch_size)
    #print('here')
    train_data = mx.io.NDArrayIter(input_np_data, label=None, shuffle=True, batch_size=batch_size, label_name=None)
    #print(train_data.type)
    #print(target_data.dtype)



    # MODEL DEFINITION
    data = mx.sym.Variable('data')

    conv1 = mx.sym.Convolution(data, name="conv1", num_filter=128, kernel=(16000,), stride=1)
    conv2 = mx.sym.Convolution(conv1, name="conv2",num_filter=64, kernel=(16000,), stride=2)
    #encoded here
    #conv3 = mx.sym.Convolution(conv2, name="conv3", num_filter=32, kernel=(8000,), stride=1)
    transp1 = mx.sym.Deconvolution(conv2, name="transp1", num_filter=16, kernel=(16000,), stride=2)
    transp2 = mx.sym.Deconvolution(transp1, name="transp2", num_filter=1, kernel=(32000,), stride=2)


    model = mx.mod.Module(symbol=transp2, data_names=('data',), label_names=None, context=mx.gpu(2))
    model.bind(for_training=True, data_shapes=train_data.provide_data, label_shapes=None)
    #model.init_params(initializer=mx.init.Xavier)
    #model.init_optimizer(optimizer='adadelta', optimizer_params={'learning_rate': lr,})


    model.fit(train_data=train_data,
    			eval_data=None,
    			initializer = mx.init.Xavier(factor_type = "in", rnd_type = "gaussian", magnitude = 2.0),
    			optimizer = "adadelta",
    			optimizer_params = learning_rate,
    			eval_metric = ['accuracy', 'loss','mse'],
                num_epoch = 30,
                batch_end_callback = stats_per_batch,
                epoch_end_callback = mx.callback.do_checkpoint("~/Desktop/checkpoint",1)
    			)