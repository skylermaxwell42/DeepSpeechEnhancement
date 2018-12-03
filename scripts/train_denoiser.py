import argparse
import tensorflow as tf
from AudioProcessing.DataUtils import data_input_fn
from AudioProcessing.TrainUtils import speech_enhancement_cnn, configure_denoiser_run

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a Speech Enhancement Neural Network')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--model_dir',
                        help='Path to write persistent training data (aka checkpoints and metadata)',
                        required=True,
                        type=str)

    rgroup.add_argument('--train_record_path',
                        help='Exact path to training TF Record',
                        required=True,
                        type=str)

    rgroup.add_argument('--eval_record_path',
                        help='Exact path to evaluation TF Record',
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ =='__main__':
    args = parse_args()

    classifier, train_spec, eval_spec = configure_denoiser_run(model_dir=args.model_dir,
                                                               train_record_path=args.train_record_path,
                                                               eval_record_path=args.eval_record_path,
                                                               data_input_fn=data_input_fn,
                                                               cnn_model_fn=speech_enhancement_cnn)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)