import argparse

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