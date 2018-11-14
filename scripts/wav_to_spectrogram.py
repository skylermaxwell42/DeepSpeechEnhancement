import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Script to convert .wav files to spectrograms')


    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--input_path',
                        help='Input Path',
                        required=True,
                        type=str)
    return parser.parse_args()

def main():

    return

if __name__ == '__main__':
    args = parse_args()

    main()