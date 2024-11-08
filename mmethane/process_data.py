from utilities.data import *
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default='./config_files/erawijantari.cfg')
    args = parser.parse_args()

    pData = ProcessData(args.config_file)