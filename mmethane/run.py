import argparse
import os
from utilities.data import *
import subprocess


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='./config_files/erawijantari.cfg')
    args = parser.parse_args()
    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)


    if "data" in config:
        ProcessData(args.config_file)

    if "run" in config:
        save_path = config['description']['out_path'] + config['description']['tag']

        if "data_met" not in config["run"] or config["run"]["data_met"]=="":
            config["run"]["data_met"] = f"{save_path}/mets.pkl"

        if "data_otu" not in config["run"] or config["run"]["data_otu"] == "":
            config["run"]["data_met"] = f"{save_path}/seqs.pkl"

        if config["run"]["model"].lower()=="mmethane":
            run_str = "python3 ./lightning_trainer.py "

        elif config["run"]["model"].lower()=="ffn":
            run_str = "python3 ./lightning_trainer_full_nn.py "

        else:
            run_str = f"python3 ./benchmarker.py --model {config['run']['model']} "

        for (key, val) in config.items("run"):
            if val!='' and key!="model":
                run_str += f"--{key} {val}"

        pid = subprocess.Popen(run_str.split(' '))
