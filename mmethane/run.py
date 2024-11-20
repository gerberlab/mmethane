import argparse
import os
from utilities.data import *
import subprocess


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='../config_files/sample.cfg')
    args = parser.parse_args()
    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)


    if "data" in config.sections():
        if "process_data" in config['description'] and config['description']['process_data'].lower()!='false':
            ProcessData(args.config_file)

    if "run" in config:
        if "run_model" in config['description'] and config['description']['run_model'].lower()=='false':
            sys.exit()
        save_path = config['description']['out_path'] + config['description']['tag']

        if "data_met" not in config["run"] or config["run"]["data_met"]=="":
            config["run"]["data_met"] = f"{save_path}/mets.pkl"

        if "data_otu" not in config["run"] or config["run"]["data_otu"] == "":
            config["run"]["data_otu"] = f"{save_path}/seqs.pkl"

        if config["run"]["model"].lower()=="mmethane":
            run_str = "python3 ./lightning_trainer.py "

        elif config["run"]["model"].lower()=="ffn":
            run_str = "python3 ./lightning_trainer_full_nn.py "

        else:
            run_str = f"python3 ./benchmarker.py --model {config['run']['model']}"

        for (key, val) in config.items("run"):
            if val!='' and key!="model":
                if ', ' in val:
                    val = " ".join(val.split(', '))
                elif ',' in val:
                    val = " ".join(val.split(','))
                run_str += f" --{key} {val}"

        output_path = f"{config['run']['out_path']}/{config['run']['run_name']}/seed_{config['run']['seed']}/"
        print(f"Command: {run_str}")
        pid = subprocess.Popen(run_str.split(' '))

        while pid.poll() is None:
            time.sleep(1)

        if 'data' not in config:
            op = "1"
            on = "0"
        else:
            op = config['data']['outcome_positive_value']
            on = config['data']['outcome_negative_value']

        plot_str = f"python3 ./plot_results.py --path {output_path}/ " \
                   f"--outcome_positive_value {op} --outcome_negative_value {on}"

        with open(output_path + 'commands_run.txt', 'w') as f:
            f.write(run_str + '\n\n')
            f.write(plot_str)
        pid = subprocess.Popen(plot_str.split(' '))
