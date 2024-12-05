import sys
import argparse
import os
from utilities.data import *
import subprocess




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='config_files/sample.cfg')
    parser.add_argument('-o', '--out_path', type=str, default='logs/')
    args,_ = parser.parse_known_args()
    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)


    config['description']['out_path'] = args.out_path
    if "data" in config.sections():
        if "process_data" in config['description'] and config['description']['process_data'].lower()!='false':
            ProcessData(config)

    if "run" in config:
        benchmark_model=False
        cmds = []
        if "run_model" in config['description'] and config['description']['run_model'].lower()=='false':
            sys.exit()

        if "run_name" not in config["run"] or config["run"]["run_name"]=='':
            config["run"]["run_name"] = config['description']['tag']

        save_path = config['description']['out_path'] + config['description']['tag']
        if 'out_path' not in config['run'] or config['run']['out_path']=='':
            config['run']['out_path'] = config['description']['out_path']


        if "data_met" not in config["run"] or config["run"]["data_met"]=="":
            config["run"]["data_met"] = f"{save_path}/mets.pkl"

        if "data_otu" not in config["run"] or config["run"]["data_otu"] == "":
            config["run"]["data_otu"] = f"{save_path}/seqs.pkl"

        if config["run"]["model"].lower()=="mmethane":
            run_str = "python3 ./mmethane/lightning_trainer.py "

        elif config["run"]["model"].lower()=="ffn":
            run_str = "python3 ./mmethane/lightning_trainer_full_nn.py "

        else:
            run_str = f"python3 ./mmethane/benchmarker.py --model {config['run']['model']}"
            benchmark_model=True
            for (key, val) in config.items("run"):
                if val != '' and key != "model":
                    if ', ' in val:
                        val = " ".join(val.split(', '))
                    elif ',' in val:
                        val = " ".join(val.split(','))
                    run_str += f" --{key} {val}"

            cmds.append(run_str)
            print(f"Command: {run_str}")
            pid = subprocess.Popen(run_str.split(' '))
            output_path = f"{config['run']['out_path']}/{config['run']['run_name']}/"

            while pid.poll() is None:
                time.sleep(1)


        if not benchmark_model:
            seeds = config['run']['seed'].split(',')
            for seed in seeds:
                for (key, val) in config.items("run"):
                    if val!='' and key!="model" and key!='seed':
                        if ', ' in val:
                            val = " ".join(val.split(', '))
                        elif ',' in val:
                            val = " ".join(val.split(','))
                        run_str += f" --{key} {val}"

                run_str += f" --seed {seed.strip()}"
                cmds.append(run_str)
                output_path = f"{config['run']['out_path']}/{config['run']['run_name']}/seed_{seed.strip()}/"
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

                if config["run"]["model"].lower() == "mmethane":
                    plot_str = f"python3 ./mmethane/visualize.py --path {output_path}/ " \
                               f"--outcome_positive_value {op} --outcome_negative_value {on}"
                    cmds.append(plot_str)
                    pid = subprocess.Popen(plot_str.split(' '))

            if len(seeds) > 1:
                res_str = f"python3 ./mmethane/process_multi_seed_results.py --path {config['run']['out_path']}/{config['run']['run_name']}/"
                pid = subprocess.Popen(res_str.split(' '))
                cmds.append(res_str)

        with open(output_path + 'commands_run.txt', 'w') as f:
            for l in cmds:
                f.write(l + '\n\n')
