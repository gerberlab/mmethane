from benchmarker import *

logs_dirs = ['logs/RF_logs', 'logs/LR_logs', 'logs/AdaBoost_logs','logs/GradBoost_logs']
for log in logs_dirs:
    for file in os.listdir(log):
        path = log + '/' + file
        if os.path.isdir(path):
            process_benchmark_results(path)