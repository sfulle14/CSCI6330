import os
import time


def main():
    script_to_run = 'python parallel.py' # set the script to run here
    num_runs = 10 # set the number of runs here
    run_times = [None] * num_runs # variable to hold the run times

    # loop through num_runs and track time
    for i in range(num_runs): 
        start_time = time.time()
        os.system(script_to_run)

        end_time = time.time()

        run_times[i] = end_time - start_time
    
    print(run_times)


if __name__ == '__main__':
    main()
