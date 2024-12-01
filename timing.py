"""
Timing script
Created by: Steven Fuller
Date: 11/15/24
"""

import os
import time

# find the average of the run times
def average_time(run_times):
    return sum(run_times) / len(run_times) if run_times else 0

# time the running of the scripts
def time_runs(script_to_run):
    run_times = []  # store run times for the current script
    for i in range(num_runs): 
        start_time = time.time()  # get time at the start of the run

        os.system(script_to_run)  # run the script

        end_time = time.time()  # get time at the end of the run

        run_times.append(end_time - start_time)  # store the difference in the times

    return run_times  # return the collected run times

# global variables
script_to_run = ['python parallel.py', 'python serial.py']  # set the script to run here
num_runs = 2 # set the number of runs here
average_times = [None] * 2

"""
The purpose of this program is to run our scripts multiple times timing each.
These times will be stored in a list and averaged later.
"""
def main():
    # loop through scripts and get average time of each
    for num, script in enumerate(script_to_run):
        run_times = time_runs(script)  # Get run times for the current script
        average_times[num] = average_time(run_times)  # calculate average time

    print(f"Parallel average time: {average_times[0]}\n")
    print(f"Serial average time: {average_times[1]}\n")
    

if __name__ == '__main__':
    main()
