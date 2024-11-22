"""
Timing script
Created by: Steven Fuller
Date: 11/15/24
"""

import os
import time

# find the average of the run times
def average_time(num_runs, run_times):
    parallel_total = 0
    serial_total = 0
    for i in range(num_runs):
        if i%2 ==0:
            parallel_total += run_times[i]
        else:
            serial_total += run_times[i]

    average_time[0] = parallel_total / num_runs
    average_time[1] = serial_total / num_runs

    # print("\n")
    # print(f"Average time: {average_time}\n")

    return average_time


def time_runs(script_to_run,  num):
    # loop through num_runs and track time
    for i in range(num_runs): 
        start_time = time.time() # get time at the start of the run

        os.system(script_to_run) # run the script

        end_time = time.time() # get time at the end of the run

        run_times[i] = end_time - start_time # store the difference in the times

    average_times[num] = average_time(num_runs, run_times)


# global variables
script_to_run = ['python parallel.py', 'python serial.py']  # set the script to run here
num_runs = 10 # set the number of runs here
run_times = [None] * num_runs # variable to hold the run times
average_times = [None] * 2

"""
The purpose of this program is to run our scripts multiple times timing each.
These times will be stored in a list and averaged later.
"""
def main():
    # loop through scripts and get average time of e
    for num, script in enumerate(script_to_run):
        time_runs(script,  num)
    

    print(f"Parallel average time: {average_times[0]}\n")
    print(f"Serial average time: {average_times[1]}\n")
    


if __name__ == '__main__':
    main()
