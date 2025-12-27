import csv
import subprocess
from datetime import datetime
import time
import os 
import json


FREQS = [1600000, 2000000, 2500000, 3000000]
POWER_DATA_FILE = "power_data.json"
SENSORS_CSV = "internal_data.csv"
start_time = datetime.now()
print(start_time)
time_wait = 60


def temperature_log():
     while True:
        result = subprocess.run(
            ["sensors"],
            capture_output=True,
            text=True
            )


        with open("internal_data.csv","a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([start_time, result])
        
        time.sleep(time_wait)

def read_cpu_power():
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
            energy = int(f.read())
        return energy
    except:
        return 0

#testing with diff freq
def profile_frequency(freq):
    os.system(f"sudo cpupower frequency-set -u {freq//1000}MHz")
    time.sleep(0.1)  

    energy_start = read_cpu_power()
    start = time.time()

    for _ in range(10**6):
        pass

    energy_end = read_cpu_power()   
    elapsed = time.time() - start
    power = (energy_end - energy_start) / elapsed  
    power = power * 1e-6 
    return power, elapsed


def offline_profiling():
    data = {}
    for freq in FREQS:
        power, exec_time = profile_frequency(freq)
        data[freq] = {"power": power, "execution_time": exec_time}
        print(f"Freq {freq}: Power={power:.4f} W, Time={exec_time:.4f} s")
    with open(POWER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
        

if __name__ == "__main__":
    offline_profiling()
    temperature_log()