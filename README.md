![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# DVFS Temperature Control Reactive Improving 
**Paper:** [Reactive DVFS Control for Multicore Processors](https://www.researchgate.net/publication/261265827_Reactive_DVFS_Control_for_Multicore_Processors)

**Internal File:** Python & Ubuntuu
**External File** C++ & Arduino

## Motivation

In everyday computing, users often run multiple applications simultaneously—like playing video games while leaving a web browser open. Web browsers, especially with many tabs or extensions, can consume significant CPU resources even when idle or in the background. This unnecessary CPU usage can lead to thermal spikes, causing the CPU to throttle and the game to lag, creating a frustrating user experience.

Dynamic Voltage and Frequency Scaling (DVFS) offers a way to adapt CPU performance to workload demand, reducing power consumption and heat generation. By improving DVFS reactive to real-time workloads and temperature, we can prevent overheating, maintain smooth performance, and extend hardware longevity. On Linux/Ubuntu, tools like cpupower and the Intel RAPL interface allow precise monitoring and control of CPU frequency and energy consumption, making it an ideal platform for studying reactive DVFS.

## Overview

This research focuses on improving CPU temperature management using Dynamic Voltage and Frequency Scaling (DVFS) on Linux/Ubuntu systems. Unlike most existing studies that rely solely on internal CPU sensors, this work incorporates external temperature measurements using Arduino and the TMP sensor, allowing environmental conditions to be factored into DVFS decisions.

The project will implement DVFS algorithms in C++ on Linux, with the possibility of using Gem5 for hardware simulation and algorithm testing. Python will be used to collect, analyze, and visualize the data, providing clear insights into CPU performance, power consumption, and temperature trends. 

## Advantage & Disadvantage
**Advantage:**
No extra hardware needed – uses built-in sensors on Ubuntuu (RAPL, sensors) to measure CPU energy and temperature.

Reactive and real-time – can adjust CPU frequency on-the-fly based on actual workload, improving performance-per-watt.

Better thermal management – prevents CPU from overheating during unexpected high-load scenarios (like leaving a browser open while gaming).

Internal and External temperature - Most paper measure CPU temperature but ignore environment condition, therefore including external temperature can improve DVFS efficiency

**Disadvantage**
Energy resolution limits – microjoule readings may not capture very small changes for short workloads accurately.

Frequency switching overhead – changing CPU frequency too often can slightly reduce performance or create small lag spikes.

CPU test limits - due to the budget, this research might not be able to test on more than 4-5 CPU

## Novelty

Reactive DVFS using software-only Linux tools – shows that it’s possible to dynamically manage temperature without specialized hardware.

Bridges energy profiling and thermal management – combines RAPL energy readings and CPU frequency scaling to optimize CPU usage.

Practical for everyday users – can prevent lag spikes in gaming or multitasking by throttling CPU frequency reactively.

Framework for further research – provides a foundation to test more sophisticated policies like workload-aware or predictive DVFS.