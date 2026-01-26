#include <iostream>
#include <vector>
#include <algorithm>

struct ThermalSample {
    double time;        
    double cpu_temp;    
    double ambient_temp;
};


const double T_TARGET = 70.0;     
const double T_LIMIT  = 80.0;     
const double HORIZON  = 2.0;      
const double ALPHA    = 0.7;      
const double Kp       = 0.015;    

const double FREQ_MIN = 1.2;      
const double FREQ_MAX = 3.6;      
const double MAX_STEP = 0.2;      
int main() {
    std::vector<ThermalSample> samples = {
        {0, 45, 22},
        {1, 47, 22},
        {2, 50, 24},
        {3, 54, 26},
        {4, 58, 28},
        {5, 62, 30},
        {6, 67, 31}
    };

    double frequency = 3.4;
    double trend = 0.0; 

    for (int i = 1; i < samples.size(); i++) {
        double dT = samples[i].cpu_temp - samples[i-1].cpu_temp;


        trend = ALPHA * trend + (1 - ALPHA) * dT;

        // short-horizon prediction
        double T_pred = samples[i].cpu_temp + trend * HORIZON;

        // proportional DVFS control
        double error = T_pred - T_TARGET;
        double delta_f = -Kp * error;

        // clamp frequency step (prevents oscillation)
        delta_f = std::clamp(delta_f, -MAX_STEP, MAX_STEP);

        // apply frequency change
        frequency += delta_f;
        frequency = std::clamp(frequency, FREQ_MIN, FREQ_MAX);

        // emergency clamp
        if (T_pred > T_LIMIT) {
            frequency = std::max(FREQ_MIN, frequency - MAX_STEP);
        }

        std::cout
            << "t=" << samples[i].time << "s | "
            << "T=" << samples[i].cpu_temp << "°C | "
            << "Trend=" << trend << "°C/s | "
            << "Pred=" << T_pred << "°C | "
            << "Freq=" << frequency << " GHz"
            << std::endl;
    }

    return 0;
}
