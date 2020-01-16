from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani

from quadsim import quadSimulator

class bigLogger:
    def __init__(self, repetitions, time):

        self.n = repetitions

        self.xyz_h = np.empty((self.n, time, 3))
        self.formation_error = np.empty((self.n, time))
        self.circle_error = np.empty((self.n, time))

    def calculate_descriptive_statistics(self):
        print("Calculating descriptive statistics from " + str(self.n) + " repetitions") 
        return self.n

    def add_stats(self, rep, xyz, formation, circle):
        self.xyz_h[rep] = xyz
        self.formation_error[rep] = formation
        self.circle_error[rep] = circle



# Simulation parameters
tf = 300
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50
n_drones = 3

# Statistics logging parameters
n = 10
logger = bigLogger(n, int(tf/dt))


for repetition in range(0, n):
    print("Run number " + str(repetition+1))

    sim = quadSimulator(tf, dt, time, it, frames, n_drones)
    q_log_list, qt_log, drone_id_list = sim.run_simulation()    
    quadcolor = sim.quadcolor
    number_of_drones = sim.number_of_drones

    for idx in range(0, number_of_drones):
        logger.add_stats(repetition, q_log_list[idx].xyz_h, q_log_list[idx].formation_error, q_log_list[idx].circle_error)


print(logger.formation_error.shape)

stats = logger.calculate_descriptive_statistics()

