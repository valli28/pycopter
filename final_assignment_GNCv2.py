from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani

from quadsim import quadSimulator

# Simulation parameters
tf = 1000
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50

n_drones = 6


# Initialize simulation class (all the stuff we did)
sim = quadSimulator(tf, dt, time, it, frames, n_drones)
q_log_list, qt_log, drone_id_list = sim.run_simulation()
quadcolor = sim.quadcolor
# redefine what the current number of drones is as it might have changed.
number_of_drones = sim.number_of_drones



# Do some pretty plots :D
pl.figure(1)
pl.title("2D Position [m]")
pl.plot(qt_log.xyz_h[:, 0], qt_log.xyz_h[:, 1], label="qt", color=quadcolor[0])
for idx in drone_id_list:
    pl.plot(q_log_list[idx].xyz_h[:, 0], q_log_list[idx].xyz_h[:, 1], label="q" + str(idx+1), color=quadcolor[idx+1])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()

# Calculate the formation error for all drones
formation_error_total = np.array([])
for idx in drone_id_list:
    formation_error_total = np.append(formation_error_total, q_log_list[idx].formation_error)
formation_error_total = np.reshape(formation_error_total, (number_of_drones, formation_error_total.size/number_of_drones))
formation_error_total = np.sum(formation_error_total, axis=0)


circle_error_total = np.array([])
for idx in drone_id_list:
    circle_error_total = np.append(circle_error_total, q_log_list[idx].circle_error)
print(circle_error_total.shape)
circle_error_total = np.reshape(circle_error_total, (number_of_drones, circle_error_total.size/number_of_drones))
print(circle_error_total.shape)

circle_error_total = np.sum(circle_error_total, axis=0)
print(circle_error_total.shape)


# Formation error plot
pl.figure(2)
pl.title("Formation Error of " + str(number_of_drones) + " Drones Over Time [m]")
for idx in drone_id_list:
    pl.plot(time, q_log_list[idx].formation_error, label="q"+str(idx+1), color=quadcolor[idx+1])
pl.plot(time, formation_error_total, label="Total", color=quadcolor[0], linewidth=2)
pl.xlabel("Time [s]")
pl.ylabel("Formation Error [m]")
pl.grid()
pl.legend()
 
# Circle error plot
pl.figure(3)
pl.title("Circle Error of " + str(number_of_drones) + " Drones Over Time [m]")
for idx in drone_id_list:
    pl.plot(time, q_log_list[idx].circle_error, label="q"+str(idx+1), color=quadcolor[idx+1])
pl.plot(time, circle_error_total, label="Circle error total", color=quadcolor[0], linewidth=2)
pl.xlabel("Time [s]")
pl.ylabel("Circle Error Squared [m]")
pl.grid()
pl.legend()


pl.pause(0)
