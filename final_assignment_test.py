from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani

from quadsim import quadSimulator

class bigLogger:
    def __init__(self, repetitions, time, number_of_drones):

        self.n = repetitions
        self.t = time
        self.n_drones = number_of_drones

        #self.xyz_h = np.empty((self.n, time, 3, 1))
        #self.formation_error = np.empty((self.n, 1, time))
        #self.circle_error = np.empty((self.n, 0, time))


        self.xyz_h = np.empty((self.t, 3, self.n_drones, self.n))
        self.formation_error = np.empty((self.t, self.n_drones, self.n))
        self.circle_error = np.empty((self.t, self.n_drones, self.n))


    def add_stats(self, rep, number_of_drones, q_logger):
        # Input the data into the log arrays        
        if rep == 0: # If this is the first run, initialize the number of drones again.
            self.n_drones = number_of_drones
            self.xyz_h = np.empty((self.t, 3, self.n_drones, self.n))
            self.formation_error = np.empty((self.t, self.n_drones, self.n))
            self.circle_error = np.empty((self.t, self.n_drones, self.n))

        for idx in range(0, self.n_drones):
            self.xyz_h[:, :, idx, rep] = q_logger[idx].xyz_h
            self.formation_error[:, idx, rep] = q_logger[idx].formation_error
            self.circle_error[:, idx, rep] = q_logger[idx].circle_error

        

    def calculate_descriptive_statistics(self):
        print("Calculating descriptive statistics from " + str(self.n) + " repetitions") 

        # Find faulty data where the simulation went very wrong and drones flipped out and delete that test.
        # arr = arr[ (arr >= 6) & (arr <= 10) ]
        indeces = np.array([])
        for i in range(0, n):
            for idx in range(0, self.n_drones):
                print(np.amax(np.abs(self.circle_error[:, idx, i])))
                if np.amax(np.abs(self.circle_error[:, idx, i])) > 10:
                    if i not in indeces:
                        indeces = np.append(indeces, i)
                        print("Delete test number " + str(i))

        
        self.xyz_h = np.delete(self.xyz_h, indeces, axis = 3)
        self.formation_error = np.delete(self.formation_error, indeces, axis = 2)
        self.circle_error = np.delete(self.circle_error, indeces, axis = 2)

        # Means
        pos_mean = np.mean(self.xyz_h, axis=3)

        formation_mean = np.mean(self.formation_error, axis=2)
        formation_mean = np.mean(formation_mean, axis=1)

        circle_mean = np.mean(self.circle_error, axis=2)
        circle_mean = np.mean(circle_mean, axis=1)

        pos_var = np.var(self.xyz_h, axis=3)

        formation_var = np.var(self.formation_error, axis=2)
        formation_var = np.var(formation_var, axis=1)

        circle_var = np.var(self.circle_error, axis=2)
        circle_var = np.var(circle_var, axis=1)

        return pos_mean, pos_var, formation_mean, formation_var, circle_mean, circle_var


# Simulation parameters
tf = 500
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50
n_drones = 4

# Statistics logging parameters
n = 50
logger = bigLogger(n, int(tf/dt), n_drones)

try:
    logger.xyz_h = np.load('data_xyz.npy')
    logger.formation_error = np.load('data_formation.npy')
    logger.circle_error = np.load('data_circle.npy')
except (IOError):
    print("There is no saved data from previous simulations. Starting simulations.")
    for repetition in range(0, n):
        print("Run number " + str(repetition+1))
        n_drones = 4

        sim = quadSimulator(tf, dt, time, it, frames, n_drones)
        q_log_list, qt_log, drone_id_list = sim.run_simulation()    
        quadcolor = sim.quadcolor
        n_drones = sim.number_of_drones

        del sim
        
        logger.add_stats(repetition, n_drones, q_log_list)

    np.save('data_xyz', logger.xyz_h) 
    np.save('data_formation', logger.formation_error) 
    np.save('data_circle', logger.circle_error) 

pos_mean, pos_var, formation_mean, formation_var, circle_mean, circle_var = logger.calculate_descriptive_statistics()


# Formation error plot
pl.figure(1)
pl.title("Formation Error of " + str(n_drones) + " Drones Over Time [m]")
pl.plot(time, formation_mean, label="q", color='r')
pl.fill_between(time, formation_mean + formation_var, formation_mean - formation_var, color='r', alpha=0.5)
pl.xlabel("Time [s]")
pl.ylabel("Formation Error [m]")
pl.grid()
pl.legend()
 
# Circle error plot
pl.figure(2)
pl.title("Circle Error of " + str(n_drones) + " Drones Over Time [m]")
pl.plot(time, circle_mean, label="q", color='r')
pl.fill_between(time, circle_mean + circle_var, circle_mean - circle_var, color='r', alpha=0.5)
pl.xlabel("Time [s]")
pl.ylabel("Circle Error Squared [m]")
pl.grid()
pl.legend()


# 2D position plot
pl.figure(3)
pl.title("2D position of" + str(n_drones))
pl.plot(pos_mean[:, 0], pos_mean[:, 1], label="q")
pl.xlabel("x-position [m]")
pl.ylabel("y-position [m]")
pl.grid()
pl.legend()

pl.pause(0)

