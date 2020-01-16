from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani


# Simulation parameters
tf = 700
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50

n_drones = 3

class quadSimulator:
    def __init__(self, tf, dt, time, it, frames, number_of_drones):

        self.number_of_drones = number_of_drones
        self.frames = frames
        self.it = it
        self.time = time
        self.dt = dt
        self.tf = tf

        # Quadrotor
        self.m = 0.65  # Kg
        self.l = 0.23  # m
        Jxx = 7.5e-3  # Kg/m^2
        Jyy = Jxx
        Jzz = 1.3e-2
        Jxy = 0
        Jxz = 0
        Jyz = 0
        self.J = np.array([[Jxx, Jxy, Jxz],
                    [Jxy, Jyy, Jyz],
                    [Jxz, Jyz, Jzz]])
        self.CDl = 9e-3
        self.CDr = 9e-4
        self.kt = 3.13e-5  # Ns^2
        self.km = 7.5e-7   # Ns^2
        self.kw = 1.0/0.18   # rad/s

        # Initial conditions
        self.att_0 = np.array([0.0, 0.0, 0.0])
        self.pqr_0 = np.array([0.0, 0.0, 0.0])
        self.xyzt_0 = np.array([0.0, 0.0, -0.0])
        self.v_ned_0 = np.array([0.0, 0.0, 0.0])
        self.w_0 = np.array([0.0, 0.0, 0.0, 0.0])

        # Setting quads
        self.qt = quad.quadrotor(0, self.m, self.l, self.J, self.CDl, self.CDr, self.kt, self.km, self.kw, self.att_0, self.pqr_0, self.xyzt_0, self.v_ned_0, self.w_0, 't')

        self.alt_d = 2
        self.qc_list = np.empty((0, 1))
        for idx in range(number_of_drones):
            self.qc_list = np.append(self.qc_list, quad.quadrotor(0, self.m, self.l, self.J, self.CDl, self.CDr, self.kt, self.km, self.kw, self.att_0, self.pqr_0, self.xyzt_0 + np.append([np.random.rand(1,2)], 0.0), self.v_ned_0, self.w_0, idx))
            self.qc_list[idx].yaw_d = 0

        # Data log
        self.qt_log = quadlog.quadlog(time)
        self.q_log_list = np.empty((0,1))
        for idx in range(number_of_drones):
            self.q_log_list = np.append(self.q_log_list, quadlog.quadlog(time))
        #q1_log = quadlog.quadlog(time)
        #q2_log = quadlog.quadlog(time)
        #q3_log = quadlog.quadlog(time)

        # Plots
        self.quadcolor = ['m', 'r', 'g', 'b', 'c', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'y', 'k', 'w']
        pl.close("all")
        pl.ion()
        fig = pl.figure(0)
        self.axis3d = fig.add_subplot(111, projection='3d')

        self.init_area = 5
        self.s = 2
        
        self.number_of_swaps = 0

        # Constants
        self.tangentHelp = np.array([[0, -1], [1, 0]])

    # Calculation and helper functions 
    def impPath(self, x, y, r):
        return ((x-self.qt.xyz[0])**2 + (y-self.qt.xyz[1])**2 - r**2)

    def angle_to_chord(self, radius, angle):
        return (2 * radius * np.sin((angle / 57.3) / 2))

    def to_euclid(self, xyz):
        return (np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]))

    def sigmoid(self, u):
        mag = np.linalg.norm(u)
        range_s = 7.0
        curv = -0.5
        new_mag = (range_s / (1 + 2.73 ** ((curv) * mag))) - range_s/2
        mag_scale = new_mag / mag
        new_u = u * mag_scale
        return new_u

    def angle_between_unit_vectors(self, v0, v1):
        angle = np.math.atan2(np.linalg.det([v0[0:2],v1[0:2]]),np.dot(v0[0:2],v1[0:2]))
        return np.abs(np.degrees(angle))
        
    #def on_key(self, event): # Add a new drone here.
    #    #print('you pressed', event.key, event.xdata, event.ydata)
    #    if event.key == 'a':
    #        add_new_drone()

    def add_new_drone(self):
        self.qc_list = np.append(self.qc_list, quad.quadrotor(0, self.m, self.l, self.J, self.CDl, self.CDr, self.kt, self.km, self.kw, self.att_0, self.pqr_0, self.xyzt_0 + np.append([np.random.rand(1,2)], 0.0), self.v_ned_0, self.w_0, self.number_of_drones))
        self.qc_list[self.number_of_drones - 1].yaw_d = 0

    def run_simulation(self):
        radius = 2

        angle = 360 / self.number_of_drones
        angle_list = np.ones(self.number_of_drones)*angle
        des_dist_list = np.zeros(self.number_of_drones)
        for idx in range(0,len(angle_list)): # This only works for equal angles for the whole formation.
            des_dist_list[idx] = self.angle_to_chord(radius, angle_list[idx])

        act_dist_list = np.zeros(self.number_of_drones)
        err_dist_list = np.zeros(self.number_of_drones)
        unit_vector_list = np.zeros((self.number_of_drones, 3))
        inter_angle_list = np.zeros(self.number_of_drones)

        ke = 0.00285*2 *0.7  # Gain for going towards circle
        kt = 0.001  # Gain for orbiting
        kf = 0.08 *0.5 # Gain for formation

        drone_added = False
        #drone_added = True

        do_animation = False

        for self.t in self.time:
            # If we want to add a new drone
            if drone_added == False and int(self.t % 100) == 0 and self.t > 1.0: 
                self.add_new_drone()
                drone_added = True

            self.number_of_drones = len(self.qc_list) # Recalculate number of drones for all for-loops in the code.

            if self.number_of_drones > len(act_dist_list):
                print("[" + str(self.t) + "] - Adding new drone and reinitializing vectors")
                act_dist_list = np.zeros((self.number_of_drones))
                err_dist_list = np.zeros(self.number_of_drones)
                unit_vector_list = np.zeros((self.number_of_drones, 3))
                inter_angle_list = np.zeros(self.number_of_drones)

                angle = 360 / self.number_of_drones
                angle_list = np.ones(self.number_of_drones)*angle
                des_dist_list = np.zeros(self.number_of_drones)
                for idx in range(0,len(angle_list)): # This only works for equal angles for the whole formation.
                    des_dist_list[idx] = self.angle_to_chord(radius, angle_list[idx])

                self.q_log_list = np.append(self.q_log_list, quadlog.quadlog(time))

            drone_id_list = []
            for idx in range(0, self.number_of_drones):
                drone_id_list.append(self.qc_list[idx].identification)

            for i in drone_id_list:
                copter = self.qc_list[i]
                copter.broadcast_rel_xyz(self.qt)

            for idx in drone_id_list:
                act_dist_list[idx] = self.to_euclid(self.qc_list[idx].calc_neighbour_vector(self.qc_list[(idx + 1) % self.number_of_drones]))

            for idx in drone_id_list:
                err_dist_list[idx] = act_dist_list[idx] - des_dist_list[idx]


            #print(unit_vector_list.shape)
            #print(len(qc_list))
            #print(act_dist_list)
            
            for idx in drone_id_list:
                for i in range (0, 3):
                    unit_vector_list[idx][i] = self.qc_list[idx].calc_neighbour_vector(self.qc_list[(idx + 1) % self.number_of_drones])[i] / act_dist_list[idx]

            # Formation adjustment by rearranging drones in the list depending on the angles between them
            
            for idx in drone_id_list:
                i = idx - 1 # A delayed counter that wraps in the beginning.
                o = idx + 1 # A hastened counter that wraps in the end.
                if idx == 0:
                    i = self.number_of_drones - 1
                if idx == self.number_of_drones - 1:
                    o = 0 
                inter_angle_list[idx] = self.angle_between_unit_vectors(self.qc_list[i].xyz - self.qc_list[idx].xyz, self.qc_list[o].xyz - self.qc_list[idx].xyz)
            for idx in range (0, self.number_of_drones):
                if inter_angle_list[idx] < 35:
                    if self.qc_list[idx].cooldown < 0 and self.qc_list[i].cooldown < 0 and self.qc_list[o].cooldown < 0:
                        i = idx - 1 # A delayed counter that wraps in the beginning.
                        o = idx + 1 # A hastened counter that wraps in the end.
                        if idx == 0:
                            i = self.number_of_drones - 1
                        if idx == self.number_of_drones - 1:
                            o = 0 
                        print("[" + str(self.t) + "] - Swapping drone q" + str(o+1) + " and q" + str(idx+1))
                        # Swap the drones
                        get = self.qc_list[o], self.qc_list[idx]
                        self.qc_list[idx], self.qc_list[o] = get
                        # Reset the cooldown on these drones
                        self.qc_list[idx].cooldown = self.qc_list[o].cooldown = self.qc_list[i].cooldown = 120*20
                        self.number_of_swaps += 1
                        # Swap the colors as well, so it won't look weird in the graphs
                        #get_color = quadcolor[o+1], quadcolor[idx+1]
                        #quadcolor[idx+1], quadcolor[o+1] = get_color
            

            #print("Desired: ", des_dist_list)
            #print("Actual: ", act_dist_list)
            #print("Errors: ", err_dist_list)
            #print("unit_vectors: " , unit_vector_list)

            # Simulation
            X = V = []
            for idx in drone_id_list:
                X = np.append(X, self.qc_list[idx].xyz[0:2])
                V = np.append(V, self.qc_list[idx].v_ned[0:2])

            grad_list = np.empty((0,2))
            for i in drone_id_list:    
                grad_list = np.append(grad_list, np.array([[(X[i*2] - self.qt.xyz[0])*2, (X[i*2 + 1] - self.qt.xyz[1])*2]]), axis=0)

            tangent_list = np.empty((0,2))
            for idx in drone_id_list:
                tangent_list = np.append(tangent_list, np.array([self.tangentHelp.dot((grad_list[idx]/la.norm(grad_list[idx])))]), axis=0)

            e_list = np.array([])
            for i in drone_id_list:
                e_list = np.append(e_list, self.impPath(X[i*2], X[i*2 + 1], radius))

            # This small u is our circle following control input (velocity)
            # The form is: 
            # ControlInput = FollowCircleError + RotateOnCircle + StayInFormationDistance
            formation_list = np.empty((0,2))
            for idx in drone_id_list:
                i = idx - 1
                if idx == 0:
                    i = self.number_of_drones - 1 # A delayed counter that wraps in the beginning.
                formation_list = np.append(formation_list, np.array([[(err_dist_list[idx]*unit_vector_list[idx][0] + err_dist_list[i]*unit_vector_list[idx][0])*kf, (err_dist_list[idx]*unit_vector_list[idx][1] + err_dist_list[idx]*unit_vector_list[i][1])*kf]]), axis=0)

            #print(err_dist_list)
            u_list = np.empty((0,2))
            for idx in drone_id_list:
                u_list = np.append(u_list, np.array([ke*(-e_list[idx])*grad_list[idx] + kt*tangent_list[idx] + formation_list[idx]]), axis=0)
            
            for idx in drone_id_list:
                u_list[idx] = self.sigmoid(u_list[idx])
            
            zero = np.array([0, 0])
            #ut = np.array([0.25, 0.25]) # move the middle drone qt a little bit in x and y
            ut = np.array([0, 0])
            #ut = impPath(qt.xyz[0], qt.xyz[1], 2)

            self.qt.set_v_2D_alt_lya(ut, -self.alt_d)
            
            for idx in drone_id_list:
                self.qc_list[idx].set_v_2D_alt_lya(u_list[idx] + ut, -self.alt_d)

            self.qt.step(self.dt)
            for idx in drone_id_list:
                self.qc_list[idx].step(self.dt)

            # Animation
            if self.it % self.frames == 0 and do_animation == True:

                pl.figure(0)
                self.axis3d.cla()
                ani.draw3d(self.axis3d, self.qt.xyz, self.qt.Rot_bn(), self.quadcolor[0])
                for idx in drone_id_list:
                    ani.draw3d(self.axis3d, self.qc_list[idx].xyz, self.qc_list[idx].Rot_bn(), self.quadcolor[idx+1])
                    #ani.draw3d(axis3d, q1.xyz, q1.Rot_bn(), quadcolor[1])
                    #ani.draw3d(axis3d, q2.xyz, q2.Rot_bn(), quadcolor[2])
                    #ani.draw3d(axis3d, q3.xyz, q3.Rot_bn(), quadcolor[3])
                self.axis3d.set_xlim(-10, 10)
                self.axis3d.set_ylim(-10, 10)
                self.axis3d.set_zlim(0, 10)
                self.axis3d.set_xlabel('South [m]')
                self.axis3d.set_ylabel('East [m]')
                self.axis3d.set_zlabel('Up [m]')
                self.axis3d.set_title("Time %.3f s" % self.t)
                pl.pause(0.001)
                pl.draw()
                # namepic = '%i'%it
                # digits = len(str(it))
                # for j in range(0, 5-digits):
                #    namepic = '0' + namepic
                # pl.savefig("./images/%s.png"%namepic)

                #cid = fig.canvas.mpl_connect('key_press_event', on_key)

            # Log
            p = 0
            for idx in drone_id_list: 
                self.q_log_list[p].xyz_h[self.it, :] = self.qc_list[idx].xyz
                self.q_log_list[p].att_h[self.it, :] = self.qc_list[idx].att
                self.q_log_list[p].w_h[self.it, :] = self.qc_list[idx].w
                self.q_log_list[p].v_ned_h[self.it, :] = self.qc_list[idx].v_ned
                self.q_log_list[p].formation_error[self.it] = err_dist_list[idx]
                self.q_log_list[p].circle_error[self.it] = e_list[idx]
                p += 1
                
            self.qt_log.xyz_h[self.it, :] = self.qt.xyz
            self.qt_log.att_h[self.it, :] = self.qt.att
            self.qt_log.w_h[self.it, :] = self.qt.w
            self.qt_log.v_ned_h[self.it, :] = self.qt.v_ned

            self.it += 1

            # Stop if crash
            for q in self.qc_list:
                if q.crashed==1:
                    break
        print(self.number_of_drones)
        return self.q_log_list, self.qt_log, drone_id_list # and also return qt_log...


quadsim = quadSimulator(tf, dt, time, it, frames, n_drones)
q_log_list, qt_log, drone_id_list = quadsim.run_simulation()
quadcolor = quadsim.quadcolor
number_of_drones = quadsim.number_of_drones


pl.figure(1)
pl.title("2D Position [m]")
pl.plot(qt_log.xyz_h[:, 0], qt_log.xyz_h[:, 1], label="qt", color=quadcolor[0])
for idx in drone_id_list:
    pl.plot(q_log_list[idx].xyz_h[:, 0], q_log_list[idx].xyz_h[:, 1], label="q" + str(idx+1), color=quadcolor[idx+1])
#pl.plot(q_log_list[0].xyz_h[:, 0], q_log_list[0].xyz_h[:, 1], label="q1", color=quadcolor[1])
#pl.plot(q_log_list[1].xyz_h[:, 0], q_log_list[1].xyz_h[:, 1], label="q2", color=quadcolor[2])
#pl.plot(q_log_list[2].xyz_h[:, 0], q_log_list[2].xyz_h[:, 1], label="q3", color=quadcolor[3])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()

# Calculate the formation error for all drones
formation_error_total = np.array([])
for idx in drone_id_list:
    formation_error_total = np.append(formation_error_total, q_log_list[idx].formation_error)
formation_error_total = np.reshape(formation_error_total, (number_of_drones, formation_error_total.size/number_of_drones))
formation_error_total = np.sum(formation_error_total, axis=0)

# Calculate the formation error for the first 3 drones only.
#formation_error_total = np.array([])
#for i in range(0, len(q_log_list[0].formation_error)):
#    formation_error_total = np.append(formation_error_total, (la.norm(q_log_list[0].formation_error[i]) + la.norm(q_log_list[1].formation_error[i]) + la.norm(q_log_list[2].formation_error[i])))
 
circle_error_total = np.array([])
for idx in drone_id_list:
    circle_error_total = np.append(circle_error_total, q_log_list[idx].circle_error)
print(circle_error_total.shape)
circle_error_total = np.reshape(circle_error_total, (number_of_drones, circle_error_total.size/number_of_drones))
print(circle_error_total.shape)

circle_error_total = np.sum(circle_error_total, axis=0)
print(circle_error_total.shape)


# Calculate the circle error for the first 3 drones only.
#circle_error_total = np.array([])
#for i in range(0, len(q_log_list[0].formation_error)):
#    circle_error_total = np.append(circle_error_total, ((la.norm(q_log_list[0].circle_error[i]) + la.norm(q_log_list[1].circle_error[i]) + la.norm(q_log_list[2].circle_error[i]))))
 

# Formation error plot
pl.figure(2)
pl.title("Formation Error of " + str(number_of_drones) + " Drones Over Time [m]")
for idx in drone_id_list:
    pl.plot(time, q_log_list[idx].formation_error, label="q"+str(idx+1), color=quadcolor[idx+1])
#pl.plot(time, la.norm(q_log_list[0].formation_error, axis=1), label="Formation error q1")
#pl.plot(time, la.norm(q_log_list[1].formation_error, axis=1), label="Formation error q2")
#pl.plot(time, la.norm(q_log_list[2].formation_error, axis=1), label="Formation error q3")
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
#pl.plot(time, q_log_list[0].circle_error[:,0], label="Circle error q1")
#pl.plot(time, q_log_list[1].circle_error[:,0], label="Circle error q2")
#pl.plot(time, q_log_list[2].circle_error[:,0], label="Circle error q3")
pl.plot(time, circle_error_total, label="Circle error total", color=quadcolor[0], linewidth=2)
pl.xlabel("Time [s]")
pl.ylabel("Circle Error Squared [m]")
pl.grid()
pl.legend()


pl.pause(0)
