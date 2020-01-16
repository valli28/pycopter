from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani

# Quadrotor
m = 0.65  # Kg
l = 0.23  # m
Jxx = 7.5e-3  # Kg/m^2
Jyy = Jxx
Jzz = 1.3e-2
Jxy = 0
Jxz = 0
Jyz = 0
J = np.array([[Jxx, Jxy, Jxz],
              [Jxy, Jyy, Jyz],
              [Jxz, Jyz, Jzz]])
CDl = 9e-3
CDr = 9e-4
kt = 3.13e-5  # Ns^2
km = 7.5e-7   # Ns^2
kw = 1.0/0.18   # rad/s

# Initial conditions
att_0 = np.array([0.0, 0.0, 0.0])
pqr_0 = np.array([0.0, 0.0, 0.0])
xyzt_0 = np.array([0.0, 0.0, -0.0])
xyz1_0 = np.array([1.0, 1.2, -0.0])
xyz2_0 = np.array([1.2, 2.0, -0.0])
xyz3_0 = np.array([-1.1, 2.6, -0.0])
v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
qt = quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw, att_0, pqr_0, xyzt_0, v_ned_0, w_0)

number_of_drones = 3
alt_d = 2
qc_list = np.empty((0, 1))
for idx in range(number_of_drones):
    qc_list = np.append(qc_list, quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw, att_0, pqr_0, xyzt_0 + np.append([np.random.rand(1,2)], 0.0), v_ned_0, w_0))
    qc_list[idx].yaw_d = 0

# Simulation parameters
tf = 400
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50

# Data log
qt_log = quadlog.quadlog(time)
q_log_list = np.empty((0,1))
for idx in range(number_of_drones):
    q_log_list = np.append(q_log_list, quadlog.quadlog(time))
#q1_log = quadlog.quadlog(time)
#q2_log = quadlog.quadlog(time)
#q3_log = quadlog.quadlog(time)

# Plots
quadcolor = ['m', 'r', 'g', 'b', 'c', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'y', 'k', 'w']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

init_area = 5
s = 2

# Constants
tangentHelp = np.array([[0, -1], [1, 0]])

# Calculation and helper functions 
def impPath(x, y, r):
	return ((x-qt.xyz[0])**2 + (y-qt.xyz[1])**2 - r**2)

def angle_to_chord(radius, angle):
    return (2 * radius * np.sin((angle / 57.3) / 2))

def to_euclid(xyz):
	return (np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]))

def sigmoid(u):
    mag = np.linalg.norm(u)
    range_s = 7.0
    curv = -0.5
    new_mag = (range_s / (1 + 2.73 ** ((curv) * mag))) - range_s/2
    mag_scale = new_mag / mag
    new_u = u * mag_scale
    return new_u

def angle_between_unit_vectors(v0, v1):
    angle = np.math.atan2(np.linalg.det([v0[0:2],v1[0:2]]),np.dot(v0[0:2],v1[0:2]))
    return np.abs(np.degrees(angle))
	
def on_key(event): # Add a new drone here.
    #print('you pressed', event.key, event.xdata, event.ydata)
    if event.key == 'a':
        add_new_drone()

radius = 2

angle = 360 / number_of_drones
angle_list = np.ones(number_of_drones)*angle
des_dist_list = np.zeros(number_of_drones)
for idx in range(0,len(angle_list)): # This only works for equal angles for the whole formation.
	des_dist_list[idx] = angle_to_chord(radius, angle_list[idx])

act_dist_list = np.zeros((number_of_drones))
err_dist_list = np.zeros(number_of_drones)
unit_vector_list = np.zeros((number_of_drones, 3))
inter_angle_list = np.zeros(number_of_drones)

def add_new_drone():
    global qc_list
    qc_list = np.append(qc_list, quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw, att_0, pqr_0, xyzt_0 + np.append([np.random.rand(1,2)], 0.0), v_ned_0, w_0))
    qc_list[number_of_drones - 1].yaw_d = 0


ke = 0.00285*2  # Gain for going towards circle
kt = 0.01  # Gain for orbiting
kf = 0.08  # Gain for formation

drone_added = False

for t in time:
    if drone_added == False and t > 100: 
        add_new_drone()
        drone_added = True

    number_of_drones = len(qc_list) # Recalculate number of drones for all for-loops in the code.

    if number_of_drones > len(act_dist_list):
        print("[" + str(t) + "] - Adding new drone and reinitializing vectors")
        act_dist_list = np.zeros((number_of_drones))
        err_dist_list = np.zeros(number_of_drones)
        unit_vector_list = np.zeros((number_of_drones, 3))
        inter_angle_list = np.zeros(number_of_drones)

        angle = 360 / number_of_drones
        angle_list = np.ones(number_of_drones)*angle
        des_dist_list = np.zeros(number_of_drones)
        for idx in range(0,len(angle_list)): # This only works for equal angles for the whole formation.
            des_dist_list[idx] = angle_to_chord(radius, angle_list[idx])

        q_log_list = np.append(q_log_list, quadlog.quadlog(time))

    for i in range(0, number_of_drones):
        copter = qc_list[i]
        copter.broadcast_rel_xyz(qt)

    for idx in range(0, number_of_drones):
        act_dist_list[idx] = to_euclid(qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % number_of_drones]))

    for idx in range(0, number_of_drones):
        err_dist_list[idx] = act_dist_list[idx] - des_dist_list[idx]


    #print(unit_vector_list.shape)
    #print(len(qc_list))
    #print(act_dist_list)
    
    for idx in range(0, number_of_drones):
        for i in range (0, 3):
            unit_vector_list[idx][i] = qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % number_of_drones])[i] / act_dist_list[idx]

    # Formation adjustment by rearranging drones in the list depending on the angles between them
    
    for idx in range(0, number_of_drones):
        i = idx - 1 # A delayed counter that wraps in the beginning.
        o = idx + 1 # A hastened counter that wraps in the end.
        if idx == 0:
            i = number_of_drones - 1
        if idx == number_of_drones - 1:
            o = 0 
        inter_angle_list[idx] = angle_between_unit_vectors(qc_list[i].xyz - qc_list[idx].xyz, qc_list[o].xyz - qc_list[idx].xyz)
    for idx in range (0, number_of_drones):
        if inter_angle_list[idx] < 30:
            if qc_list[idx].cooldown < 0 and qc_list[i].cooldown < 0 and qc_list[o].cooldown < 0:
                i = idx - 1 # A delayed counter that wraps in the beginning.
                o = idx + 1 # A hastened counter that wraps in the end.
                if idx == 0:
                    i = number_of_drones - 1
                if idx == number_of_drones - 1:
                    o = 0 
                print("[" + str(t) + "] - Swapping drone q" + str(o+1) + " and q" str(idx+1))
                get = qc_list[o], qc_list[idx]
                qc_list[idx], qc_list[o] = get
                qc_list[idx].cooldown = qc_list[o].cooldown = qc_list[i].cooldown = 75*20
    

    #print("Desired: ", des_dist_list)
    #print("Actual: ", act_dist_list)
    #print("Errors: ", err_dist_list)
    #print("unit_vectors: " , unit_vector_list)

    # Simulation
    X = V = []
    for idx in range(0, number_of_drones):
        X = np.append(X, qc_list[idx].xyz[0:2])
        V = np.append(V, qc_list[idx].v_ned[0:2])

    grad_list = np.empty((0,2))
    for i in range(0, number_of_drones):    
        grad_list = np.append(grad_list, np.array([[(X[i*2] - qt.xyz[0])*2, (X[i*2 + 1] - qt.xyz[1])*2]]), axis=0)

    tangent_list = np.empty((0,2))
    for idx in range(0, number_of_drones):
        tangent_list = np.append(tangent_list, np.array([tangentHelp.dot((grad_list[idx]/la.norm(grad_list[idx])))]), axis=0)

    e_list = np.array([])
    for i in range(0, number_of_drones):
        e_list = np.append(e_list, impPath(X[i*2], X[i*2 + 1], radius))

    # This small u is our circle following control input (velocity)
    # The form is: 
    # ControlInput = FollowCircleError + RotateOnCircle + StayInFormationDistance
    formation_list = np.empty((0,2))
    for idx in range(0, number_of_drones):
        i = idx - 1
        if idx == 0:
            i = number_of_drones - 1 # A delayed counter that wraps in the beginning.
        formation_list = np.append(formation_list, np.array([[(err_dist_list[idx]*unit_vector_list[idx][0] + err_dist_list[i]*unit_vector_list[idx][0])*kf, (err_dist_list[idx]*unit_vector_list[idx][1] + err_dist_list[idx]*unit_vector_list[i][1])*kf]]), axis=0)

    #print(err_dist_list)
    u_list = np.empty((0,2))
    for idx in range(0, number_of_drones):
        u_list = np.append(u_list, np.array([ke*(-e_list[idx])*grad_list[idx] + kt*tangent_list[idx] + formation_list[idx]]), axis=0)
        u_list[idx] = sigmoid(u_list[idx])
    
    zero = np.array([0, 0])
    #ut = np.array([0.25, 0.25]) # move the middle drone qt a little bit in x and y
    ut = np.array([0, 0])
    #ut = impPath(qt.xyz[0], qt.xyz[1], 2)

    qt.set_v_2D_alt_lya(ut, -alt_d)
    
    for idx in range(0, number_of_drones):
        qc_list[idx].set_v_2D_alt_lya(u_list[idx] + ut, -alt_d)

    qt.step(dt)
    for idx in range(0, number_of_drones):
        qc_list[idx].step(dt)


    # Animation
    if it % frames == 0:

        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, qt.xyz, qt.Rot_bn(), quadcolor[0])
        for idx in range(0, number_of_drones):
            ani.draw3d(axis3d, qc_list[idx].xyz, qc_list[idx].Rot_bn(), quadcolor[idx+1])
            #ani.draw3d(axis3d, q1.xyz, q1.Rot_bn(), quadcolor[1])
            #ani.draw3d(axis3d, q2.xyz, q2.Rot_bn(), quadcolor[2])
            #ani.draw3d(axis3d, q3.xyz, q3.Rot_bn(), quadcolor[3])
        axis3d.set_xlim(-10, 10)
        axis3d.set_ylim(-10, 10)
        axis3d.set_zlim(0, 10)
        axis3d.set_xlabel('South [m]')
        axis3d.set_ylabel('East [m]')
        axis3d.set_zlabel('Up [m]')
        axis3d.set_title("Time %.3f s" % t)
        pl.pause(0.001)
        pl.draw()
        # namepic = '%i'%it
        # digits = len(str(it))
        # for j in range(0, 5-digits):
        #    namepic = '0' + namepic
        # pl.savefig("./images/%s.png"%namepic)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Log
    for idx in range(0, number_of_drones): 
        q_log_list[idx].xyz_h[it, :] = qc_list[idx].xyz
        q_log_list[idx].att_h[it, :] = qc_list[idx].att
        q_log_list[idx].w_h[it, :] = qc_list[idx].w
        q_log_list[idx].v_ned_h[it, :] = qc_list[idx].v_ned
        q_log_list[idx].formation_error[it, :] = formation_list[idx]
        q_log_list[idx].circle_error[it] = e_list[idx]
        
    qt_log.xyz_h[it, :] = qt.xyz
    qt_log.att_h[it, :] = qt.att
    qt_log.w_h[it, :] = qt.w
    qt_log.v_ned_h[it, :] = qt.v_ned

    it += 1

    # Stop if crash
    for q in qc_list:
        if q.crashed==1:
            break



pl.figure(1)
pl.title("2D Position [m]")
pl.plot(qt_log.xyz_h[:, 0], qt_log.xyz_h[:, 1], label="qt", color=quadcolor[0])
for idx in range(0, number_of_drones):
    pl.plot(q_log_list[idx].xyz_h[:, 0], q_log_list[idx].xyz_h[:, 1], label="q" + str(idx+1), color=quadcolor[idx+1])
#pl.plot(q_log_list[0].xyz_h[:, 0], q_log_list[0].xyz_h[:, 1], label="q1", color=quadcolor[1])
#pl.plot(q_log_list[1].xyz_h[:, 0], q_log_list[1].xyz_h[:, 1], label="q2", color=quadcolor[2])
#pl.plot(q_log_list[2].xyz_h[:, 0], q_log_list[2].xyz_h[:, 1], label="q3", color=quadcolor[3])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()


# Calculate the formation error for the first 3 drones only.
formation_error_total = np.array([])
for i in range(0, len(q_log_list[0].formation_error)):
    formation_error_total = np.append(formation_error_total, (la.norm(q_log_list[0].formation_error[i]) + la.norm(q_log_list[1].formation_error[i]) + la.norm(q_log_list[2].formation_error[i])))
 
# Calculate the circle error for the first 3 drones only.
circle_error_total = np.array([])
for i in range(0, len(q_log_list[0].formation_error)):
    circle_error_total = np.append(circle_error_total, ((la.norm(q_log_list[0].circle_error[i]) + la.norm(q_log_list[1].circle_error[i]) + la.norm(q_log_list[2].circle_error[i]))))
 
 
# Formation error plot
pl.figure(2)
pl.title("Formation Error of " + str(number_of_drones) + " Drones Over Time [m]")
for idx in range(0, number_of_drones):
    pl.plot(time, la.norm(q_log_list[idx].formation_error, axis=1), label="q"+str(idx+1), color=quadcolor[idx+1])
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
for idx in range(0, number_of_drones):
    pl.plot(time, q_log_list[idx].circle_error[:,0], label="q"+str(idx+1), color=quadcolor[idx+1])
#pl.plot(time, q_log_list[0].circle_error[:,0], label="Circle error q1")
#pl.plot(time, q_log_list[1].circle_error[:,0], label="Circle error q2")
#pl.plot(time, q_log_list[2].circle_error[:,0], label="Circle error q3")
pl.plot(time, circle_error_total, label="Circle error total", color=quadcolor[0], linewidth=2)
pl.xlabel("Time [s]")
pl.ylabel("Circle Error Squared [m]")
pl.grid()
pl.legend()


pl.pause(0)
