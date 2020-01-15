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
kw = 1/0.18   # rad/s

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
qt = quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw,
                    att_0, pqr_0, xyzt_0, v_ned_0, w_0)

q1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw,
                    att_0, pqr_0, xyz1_0, v_ned_0, w_0)

q2 = quad.quadrotor(2, m, l, J, CDl, CDr, kt, km, kw,
                    att_0, pqr_0, xyz2_0, v_ned_0, w_0)

q3 = quad.quadrotor(3, m, l, J, CDl, CDr, kt, km, kw,
                    att_0, pqr_0, xyz3_0, v_ned_0, w_0)


# Simulation parameters
tf = 500
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50

# Data log
qt_log = quadlog.quadlog(time)
q1_log = quadlog.quadlog(time)
q2_log = quadlog.quadlog(time)
q3_log = quadlog.quadlog(time)

# Plots
quadcolor = ['m', 'r', 'g', 'b']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

init_area = 5
s = 2

# Desired altitude and heading
alt_d = 2
q1.yaw_d = 0
q2.yaw_d = 0
q3.yaw_d = 0

# Constants
tangentHelp = np.array([[0, -1], [1, 0]])

def impPath(x, y, r):
	return ((x-qt.xyz[0])**2 + (y-qt.xyz[1])**2 - r**2)

def angle_to_chord(radius, angle):
    return (2 * radius * np.sin((angle / 57.3) / 2))

def to_euclid(xyz):
	return (np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]))
	
radius = 2
angle_list = np.array([120, 120, 120])
des_dist_list = np.array([0, 0, 0])
for idx in range(0,len(angle_list)):
	des_dist_list[idx] = angle_to_chord(radius, angle_list[idx])

act_dist_list = np.array([0, 0, 0])
qc_list = np.array([q1, q2, q3])
err_dist_list = np.array([0, 0, 0])
unit_vector_list = np.array([[0,0,0],[0,0,0],[0,0,0]])

for t in time:
    
    q1.broadcast_rel_xyz(qt)
    q2.broadcast_rel_xyz(qt)
    q3.broadcast_rel_xyz(qt)
    for idx in range(0,len(qc_list)):
        act_dist_list[idx] = to_euclid(qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % len(qc_list)]))

    for idx in range(0, len(qc_list)):
        err_dist_list[idx] = act_dist_list[idx] - des_dist_list[idx]

    for idx in range(0, len(qc_list)):
        unit_vector_list[idx][0] = qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % len(qc_list)])[0] / act_dist_list[idx]
        unit_vector_list[idx][1] = qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % len(qc_list)])[1] / act_dist_list[idx]
        unit_vector_list[idx][2] = qc_list[idx].calc_neighbour_vector(qc_list[(idx + 1) % len(qc_list)])[2] / act_dist_list[idx]

    #print("Desired: ", des_dist_list)
    #print("Actual: ", act_dist_list)
    #print("Errors: ", err_dist_list)
    #print("unit_vectors: " , unit_vector_list)


    # Simulation
    X = np.append(q1.xyz[0:2], np.append(q2.xyz[0:2], q3.xyz[0:2]))
    V = np.append(q1.v_ned[0:2], np.append(q2.v_ned[0:2], q3.v_ned[0:2]))

    gradF1 = np.array([(X[0] - qt.xyz[0])*2, (X[1] - qt.xyz[1])*2])
    gradF2 = np.array([(X[2] - qt.xyz[0])*2, (X[3] - qt.xyz[1])*2])
    gradF3 = np.array([(X[4] - qt.xyz[0])*2, (X[5] - qt.xyz[1])*2])

    tangent1 = tangentHelp.dot((gradF1/la.norm(gradF1)))
    tangent2 = tangentHelp.dot((gradF2/la.norm(gradF2)))
    tangent3 = tangentHelp.dot((gradF3/la.norm(gradF3)))

    e1 = impPath(X[0], X[1], radius)
    e2 = impPath(X[2], X[3], radius)
    e3 = impPath(X[4], X[5], radius)
    ke = 0.00285  # Gain for going towards circle
    kt = 0.0  # Gain for velocity
    kf = 0.15 * 0.5

    # This small u is our circle following control input (velocity)
    # The form is: 
    # ControlInput = FollowCircleError + RotateOnCircle + StayInFormationDistance
    formation1 = np.array([(err_dist_list[0]*unit_vector_list[0][0] + err_dist_list[2]*unit_vector_list[0][0])*kf, (err_dist_list[0]*unit_vector_list[0][1] + err_dist_list[0]*unit_vector_list[2][1])*kf])
    formation2 = np.array([(err_dist_list[1]*unit_vector_list[1][0] + err_dist_list[0]*unit_vector_list[1][0])*kf, (err_dist_list[1]*unit_vector_list[1][1] + err_dist_list[1]*unit_vector_list[0][1])*kf])
    formation3 = np.array([(err_dist_list[2]*unit_vector_list[2][0] + err_dist_list[1]*unit_vector_list[2][0])*kf, (err_dist_list[2]*unit_vector_list[2][1] + err_dist_list[2]*unit_vector_list[1][1])*kf])

    print(err_dist_list)

    u1 = ke*(-e1)*gradF1 + kt*tangent1 + formation1
    u2 = ke*(-e2)*gradF2 + kt*tangent2 + formation2
    u3 = ke*(-e3)*gradF3 + kt*tangent3 + formation3

    zero = np.array([0, 0])
    ut = np.array([0.025, 0.025]) # move the middle drone qt a little bit in x and y
    qt.set_v_2D_alt_lya(ut, -alt_d)
    q1.set_v_2D_alt_lya(u1, -alt_d)
    q2.set_v_2D_alt_lya(u2, -alt_d)
    q3.set_v_2D_alt_lya(u3, -alt_d)

    qt.step(dt)
    q1.step(dt)
    q2.step(dt)
    q3.step(dt)

    # Animation
    if it % frames == 0:

        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, qt.xyz, qt.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, q1.xyz, q1.Rot_bn(), quadcolor[1])
        ani.draw3d(axis3d, q2.xyz, q2.Rot_bn(), quadcolor[2])
        ani.draw3d(axis3d, q3.xyz, q3.Rot_bn(), quadcolor[3])
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


    # Log
    qt_log.xyz_h[it, :] = qt.xyz
    qt_log.att_h[it, :] = qt.att
    qt_log.w_h[it, :] = qt.w
    qt_log.v_ned_h[it, :] = qt.v_ned

    q1_log.xyz_h[it, :] = q1.xyz
    q1_log.att_h[it, :] = q1.att
    q1_log.w_h[it, :] = q1.w
    q1_log.v_ned_h[it, :] = q1.v_ned

    q2_log.xyz_h[it, :] = q2.xyz
    q2_log.att_h[it, :] = q2.att
    q2_log.w_h[it, :] = q2.w
    q2_log.v_ned_h[it, :] = q2.v_ned

    q3_log.xyz_h[it, :] = q3.xyz
    q3_log.att_h[it, :] = q3.att
    q3_log.w_h[it, :] = q3.w
    q3_log.v_ned_h[it, :] = q3.v_ned

    it += 1

    # Stop if crash
    if (qt.crashed == 1 or q1.crashed == 1 or q2.crashed == 1 or q3.crashed == 1):
        break

pl.figure(1)
pl.title("2D Position [m]")
pl.plot(qt_log.xyz_h[:, 0], qt_log.xyz_h[:, 1], label="qt", color=quadcolor[0])
pl.plot(q1_log.xyz_h[:, 0], q1_log.xyz_h[:, 1], label="q1", color=quadcolor[1])
pl.plot(q2_log.xyz_h[:, 0], q2_log.xyz_h[:, 1], label="q2", color=quadcolor[2])
pl.plot(q3_log.xyz_h[:, 0], q3_log.xyz_h[:, 1], label="q3", color=quadcolor[3])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()

pl.figure(2)
pl.plot(time, q1_log.att_h[:, 2], label="yaw q1")
pl.plot(time, q2_log.att_h[:, 2], label="yaw q2")
pl.plot(time, q3_log.att_h[:, 2], label="yaw q3")
pl.xlabel("Time [s]")
pl.ylabel("Yaw [rad]")
pl.grid()
pl.legend()

pl.figure(3)
pl.plot(time, -q1_log.xyz_h[:, 2], label="$q_1$")
pl.plot(time, -q2_log.xyz_h[:, 2], label="$q_2$")
pl.plot(time, -q3_log.xyz_h[:, 2], label="$q_3$")
pl.xlabel("Time [s]")
pl.ylabel("Altitude [m]")
pl.grid()
pl.legend(loc=2)


pl.pause(0)
