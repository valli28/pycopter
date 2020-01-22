QUADCOPTER(s) SIMULATOR
--------------------

Author: H. Garcia de Marina  
Contact mail: noeth3r@gmail.com  
Webpage/Forum: http://dobratech.com/tools/pycopter  
Licence: GNU GPL V2, http://www.gnu.org/licenses/gpl-2.0.en.html

Easy and intuitive simulator of quadcopters in Python.

Requirements: Matplotlib 1.5.1 , Numpy 1.11.0 and Scipy 0.14.1


1. EXAMPLES
-----------

ex_position.py: One quadcopter flies to a fixed 3D position  
ex_vel_ned.py: One quadcopter flies with a constant velocity  
ex_triangle_2D.py: A team of three quadcopters forming an equilateral triangle


2. ADDITIONS FOR GNC COURSE BY STUDENTS
Additions for the GNC course made by Martin S. Metze, Nicolai H. Malle, Thor M Fischer and Valthor B. Gudmundsson:
Scripts for the final assignment due 17th of January 2020:

Added files:
final_assignment_GNC.py
final_assignment_GNCv2.py
final_assigmnent_test.py
quadsim.py

Edited files:
quadlog.py
quadrotor.py

3. GUIDE TO RUN CODE BY STUDENTS
The script that performs many simulations and collects all the results is final_assigment_test.py where different parameters can be configured for the simulation. Mind that the number of drones n_drones has to be configured both before and inside the for-loop. After the n-number of simulations have been performed, the results are saved as .npy files in the same folder. If these files are present in the folder when the script is run, the simulations are not run, but the statistics are calculated and plotted instead. 

The final_assigment_GNCv2 runs only one simulation and was created for development purposes.

The quadsim.py file includes a new class the performs one simulation.