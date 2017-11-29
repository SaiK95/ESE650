Hello, 

	Only MDP has been completed.
1. To run the code, just run run_FrozenLake.py
2. This code is designed for 4*4 map. To run for 8*8, there are a lot of variables to be changed. 
3. However, to test PGO for 8*8 map, please change the following parameters:
 In frozen_lake.py, change the variable map_name in to 8x8.
 In run_FrozenLake.py, change the following variables in test_PGO:
   n_iter to 3000, horizon to 50
   
4. !!!!IMPORTANT NOTE!!!!:
    This code ran on a friend's laptop, while in mine, it threw an error in line 101 in test_policy_gradient.py. 
    I think there's some compatibility issue with numdifftools, it threw an error in the line ndt.derivative(auxfunc)(0)
	If this error occurs when you test my code, please test it on another system or let me know, I'll run it in a friend's laptop. 
	

COLLABORATORS: Collaborated with Sakthivel Sivaraman and Anvith Ekkati.