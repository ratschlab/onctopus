import optimization
import cplex
from cplex.exceptions import CplexSolverError
from scipy.optimize import line_search
import logging
import exceptions_onctopus as eo

# given the average CN ci of a segment i, the objective of the
# optimization is returned
def obj_val_for_ci(ci, cplex_obj, seg_index, warm_start_dc_binary=None, warm_start_dsnp=None, 
	warm_start_dssm=None):

	# set CN of segment to ci
	cplex_obj.cn[seg_index] = ci

	# set specific constraints
	cplex_obj.constraint_fix_single_avg_cn(ci, seg_index)

	# do the optimization
	obj = -float("inf")
	try:
		#TODO: save time here, don't create new CPLEX obj my_prob again and again but recycle it!
		cplex_obj.start_CPLEX(warm_start_dc_binary, warm_start_dsnp, warm_start_dssm)
		obj = cplex_obj.my_prob.solution.get_objective_value()
	# if CPLEX doesn't find a solution, nothing happens and the objective value
	# stays at -inf
	except CplexSolverError, exc:
		if exc.args[2] != cplex.exceptions.error_codes.CPXERR_NO_SOLN:
			raise

	# remove specific constraint
	cplex_obj.remove_specific_constraints(cplex_obj.start_index_constraint_fix_single_avg_cn, 
		cplex_obj.start_index_constraint_fix_single_avg_cn+1)

	logging.info("Objective value for segment {0} with copy number {1} is: {2}".format(
		seg_index, ci, obj))

	# return negative objective because line search minimizes
	return obj

# computes the gradient of the ci function, where different ci values are tried
# for one segment
#def gradient_for_ci_function(ci, cplex_obj, seg_index, warm_start_dc_binary=None,
#	warm_start_dsnp=None, warm_start_dssm=None, h=1e-5):
#	obj_ci_minus = obj_val_for_ci(ci - h, cplex_obj, seg_index, warm_start_dc_binary,
#		warm_start_dsnp, warm_start_dssm)	
#	obj_ci_plus = obj_val_for_ci(ci + h, cplex_obj, seg_index, warm_start_dc_binary,
#		warm_start_dsnp, warm_start_dssm)	
#
#	my_gradient = 1
#
#	# no gradients for both points next to ci
#	if (obj_ci_minus == float("inf")) and (obj_ci_plus == float("inf")):
#		error = "No gradient exists for ci {0} in segment {1}.".format(ci, seg_index)
#		raise(eo.NoGradientException(error))
#	# only one value couldn't be computed
#	elif (obj_ci_minus == float("inf")) or (obj_ci_plus == float("inf")):  
#		obj_ci = obj_val_for_ci(ci, cplex_obj, seg_index)
#		# no objective value for ci could be computed
#		if obj_ci == float("inf"):
#			error = "No gradient exists for ci {0} in segment {1}.".format(ci, seg_index) 
#			raise(eo.NoGradientException(error))
#		# compute gradient with ci - h
#		if obj_ci_plus == float("inf"):
#			my_gradient = (float(obj_ci) - obj_ci_minus) / h
#		# compute gradient with ci + h
#		else:
#			my_gradient = (float(obj_ci_plus)  - obj_ci) / h
#	# both values were computed
#	else:
#		my_gradient = (float(obj_ci_plus) - obj_ci_minus) / (2 * h)
#
#	logging.info("Gradient for segment {0} with copy number {1} is: {2}". format(
#		seg_index, ci, my_gradient))
#
#	return my_gradient

# Function that determines the initial direction in which the line search
# for the ci should go.
#
# func: function that's used to compute the objective
# ci: starting value
# obj_ci: objective of ci
# epsilon: how far other values are around ci
# epsilon_plateau_round: if values around ci are equal to ci, how often should new
#	values with other values of epilon be computed
# *args: parameters for function
def initialize_direction(func, ci, obj_ci, epsilon, epsilon_plateau_rounds, 
	start_direction, *args):
	
	optimum_found = False
	direction = 0

	ci_left = ci - epsilon
	ci_right = ci + epsilon
	obj_ci_left = func(ci_left, *args)
	obj_ci_right = func(ci_right, *args)

	# when the objectives for the points left and right of ci don't
	# differ from the value for ci, it is tried to find points that
	# lie further away from ci until a certain number of rounds
	# is reached
	(ci_left, obj_ci_left, ci_right, obj_ci_right) = plateau_loop(
		func, ci, obj_ci, ci_left, obj_ci_left, ci_right, obj_ci_right, epsilon, 
		epsilon_plateau_rounds, 1, *args)
	
	# choose direction
	(optimum_found, direction) = choose_direction(obj_ci, obj_ci_left, obj_ci_right, 
		start_direction)
	return (optimum_found, direction)

# given the objective of point ci, the objectives of the points next to it
# and the direction that was used before, choose the next direction to go in
def choose_direction(obj_ci, obj_ci_1, obj_ci_2, old_direction):

	optimum_found = False
	direction = 0

	# when the objectives for the point left and right of ci
	# are smaller than the objective of ci, ci is an optimum
	if (obj_ci > obj_ci_1) and (obj_ci > obj_ci_2):
		optimum_found = True
	
	# when the objectives for the point left and right of ci
	# are higher than the objective of ci (implicit here)
	# and are equal, choose default direction
	elif obj_ci_1 == obj_ci_2:
		direction = old_direction

	# when objectiv of point left of ci is higher than objective
	# of point right from ci, choose direction "left"
	elif obj_ci_1 > obj_ci_2:
		direction = - old_direction
		
	# when objectiv of point right of ci is higher than objective
	# of point left from ci, choose direction "right"
	else:
		direction = old_direction

	return (optimum_found, direction)

# when the objectives of the points next to ci don't
# differ from the value for ci, it is tried to find points that
# lie further away from ci until a certain number of rounds
# is reached
def plateau_loop(func, ci, obj_ci, ci_1, obj_ci_1, ci_2,
	obj_ci_2, epsilon, epsilon_plateau_rounds, direction, *args):

	plateau_round_count = 0

	while (obj_ci == obj_ci_1 and obj_ci == obj_ci_2):

		if plateau_round_count > epsilon_plateau_rounds:
			error = "After {0} rounds with an epsilon of {1} ci still lies on a plateau.".format(
				epsilon_plateau_rounds, epsilon)
			raise eo.CiLineSearchEpsilonPlateau(error)

		epsilon = 2 * epsilon
		ci_1 = ci - (direction * epsilon)
		ci_2 = ci + (direction * epsilon)
		obj_ci_1 = func(ci_1, *args)
		obj_ci_2 = func(ci_2, *args)

		plateau_round_count += 1

	return (ci_1, obj_ci_1, ci_2, obj_ci_2)

# compute the next point for the line search that lies in the same direction as the last one
def step_same_direction(func, ci_2, obj_ci_2, first_direction, direction, delta, *args):
	# if the direction was changed already once, the step size must
	# be taken smaller
	if not first_direction:
		delta = float(delta) / 2
	# compute new values
	ci_3 = ci_2 + (direction * delta)
	obj_ci_3 = func(ci_3, *args)

	return (ci_2, obj_ci_2, ci_3, obj_ci_3, delta)

# compute the next point for the line search that lies in the other direction compared
# to the last one
def step_other_direction(func, ci_2, direction, delta, *args):
	delta = float(delta) / 2
	ci_3 = ci_2 - (direction * delta)
	obj_ci_3 = func(ci_3, *args)

	first_direction = False
	return (ci_3, obj_ci_3, delta, first_direction)


# ...
# func: function that's used to compute the objective
# ci_1, ci_2: value 1 and 2
# obj_ci_1, obj_ci_2: objectives of value 1 and 2
# direction: direction in which the search should go
# first_direction: if this is the first direction (otherwise it was changed
#	already)
# delta: step size for the search
# epsilon: how far other values are around ci
# epsilon_plateau_round: if values around ci are equal to ci, how often should new
#	values with other values of epilon be computed
# *args: parameters for function
def find_new_interval(func, ci_1, obj_ci_1, ci_2, obj_ci_2, direction, first_direction,
	delta, epsilon, epsilon_plateau_rounds, *args):
	
	optimum_found = False
	new_direction = 0

	# when direction goes to the left, swap ci_1 and ci_2
	if direction == -1:
		(ci_1, obj_ci_1, ci_2, obj_ci_2) = (ci_2, obj_ci_2, ci_1, obj_ci_1)

	# if objective of ci_2 is smaller than objective of ci_1
	# previous taken step was too big, take smaller step
	if obj_ci_2 < obj_ci_1:
		delta = float(delta) / 2
		ci_2 = ci_1 + (direction * delta)
		obj_ci_2 = func(ci_2, *args)
		new_direction = direction

	# if objective of ci_2 is bigger or equal than objective of ci_1
	# the direction of ci_2 is checked
	else:
		ci_2_prime = ci_2 + (direction * epsilon)
		obj_ci_2_prime = func(ci_2_prime, *args)

		# direction stays the same
		if obj_ci_2_prime > obj_ci_2:
			new_direction = direction
			(ci_1, obj_ci_1, ci_2, obj_ci_2, delta) = step_same_direction(
				func, ci_2, obj_ci_2, first_direction, direction, delta, *args)

		# obj_ci_2_prime is not bigger than obj_ci_2, direction might change
		# check other side of ci_2
		else:
			ci_2_pp = ci_2 - (direction * epsilon)
			obj_ci_2_pp = func(ci_2_pp, *args)

			# ci_2 lies on a plateau
			if (obj_ci_2_pp == obj_ci_2) and (obj_ci_2_prime == obj_ci_2):
				(ci_2_pp, obj_ci_2_pp, ci_2_prime, obj_ci_2_prime) = plateau_loop(
					func, ci_2, obj_ci_2, ci_2_pp,  obj_ci_2_pp, ci_2_prime, 
					obj_ci_2_prime, epsilon, epsilon_plateau_rounds, 
					direction, *args)
			# after borders of plateau is reached (if there is one), continue

			# look for new direction
			(optimum_found, new_direction) = choose_direction(obj_ci_2, obj_ci_2_pp, 
				obj_ci_2_prime, direction)

			# optimum wasn't found, define next step
			if not optimum_found:
				# step in same direction
				if new_direction == direction:
					(ci_1, obj_ci_1, ci_2, obj_ci_2, delta) = (
						step_same_direction(func, ci_2, obj_ci_2, 
						first_direction, direction, delta, *args))
				# step in other direction
				else:
					(ci_1, obj_ci_1, delta, first_direction) = (
						step_other_direction(func, ci_2, direction, 
						delta, *args))

	# when direction goes to the left, swap ci_1 and ci_2 back
	if direction == -1:
		(ci_1, obj_ci_1, ci_2, obj_ci_2) = (ci_2, obj_ci_2, ci_1, obj_ci_1)

	return (optimum_found, ci_1, obj_ci_1, ci_2, obj_ci_2, new_direction, 
		first_direction, delta)

# start_ci: initial value for ci
# start_objective: objective for start_ci
# cplex_object: object that used for optimization with CPLEX
# epsilon: how far other points lie around ci
# delta: distance from ci in which is looked for new points
# epsilon_plateau_rounds: if values around ci are equal to ci, how often should new
#	values with other values of epilon be computed
# start_direction: direction in which the search is started
# seg_index: index of segments in which is looked for ci
# general_round_number: total number of rounds in that it is tried to update ci
# delta_end_size: when delta got small enough, search for new ci is stopped
# interval_shortend_round_number: number of rounds in that is is tried to update ci, when
# 	the interval is actually shortened
# warm_start_dc_binary, warm_start_dsnp, warm_start_dssm: values for warm start
def do_line_search(start_ci, start_objective, cplex_object, epsilon, delta, epsilon_plateau_rounds, 
	seg_index, start_direction=1,
	general_round_number=50, delta_end_size=0.0001, interval_shortend_round_number=50,
	warm_start_dc_binary=None, warm_start_dsnp=None, warm_start_dssm=None):

	# direction is initialized
	(optimum_found, direction) = initialize_direction(obj_val_for_ci, start_ci, 
		start_objective, epsilon, epsilon_plateau_rounds, 
		start_direction, cplex_object, seg_index, warm_start_dc_binary, 
		warm_start_dsnp, warm_start_dssm)
	if optimum_found:
		#TODO: best variable assignment
		return (True, start_ci)

	# compute other side of interval
	ci_1 = obj_ci_1 = ci_2 = obj_ci_2 = 0.0
	if direction == 1:
		ci_1 = start_ci
		obj_ci_1 = start_objective
		ci_2 = start_ci + delta
		obj_ci_2 = obj_val_for_ci(ci_2, cplex_object, seg_index, warm_start_dc_binary,
			warm_start_dsnp, warm_start_dssm)
	else:
		ci_1 = start_ci - delta
		obj_ci_1 = obj_val_for_ci(ci_1, cplex_object, seg_index, warm_start_dc_binary,
			warm_start_dsnp, warm_start_dssm)
		ci_2 = start_ci
		obj_ci_2 = start_objective

	# look for best ci
	current_general_round_number = 0
	current_interval_shortend_round_number = 0
	optimum_found = False
	first_direction = True
	plateau = False
	while ((optimum_found == False) and
		(current_general_round_number < general_round_number) and
		(delta > delta_end_size) and
		(current_interval_shortend_round_number < interval_shortend_round_number) and
		plateau == False):

		try:
			(optimum_found, ci_1, obj_ci_1, ci_2, obj_ci_2, direction, first_direction, 
				delta) = find_new_interval(obj_val_for_ci, ci_1, obj_ci_1, ci_2, 
				obj_ci_2, 
				direction, first_direction, delta, epsilon, epsilon_plateau_rounds, 
				cplex_object, seg_index, warm_start_dc_binary, warm_start_dsnp, 
				warm_start_dssm)
		# end of interval lies on a plateau, no better value for ci was found
		# after a certain number of iterations
		except eo.CiLineSearchEpsilonPlateau, exc:
			plateau = True	

		current_general_round_number += 1
		if first_direction == False:
			current_interval_shortend_round_number += 1

	# choose best ci
	best_ci = 0
	if obj_ci_1 > obj_ci_2:
		best_ci = ci_1
	else:
		best_ci = ci_2

	#TODO: best variable assignment
	return (optimum_found, best_ci)


