import unittest
import optimization
import onctopus_io as oio
import model
import ci_line_search
import exceptions_onctopus as eo

class CiLineSearchTest(unittest.TestCase):

	def line_search_create_cplex_object(self):
		seg_file = "testdata/unittests/line_search_seg_1"
		snp_file = "testdata/unittests/line_search_no_snps"
		ssm_file = "testdata/unittests/line_search_no_ssms"
		number_spline_points = 50
		sublin_num = 2

		# create data for first segment fixed
		my_lineages = oio.read_result_file("testdata/unittests/line_search_result_1")
		cn_state_num = 2
		fixed_seg = model.create_fixed_CNV_data(my_lineages, cn_state_num)[0]
		unfixed_seg_1 = (fixed_seg[0, 1, :].flatten().tolist() 
			+ fixed_seg[1, 1, :].flatten().tolist()
			+ fixed_seg[2, 1, :].flatten().tolist()
			+ fixed_seg[3, 1, :].flatten().tolist())

		# create segment and mutations lists and splines
		(seg_list, snp_list, ssm_list, seg_splines, snp_splines, ssm_splines) = (
			model.create_segment_and_mutation_lists_and_splines(seg_file, snp_file,
			ssm_file, number_spline_points))

		# prepare optimization
		cplex_obj = optimization.Optimization_with_CPLEX(seg_splines, snp_splines, ssm_splines)
		cplex_obj.set_other_parameter(sublin_num, snp_list, ssm_list, seg_list)
		cplex_obj.cplex_log_file = None
		# create variables and constraints
		cplex_obj.create_variables()
		cplex_obj.create_standard_constraints(snp_list, ssm_list, seg_list)
		cplex_obj.create_fixed_variables_constraints(
			fixed_cnv=unfixed_seg_1,
			unfixed_cnv_start=1, unfixed_cnv_stop=1)

		return cplex_obj

	def test_do_line_search(self):

		cplex_obj = self.line_search_create_cplex_object()  
		seg_index = 1

		# ci is good at beginning, ci = 2.00000155
		ci = 2.00000155
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.5
		epsilon_plateau_rounds = 3
		start_direction = 1
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction)
		self.assertTrue(finished)
		self.assertEqual(best_ci, ci)

		# ci is not good at beginning, but can be found, search from left
		ci = 1.5
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.5
		epsilon_plateau_rounds = 3
		start_direction = 1
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction)
		self.assertTrue(finished)
		self.assertEqual(best_ci, 2.0)

		# ci is not good at beginning, but can be found, search from right
		ci = 2.5
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.5
		epsilon_plateau_rounds = 3
		start_direction = 1
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction)
		self.assertTrue(finished)
		self.assertEqual(best_ci, 2.0)

		# ci is not good at beginning, is not found, because general number of steps is too low
		ci = 1.5
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.1
		epsilon_plateau_rounds = 3
		start_direction = 1
		general_round_number = 2
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction,
			general_round_number=general_round_number)
		self.assertFalse(finished)
		self.assertAlmostEqual(best_ci, 1.8)

		# ci is not good at beginning, is not found, delta end size reached
		ci = 1.9
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.5
		epsilon_plateau_rounds = 3
		start_direction = 1
		delta_end_size = 0.25
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction,
			delta_end_size=delta_end_size)
		self.assertFalse(finished)
		self.assertAlmostEqual(best_ci, 1.9)

		# ci is not good at beginning, is not found, delta end size reached
		ci = 1.7
		ci_objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		epsilon = 0.001
		delta = 0.5
		epsilon_plateau_rounds = 3
		start_direction = 1
		interval_shortend_round_number = 1
		(finished, best_ci) = ci_line_search.do_line_search(ci, ci_objective, cplex_obj,
			epsilon, delta, epsilon_plateau_rounds, seg_index, start_direction=start_direction,
			interval_shortend_round_number=interval_shortend_round_number)
		self.assertFalse(finished)
		self.assertAlmostEqual(best_ci, 1.95)

	#def test_gradient_for_ci_function(self):
	#	
	#	cplex_obj = self.line_search_create_cplex_object()
	#	seg_index = 1

	#	# test on both sides infeasibilty, ci = 0.5
	#	ci = 0.5
	#	with self.assertRaises(eo.NoGradientException):
	#		ci_line_search.gradient_for_ci_function(ci, cplex_obj, seg_index)

	#	# on one side infeasible, but also at position ci
	#	h = 1e-5
	#	ci = 1.0 - h
	#	with self.assertRaises(eo.NoGradientException):
	#		model.gradient_for_ci_function(ci, cplex_obj, seg_index)


	#	# right of ci okay, left not
	#	ci = 1.0
	#	gradient = model.gradient_for_ci_function(ci, cplex_obj, seg_index)
	#	self.assertTrue(gradient < 0)
	#	
	#	# left of ci okay, right not
	#	ci = 3.0
	#	gradient = model.gradient_for_ci_function(ci, cplex_obj, seg_index)
	#	self.assertTrue(gradient > 0)

	#	# point okay, is minima
	#	ci = 2.00000155
	#	epsilon = 1e-2
	#	gradient = model.gradient_for_ci_function(ci, cplex_obj, seg_index)
	#	self.assertTrue(abs(gradient) < epsilon)

	def test_find_new_interval(self):

		def my_func(x):
			if x < 2:
				return -4
			if x < 5:
				return x - 6
			if x < 8:
				return -x + 4
			else:
				return -4

		# direction to the right, step too far, lower stepsize, new ci_2
		ci_1 = 5
		obj_ci_1 = -1
		ci_2 = 6
		obj_ci_2 = -2
		direction = 1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 5, -1, 5.5, -1.5, 1, True, 0.5))

		# direction to the left, step too far, lower stepsize, new ci_1
		ci_1 = 4
		obj_ci_1 = -2
		ci_2 = 5
		obj_ci_2 = -1
		direction = -1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 4.5, -1.5, 5, -1, -1, True, 0.5))

		# direction to the right, step good, same stepsize, new values
		ci_1 = 2
		obj_ci_1 = -4
		ci_2 = 3
		obj_ci_2 = -3
		direction = 1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 3, -3, 4, -2, 1, True, 1))

		# direction to the left, step good, same stepsize, new values
		ci_1 = 6
		obj_ci_1 = -2
		ci_2 = 7
		obj_ci_2 = -3
		direction = -1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 5, -1, 6, -2, -1, True, 1))

		# direction to the right, step good, same stepsize, new values,
		# direction got changed before
		ci_1 = 2
		obj_ci_1 = -4
		ci_2 = 3
		obj_ci_2 = -3
		direction = 1
		first_direction = False
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 3, -3, 3.5, -2.5, 1, False, 0.5))

		# direction to the left, step good, same stepsize, new values,
		# direction got changed before
		ci_1 = 6
		obj_ci_1 = -2
		ci_2 = 7
		obj_ci_2 = -3
		direction = -1
		first_direction = False
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 5.5, -1.5, 6, -2, -1, False, 0.5))

		# direction to the right, change direction, new values
		ci_1 = 3.5
		obj_ci_1 = -2.5
		ci_2 = 5.5
		obj_ci_2 = -1.5
		direction = 1
		first_direction = True
		delta = 2
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 4.5, -1.5, 5.5, -1.5, -1, False, 1))

		# direction to the right, change direction, new values
		ci_1 = 4.5
		obj_ci_1 = -1.5
		ci_2 = 6.5
		obj_ci_2 = -2.5
		direction = -1
		first_direction = True
		delta = 2
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 4.5, -1.5, 5.5, -1.5, 1, False, 1))

		# direction to the right, plateau, same direction, new values
		ci_1 = 0.9
		obj_ci_1 = -4
		ci_2 = 1.9
		obj_ci_2 = -4
		direction = 1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (False, 1.9, -4, 2.9, -3.1, 1, True, 1))

		# direction to the left, plateau, same direction, new values
		ci_1 = 8.1
		obj_ci_1 = -4
		ci_2 = 9.1
		obj_ci_2 = -4
		direction = -1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		(optimum_found, ci_1, obj_ci_1, ci_2, obj_ci_2, new_direction,
			first_direction, delta) = ci_line_search.find_new_interval(my_func, 
			ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual((optimum_found, ci_1, ci_2, obj_ci_2, new_direction,
			first_direction, delta), (False, 7.1, 8.1, -4, -1, True, 1))
		self.assertAlmostEqual(obj_ci_1, -3.1)

		# direction to the right, optimum found
		ci_1 = 4
		obj_ci_1 = -2
		ci_2 = 5
		obj_ci_2 = -1
		direction = 1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (True, 4, -2, 5, -1, 0, True, 1))

		# direction to the left, optimum found
		ci_1 = 5
		obj_ci_1 = -1
		ci_2 = 6
		obj_ci_2 = -2
		direction = -1
		first_direction = True
		delta = 1
		epsilon = 0.1
		epsilon_plateau_rounds = 3
		answer = ci_line_search.find_new_interval(my_func, ci_1, obj_ci_1, ci_2, obj_ci_2, 
			direction, first_direction, delta, epsilon, epsilon_plateau_rounds)
		self.assertEqual(answer, (True, 5, -1, 6, -2, 0, True, 1))

	def test_choose_direction(self):

		direction = 1

		# optimum found
		obj = -2
		obj_1 = -3
		obj_2 = -4
		self.assertEqual((True, 0), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))

		# direction the same as before
		obj = -2
		obj_1 = -1
		obj_2 = -1
		self.assertEqual((False, direction), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))

		# direction to the right
		obj = -2
		obj_1 = -3
		obj_2 = -1
		self.assertEqual((False, direction), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))

		# direction to the right
		obj = -2
		obj_1 = -3
		obj_2 = -1.5
		self.assertEqual((False, direction), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))

		# direction to the left
		obj = -2
		obj_1 = -1
		obj_2 = -3
		self.assertEqual((False, - direction), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))

		# direction to the left
		obj = -2
		obj_1 = -1
		obj_2 = -1.5
		self.assertEqual((False, - direction), ci_line_search.choose_direction(
			obj, obj_1, obj_2, direction))
	
	def test_plateau_loop(self):

		def my_func(x):
			if x <= 1:
				return -1
			else:
				return -2

		epsilon = 0.1
		epsilon_plateau_rounds = 3
		direction = 1

		# extentions works
		ci = 0.9
		ci_1 = 0.8
		ci_2 = 1
		obj_ci = obj_ci_1 = obj_ci_2 = -1

		(ci_1, obj_ci_1, ci_2, obj_ci_2) = ci_line_search.plateau_loop(
			my_func, ci, obj_ci, ci_1, obj_ci_1, ci_2, obj_ci_2, epsilon, 
			epsilon_plateau_rounds, direction)
		self.assertEqual((ci_1, obj_ci_1, ci_2, obj_ci_2), (0.7, -1, 1.1, -2))

		# extentions works, other direction
		direction = -1
		ci = 1.2
		ci_1 = 1.3
		ci_2 = 1.1
		obj_ci = obj_ci_1 = obj_ci_2 = -2

		(ci_1, obj_ci_1, ci_2, obj_ci_2) = ci_line_search.plateau_loop(
			my_func, ci, obj_ci, ci_1, obj_ci_1, ci_2, obj_ci_2, epsilon, 
			epsilon_plateau_rounds, direction)
		self.assertEqual((ci_1, obj_ci_1, ci_2, obj_ci_2), (1.4, -2, 1.0, -1))

		# extention is not needed
		direction = 1
		ci = 1
		ci_1 = 0.9
		ci_2 = 1.1
		obj_ci = obj_ci_1 = -1
		obj_ci_2 = -2
		(ci_1, obj_ci_1, ci_2, obj_ci_2) = ci_line_search.plateau_loop(
			my_func, ci, obj_ci, ci_1, obj_ci_1, ci_2, obj_ci_2, epsilon, 
			epsilon_plateau_rounds, direction)
		self.assertEqual((ci_1, obj_ci_1, ci_2, obj_ci_2), (0.9, -1, 1.1, -2))

		# extention doesn't work
		ci = 10
		ci_1 = 10.1
		ci_2 = 9.9
		obj_ci = obj_ci_1 = obj_ci_2 = -2

		with self.assertRaises(eo.CiLineSearchEpsilonPlateau):	
			(ci_1, obj_ci_1, ci_2, obj_ci_2) = ci_line_search.plateau_loop(
				my_func, ci, obj_ci, ci_1, obj_ci_1, ci_2, obj_ci_2, epsilon, 
				epsilon_plateau_rounds, direction)



	def test_obj_val_for_ci(self):

		cplex_obj = self.line_search_create_cplex_object()

		# test with ci = 2, objective < 0
		ci = 2
		seg_index = 1
		objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		self.assertTrue(objective < 0 and objective != float('inf'))

		# test with other value, that leads to feasible solution, ci = 1.8, objective < 0
		ci = 1.8
		seg_index = 1
		objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		self.assertTrue(objective < 0 and objective != float('inf'))

		# test with ci = 10, infeasible
		ci = 10
		seg_index = 1
		objective = ci_line_search.obj_val_for_ci(ci, cplex_obj, seg_index)
		self.assertTrue(objective == - float('inf'))

def suite():
	 return unittest.TestLoader().loadTestsFromTestCase(CiLineSearchTest)
