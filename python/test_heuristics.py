import unittest
import main
import onctopus_io as oio
import lineage
import cnv
import snp_ssm
import os
import sys
import model
import logging
import constants as cons

class HeuristicsTest(unittest.TestCase):
	
	def setUp(self):
		self.time = 1e+75
		self.threads = 1
		self.probing = 0
		self.emph_switch = 0
		self.coef_reduc = -1
		self.mipgap = 1e-04
		self.symmetry = 4
		self.strategy_file = 1
		self.workmem = 128.0 
		self.workdir = "/scratch"
		self.treememory = 1e+75 
		self.emphasis_memory = 0 
		self.nodeselect = 1
		self.cplex_log_file = None

		numeric_logging_info = getattr(logging, "DEBUG".upper(), None)
		logging.basicConfig(filename="testdata/unittests/heuristic_test_1_logger",
			filemode='w', level=numeric_logging_info)

		self.number_spline_points = 50

		#self.f = open(os.devnull, 'w')
		#sys.stdout = self.f

	def test_iterative_heuristic_all_in_1_clustering(self):
		seg_file = "testdata/unittests/heuristic_test_1_seg_as"
		ssm_file = "testdata/unittests/heuristic_test_1_2_ssm"
		sublin_num = 3
		output_file_optimal = "testdata/unittests/heuristic_test_1_2_result_file_optimal_as"
		simple_CN_changes = False
		cluster_SSM = True
		fixed_phi_file = "testdata/unittests/heuristic_test_1_2_phis"

		max_rounds = 3
		epsilon = 0.00001
		logging_info = logging.NOTSET

		(lineages_after_heuristic, used_rounds, heuristic_objective, 
			initial_lineages, initial_objective,
			z_matrix_list, new_lineage_list, lin_div_rule_feasibility, cluster_SSM) = (
			main.start_iterative_heuristic_all_in(
			seg_file, ssm_file, sublin_num, output_file_optimal, 
			self.time, self.time, self.threads, self.probing,
			self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, self.strategy_file,
			self.workmem, self.workdir, self.treememory, self.emphasis_memory,
			self.nodeselect, self.number_spline_points, max_rounds, 
			epsilon, self.cplex_log_file, logging_info, test_run=True,
			simple_CN_changes=simple_CN_changes, cluster_SSM=cluster_SSM, 
			fixed_phi_file=fixed_phi_file))

		# tests
		self.assertEqual(len(lineages_after_heuristic[1].ssms) + len(lineages_after_heuristic[2].ssms), 6)
		self.assertEqual(len(initial_lineages[1].ssms) + len(initial_lineages[2].ssms), 4)



	# Test with 3 segments and different numbers of SSMs
	# Not all segment have SSMs
	# best solution is given as starting point to iterative heuristic, so after
	#	two rounds covergence is reached
	def test_iterative_heuristic_all_in_1_seg(self):
		# test with allele-specifc CN and with non-simple CN changes, fix fixed phis
		seg_file = "testdata/unittests/heuristic_test_1_seg_as"
		ssm_file = "testdata/unittests/heuristic_test_1_ssm"
		fixed_phi_file = "testdata/unittests/heuristic_test_1_phis"
		sublin_num = 3
		output_file_optimal = "testdata/unittests/heuristic_test_1_result_file_optimal_as"
		simple_CN_changes = False

		max_rounds = 3
		epsilon = 0.00001
		logging_info = logging.NOTSET

		(lineages_after_heuristic, used_rounds, heuristic_objective, 
			initial_lineages, initial_objective,
			z_matrix_list, new_lineage_list, lin_div_rule_feasibility, cluster_SSM) = (
			main.start_iterative_heuristic_all_in(
			seg_file, ssm_file, sublin_num, output_file_optimal, 
			self.time, self.time, self.threads, self.probing,
			self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, self.strategy_file,
			self.workmem, self.workdir, self.treememory, self.emphasis_memory,
			self.nodeselect, self.number_spline_points, max_rounds, 
			epsilon, self.cplex_log_file, logging_info, test_run=True,
			simple_CN_changes=simple_CN_changes, fixed_phi_file=fixed_phi_file))

		# tests
		self.assertEqual(lineages_after_heuristic[1].freq, 0.5)
		self.assertEqual(lineages_after_heuristic[2].freq, 0.2)

		# test with allele-specifc CN and with non-simple CN changes
		seg_file = "testdata/unittests/heuristic_test_1_seg_as"
		ssm_file = "testdata/unittests/heuristic_test_1_ssm"
		#z_file = "testdata/unittests/heuristic_test_1_z_matrix"
		sublin_num = 3
		output_file_optimal = "testdata/unittests/heuristic_test_1_result_file_optimal_as"
		simple_CN_changes = False
		cplex_log_file = "testdata/unittests/heuristic_test_1.cplex.log"

		max_rounds = 3
		epsilon = 0.00001
		logging_info = logging.NOTSET

		(lineages_after_heuristic, used_rounds, heuristic_objective, 
			initial_lineages, initial_objective,
			z_matrix_list, new_lineage_list, lin_div_rule_feasibility, cluster_SSM) = (
			main.start_iterative_heuristic_all_in(
			seg_file, ssm_file, sublin_num, output_file_optimal, 
			self.time, self.time, self.threads, self.probing,
			self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, self.strategy_file,
			self.workmem, self.workdir, self.treememory, self.emphasis_memory,
			self.nodeselect, self.number_spline_points, max_rounds, 
			epsilon, cplex_log_file, logging_info, test_run=True,
			simple_CN_changes=simple_CN_changes))

		# tests
		self.assertEqual(used_rounds, 1)
		self.assertEqual(lineages_after_heuristic[0], initial_lineages[0])
		self.assertEqual(lineages_after_heuristic[1], initial_lineages[1])
		self.assertEqual(lineages_after_heuristic[2], initial_lineages[2])
		self.assertAlmostEqual(heuristic_objective, initial_objective)
		
		##########################
		# test with allele-specifc CN and with simple CN changes
		seg_file = "testdata/unittests/heuristic_test_1_seg_as"
		ssm_file = "testdata/unittests/heuristic_test_1_ssm"
		#z_file = "testdata/unittests/heuristic_test_1_z_matrix"
		sublin_num = 3
		output_file_optimal = "testdata/unittests/heuristic_test_1_result_file_optimal_as"
		simple_CN_changes = True

		max_rounds = 3
		epsilon = 0.00001
		logging_info = logging.NOTSET

		(lineages_after_heuristic, used_rounds, heuristic_objective, 
			initial_lineages, initial_objective,
			z_matrix_list, new_lineage_list, lin_div_rule_feasibility, cluster_SSM) = (
			main.start_iterative_heuristic_all_in(
			seg_file, ssm_file, sublin_num, output_file_optimal, 
			self.time, self.time, self.threads, self.probing,
			self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, self.strategy_file,
			self.workmem, self.workdir, self.treememory, self.emphasis_memory,
			self.nodeselect, self.number_spline_points, max_rounds, 
			epsilon, self.cplex_log_file, logging_info, test_run=True,
			simple_CN_changes=simple_CN_changes))

		# tests
		self.assertEqual(used_rounds, 1)
		self.assertEqual(lineages_after_heuristic[0], initial_lineages[0])
		self.assertEqual(lineages_after_heuristic[1], initial_lineages[1])
		self.assertEqual(lineages_after_heuristic[2], initial_lineages[2])
		self.assertAlmostEqual(heuristic_objective, initial_objective)



def suite():
	return unittest.TestLoader().loadTestsFromTestCase(HeuristicsTest)
