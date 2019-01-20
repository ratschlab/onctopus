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
import optimization

class MiniTestCasesTest_2(unittest.TestCase):
	
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
		logging.basicConfig(filename="testdata/unittests/mini_test_cases/logger_4_12",
			filemode='w', level=numeric_logging_info)

		self.number_spline_points = 50

		#self.f = open(os.devnull, 'w')
		#sys.stdout = self.f

	#def test_14_1_excat_mode_clustering(self):
	#	input_seg = "testdata/unittests/mini_test_cases/14_1_seg_as"
	#	input_snp = None
	#	input_ssm = "testdata/unittests/mini_test_cases/14_1_ssm"
	#	num = 3
	#	out_results = "testdata/unittests/mini_test_cases/14_1_out_results"
	#	simple_CN_changes = False
	#	max_x_CN_changes = 1
	#	only_one_loss = False
	#	only_gains_losses_LOH = True
	#	cluster_SSM = True
	#	cluster_num_param = 1
	#	self.time = 600
	#	
	#	# do optimization
	#	(my_lineages, cplex_obj) = main.go_onctopus(
	#		input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
	#		self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
	#		self.strategy_file, self.workmem, self.workdir, self.treememory, 
	#		self.emphasis_memory, self.nodeselect,
	#		self.cplex_log_file, self.number_spline_points,
	#		test_run=True, allele_specific=True,
	#		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
	#		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
	#		cluster_SSM=cluster_SSM, cluster_num_param=cluster_num_param,
	#		write_output_to_disk=True)

	#	cplex_status = cplex_obj.my_prob.solution.status[cplex_obj.my_prob.solution.get_status()]
	#	self.assertEqual(cplex_status, "MIP_optimal")

	def test_14_2_excat_mode_clustering_CNV_fixation(self):
		input_seg = "testdata/unittests/mini_test_cases/14_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/14_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/14_2_out_results"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/14_1_fixed_cnvs"
		simple_CN_changes = False
		max_x_CN_changes = 5
		only_one_loss = False
		only_gains_losses_LOH = True
		cluster_SSM = True
		cluster_num_param = 2
		self.time = 600
		
		# do optimization
		(my_lineages, cplex_obj) = main.go_onctopus(
			input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cluster_SSM=cluster_SSM, use_super_SSMs=True,
			write_output_to_disk=True, fixed_cnv_file=fixed_cnv_file, review_ambiguous_relations=False)

		cplex_status = cplex_obj.my_prob.solution.status[cplex_obj.my_prob.solution.get_status()]
		self.assertEqual(cplex_status, "MIP_optimal")
	
def suite():
	return unittest.TestLoader().loadTestsFromTestCase(MiniTestCasesTest_2)
