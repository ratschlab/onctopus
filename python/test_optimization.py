import unittest
import optimization
import log_pdf
import constants as cons
import snp_ssm
import segment
import cplex
import pdb
import numpy as np
import model
import main

class OptimizationTest(unittest.TestCase):

	def setUp(self):

		self.seg_num = 1
		self.snp_num = 1
		self.ssm_num = 1
		
		self.number_spline_points = 30

		# create splines
		self.seg_spl_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 300,400,1000,
			log_pdf.neg_binomial,340,1000)[0]
		self.snp_spl_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0.5,0.7,1000,
			log_pdf.beta_binomial,220,340,1000)[0]
		self.ssm_spl_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0.2,0.4,1000,
			log_pdf.beta_binomial, 100, 340, 1000)[0]
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num

		# create segment and SNPs/SSMs
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		snp = snp_ssm.SNP()
		self.snp_list = [snp] * self.snp_num
		ssm = snp_ssm.SSM()
		self.ssm_list = [ssm] * self.ssm_num

	# following part is not needed, 
	# function put in comments because they don't give improvement in terms of
	# speed and memory usage
	#def test_get_values_for_dc_ancestral(self):
	#	sublin_num = 4
	#	opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
	#		self.snp_spl_list, self.ssm_spl_list)
	#	opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	#	opt.seg_num = 2

	#	z_list = [1, 0, 1]
	#	z_matrix = model.create_z_as_maxtrix_w_values(sublin_num, z_list)

	#	dc_binary_list = [
	#		0, 1, 1, 0,
	#		0, 1, 0, 1,

	#		0, 1, 0, 1,
	#		0, 1, 1, 0,

	#		0, 1, 0, 1,
	#		0, 0, 1, 0,
	#		
	#		0, 0, 1, 0,
	#		0, 1, 0, 1]
	#	dc_binary_matrix = np.array(dc_binary_list).reshape(4, 2, 4)

	#	wanted_dc_ancestral_list = [
	#		1, 0, 0,
	#		0, 0, 1,

	#		0, 0, 1,
	#		1, 0, 0]
	#	dc_ancestral_list = opt.get_values_for_dc_ancestral(
	#		dc_binary_matrix, z_matrix)
	#	self.assertListEqual(wanted_dc_ancestral_list, dc_ancestral_list)

	#def test_get_values_for_dc_descendant(self):
	#	sublin_num = 4
	#	opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
	#		self.snp_spl_list, self.ssm_spl_list)
	#	opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	#	opt.seg_num = 2

	#	z_list = [1, 0, 1]
	#	z_matrix = model.create_z_as_maxtrix_w_values(sublin_num, z_list)

	#	dc_binary_list = [
	#		0, 1, 1, 0,
	#		0, 1, 0, 1,

	#		0, 1, 0, 1,
	#		0, 1, 1, 0,

	#		0, 1, 0, 1,
	#		0, 1, 1, 0,

	#		0, 1, 1, 0,
	#		0, 1, 0, 1]
	#	dc_binary_matrix = np.array(dc_binary_list).reshape(4, 2, 4)

	#	wanted_dc_descendant_list = [
	#		1, 0, 0,
	#		0, 0, 1,
	#		
	#		0, 0, 1,
	#		1, 0, 0,
	#		
	#		0, 0, 1,
	#		1, 0, 0,
	#		
	#		1, 0, 0,
	#		0, 0, 1]
	#	dc_descendant_list = opt.get_values_for_dc_descendant(
	#		dc_binary_matrix, z_matrix)
	#	self.assertListEqual(wanted_dc_descendant_list, dc_descendant_list)


	def test_add_values_for_warm_start(self):
		# prepare
		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_snp_matrices()
		opt.vars_aux_dc()
		opt.vars_phi()
		opt.vars_dc_descendant()

		opt.my_prob = cplex.Cplex()
		opt.my_prob.variables.add(obj = opt.my_obj, lb = opt.my_lb, ub = opt.my_ub,
			types = opt.my_ctype, names = opt.my_colnames)

		# add warm start values
		values = [0, 0, 0,
			0, 1, 0,
			0, 0, 0,
			0, 0, 0]
		values_matrix = np.array(values).reshape(4, 1, 3)
		values_matrix_flat = values_matrix.flatten().tolist()
		opt.add_values_for_warm_start(values_matrix_flat, opt.dc_binary_index_start_p1)

		# test
		self.assertEqual(12, len(opt.my_prob.MIP_starts.get_starts()[0][0].ind))
		self.assertListEqual(values, opt.my_prob.MIP_starts.get_starts()[0][0].val)
		self.assertEqual(opt.dc_binary_index_start_p1 + 4, 
			opt.my_prob.MIP_starts.get_starts()[0][0].ind[4])
		self.assertEqual(1, opt.my_prob.MIP_starts.get_starts()[0][0].val[4])
		self.assertEqual("dc_b_p1_binary_0_1", opt.my_colnames[opt.dc_binary_index_start_p1 + 4])

		# add more warm starts
		new_values = [0, 1, 0]
		new_value_matrix = np.array(new_values).reshape(3, 1, 1)
		new_value_matrix_flat = new_value_matrix.flatten().tolist()
		opt.add_values_for_warm_start(new_value_matrix_flat, opt.dsnp_start_index)

		# test
		self.assertEqual(1, opt.my_prob.MIP_starts.get_num())
		self.assertListEqual(new_values + values, opt.my_prob.MIP_starts.get_starts()[0][0].val)

		#######################################################
		# following part is not needed, 
		# function put in comments because they don't give improvement in terms of
		# speed and memory usage
		# add values for variables dc_descendant
		# prepare, set CPLEX object up new
		#sublin_num = 4
		#self.seg_num = 2
		#self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		#seg = segment.Segment(1, 1, 1, 1, 1)
		#self.seg_list = [seg] * self.seg_num

		#opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
		#	self.snp_spl_list, self.ssm_spl_list)
		#opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		#opt.vars_three_snp_matrices()
		#opt.vars_aux_dc()
		#opt.vars_dc_ancestral()
		#opt.vars_phi()
		#opt.vars_dc_descendant()

		#opt.my_prob = cplex.Cplex()
		#opt.my_prob.variables.add(obj = opt.my_obj, lb = opt.my_lb, ub = opt.my_ub,
		#	types = opt.my_ctype, names = opt.my_colnames)

		#z_list = [1, 0, 1]
		#z_matrix = model.create_z_as_maxtrix_w_values(sublin_num, z_list)
		#dc_binary_list = [
		#	0, 1, 1, 0,
		#	0, 1, 0, 1,

		#	0, 1, 0, 1,
		#	0, 1, 1, 0,

		#	0, 1, 0, 1,
		#	0, 1, 1, 0,

		#	0, 1, 1, 0,
		#	0, 1, 0, 1]
		#wanted_dc_descendant_list = [
		#	1, 0, 0,
		#	0, 0, 1,
		#	
		#	0, 0, 1,
		#	1, 0, 0,
		#	
		#	0, 0, 1,
		#	1, 0, 0,
		#	
		#	1, 0, 0,
		#	0, 0, 1]
		#dc_binary_matrix = np.array(dc_binary_list).reshape(4, 2, 4)
		#dc_descendant_list = opt.get_values_for_dc_descendant(
		#	dc_binary_matrix, z_matrix)
		#opt.add_values_for_warm_start(dc_descendant_list, opt.dc_descdendant_start_index)

		## test
		#self.assertEqual(wanted_dc_descendant_list, opt.my_prob.MIP_starts.get_starts()[0][0].val)
		#self.assertEqual(1, opt.my_prob.MIP_starts.get_starts()[0][0].val[0])
		#self.assertEqual(0, opt.my_prob.MIP_starts.get_starts()[0][0].val[1])
		#self.assertEqual(opt.dc_descdendant_start_index, 
		#	opt.my_prob.MIP_starts.get_starts()[0][0].ind[0])
		#self.assertEqual(opt.dc_descdendant_start_index + 10, 
		#	opt.my_prob.MIP_starts.get_starts()[0][0].ind[0] + 10)
	
		## test for dc_ancestral
		#dc_binary_list = [
		#	0, 1, 1, 0,
		#	0, 1, 0, 1,

		#	0, 1, 0, 1,
		#	0, 1, 1, 0,

		#	0, 1, 0, 1,
		#	0, 0, 1, 0,
		#	
		#	0, 0, 1, 0,
		#	0, 1, 0, 1]
		#dc_binary_matrix = np.array(dc_binary_list).reshape(4, 2, 4)

		#wanted_dc_ancestral_list = [
		#	1, 0, 0,
		#	0, 0, 1,

		#	0, 0, 1,
		#	1, 0, 0]
		#dc_ancestral_list = opt.get_values_for_dc_ancestral(
		#	dc_binary_matrix, z_matrix)
		#opt.add_values_for_warm_start(dc_ancestral_list, opt.dc_ancestral_start_index)

		## test
		#self.assertEqual(wanted_dc_ancestral_list, 
		#	opt.my_prob.MIP_starts.get_starts()[0][0].val[:len(dc_ancestral_list)])
		#self.assertEqual(1, opt.my_prob.MIP_starts.get_starts()[0][0].val[0])
		#self.assertEqual(0, opt.my_prob.MIP_starts.get_starts()[0][0].val[1])
		#self.assertEqual(opt.dc_ancestral_start_index, 
		#	opt.my_prob.MIP_starts.get_starts()[0][0].ind[0])
		#self.assertEqual(opt.dc_descdendant_start_index, 
		#	opt.my_prob.MIP_starts.get_starts()[0][0].ind[len(dc_ancestral_list)])


	def test_set_other_parameter(self):
		# tests, whether CN of segments is set correctly

		sublin_num = 2

		self.seg_num = 5
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num

		# function to test, no CNs fixed
		fixed_avg_cn = None
		fixed_avg_cn_start = -1
		fixed_avg_cn_stop = -1
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list,
			fixed_avg_cn, fixed_avg_cn_start, fixed_avg_cn_stop)

		self.assertListEqual(opt.cn, [1, 1, 1, 1, 1,])

		# function to test, all CNs fixed
		fixed_avg_cn = [1, 2, 3, 4, 5]
		fixed_avg_cn_start = -1
		fixed_avg_cn_stop = -1
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list,
			fixed_avg_cn, fixed_avg_cn_start, fixed_avg_cn_stop)

		self.assertListEqual(opt.cn, [1, 2, 3, 4, 5])

		# function to test, CNs in middle unfixed
		fixed_avg_cn = [1, 2, 5]
		fixed_avg_cn_start = 2
		fixed_avg_cn_stop = 3
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list,
			fixed_avg_cn, fixed_avg_cn_start, fixed_avg_cn_stop)

		self.assertListEqual(opt.cn, [1, 2, 1, 1, 5])

		# function to test, CNs in beginning unfixed
		fixed_avg_cn = [3, 4, 5]
		fixed_avg_cn_start = 0
		fixed_avg_cn_stop = 1
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list,
			fixed_avg_cn, fixed_avg_cn_start, fixed_avg_cn_stop)

		self.assertListEqual(opt.cn, [1, 1, 3, 4, 5])

		# function to test, CN at end unfixed
		fixed_avg_cn = [1, 2, 3, 4]
		fixed_avg_cn_start = 4
		fixed_avg_cn_stop = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list,
			fixed_avg_cn, fixed_avg_cn_start, fixed_avg_cn_stop)

		self.assertListEqual(opt.cn, [1, 2, 3, 4, 1])

	def test_constraint_fix_dc_binary_or_SSMs(self):
		# for CNVs
		# do for 3 lineages and 4 segments
		sublin_num = 3
		seg_num = 4
		seg_spl_list = [self.seg_spl_1] * seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		seg_list = [seg] * seg_num

		opt = optimization.Optimization_with_CPLEX(seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, seg_list)
		opt.vars_aux_dc()

		# function to test
		fixed_mutations = [
			[0, [[0, 0, cons.A]]],
			[2, [[1, 1, cons.A], [1, 1, cons.B]]],
			[3, [[2, -1, cons.A], [1, -1, cons.B]]]
			]
		opt.constraint_fix_dc_binary_or_SSMs(fixed_mutations, cons.CNV)

		entries = 36

		# test indices of constraints
		self.assertEqual(0, opt.start_index_constraint_fix_dc_binary)
		self.assertEqual(36, opt.end_index_constraint_fix_dc_binary)

		# rhs
		my_rhs = [
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0
			]
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"fixed_constraint_cnv_a_p1_0_0", "fixed_constraint_cnv_a_p1_0_1", "fixed_constraint_cnv_a_p1_0_2",
			"fixed_constraint_cnv_b_p1_0_0", "fixed_constraint_cnv_b_p1_0_1", "fixed_constraint_cnv_b_p1_0_2",
			"fixed_constraint_cnv_a_m1_0_0", "fixed_constraint_cnv_a_m1_0_1", "fixed_constraint_cnv_a_m1_0_2",
			"fixed_constraint_cnv_b_m1_0_0", "fixed_constraint_cnv_b_m1_0_1", "fixed_constraint_cnv_b_m1_0_2",
			"fixed_constraint_cnv_a_p1_2_0", "fixed_constraint_cnv_a_p1_2_1", "fixed_constraint_cnv_a_p1_2_2",
			"fixed_constraint_cnv_b_p1_2_0", "fixed_constraint_cnv_b_p1_2_1", "fixed_constraint_cnv_b_p1_2_2",
			"fixed_constraint_cnv_a_m1_2_0", "fixed_constraint_cnv_a_m1_2_1", "fixed_constraint_cnv_a_m1_2_2",
			"fixed_constraint_cnv_b_m1_2_0", "fixed_constraint_cnv_b_m1_2_1", "fixed_constraint_cnv_b_m1_2_2",
			"fixed_constraint_cnv_a_p1_3_0", "fixed_constraint_cnv_a_p1_3_1", "fixed_constraint_cnv_a_p1_3_2",
			"fixed_constraint_cnv_b_p1_3_0", "fixed_constraint_cnv_b_p1_3_1", "fixed_constraint_cnv_b_p1_3_2",
			"fixed_constraint_cnv_a_m1_3_0", "fixed_constraint_cnv_a_m1_3_1", "fixed_constraint_cnv_a_m1_3_2",
			"fixed_constraint_cnv_b_m1_3_0", "fixed_constraint_cnv_b_m1_3_1", "fixed_constraint_cnv_b_m1_3_2"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["dc_a_p1_binary_0_0"], ["dc_a_p1_binary_0_1"], ["dc_a_p1_binary_0_2"],
			["dc_b_p1_binary_0_0"], ["dc_b_p1_binary_0_1"], ["dc_b_p1_binary_0_2"],
			["dc_a_m1_binary_0_0"], ["dc_a_m1_binary_0_1"], ["dc_a_m1_binary_0_2"],
			["dc_b_m1_binary_0_0"], ["dc_b_m1_binary_0_1"], ["dc_b_m1_binary_0_2"],
			["dc_a_p1_binary_2_0"], ["dc_a_p1_binary_2_1"], ["dc_a_p1_binary_2_2"],
			["dc_b_p1_binary_2_0"], ["dc_b_p1_binary_2_1"], ["dc_b_p1_binary_2_2"],
			["dc_a_m1_binary_2_0"], ["dc_a_m1_binary_2_1"], ["dc_a_m1_binary_2_2"],
			["dc_b_m1_binary_2_0"], ["dc_b_m1_binary_2_1"], ["dc_b_m1_binary_2_2"],
			["dc_a_p1_binary_3_0"], ["dc_a_p1_binary_3_1"], ["dc_a_p1_binary_3_2"],
			["dc_b_p1_binary_3_0"], ["dc_b_p1_binary_3_1"], ["dc_b_p1_binary_3_2"],
			["dc_a_m1_binary_3_0"], ["dc_a_m1_binary_3_1"], ["dc_a_m1_binary_3_2"],
			["dc_b_m1_binary_3_0"], ["dc_b_m1_binary_3_1"], ["dc_b_m1_binary_3_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

		# for SSMs, only simple CN changes
		sublin_num = 3
		ssm_num = 4
		ssm_spl_list = [self.ssm_spl_1] * ssm_num
		ssm = snp_ssm.SSM()
		ssm_list = [ssm] * ssm_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list,
			ssm_spl_list, simple_CN_changes=True)

		opt.set_other_parameter(sublin_num, self.snp_list, ssm_list, self.seg_list)
		opt.vars_three_ssm_matrices()

		# function to test
		fixed_mutations = [
			[0, 1, cons.A], [1, 2, cons.B], [3, 1, cons.UNPHASED]
			]
		opt.constraint_fix_dc_binary_or_SSMs(fixed_mutations, cons.SSM)

		entries = 27

		# rhs
		my_rhs = [
			0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 1, 0
			]
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"fixed_constraint_ssm_a_0_0", "fixed_constraint_ssm_a_0_1", "fixed_constraint_ssm_a_0_2",
			"fixed_constraint_ssm_b_0_0", "fixed_constraint_ssm_b_0_1", "fixed_constraint_ssm_b_0_2",
			"fixed_constraint_ssm_unphased_0_0", "fixed_constraint_ssm_unphased_0_1", 
			"fixed_constraint_ssm_unphased_0_2",
			"fixed_constraint_ssm_a_1_0", "fixed_constraint_ssm_a_1_1", "fixed_constraint_ssm_a_1_2",
			"fixed_constraint_ssm_b_1_0", "fixed_constraint_ssm_b_1_1", "fixed_constraint_ssm_b_1_2",
			"fixed_constraint_ssm_unphased_1_0", "fixed_constraint_ssm_unphased_1_1", 
			"fixed_constraint_ssm_unphased_1_2",
			"fixed_constraint_ssm_a_3_0", "fixed_constraint_ssm_a_3_1", "fixed_constraint_ssm_a_3_2",
			"fixed_constraint_ssm_b_3_0", "fixed_constraint_ssm_b_3_1", "fixed_constraint_ssm_b_3_2",
			"fixed_constraint_ssm_unphased_3_0", "fixed_constraint_ssm_unphased_3_1", 
			"fixed_constraint_ssm_unphased_3_2"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["dssm_a_0_0"], ["dssm_a_0_1"], ["dssm_a_0_2"], ["dssm_b_0_0"], ["dssm_b_0_1"], ["dssm_b_0_2"],
			["dssm_0_0"], ["dssm_0_1"], ["dssm_0_2"],
			["dssm_a_1_0"], ["dssm_a_1_1"], ["dssm_a_1_2"], ["dssm_b_1_0"], ["dssm_b_1_1"], ["dssm_b_1_2"],
			["dssm_1_0"], ["dssm_1_1"], ["dssm_1_2"],
			["dssm_a_3_0"], ["dssm_a_3_1"], ["dssm_a_3_2"], ["dssm_b_3_0"], ["dssm_b_3_1"], ["dssm_b_3_2"],
			["dssm_3_0"], ["dssm_3_1"], ["dssm_3_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)
		
		# for SSMs, no simple CN changes
		sublin_num = 3
		ssm_num = 4
		ssm_spl_list = [self.ssm_spl_1] * ssm_num
		ssm = snp_ssm.SSM()
		ssm_list = [ssm] * ssm_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list,
			ssm_spl_list, simple_CN_changes=False)

		opt.set_other_parameter(sublin_num, self.snp_list, ssm_list, self.seg_list)
		opt.vars_three_ssm_matrices()

		# function to test
		fixed_mutations = [
			[0, 1, cons.A], [1, 2, cons.B], [3, 1, cons.UNPHASED]
			]
		opt.constraint_fix_dc_binary_or_SSMs(fixed_mutations, cons.SSM)

		entries = 18

		# rhs
		my_rhs = [
			0, 1, 0, 0, 0, 0, 
			0, 0, 0, 0, 0, 1, 
			0, 1, 0, 0, 0, 0, 
			]
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"fixed_constraint_ssm_a_0_0", "fixed_constraint_ssm_a_0_1", "fixed_constraint_ssm_a_0_2",
			"fixed_constraint_ssm_b_0_0", "fixed_constraint_ssm_b_0_1", "fixed_constraint_ssm_b_0_2",
			"fixed_constraint_ssm_a_1_0", "fixed_constraint_ssm_a_1_1", "fixed_constraint_ssm_a_1_2",
			"fixed_constraint_ssm_b_1_0", "fixed_constraint_ssm_b_1_1", "fixed_constraint_ssm_b_1_2",
			"fixed_constraint_ssm_a_3_0", "fixed_constraint_ssm_a_3_1", "fixed_constraint_ssm_a_3_2",
			"fixed_constraint_ssm_b_3_0", "fixed_constraint_ssm_b_3_1", "fixed_constraint_ssm_b_3_2",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["dssm_a_0_0"], ["dssm_a_0_1"], ["dssm_a_0_2"], ["dssm_b_0_0"], ["dssm_b_0_1"], ["dssm_b_0_2"],
			["dssm_a_1_0"], ["dssm_a_1_1"], ["dssm_a_1_2"], ["dssm_b_1_0"], ["dssm_b_1_1"], ["dssm_b_1_2"],
			["dssm_a_3_0"], ["dssm_a_3_1"], ["dssm_a_3_2"], ["dssm_b_3_0"], ["dssm_b_3_1"], ["dssm_b_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

		
	def test_constraint_ldr(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_child_freq()
		opt.vars_phi()

		# function to test
		opt.constraint_ldr()

		entries = 2

		# rhs 
		my_rhs = [0] * entries
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = ["constraint_ldr_0", "constraint_ldr_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["child_0_1_freq", "child_0_2_freq", "child_0_3_freq", "phi_0"],
			["child_1_2_freq", "child_1_3_freq", "phi_1"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)
		

	def test_constraint_ldr_active(self):
		# do for 5 lineages
		sublin_num = 5
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_child_freq()
		opt.vars_sibling_freq()
		opt.vars_parent_freq()
		opt.vars_child_freq_minus_par_freq()
		opt.vars_chf_m_pf_LDRa()
		opt.vars_chf_m_pf_LDRi()
		opt.vars_child()
		opt.vars_ldr_active()

		# function to test
		opt.constraint_ldr_active()

		entries = 6
		
		# rhs 
		my_rhs = [0] * entries * 3
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["E"] * entries * 2 + ["L"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_ldr_active_long_1_2", "constraint_ldr_active_long_1_3", "constraint_ldr_active_long_1_4",
			"constraint_ldr_active_long_2_3", "constraint_ldr_active_long_2_4", "constraint_ldr_active_long_3_4",
			"constraint_ldr_active_short_1_2", "constraint_ldr_active_short_1_3", "constraint_ldr_active_short_1_4",
			"constraint_ldr_active_short_2_3", "constraint_ldr_active_short_2_4", "constraint_ldr_active_short_3_4",
			"constraint_ldr_active_child_1_2", "constraint_ldr_active_child_1_3", 
			"constraint_ldr_active_child_1_4",
			"constraint_ldr_active_child_2_3", "constraint_ldr_active_child_2_4", 
			"constraint_ldr_active_child_3_4"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["child_freq_minus_par_freq_1_2", "child_1_2_freq", "phi_1", "sibling_0_1_3_freq",
			"sibling_0_1_4_freq", "parent_0_1_freq"],
			["child_freq_minus_par_freq_1_3", "child_1_3_freq", "phi_1", "sibling_0_1_2_freq",
			"sibling_0_1_4_freq", "parent_0_1_freq"],
			["child_freq_minus_par_freq_1_4", "child_1_4_freq", "phi_1", "sibling_0_1_2_freq",
			"sibling_0_1_3_freq", "parent_0_1_freq"],
			["child_freq_minus_par_freq_2_3", "child_2_3_freq", "phi_2", "sibling_0_2_1_freq",
			"sibling_0_2_4_freq", "sibling_1_2_4_freq", "parent_0_2_freq", "parent_1_2_freq"],
			["child_freq_minus_par_freq_2_4", "child_2_4_freq", "phi_2", "sibling_0_2_1_freq",
			"sibling_0_2_3_freq", "sibling_1_2_3_freq", "parent_0_2_freq", "parent_1_2_freq"],
			["child_freq_minus_par_freq_3_4", "child_3_4_freq", "phi_3", "sibling_0_3_1_freq",
			"sibling_0_3_2_freq", "sibling_1_3_2_freq", "parent_0_3_freq", "parent_1_3_freq",
			"parent_2_3_freq"],
			["child_freq_minus_par_freq_1_2", "chf_m_pf_LDRa_1_2", "chf_m_pf_LDRi_1_2"],
			["child_freq_minus_par_freq_1_3", "chf_m_pf_LDRa_1_3", "chf_m_pf_LDRi_1_3"],
			["child_freq_minus_par_freq_1_4", "chf_m_pf_LDRa_1_4", "chf_m_pf_LDRi_1_4"],
			["child_freq_minus_par_freq_2_3", "chf_m_pf_LDRa_2_3", "chf_m_pf_LDRi_2_3"],
			["child_freq_minus_par_freq_2_4", "chf_m_pf_LDRa_2_4", "chf_m_pf_LDRi_2_4"],
			["child_freq_minus_par_freq_3_4", "chf_m_pf_LDRa_3_4", "chf_m_pf_LDRi_3_4"],
			["LDR_active_1_2", "child_1_2"], ["LDR_active_1_3", "child_1_3"],
			["LDR_active_1_4", "child_1_4"], ["LDR_active_2_3", "child_2_3"],
			["LDR_active_2_4", "child_2_4"], ["LDR_active_3_4", "child_3_4"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = ([[1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]
			+ [[1.0, -1.0, -1.0]] * entries
			+ [[1.0, -1.0]] * entries)
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)
		

	def test_constraint_chf_m_pf_LDRi(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_ldr_inactive()
		opt.vars_chf_m_pf_LDRi()

		# function to test
		opt.constraint_chf_m_pf_LDRi()

		entries = 3

		# rhs
		my_rhs = [0] * entries
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_chf_m_pf_LDRi_1_2", "constraint_chf_m_pf_LDRi_1_3", "constraint_chf_m_pf_LDRi_2_3"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["LDR_inactive_1_2", "chf_m_pf_LDRi_1_2"], ["LDR_inactive_1_3", "chf_m_pf_LDRi_1_3"],
			["LDR_inactive_2_3", "chf_m_pf_LDRi_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[-1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

	def test_constraint_ldr_inactive(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_ldr_active()
		opt.vars_ldr_inactive()
		
		# function to test
		opt.constraint_ldr_inactive()

		entries = 3

		# rhs
		my_rhs = [1] * entries
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_ldr_inactive_1_2", "constraint_ldr_inactive_1_3", "constraint_ldr_inactive_2_3"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["LDR_inactive_1_2", "LDR_active_1_2"], ["LDR_inactive_1_3", "LDR_active_1_3"],
			["LDR_inactive_2_3", "LDR_active_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0, 1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

	def test_constraint_chf_m_pf_LDRa(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_ldr_active()
		opt.vars_chf_m_pf_LDRa()

		# function to test
		opt.constraint_chf_m_pf_LDRa()

		entries = 3

		# rhs
		my_rhs = [0] * entries * 2
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * entries * 2
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_chf_m_pf_LDRa_lower_bound_1_2", "constraint_chf_m_pf_LDRa_lower_bound_1_3",
			"constraint_chf_m_pf_LDRa_lower_bound_2_3",
			"constraint_chf_m_pf_LDRa_upper_bound_1_2", "constraint_chf_m_pf_LDRa_upper_bound_1_3",
			"constraint_chf_m_pf_LDRa_upper_bound_2_3"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["LDR_active_1_2", "chf_m_pf_LDRa_1_2"], ["LDR_active_1_3", "chf_m_pf_LDRa_1_3"],
			["LDR_active_2_3", "chf_m_pf_LDRa_2_3"],
			["chf_m_pf_LDRa_1_2", "LDR_active_1_2"], ["chf_m_pf_LDRa_1_3", "LDR_active_1_3"],
			["chf_m_pf_LDRa_2_3", "LDR_active_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[0.00001, -1.0]] * entries + [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)


	def test_constraint_sibling_frequency(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_child()
		opt.vars_sibling_freq()

		# function to test
		opt.constraint_sibling_frequency()

		entries = 8

		# rhs
		my_rhs = [0] * (entries * 3) + [-2] * entries
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * (entries * 3) + ["G"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_sibling_0_1_2_frequency_child_circ", "constraint_sibling_0_1_3_frequency_child_circ",
			"constraint_sibling_0_2_1_frequency_child_circ", "constraint_sibling_0_2_3_frequency_child_circ",
			"constraint_sibling_0_3_1_frequency_child_circ", "constraint_sibling_0_3_2_frequency_child_circ",
			"constraint_sibling_1_2_3_frequency_child_circ", "constraint_sibling_1_3_2_frequency_child_circ",
			"constraint_sibling_0_1_2_frequency_child_bullet", "constraint_sibling_0_1_3_frequency_child_bullet",
			"constraint_sibling_0_2_1_frequency_child_bullet", "constraint_sibling_0_2_3_frequency_child_bullet",
			"constraint_sibling_0_3_1_frequency_child_bullet", "constraint_sibling_0_3_2_frequency_child_bullet",
			"constraint_sibling_1_2_3_frequency_child_bullet", "constraint_sibling_1_3_2_frequency_child_bullet",
			"constraint_sibling_0_1_2_frequency_phi", "constraint_sibling_0_1_3_frequency_phi",
			"constraint_sibling_0_2_1_frequency_phi", "constraint_sibling_0_2_3_frequency_phi",
			"constraint_sibling_0_3_1_frequency_phi", "constraint_sibling_0_3_2_frequency_phi",
			"constraint_sibling_1_2_3_frequency_phi", "constraint_sibling_1_3_2_frequency_phi",
			"constraint_sibling_0_1_2_frequency_ge_all", "constraint_sibling_0_1_3_frequency_ge_all",
			"constraint_sibling_0_2_1_frequency_ge_all", "constraint_sibling_0_2_3_frequency_ge_all",
			"constraint_sibling_0_3_1_frequency_ge_all", "constraint_sibling_0_3_2_frequency_ge_all",
			"constraint_sibling_1_2_3_frequency_ge_all", "constraint_sibling_1_3_2_frequency_ge_all",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		my_row = [
			["sibling_0_1_2_freq", "child_0_1"], ["sibling_0_1_3_freq", "child_0_1"],
			["sibling_0_2_1_freq", "child_0_2"], ["sibling_0_2_3_freq", "child_0_2"],
			["sibling_0_3_1_freq", "child_0_3"], ["sibling_0_3_2_freq", "child_0_3"],
			["sibling_1_2_3_freq", "child_1_2"], ["sibling_1_3_2_freq", "child_1_3"],

			["sibling_0_1_2_freq", "child_0_2"], ["sibling_0_1_3_freq", "child_0_3"],
			["sibling_0_2_1_freq", "child_0_1"], ["sibling_0_2_3_freq", "child_0_3"],
			["sibling_0_3_1_freq", "child_0_1"], ["sibling_0_3_2_freq", "child_0_2"],
			["sibling_1_2_3_freq", "child_1_3"], ["sibling_1_3_2_freq", "child_1_2"],

			["sibling_0_1_2_freq", "phi_2"], ["sibling_0_1_3_freq", "phi_3"],
			["sibling_0_2_1_freq", "phi_1"], ["sibling_0_2_3_freq", "phi_3"],
			["sibling_0_3_1_freq", "phi_1"], ["sibling_0_3_2_freq", "phi_2"],
			["sibling_1_2_3_freq", "phi_3"], ["sibling_1_3_2_freq", "phi_2"],

			["sibling_0_1_2_freq", "phi_2", "child_0_1", "child_0_2"], 
			["sibling_0_1_3_freq", "phi_3", "child_0_1", "child_0_3"],
			["sibling_0_2_1_freq", "phi_1", "child_0_2", "child_0_1"], 
			["sibling_0_2_3_freq", "phi_3", "child_0_2", "child_0_3"],
			["sibling_0_3_1_freq", "phi_1", "child_0_3", "child_0_1"], 
			["sibling_0_3_2_freq", "phi_2", "child_0_3", "child_0_2"],
			["sibling_1_2_3_freq", "phi_3", "child_1_2", "child_1_3"], 
			["sibling_1_3_2_freq", "phi_2", "child_1_3", "child_1_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0, -1.0]] * (entries * 3) + [[1.0, -1.0, -1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)


	def test_constraint_parent_frequency(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_child()
		opt.vars_parent_freq()

		# function to test
		opt.constraint_parent_frequency()

		entries = 6

		# rhs
		my_rhs = [0] * (entries + entries) + [-1] * entries
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * (entries + entries) + ["G"] * entries
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_parent_0_1_frequency_phi", "constraint_parent_0_2_frequency_phi",
			"constraint_parent_0_3_frequency_phi", "constraint_parent_1_2_frequency_phi",
			"constraint_parent_1_3_frequency_phi", "constraint_parent_2_3_frequency_phi",
			"constraint_parent_0_1_frequency_child", "constraint_parent_0_2_frequency_child",
			"constraint_parent_0_3_frequency_child", "constraint_parent_1_2_frequency_child",
			"constraint_parent_1_3_frequency_child", "constraint_parent_2_3_frequency_child",
			"constraint_parent_0_1_frequency_phi_child", "constraint_parent_0_2_frequency_phi_child",
			"constraint_parent_0_3_frequency_phi_child", "constraint_parent_1_2_frequency_phi_child",
			"constraint_parent_1_3_frequency_phi_child", "constraint_parent_2_3_frequency_phi_child"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		my_row = [
			["parent_0_1_freq", "phi_0"], ["parent_0_2_freq", "phi_0"], ["parent_0_3_freq", "phi_0"],
			["parent_1_2_freq", "phi_1"], ["parent_1_3_freq", "phi_1"], ["parent_2_3_freq", "phi_2"],
			["parent_0_1_freq", "child_0_1"], ["parent_0_2_freq", "child_0_2"], ["parent_0_3_freq", "child_0_3"],
			["parent_1_2_freq", "child_1_2"], ["parent_1_3_freq", "child_1_3"], ["parent_2_3_freq", "child_2_3"],
			["parent_0_1_freq", "phi_0", "child_0_1"], ["parent_0_2_freq", "phi_0", "child_0_2"],
			["parent_0_3_freq", "phi_0", "child_0_3"], ["parent_1_2_freq", "phi_1", "child_1_2"],
			["parent_1_3_freq", "phi_1", "child_1_3"], ["parent_2_3_freq", "phi_2", "child_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0, -1.0]] * (entries + entries) + [[1.0, -1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

	def test_constraint_child_frequency(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_child()
		opt.vars_child_freq()

		# function to test
		opt.constraint_child_frequency()

		entries1 = 6
		entries2 = 6
		entries3 = 6

		# rhs
		my_rhs = [0] * entries1 + [0] * entries2 + [-1] * entries3
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * entries1 + ["L"] * entries2 + ["G"] * entries3
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_child_0_1_freq_phi", "constraint_child_0_2_freq_phi", "constraint_child_0_3_freq_phi",
			"constraint_child_1_2_freq_phi", "constraint_child_1_3_freq_phi", "constraint_child_2_3_freq_phi",
			"constraint_child_freq_0_1_child", "constraint_child_freq_0_2_child", "constraint_child_freq_0_3_child",
			"constraint_child_freq_1_2_child", "constraint_child_freq_1_3_child", "constraint_child_freq_2_3_child",
			"constraint_child_freq_0_1_phi_child", "constraint_child_freq_0_2_phi_child",
			"constraint_child_freq_0_3_phi_child", "constraint_child_freq_1_2_phi_child",
			"constraint_child_freq_1_3_phi_child", "constraint_child_freq_2_3_phi_child"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		my_row = [
			["child_0_1_freq", "phi_1"], ["child_0_2_freq", "phi_2"], ["child_0_3_freq", "phi_3"],
			["child_1_2_freq", "phi_2"], ["child_1_3_freq", "phi_3"], ["child_2_3_freq", "phi_3"],
			["child_0_1_freq", "child_0_1"], ["child_0_2_freq", "child_0_2"],
			["child_0_3_freq", "child_0_3"], ["child_1_2_freq", "child_1_2"],
			["child_1_3_freq", "child_1_3"], ["child_2_3_freq", "child_2_3"],
			["child_0_1_freq", "phi_1", "child_0_1"], ["child_0_2_freq", "phi_2", "child_0_2"],
			["child_0_3_freq", "phi_3", "child_0_3"], ["child_1_2_freq", "phi_2", "child_1_2"],
			["child_1_3_freq", "phi_3", "child_1_3"], ["child_2_3_freq", "phi_3", "child_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = [[1.0, -1.0]] * entries1 + [[1.0, -1.0]] * entries2 + [[1.0, -1.0, -1.0]] * entries3
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

	def test_constraint_define_children(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_z()
		opt.vars_child()

		# function to test
		opt.constraint_define_children()

		entries1 = 6
		entries2 = 4
		entries3 = 6

		# rhs
		my_rhs = [0.0] * entries1 + [1.0] * entries2 + [0.0] * entries3
		self.assertListEqual(opt.my_rhs, my_rhs)

		# senses
		my_senses = ["L"] * (entries1 + entries2) + ["G"] * entries3
		self.assertListEqual(opt.my_sense, my_senses)

		# rownames
		rownames = [
			"constraint_child_ancestral_relation_0_1", "constraint_child_ancestral_relation_0_2",
			"constraint_child_ancestral_relation_0_3", "constraint_child_ancestral_relation_1_2",
			"constraint_child_ancestral_relation_1_3", "constraint_child_ancestral_relation_2_3",
			"constraint_child_0_2_other_parents_1", "constraint_child_0_3_other_parents_1",
			"constraint_child_0_3_other_parents_2", "constraint_child_1_3_other_parents_2",
			"constraint_child_ge_0_1", "constraint_child_ge_0_2", "constraint_child_ge_0_3",
			"constraint_child_ge_1_2", "constraint_child_ge_1_3", "constraint_child_ge_2_3"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# row
		my_row = [
			["child_0_1", "z_0_1"], ["child_0_2", "z_0_2"], ["child_0_3", "z_0_3"],
			["child_1_2", "z_1_2"], ["child_1_3", "z_1_3"], ["child_2_3", "z_2_3"],
			["child_0_2", "z_1_2"], ["child_0_3", "z_1_3"], ["child_0_3", "z_2_3"],
			["child_1_3", "z_2_3"],
			["child_0_1", "z_0_1"], ["child_0_2", "z_0_2", "z_1_2"],
			["child_0_3", "z_0_3", "z_1_3", "z_2_3"], ["child_1_2", "z_1_2"],
			["child_1_3", "z_1_3", "z_2_3"], ["child_2_3", "z_2_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)

		my_row = ([[1.0, -1.0]] * entries1 + [[1.0, 1.0]] * entries2 
			+ [[1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [1.0, -1.0],
			[1.0, -1.0, 1.0], [1.0, -1.0]])
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)

	def test_constraint_fix_single_avg_cn(self):
		# 2 segments
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num

		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		fixed_value = 2
		seg_index = 1
		opt.constraint_fix_single_avg_cn(fixed_value, seg_index)

		# rhs
		self.assertListEqual(opt.my_rhs, [0])

		# senses
		self.assertListEqual(opt.my_sense, ["E"])

		# rowname
		self.assertListEqual(opt.my_rownames, ["fixed_single_avg_cn_seg_1"])

		#row
		my_row = [["dc_a_p1_float_1_0", "dc_a_p1_float_1_1", "dc_a_p1_float_1_2",
			"dc_b_p1_float_1_0", "dc_b_p1_float_1_1", "dc_b_p1_float_1_2",
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1", "dc_a_m1_float_1_2",
			"dc_b_m1_float_1_0", "dc_b_m1_float_1_1", "dc_b_m1_float_1_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), my_row)
		
		my_row = [[1.0] * 6 + [-1.0] * 6]
		self.assertListEqual(self.get_opt_rows_values(opt), my_row)
		

	def test_constraint_fix_avg_cn(self):

		# more segments
		self.seg_num = 4
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num 
		seg = segment.Segment(1, 1, 1, 1, 1) 
		self.seg_list = [seg] * self.seg_num

		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test, all fixed
		fixed_values = [2, 3, 4, 5]
		unfixed_start = -1
		unfixed_stop = -1
		opt.constraint_fix_avg_cn(fixed_values, unfixed_start, unfixed_stop)
		
		# rhs
		self.assertListEqual(opt.my_rhs, [x - 2 for x in fixed_values])

		# senses
		senses = ["E"] * self.seg_num
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_avg_cn_0", "fixed_avg_cn_1", "fixed_avg_cn_2", "fixed_avg_cn_3"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2",
			"dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2",
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2",
			"dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"],
			["dc_a_p1_float_1_0", "dc_a_p1_float_1_1", "dc_a_p1_float_1_2",
			"dc_b_p1_float_1_0", "dc_b_p1_float_1_1", "dc_b_p1_float_1_2",
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1", "dc_a_m1_float_1_2",
			"dc_b_m1_float_1_0", "dc_b_m1_float_1_1", "dc_b_m1_float_1_2"],
			["dc_a_p1_float_2_0", "dc_a_p1_float_2_1", "dc_a_p1_float_2_2",
			"dc_b_p1_float_2_0", "dc_b_p1_float_2_1", "dc_b_p1_float_2_2",
			"dc_a_m1_float_2_0", "dc_a_m1_float_2_1", "dc_a_m1_float_2_2",
			"dc_b_m1_float_2_0", "dc_b_m1_float_2_1", "dc_b_m1_float_2_2"],
			["dc_a_p1_float_3_0", "dc_a_p1_float_3_1", "dc_a_p1_float_3_2",
			"dc_b_p1_float_3_0", "dc_b_p1_float_3_1", "dc_b_p1_float_3_2",
			"dc_a_m1_float_3_0", "dc_a_m1_float_3_1", "dc_a_m1_float_3_2",
			"dc_b_m1_float_3_0", "dc_b_m1_float_3_1", "dc_b_m1_float_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# function to test, middle unfixed
		fixed_values = [2, 5]
		unfixed_start = 1
		unfixed_stop = 2
		opt.empty_CPLEX_optimization_lists()
		opt.vars_aux_dc()
		opt.constraint_fix_avg_cn(fixed_values, unfixed_start, unfixed_stop)
		
		# rhs
		self.assertListEqual(opt.my_rhs, [x - 2 for x in fixed_values])

		# senses
		senses = ["E"] * len(fixed_values)
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_avg_cn_0", "fixed_avg_cn_3"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2",
			"dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2",
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2",
			"dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"],
			["dc_a_p1_float_3_0", "dc_a_p1_float_3_1", "dc_a_p1_float_3_2",
			"dc_b_p1_float_3_0", "dc_b_p1_float_3_1", "dc_b_p1_float_3_2",
			"dc_a_m1_float_3_0", "dc_a_m1_float_3_1", "dc_a_m1_float_3_2",
			"dc_b_m1_float_3_0", "dc_b_m1_float_3_1", "dc_b_m1_float_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# function to test, beginning ufixed
		fixed_values = [3, 4, 5]
		unfixed_start = 0
		unfixed_stop = 0
		opt.empty_CPLEX_optimization_lists()
		opt.vars_aux_dc()
		opt.constraint_fix_avg_cn(fixed_values, unfixed_start, unfixed_stop)
		
		# rhs
		self.assertListEqual(opt.my_rhs, [x - 2 for x in fixed_values])

		# senses
		senses = ["E"] * len(fixed_values)
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_avg_cn_1", "fixed_avg_cn_2", "fixed_avg_cn_3"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [
			["dc_a_p1_float_1_0", "dc_a_p1_float_1_1", "dc_a_p1_float_1_2",
			"dc_b_p1_float_1_0", "dc_b_p1_float_1_1", "dc_b_p1_float_1_2",
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1", "dc_a_m1_float_1_2",
			"dc_b_m1_float_1_0", "dc_b_m1_float_1_1", "dc_b_m1_float_1_2"],
			["dc_a_p1_float_2_0", "dc_a_p1_float_2_1", "dc_a_p1_float_2_2",
			"dc_b_p1_float_2_0", "dc_b_p1_float_2_1", "dc_b_p1_float_2_2",
			"dc_a_m1_float_2_0", "dc_a_m1_float_2_1", "dc_a_m1_float_2_2",
			"dc_b_m1_float_2_0", "dc_b_m1_float_2_1", "dc_b_m1_float_2_2"],
			["dc_a_p1_float_3_0", "dc_a_p1_float_3_1", "dc_a_p1_float_3_2",
			"dc_b_p1_float_3_0", "dc_b_p1_float_3_1", "dc_b_p1_float_3_2",
			"dc_a_m1_float_3_0", "dc_a_m1_float_3_1", "dc_a_m1_float_3_2",
			"dc_b_m1_float_3_0", "dc_b_m1_float_3_1", "dc_b_m1_float_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			[1.0] * 6 + [-1.0] * 6,
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# function to test, end unfixed
		fixed_values = [2]
		unfixed_start = 1
		unfixed_stop = 3
		opt.empty_CPLEX_optimization_lists()
		opt.vars_aux_dc()
		opt.constraint_fix_avg_cn(fixed_values, unfixed_start, unfixed_stop)
		
		# rhs
		self.assertListEqual(opt.my_rhs, [x - 2 for x in fixed_values])

		# senses
		senses = ["E"]
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_avg_cn_0"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2",
			"dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2",
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2",
			"dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)
	
		rows = [
			[1.0] * 6 + [-1.0] * 6
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_clustered_ssms_infl_cnv_same_lineage(self):
		sublin_num = 3
		self.ssm_num = 6
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_dssm_infl_cnv_same_lineage()

		ssm_indices_per_cluster = [[[3], []], [[1, 5], [0, 2, 4]], [[], []]]

		# function to test
		opt.constraint_clustered_ssms_infl_cnv_same_lineage(ssm_indices_per_cluster)

		entries = 3 * (sublin_num - 1) * cons.PHASE_NUMBER

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["constraint_clustered_ssms_infl_cnv_same_lineage_x"] * entries
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_a_p1_5_1"],
			["dssm_infl_cnv_same_lineage_a_p1_1_2", "dssm_infl_cnv_same_lineage_a_p1_5_2"],
			["dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_a_p1_2_1"],
			["dssm_infl_cnv_same_lineage_a_p1_0_2", "dssm_infl_cnv_same_lineage_a_p1_2_2"],
			["dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_a_p1_4_1"],
			["dssm_infl_cnv_same_lineage_a_p1_0_2", "dssm_infl_cnv_same_lineage_a_p1_4_2"],
			["dssm_infl_cnv_same_lineage_b_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_5_1"],
			["dssm_infl_cnv_same_lineage_b_p1_1_2", "dssm_infl_cnv_same_lineage_b_p1_5_2"],
			["dssm_infl_cnv_same_lineage_b_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_2_1"],
			["dssm_infl_cnv_same_lineage_b_p1_0_2", "dssm_infl_cnv_same_lineage_b_p1_2_2"],
			["dssm_infl_cnv_same_lineage_b_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_4_1"],
			["dssm_infl_cnv_same_lineage_b_p1_0_2", "dssm_infl_cnv_same_lineage_b_p1_4_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), rows)


	def test_constraint_clustered_ssms(self):
		sublin_num = 3
		self.ssm_num = 6
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()

		ssm_indices_per_cluster = [[[3], []], [[1, 5], [0, 2, 4]], [[], []]]

		# function to test
		opt.constraint_clustered_ssms(ssm_indices_per_cluster)

		entries = 3 * 3 * (sublin_num - 1)

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["constraint_clustered_ssms_x"] * entries
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_1_1", "dssm_5_1"],
			["dssm_1_2", "dssm_5_2"],
			["dssm_0_1", "dssm_2_1"],
			["dssm_0_2", "dssm_2_2"],
			["dssm_0_1", "dssm_4_1"],
			["dssm_0_2", "dssm_4_2"],
			["dssm_a_1_1", "dssm_a_5_1"],
			["dssm_a_1_2", "dssm_a_5_2"],
			["dssm_a_0_1", "dssm_a_2_1"],
			["dssm_a_0_2", "dssm_a_2_2"],
			["dssm_a_0_1", "dssm_a_4_1"],
			["dssm_a_0_2", "dssm_a_4_2"],
			["dssm_b_1_1", "dssm_b_5_1"],
			["dssm_b_1_2", "dssm_b_5_2"],
			["dssm_b_0_1", "dssm_b_2_1"],
			["dssm_b_0_2", "dssm_b_2_2"],
			["dssm_b_0_1", "dssm_b_4_1"],
			["dssm_b_0_2", "dssm_b_4_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# test for case with for non-simple CN changes and allele-specific
		sublin_num = 3
		self.ssm_num = 6
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		seg_spl_1 = log_pdf.find_piecewise_linear(10, 0, 5, 100, log_pdf.log_normal,
			2.3, 0.25)[0]
		self.seg_A_spl_list = [seg_spl_1, seg_spl_1, seg_spl_1] 
		self.seg_B_spl_list = [seg_spl_1, seg_spl_1, seg_spl_1] 
		seg_1 = segment.Segment_allele_specific(1, 1, 2, 2.3, 0.25, 1.3, 0.125)
		seg_list = [seg_1, seg_1, seg_1]

		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list,
			allele_specific=True, seg_splines_A=self.seg_A_spl_list, 
			seg_splines_B=self.seg_B_spl_list, simple_CN_changes=False)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, seg_list)
	
		opt.vars_three_ssm_matrices()

		ssm_indices_per_cluster = [[[3], []], [[1, 5], [0, 2, 4]], [[], []]]

		# function to test
		opt.constraint_clustered_ssms(ssm_indices_per_cluster)

		entries = 3 * 2 * (sublin_num - 1)

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, senses)

		# rows
		rows = [
			["dssm_a_1_1", "dssm_a_5_1"],
			["dssm_a_1_2", "dssm_a_5_2"],
			["dssm_a_0_1", "dssm_a_2_1"],
			["dssm_a_0_2", "dssm_a_2_2"],
			["dssm_a_0_1", "dssm_a_4_1"],
			["dssm_a_0_2", "dssm_a_4_2"],
			["dssm_b_1_1", "dssm_b_5_1"],
			["dssm_b_1_2", "dssm_b_5_2"],
			["dssm_b_0_1", "dssm_b_2_1"],
			["dssm_b_0_2", "dssm_b_2_2"],
			["dssm_b_0_1", "dssm_b_4_1"],
			["dssm_b_0_2", "dssm_b_4_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_lineage_divergence_z_fixed(self):
		
		sublin_num = 8

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_z()
		opt.vars_phi()

		# function to test
		direct_descendants_for_constraints = [[0, 1, 2], [2, 4, 5, 7]]
		opt.constraint_lineage_divergence_z_fixed(direct_descendants_for_constraints)

		# rhs
		rhs = [0.0, 0.0]
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["G", "G"]
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["lineage_divergence_z_fixed_phi_0", "lineage_divergence_z_fixed_phi_2"]
		self.assertListEqual(opt.my_rownames, rownames)
		
		# rows
		rows = [
			["phi_0", "phi_1", "phi_2"],
			["phi_2", "phi_4", "phi_5", "phi_7"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0, -1.0], [1.0, -1.0, -1.0, -1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_fix_phi(self):

		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_snp_matrices()
		opt.vars_phi()

		# function to test
		fixed_values = [0.7, 0.5]
		opt.constraint_fix_phi(fixed_values, -1, -1)

		# rhs 
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense 
		senses = ["E"] * (sublin_num - 1)
		self.assertListEqual(opt.my_sense, senses) 

		# rownames
		rownames = ["fixed_phi_1", "fixed_phi_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [["phi_1"], ["phi_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# only first fixed
		# function to test
		opt.empty_CPLEX_optimization_lists()
		opt.vars_phi()
		fixed_values = [0.5]
		opt.constraint_fix_phi(fixed_values, 1, 1)

		# rhs 
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense 
		senses = ["E"] * 1
		self.assertListEqual(opt.my_sense, senses) 

		# rownames
		rownames = ["fixed_phi_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [["phi_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# only second fixed
		# function to test
		opt.empty_CPLEX_optimization_lists()
		opt.vars_phi()
		fixed_values = [0.7]
		opt.constraint_fix_phi(fixed_values, 2, 2)

		# rhs 
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense 
		senses = ["E"] * 1
		self.assertListEqual(opt.my_sense, senses) 

		# rownames
		rownames = ["fixed_phi_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [["phi_1"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

 	def test_constraint_fix_z_matrix(self):

		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_snp_matrices()
		opt.vars_z()

		# function to test
		fixed_values = [1]
		opt.constraint_fix_z_matrix(fixed_values, -1, -1)

		# rhs 
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"]
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_z_1_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [["z_1_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)


		# now with bigger sublin size
		sublin_num = 4
		opt.empty_CPLEX_optimization_lists()
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_z()

		# function to test
		fixed_values = [
			1, 0,
			0
			]
		opt.constraint_fix_z_matrix(fixed_values, -1, -1)

		# rhs 
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 3
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["fixed_z_1_2", "fixed_z_1_3", "fixed_z_2_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [["z_1_2"],["z_1_3"], ["z_2_3"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)
		
		rows = [[1.0]] * 3
		self.assertListEqual(self.get_opt_rows_values(opt), rows)
		



	def test_constraint_fix_dssm(self):

		sublin_num = 2
		self.ssm_num = 4
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		# function to test, fix all
		fixed_values = [
			0, 0,
			0, 1,
			0, 0,
			0, 1,

			0, 1,
			0, 0,
			0, 0,
			0, 0,

			0, 0,
			0, 0,
			1, 0,
			0, 0
			]
		opt.constraint_fix_dssm(fixed_values, -1, -1)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * opt.all_delta_s_ssm_entries_num
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dssm_0_0", "fixed_dssm_0_1",
			"fixed_dssm_1_0", "fixed_dssm_1_1",
			"fixed_dssm_2_0", "fixed_dssm_2_1",
			"fixed_dssm_3_0", "fixed_dssm_3_1",
			"fixed_dssm_a_0_0", "fixed_dssm_a_0_1",
			"fixed_dssm_a_1_0", "fixed_dssm_a_1_1",
			"fixed_dssm_a_2_0", "fixed_dssm_a_2_1",
			"fixed_dssm_a_3_0", "fixed_dssm_a_3_1",
			"fixed_dssm_b_0_0", "fixed_dssm_b_0_1",
			"fixed_dssm_b_1_0", "fixed_dssm_b_1_1",
			"fixed_dssm_b_2_0", "fixed_dssm_b_2_1",
			"fixed_dssm_b_3_0", "fixed_dssm_b_3_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_0_0"], ["dssm_0_1"],
			["dssm_1_0"], ["dssm_1_1"],
			["dssm_2_0"], ["dssm_2_1"],
			["dssm_3_0"], ["dssm_3_1"],
			["dssm_a_0_0"], ["dssm_a_0_1"],
			["dssm_a_1_0"], ["dssm_a_1_1"],
			["dssm_a_2_0"], ["dssm_a_2_1"],
			["dssm_a_3_0"], ["dssm_a_3_1"],
			["dssm_b_0_0"], ["dssm_b_0_1"],
			["dssm_b_1_0"], ["dssm_b_1_1"],
			["dssm_b_2_0"], ["dssm_b_2_1"],
			["dssm_b_3_0"], ["dssm_b_3_1"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 24
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# now unfix only first two variables
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		fixed_values = [
			0, 0,
			0, 1,

			0, 0,
			0, 0,

			1, 0,
			0, 0
			]
		opt.constraint_fix_dssm(fixed_values, 0, 1)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 12
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dssm_2_0", "fixed_dssm_2_1",
			"fixed_dssm_3_0", "fixed_dssm_3_1",
			"fixed_dssm_a_2_0", "fixed_dssm_a_2_1",
			"fixed_dssm_a_3_0", "fixed_dssm_a_3_1",
			"fixed_dssm_b_2_0", "fixed_dssm_b_2_1",
			"fixed_dssm_b_3_0", "fixed_dssm_b_3_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_2_0"], ["dssm_2_1"],
			["dssm_3_0"], ["dssm_3_1"],
			["dssm_a_2_0"], ["dssm_a_2_1"], 
			["dssm_a_3_0"], ["dssm_a_3_1"], 
			["dssm_b_2_0"], ["dssm_b_2_1"], 
			["dssm_b_3_0"], ["dssm_b_3_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 12
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# now unfix only third variable
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		fixed_values = [
			0, 0,
			0, 1,
			0, 1,

			0, 1,
			0, 0,
			0, 0,

			0, 0,
			0, 0,
			0, 0
			]
		opt.constraint_fix_dssm(fixed_values, 2, 2)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 18
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dssm_0_0", "fixed_dssm_0_1",
			"fixed_dssm_1_0", "fixed_dssm_1_1",
			"fixed_dssm_3_0", "fixed_dssm_3_1",
			"fixed_dssm_a_0_0", "fixed_dssm_a_0_1",
			"fixed_dssm_a_1_0", "fixed_dssm_a_1_1",
			"fixed_dssm_a_3_0", "fixed_dssm_a_3_1",
			"fixed_dssm_b_0_0", "fixed_dssm_b_0_1",
			"fixed_dssm_b_1_0", "fixed_dssm_b_1_1",
			"fixed_dssm_b_3_0", "fixed_dssm_b_3_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_0_0"], ["dssm_0_1"], 
			["dssm_1_0"], ["dssm_1_1"], 
			["dssm_3_0"], ["dssm_3_1"], 
			["dssm_a_0_0"], ["dssm_a_0_1"], 
			["dssm_a_1_0"], ["dssm_a_1_1"], 
			["dssm_a_3_0"], ["dssm_a_3_1"], 
			["dssm_b_0_0"], ["dssm_b_0_1"], 
			["dssm_b_1_0"], ["dssm_b_1_1"], 
			["dssm_b_3_0"], ["dssm_b_3_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 18
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# now unfix only last variable
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		fixed_values = [
			0, 0,
			0, 1,
			0, 0,

			0, 1,
			0, 0,
			0, 0,

			0, 0,
			0, 0,
			1, 0,
			]
		opt.constraint_fix_dssm(fixed_values, 3, 3)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 18
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dssm_0_0", "fixed_dssm_0_1",
			"fixed_dssm_1_0", "fixed_dssm_1_1",
			"fixed_dssm_2_0", "fixed_dssm_2_1",
			"fixed_dssm_a_0_0", "fixed_dssm_a_0_1",
			"fixed_dssm_a_1_0", "fixed_dssm_a_1_1",
			"fixed_dssm_a_2_0", "fixed_dssm_a_2_1",
			"fixed_dssm_b_0_0", "fixed_dssm_b_0_1",
			"fixed_dssm_b_1_0", "fixed_dssm_b_1_1",
			"fixed_dssm_b_2_0", "fixed_dssm_b_2_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_0_0"], ["dssm_0_1"], 
			["dssm_1_0"], ["dssm_1_1"], 
			["dssm_2_0"], ["dssm_2_1"], 
			["dssm_a_0_0"], ["dssm_a_0_1"], 
			["dssm_a_1_0"], ["dssm_a_1_1"], 
			["dssm_a_2_0"], ["dssm_a_2_1"], 
			["dssm_b_0_0"], ["dssm_b_0_1"], 
			["dssm_b_1_0"], ["dssm_b_1_1"], 
			["dssm_b_2_0"], ["dssm_b_2_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 18
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# unfix all variables
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		fixed_values = []
		opt.constraint_fix_dssm(fixed_values, 0, 3)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 0
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = []
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = []
		self.assertListEqual(opt.my_rows, rows)

		# do with more complicated CN changes
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list, simple_CN_changes=False)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_snp_matrices()
		opt.vars_three_ssm_matrices()

		fixed_values = [
			1, 0,
			0, 0,
			0, 0,
			0, 0,

			0, 0,
			0, 1,
			0, 0,
			0, 0,

			0, 0,
			0, 0,
			1, 0,
			0, 1
			]
		opt.constraint_fix_dssm(fixed_values, -1, -1)

		# rhs
		rhs = [
			1, 0,
			0, 1,
			0, 0,
			0, 0,

			0, 0,
			0, 0,
			1, 0,
			0, 1
			]
		self.assertListEqual(opt.my_rhs, rhs)

		# sense
		senses = ["E"] * 16
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dssm_a_0_0", "fixed_dssm_a_0_1",
			"fixed_dssm_a_1_0", "fixed_dssm_a_1_1",
			"fixed_dssm_a_2_0", "fixed_dssm_a_2_1",
			"fixed_dssm_a_3_0", "fixed_dssm_a_3_1",
			"fixed_dssm_b_0_0", "fixed_dssm_b_0_1",
			"fixed_dssm_b_1_0", "fixed_dssm_b_1_1",
			"fixed_dssm_b_2_0", "fixed_dssm_b_2_1",
			"fixed_dssm_b_3_0", "fixed_dssm_b_3_1",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_a_0_0"], ["dssm_a_0_1"], 
			["dssm_a_1_0"], ["dssm_a_1_1"], 
			["dssm_a_2_0"], ["dssm_a_2_1"], 
			["dssm_a_3_0"], ["dssm_a_3_1"], 
			["dssm_b_0_0"], ["dssm_b_0_1"], 
			["dssm_b_1_0"], ["dssm_b_1_1"], 
			["dssm_b_2_0"], ["dssm_b_2_1"], 
			["dssm_b_3_0"], ["dssm_b_3_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 16
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_fix_dsnp(self):

		sublin_num = 2
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_three_snp_matrices()

		# function to test
		fixed_values = [
			0,
			1,
			0,
			0,
			1,
			0
			]
		with self.assertRaises(TypeError):
			opt.constraint_fix_dsnp(fixed_values, -1, -1)

		## rhs
		#rhs = fixed_values
		#self.assertListEqual(opt.my_rhs, rhs)

		## sense
		#senses = ["E"] * self.snp_num * 3
		#self.assertListEqual(opt.my_sense, senses)

		## rownames
		#rownames = [
		#	"fixed_dsnp_0", "fixed_dsnp_1",
		#	"fixed_dsnp_a_0", "fixed_dsnp_a_1",
		#	"fixed_dsnp_b_0", "fixed_dsnp_b_1"
		#	]
		#self.assertListEqual(opt.my_rownames, rownames)

		## rows
		#rows = [
		#	[["dsnp_0"], [1.0]], [["dsnp_1"], [1.0]],
		#	[["dsnp_a_0"], [1.0]],[["dsnp_a_1"], [1.0]], 
		#	[["dsnp_b_0"], [1.0]], [["dsnp_b_1"], [1.0]]
		#	]
		#self.assertListEqual(opt.my_rows, rows)

		## now unfix first SNP
		#opt.empty_CPLEX_optimization_lists()
		#opt.vars_three_snp_matrices()
		#opt.vars_three_ssm_matrices()

		#fixed_values = [
		#	1,
		#	0,
		#	0
		#	]
		#opt.constraint_fix_dsnp(fixed_values, 0, 0)

		## rhs
		#rhs = fixed_values
		#self.assertListEqual(opt.my_rhs, rhs)

		## sense
		#senses = ["E"] * 3
		#self.assertListEqual(opt.my_sense, senses)

		## rownames
		#rownames = [
		#	"fixed_dsnp_1",
		#	"fixed_dsnp_a_1",
		#	"fixed_dsnp_b_1"
		#	]
		#self.assertListEqual(opt.my_rownames, rownames)

		## rows
		#rows = [
		#	[["dsnp_1"], [1.0]],
		#	[["dsnp_a_1"], [1.0]], 
		#	[["dsnp_b_1"], [1.0]]
		#	]
		#self.assertListEqual(opt.my_rows, rows)

	def test_constraint_fix_dc_binary(self):

		# fix all segments via range
		sublin_num = 2
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices
		opt.vars_aux_dc()

		# function to test
		fixed_values = [
			0, 1,
			0, 0,

			0, 0,
			0, 1,

			0, 0,
			0, 0,

			0, 0,
			0, 0
			]
		opt.constraint_fix_dc_binary(fixed_values, -1, -1)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * sublin_num * self.seg_num * cons.PHASE_NUMBER * opt.cnv_state_num

		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dc_a_p1_binary_0_0", "fixed_dc_a_p1_binary_0_1",
			"fixed_dc_a_p1_binary_1_0", "fixed_dc_a_p1_binary_1_1",
			"fixed_dc_b_p1_binary_0_0", "fixed_dc_b_p1_binary_0_1",
			"fixed_dc_b_p1_binary_1_0", "fixed_dc_b_p1_binary_1_1",
			"fixed_dc_a_m1_binary_0_0", "fixed_dc_a_m1_binary_0_1",
			"fixed_dc_a_m1_binary_1_0", "fixed_dc_a_m1_binary_1_1",
			"fixed_dc_b_m1_binary_0_0", "fixed_dc_b_m1_binary_0_1",
			"fixed_dc_b_m1_binary_1_0", "fixed_dc_b_m1_binary_1_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dc_a_p1_binary_0_0"], ["dc_a_p1_binary_0_1"], 
			["dc_a_p1_binary_1_0"], ["dc_a_p1_binary_1_1"], 
			["dc_b_p1_binary_0_0"], ["dc_b_p1_binary_0_1"], 
			["dc_b_p1_binary_1_0"], ["dc_b_p1_binary_1_1"], 
			["dc_a_m1_binary_0_0"], ["dc_a_m1_binary_0_1"], 
			["dc_a_m1_binary_1_0"], ["dc_a_m1_binary_1_1"], 
			["dc_b_m1_binary_0_0"], ["dc_b_m1_binary_0_1"], 
			["dc_b_m1_binary_1_0"], ["dc_b_m1_binary_1_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 16
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# fix all segments via specific indices
		sublin_num = 2
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices
		opt.vars_aux_dc()

		# function to test
		fixed_values = [
			0, 1,
			0, 0,

			0, 0,
			0, 1,

			0, 0,
			0, 0,

			0, 0,
			0, 0
			]
		opt.constraint_fix_dc_binary(fixed_values, -1, -1, [0,1])

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * sublin_num * self.seg_num * cons.PHASE_NUMBER * opt.cnv_state_num

		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dc_a_p1_binary_0_0", "fixed_dc_a_p1_binary_0_1",
			"fixed_dc_a_p1_binary_1_0", "fixed_dc_a_p1_binary_1_1",
			"fixed_dc_b_p1_binary_0_0", "fixed_dc_b_p1_binary_0_1",
			"fixed_dc_b_p1_binary_1_0", "fixed_dc_b_p1_binary_1_1",
			"fixed_dc_a_m1_binary_0_0", "fixed_dc_a_m1_binary_0_1",
			"fixed_dc_a_m1_binary_1_0", "fixed_dc_a_m1_binary_1_1",
			"fixed_dc_b_m1_binary_0_0", "fixed_dc_b_m1_binary_0_1",
			"fixed_dc_b_m1_binary_1_0", "fixed_dc_b_m1_binary_1_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dc_a_p1_binary_0_0"], ["dc_a_p1_binary_0_1"], 
			["dc_a_p1_binary_1_0"], ["dc_a_p1_binary_1_1"], 
			["dc_b_p1_binary_0_0"], ["dc_b_p1_binary_0_1"], 
			["dc_b_p1_binary_1_0"], ["dc_b_p1_binary_1_1"], 
			["dc_a_m1_binary_0_0"], ["dc_a_m1_binary_0_1"], 
			["dc_a_m1_binary_1_0"], ["dc_a_m1_binary_1_1"], 
			["dc_b_m1_binary_0_0"], ["dc_b_m1_binary_0_1"], 
			["dc_b_m1_binary_1_0"], ["dc_b_m1_binary_1_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 16
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# unfix last variable via range
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_aux_dc()
		
		fixed_values = [
			0, 1,

			0, 0,

			0, 0,

			0, 0
			]
		opt.constraint_fix_dc_binary(fixed_values, 1, 1)

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * 8

		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dc_a_p1_binary_0_0", "fixed_dc_a_p1_binary_0_1",
			"fixed_dc_b_p1_binary_0_0", "fixed_dc_b_p1_binary_0_1",
			"fixed_dc_a_m1_binary_0_0", "fixed_dc_a_m1_binary_0_1",
			"fixed_dc_b_m1_binary_0_0", "fixed_dc_b_m1_binary_0_1",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dc_a_p1_binary_0_0"], ["dc_a_p1_binary_0_1"],
			["dc_b_p1_binary_0_0"], ["dc_b_p1_binary_0_1"],
			["dc_a_m1_binary_0_0"], ["dc_a_m1_binary_0_1"],
			["dc_b_m1_binary_0_0"], ["dc_b_m1_binary_0_1"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 8
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# unfix last variable via index
		opt.empty_CPLEX_optimization_lists()
		opt.vars_three_snp_matrices()
		opt.vars_aux_dc()
		
		fixed_values = [
			0, 1,

			0, 0,

			0, 0,

			0, 0
			]
		opt.constraint_fix_dc_binary(fixed_values, -1, -1, [0])

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * 8

		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dc_a_p1_binary_0_0", "fixed_dc_a_p1_binary_0_1",
			"fixed_dc_b_p1_binary_0_0", "fixed_dc_b_p1_binary_0_1",
			"fixed_dc_a_m1_binary_0_0", "fixed_dc_a_m1_binary_0_1",
			"fixed_dc_b_m1_binary_0_0", "fixed_dc_b_m1_binary_0_1",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dc_a_p1_binary_0_0"], ["dc_a_p1_binary_0_1"],
			["dc_b_p1_binary_0_0"], ["dc_b_p1_binary_0_1"],
			["dc_a_m1_binary_0_0"], ["dc_a_m1_binary_0_1"],
			["dc_b_m1_binary_0_0"], ["dc_b_m1_binary_0_1"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 8
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# fix some segments via specific indices
		sublin_num = 3
		self.seg_num = 20
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices
		opt.vars_aux_dc()

		# function to test
		fixed_values = [
			0, 1, 0,
			0, 0, 0,
			0, 0, 0,

			0, 0, 0,
			0, 1, 0,
			0, 1, 0,

			0, 0, 0,
			0, 0, 0,
			0, 0, 0,

			0, 0, 0,
			0, 0, 0,
			0, 0, 0
			]
		opt.constraint_fix_dc_binary(fixed_values, -1, -1, [3,6,12])

		# rhs
		rhs = fixed_values
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * sublin_num * 3 * cons.PHASE_NUMBER * opt.cnv_state_num
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"fixed_dc_a_p1_binary_3_0", "fixed_dc_a_p1_binary_3_1", "fixed_dc_a_p1_binary_3_2",
			"fixed_dc_a_p1_binary_6_0", "fixed_dc_a_p1_binary_6_1", "fixed_dc_a_p1_binary_6_2",
			"fixed_dc_a_p1_binary_12_0", "fixed_dc_a_p1_binary_12_1", "fixed_dc_a_p1_binary_12_2",
			"fixed_dc_b_p1_binary_3_0", "fixed_dc_b_p1_binary_3_1", "fixed_dc_b_p1_binary_3_2",
			"fixed_dc_b_p1_binary_6_0", "fixed_dc_b_p1_binary_6_1", "fixed_dc_b_p1_binary_6_2",
			"fixed_dc_b_p1_binary_12_0", "fixed_dc_b_p1_binary_12_1", "fixed_dc_b_p1_binary_12_2",
			"fixed_dc_a_m1_binary_3_0", "fixed_dc_a_m1_binary_3_1", "fixed_dc_a_m1_binary_3_2",
			"fixed_dc_a_m1_binary_6_0", "fixed_dc_a_m1_binary_6_1", "fixed_dc_a_m1_binary_6_2",
			"fixed_dc_a_m1_binary_12_0", "fixed_dc_a_m1_binary_12_1", "fixed_dc_a_m1_binary_12_2",
			"fixed_dc_b_m1_binary_3_0", "fixed_dc_b_m1_binary_3_1", "fixed_dc_b_m1_binary_3_2",
			"fixed_dc_b_m1_binary_6_0", "fixed_dc_b_m1_binary_6_1", "fixed_dc_b_m1_binary_6_2",
			"fixed_dc_b_m1_binary_12_0", "fixed_dc_b_m1_binary_12_1", "fixed_dc_b_m1_binary_12_2"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dc_a_p1_binary_3_0"], ["dc_a_p1_binary_3_1"], ["dc_a_p1_binary_3_2"],
			["dc_a_p1_binary_6_0"], ["dc_a_p1_binary_6_1"], ["dc_a_p1_binary_6_2"],
			["dc_a_p1_binary_12_0"], ["dc_a_p1_binary_12_1"], ["dc_a_p1_binary_12_2"],
			["dc_b_p1_binary_3_0"], ["dc_b_p1_binary_3_1"], ["dc_b_p1_binary_3_2"],
			["dc_b_p1_binary_6_0"], ["dc_b_p1_binary_6_1"], ["dc_b_p1_binary_6_2"],
			["dc_b_p1_binary_12_0"], ["dc_b_p1_binary_12_1"], ["dc_b_p1_binary_12_2"],
			["dc_a_m1_binary_3_0"], ["dc_a_m1_binary_3_1"], ["dc_a_m1_binary_3_2"],
			["dc_a_m1_binary_6_0"], ["dc_a_m1_binary_6_1"], ["dc_a_m1_binary_6_2"],
			["dc_a_m1_binary_12_0"], ["dc_a_m1_binary_12_1"], ["dc_a_m1_binary_12_2"],
			["dc_b_m1_binary_3_0"], ["dc_b_m1_binary_3_1"], ["dc_b_m1_binary_3_2"],
			["dc_b_m1_binary_6_0"], ["dc_b_m1_binary_6_1"], ["dc_b_m1_binary_6_2"],
			["dc_b_m1_binary_12_0"], ["dc_b_m1_binary_12_1"], ["dc_b_m1_binary_12_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 36
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_start_indices(self):
		
		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_three_snp_matrices()
		opt.vars_z()
		opt.vars_aux_dc()
		opt.vars_phi()

		# test start indices of variables in my_colnames
		self.assertEqual(opt.dssm_start_index, 0)
		self.assertEqual(opt.dsnp_start_index, sublin_num * self.ssm_num * 3)
		self.assertEqual(opt.z_index_start, opt.dsnp_start_index + self.snp_num * 3)
		self.assertEqual(opt.dc_binary_index_start_p1, opt.z_index_start + sublin_num * sublin_num)
		self.assertEqual(opt.phi_start_index, opt.dc_binary_index_start_p1 + self.seg_num * sublin_num * 
			opt.aux_matrices_cnv_linear_types_num * cons.PHASE_NUMBER * opt.cnv_state_num)

	def test_constraint_ssm_value_spline_frequency(self):

		# set up variables
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1, self.snp_spl_1]
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		self.ssm_list = [ssm1, ssm2]
		self.seg_num =2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_vars_mut_splines()
		opt.vars_ssm_aux_1_cn()
		opt.vars_ssm_aux_15_cn()
		opt.vars_ssm_aux_2_cn()

		# function to test
		opt.constraint_ssm_value_spline_frequency(self.ssm_list)

		# rhs
		rhs = [0.0] * self.ssm_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * self.ssm_num
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["ssm_frequency_0", "ssm_frequency_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		value_ssm1 = [opt.ssm_splines[0].get_knots()[i] * opt.cn[ssm1.seg_index] 
			for i in xrange(len(opt.ssm_splines[0].get_knots()))] + [-1.0] * 25 + [1.0] * 18
		value_ssm2 = [opt.ssm_splines[1].get_knots()[i] * opt.cn[ssm2.seg_index] 
			for i in xrange(len(opt.ssm_splines[1].get_knots()))] + [-1.0] * 25 + [1.0] * 18
		rows = [
			opt.my_colnames_ssm[0] + [opt.my_colnames_dssm_aux_1_cn[0][0],
			opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_colnames_dssm_aux_1_cn[0][2],
			opt.my_colnames_dssm_aux_15_cn_a_p1[0][0],
			opt.my_colnames_dssm_aux_15_cn_a_p1[0][1],
			opt.my_colnames_dssm_aux_15_cn_b_p1[0][0],
			opt.my_colnames_dssm_aux_15_cn_b_p1[0][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][0][0], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][0][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][0][2], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][0],
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][1], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2],
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][2][0], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][2][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[0][2][2],
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][0][0], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][0][1],
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][0][2], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][0],
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][1], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2],
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][2][0], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][2][1],
			opt.my_colnames_dssm_aux_2_cn_b_p1[0][2][2],
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][0][0], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][0][1],
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][0][2], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][0],
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][1], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2],
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][2][0], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][2][1],
			opt.my_colnames_dssm_aux_2_cn_a_m1[0][2][2],
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][0][0], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][0][1],
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][0][2], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][0],
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][1], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2],
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][2][0], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][2][1],
			opt.my_colnames_dssm_aux_2_cn_b_m1[0][2][2]],
			opt.my_colnames_ssm[1] + [opt.my_colnames_dssm_aux_1_cn[1][0],
			opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_colnames_dssm_aux_1_cn[1][2],
			opt.my_colnames_dssm_aux_15_cn_a_p1[1][0],
			opt.my_colnames_dssm_aux_15_cn_a_p1[1][1],
			opt.my_colnames_dssm_aux_15_cn_b_p1[1][0],
			opt.my_colnames_dssm_aux_15_cn_b_p1[1][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][0][0], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][0][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][0][2], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][0],
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][1], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2],
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][2][0], 
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][2][1],
			opt.my_colnames_dssm_aux_2_cn_a_p1[1][2][2],
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][0][0], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][0][1],
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][0][2], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][0],
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][1], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2],
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][2][0], 
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][2][1],
			opt.my_colnames_dssm_aux_2_cn_b_p1[1][2][2],
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][0][0], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][0][1],
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][0][2], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][0],
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][1], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2],
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][2][0], 
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][2][1],
			opt.my_colnames_dssm_aux_2_cn_a_m1[1][2][2],
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][0][0], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][0][1],
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][0][2], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][0],
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][1], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2],
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][2][0], 
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][2][1],
			opt.my_colnames_dssm_aux_2_cn_b_m1[1][2][2]
			]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [value_ssm1, value_ssm2]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)


	def test_constraint_ssm_aux_2_cn(self):

		# set up variables
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		self.ssm_list = [ssm1, ssm2]
		self.seg_num =2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_phi() 
		opt.vars_aux_dc()
		opt.vars_ssm_aux_2_cn()
		opt.vars_z()

		# function to test
		opt.constraint_ssm_aux_2_cn(self.ssm_list)

		entries = opt.ssm_num * 2

		# rhs
		rhs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
			-3.0, -3.0, -3.0, -3.0] * 2 + [0.0] * 64
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", 
			"G", "G", "G", "G"] * 2 + ["E"] * 64
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["ssm_aux_2_cn_a_p1_phi_0_1_2", "ssm_aux_2_cn_b_p1_phi_0_1_2",
			"ssm_aux_2_cn_a_m1_phi_0_1_2", "ssm_aux_2_cn_b_m1_phi_0_1_2",
			"ssm_aux_2_cn_a_p1_z_0_1_2", "ssm_aux_2_cn_b_p1_z_0_1_2",
			"ssm_aux_2_cn_a_m1_z_0_1_2", "ssm_aux_2_cn_b_m1_z_0_1_2",
			"ssm_aux_2_cn_a_p1_dssm_0_1_2", "ssm_aux_2_cn_b_p1_dssm_0_1_2",
			"ssm_aux_2_cn_a_m1_dssm_0_1_2", "ssm_aux_2_cn_b_m1_dssm_0_1_2",
			"ssm_aux_2_cn_a_dc_p1_binary_0_1_2", "ssm_aux_2_cn_b_dc_p1_binary_0_1_2",
			"ssm_aux_2_cn_a_dc_m1_binary_0_1_2", "ssm_aux_2_cn_b_dc_m1_binary_0_1_2",
			"ssm_aux_2_cn_a_p1_three_0_1_2", "ssm_aux_2_cn_b_p1_three_0_1_2",
			"ssm_aux_2_cn_a_m1_three_0_1_2", "ssm_aux_2_cn_b_m1_three_0_1_2",
			"ssm_aux_2_cn_a_p1_phi_1_1_2", "ssm_aux_2_cn_b_p1_phi_1_1_2",
			"ssm_aux_2_cn_a_m1_phi_1_1_2", "ssm_aux_2_cn_b_m1_phi_1_1_2",
			"ssm_aux_2_cn_a_p1_z_1_1_2", "ssm_aux_2_cn_b_p1_z_1_1_2",
			"ssm_aux_2_cn_a_m1_z_1_1_2", "ssm_aux_2_cn_b_m1_z_1_1_2",
			"ssm_aux_2_cn_a_p1_dssm_1_1_2", "ssm_aux_2_cn_b_p1_dssm_1_1_2",
			"ssm_aux_2_cn_a_m1_dssm_1_1_2", "ssm_aux_2_cn_b_m1_dssm_1_1_2",
			"ssm_aux_2_cn_a_dc_p1_binary_1_1_2", "ssm_aux_2_cn_b_dc_p1_binary_1_1_2",
			"ssm_aux_2_cn_a_dc_m1_binary_1_1_2", "ssm_aux_2_cn_b_dc_m1_binary_1_1_2",
			"ssm_aux_2_cn_a_p1_three_1_1_2", "ssm_aux_2_cn_b_p1_three_1_1_2",
			"ssm_aux_2_cn_a_m1_three_1_1_2", "ssm_aux_2_cn_b_m1_three_1_1_2"]
		rownames_2 = []
		[self.rowname_ssm_aux_2_cn_a_valid_variables(i, j, k, rownames_2) for i in range(self.ssm_num)
			for j in range(sublin_num) for k in range(sublin_num)]
		rownames.extend(rownames_2)
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			[opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2], opt.my_colnames_dssm_a[0][1]],
			[opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2], opt.my_colnames_dssm_b[0][1]],
			[opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2], opt.my_colnames_dssm_a[0][1]],
			[opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2], opt.my_colnames_dssm_b[0][1]],
			[opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2], opt.my_colnames_dc_a_p1_binary[ssm1.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2], opt.my_colnames_dc_b_p1_binary[ssm1.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2], opt.my_colnames_dc_a_m1_binary[ssm1.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2], opt.my_colnames_dc_b_m1_binary[ssm1.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_p1[0][1][2], opt.my_colnames_dssm_a[0][1], 
			opt.my_colnames_dc_a_p1_binary[ssm1.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[0][1][2], opt.my_colnames_dssm_b[0][1], 
			opt.my_colnames_dc_b_p1_binary[ssm1.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[0][1][2], opt.my_colnames_dssm_a[0][1], 
			opt.my_colnames_dc_a_m1_binary[ssm1.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[0][1][2], opt.my_colnames_dssm_b[0][1], 
			opt.my_colnames_dc_b_m1_binary[ssm1.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2], opt.my_phis[2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2], opt.my_colnames_z[1][2]],
			[opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2], opt.my_colnames_dssm_a[1][1]],
			[opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2], opt.my_colnames_dssm_b[1][1]],
			[opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2], opt.my_colnames_dssm_a[1][1]],
			[opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2], opt.my_colnames_dssm_b[1][1]],
			[opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2], opt.my_colnames_dc_a_p1_binary[ssm2.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2], opt.my_colnames_dc_b_p1_binary[ssm2.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2], opt.my_colnames_dc_a_m1_binary[ssm2.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2], opt.my_colnames_dc_b_m1_binary[ssm2.seg_index][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_p1[1][1][2], opt.my_colnames_dssm_a[1][1], 
			opt.my_colnames_dc_a_p1_binary[ssm2.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_p1[1][1][2], opt.my_colnames_dssm_b[1][1], 
			opt.my_colnames_dc_b_p1_binary[ssm2.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_a_m1[1][1][2], opt.my_colnames_dssm_a[1][1], 
			opt.my_colnames_dc_a_m1_binary[ssm2.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			[opt.my_colnames_dssm_aux_2_cn_b_m1[1][1][2], opt.my_colnames_dssm_b[1][1], 
			opt.my_colnames_dc_b_m1_binary[ssm2.seg_index][2], opt.my_phis[2], opt.my_colnames_z[1][2]], 
			]
		rows_2 = []
		[self.rows_ssm_aux_2_cn_a_valid_variables_vars(i, j, k, rows_2, opt) for i in range(self.ssm_num)
			for j in range(sublin_num) for k in range(sublin_num)]
		rows.extend(rows_2)
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			]
		rows_2 = []
		[self.rows_ssm_aux_2_cn_a_valid_variables_values(i, j, k, rows_2, opt) for i in range(self.ssm_num)
			for j in range(sublin_num) for k in range(sublin_num)]
		rows.extend(rows_2)
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def rows_ssm_aux_2_cn_a_valid_variables_vars(self, i, j, k, my_list, opt):
		if (j == 0 or j >= k):
			my_list.extend([[opt.my_colnames_dssm_aux_2_cn_a_p1[i][j][k]], 
				[opt.my_colnames_dssm_aux_2_cn_b_p1[i][j][k]],
				[opt.my_colnames_dssm_aux_2_cn_a_m1[i][j][k]],
				[opt.my_colnames_dssm_aux_2_cn_b_m1[i][j][k]]])

	def rows_ssm_aux_2_cn_a_valid_variables_values(self, i, j, k, my_list, opt):
		if (j == 0 or j >= k):
			my_list.extend([[1.0]] * 4)

	def rowname_ssm_aux_2_cn_a_valid_variables(self, i, j, k, my_list):
		if (j == 0 or j >= k):
			my_list.extend(["ssm_aux_2_cn_a_p1_valid_variables_{0}_{1}_{2}".format(i, j, k), 
				"ssm_aux_2_cn_b_p1_valid_variables_{0}_{1}_{2}".format(i, j, k),
				"ssm_aux_2_cn_a_m1_valid_variables_{0}_{1}_{2}".format(i, j, k),
				"ssm_aux_2_cn_b_m1_valid_variables_{0}_{1}_{2}".format(i, j, k)])

	def test_constraint_ssm_aux_15_cn_phi(self):
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, 
			self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_phi()
		opt.vars_dssm_infl_cnv_same_lineage()
		opt.vars_ssm_aux_15_cn()

		# function to test
		opt.constraint_ssm_aux_15_cn()

		entries = 8

		# rhs 
		rhs = [0.0] * entries * 2 + [-1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries * 2 + ["G"] * entries

		# rownames
		rownames = ["ssm_aux_15_cn_phi_a_p1_0_1", "ssm_aux_15_cn_phi_a_p1_0_2",
			"ssm_aux_15_cn_phi_a_p1_1_1", "ssm_aux_15_cn_phi_a_p1_1_2",
			"ssm_aux_15_cn_phi_b_p1_0_1", "ssm_aux_15_cn_phi_b_p1_0_2",
			"ssm_aux_15_cn_phi_b_p1_1_1", "ssm_aux_15_cn_phi_b_p1_1_2",
			"ssm_aux_15_cn_dssm_infl_a_p1_0_1", "ssm_aux_15_cn_dssm_infl_a_p1_0_2",
			"ssm_aux_15_cn_dssm_infl_a_p1_1_1", "ssm_aux_15_cn_dssm_infl_a_p1_1_2",
			"ssm_aux_15_cn_dssm_infl_b_p1_0_1", "ssm_aux_15_cn_dssm_infl_b_p1_0_2",
			"ssm_aux_15_cn_dssm_infl_b_p1_1_1", "ssm_aux_15_cn_dssm_infl_b_p1_1_2",
			"ssm_aux_15_cn_phi_dssm_infl_a_p1_0_1", "ssm_aux_15_cn_phi_dssm_infl_a_p1_0_2",
			"ssm_aux_15_cn_phi_dssm_infl_a_p1_1_1", "ssm_aux_15_cn_phi_dssm_infl_a_p1_1_2",
			"ssm_aux_15_cn_phi_dssm_infl_b_p1_0_1", "ssm_aux_15_cn_phi_dssm_infl_b_p1_0_2",
			"ssm_aux_15_cn_phi_dssm_infl_b_p1_1_1", "ssm_aux_15_cn_phi_dssm_infl_b_p1_1_2",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_aux_15_cn_a_p1_0_1", "phi_1"], ["dssm_aux_15_cn_a_p1_0_2", "phi_2"],
			["dssm_aux_15_cn_a_p1_1_1", "phi_1"], ["dssm_aux_15_cn_a_p1_1_2", "phi_2"],
			["dssm_aux_15_cn_b_p1_0_1", "phi_1"], ["dssm_aux_15_cn_b_p1_0_2", "phi_2"],
			["dssm_aux_15_cn_b_p1_1_1", "phi_1"], ["dssm_aux_15_cn_b_p1_1_2", "phi_2"],
			["dssm_aux_15_cn_a_p1_0_1", "dssm_infl_cnv_same_lineage_a_p1_0_1"], 
			["dssm_aux_15_cn_a_p1_0_2", "dssm_infl_cnv_same_lineage_a_p1_0_2"],
			["dssm_aux_15_cn_a_p1_1_1", "dssm_infl_cnv_same_lineage_a_p1_1_1"], 
			["dssm_aux_15_cn_a_p1_1_2", "dssm_infl_cnv_same_lineage_a_p1_1_2"],
			["dssm_aux_15_cn_b_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"], 
			["dssm_aux_15_cn_b_p1_0_2", "dssm_infl_cnv_same_lineage_b_p1_0_2"],
			["dssm_aux_15_cn_b_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"], 
			["dssm_aux_15_cn_b_p1_1_2", "dssm_infl_cnv_same_lineage_b_p1_1_2"],
			["dssm_aux_15_cn_a_p1_0_1", "phi_1", "dssm_infl_cnv_same_lineage_a_p1_0_1"], 
			["dssm_aux_15_cn_a_p1_0_2", "phi_2", "dssm_infl_cnv_same_lineage_a_p1_0_2"],
			["dssm_aux_15_cn_a_p1_1_1", "phi_1", "dssm_infl_cnv_same_lineage_a_p1_1_1"], 
			["dssm_aux_15_cn_a_p1_1_2", "phi_2", "dssm_infl_cnv_same_lineage_a_p1_1_2"],
			["dssm_aux_15_cn_b_p1_0_1", "phi_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"], 
			["dssm_aux_15_cn_b_p1_0_2", "phi_2", "dssm_infl_cnv_same_lineage_b_p1_0_2"],
			["dssm_aux_15_cn_b_p1_1_1", "phi_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"], 
			["dssm_aux_15_cn_b_p1_1_2", "phi_2", "dssm_infl_cnv_same_lineage_b_p1_1_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0]] * entries * 2 + [[1.0, -1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		

	def test_constraint_ssm_aux_1_cn(self):

		# only simple CN changes are allowed
		# set up variables
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()
		opt.vars_phi()
		opt.vars_ssm_aux_1_cn()

		# function to test
		opt.constraint_ssm_aux_1_cn()

		entries = (opt.sublin_num - 1) * opt.ssm_num

		#rhs
		rhs = [0.0] * entries * 2 + [-1.0] * entries + [0.0] * opt.ssm_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries * 2 + ["G"] * entries + ["E"] * opt.ssm_num
		self.assertListEqual(opt.my_sense, s)  

		# rownames
		rownames = ["ssm_aux_1_cn_dssms_0_1", "ssm_aux_1_cn_dssms_0_2",
			"ssm_aux_1_cn_dssms_1_1", "ssm_aux_1_cn_dssms_1_2",
			"ssm_aux_1_cn_phi_0_1", "ssm_aux_1_cn_phi_0_2",
			"ssm_aux_1_cn_phi_1_1", "ssm_aux_1_cn_phi_1_2",
			"ssm_aux_1_cn_phi_dssms_0_1", "ssm_aux_1_cn_phi_dssms_0_2",
			"ssm_aux_1_cn_phi_dssms_1_1", "ssm_aux_1_cn_phi_dssms_1_2",
			"ssm_aux_1_cn_not_in_normal_0", "ssm_aux_1_cn_not_in_normal_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			[opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_colnames_dssm[0][1], opt.my_colnames_dssm_a[0][1],
			opt.my_colnames_dssm_b[0][1]], 
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_colnames_dssm[0][2], opt.my_colnames_dssm_a[0][2],
			opt.my_colnames_dssm_b[0][2]], 
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_colnames_dssm[1][1], opt.my_colnames_dssm_a[1][1],
			opt.my_colnames_dssm_b[1][1]], 
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_colnames_dssm[1][2], opt.my_colnames_dssm_a[1][2],
			opt.my_colnames_dssm_b[1][2]], 
			[opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_phis[1]],
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_phis[2]],
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_phis[1]],
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_phis[2]],
			[opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_phis[1], opt.my_colnames_dssm[0][1],
			opt.my_colnames_dssm_a[0][1], opt.my_colnames_dssm_b[0][1]], 
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_phis[2], opt.my_colnames_dssm[0][2],
			opt.my_colnames_dssm_a[0][2], opt.my_colnames_dssm_b[0][2]], 
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_phis[1], opt.my_colnames_dssm[1][1],
			opt.my_colnames_dssm_a[1][1], opt.my_colnames_dssm_b[1][1]], 
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_phis[2], opt.my_colnames_dssm[1][2],
			opt.my_colnames_dssm_a[1][2], opt.my_colnames_dssm_b[1][2]], 
			[opt.my_colnames_dssm_aux_1_cn[0][0]], 
			[opt.my_colnames_dssm_aux_1_cn[1][0]], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)
		
		rows = [
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0],
			[1.0]
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)
		
		########################################
		# more complicated CN changes are allowed, no variable dssm anymore
		# set up variables
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list,
			simple_CN_changes=False) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()
		opt.vars_phi()
		opt.vars_ssm_aux_1_cn()

		# function to test
		opt.constraint_ssm_aux_1_cn()

		entries = (opt.sublin_num - 1) * opt.ssm_num

		#rhs
		rhs = [0.0] * entries * 2 + [-1.0] * entries + [0.0] * opt.ssm_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries * 2 + ["G"] * entries + ["E"] * opt.ssm_num
		self.assertListEqual(opt.my_sense, s)  

		# rownames
		rownames = ["ssm_aux_1_cn_dssms_0_1", "ssm_aux_1_cn_dssms_0_2",
			"ssm_aux_1_cn_dssms_1_1", "ssm_aux_1_cn_dssms_1_2",
			"ssm_aux_1_cn_phi_0_1", "ssm_aux_1_cn_phi_0_2",
			"ssm_aux_1_cn_phi_1_1", "ssm_aux_1_cn_phi_1_2",
			"ssm_aux_1_cn_phi_dssms_0_1", "ssm_aux_1_cn_phi_dssms_0_2",
			"ssm_aux_1_cn_phi_dssms_1_1", "ssm_aux_1_cn_phi_dssms_1_2",
			"ssm_aux_1_cn_not_in_normal_0", "ssm_aux_1_cn_not_in_normal_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			[opt.my_colnames_dssm_aux_1_cn[0][1],  opt.my_colnames_dssm_a[0][1],
			opt.my_colnames_dssm_b[0][1]], 
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_colnames_dssm_a[0][2],
			opt.my_colnames_dssm_b[0][2]],
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_colnames_dssm_a[1][1],
			opt.my_colnames_dssm_b[1][1]], 
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_colnames_dssm_a[1][2],
			opt.my_colnames_dssm_b[1][2]], 
			[opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_phis[1]],
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_phis[2]],
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_phis[1]],
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_phis[2]],
			[opt.my_colnames_dssm_aux_1_cn[0][1], opt.my_phis[1], 
			opt.my_colnames_dssm_a[0][1], opt.my_colnames_dssm_b[0][1]],
			[opt.my_colnames_dssm_aux_1_cn[0][2], opt.my_phis[2], 
			opt.my_colnames_dssm_a[0][2], opt.my_colnames_dssm_b[0][2]],
			[opt.my_colnames_dssm_aux_1_cn[1][1], opt.my_phis[1], 
			opt.my_colnames_dssm_a[1][1], opt.my_colnames_dssm_b[1][1]],
			[opt.my_colnames_dssm_aux_1_cn[1][2], opt.my_phis[2], 
			opt.my_colnames_dssm_a[1][2], opt.my_colnames_dssm_b[1][2]],
			[opt.my_colnames_dssm_aux_1_cn[0][0]], 
			[opt.my_colnames_dssm_aux_1_cn[1][0]], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			[1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0],
			[1.0],
			[1.0]
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_dssm_infl_cnv_same_lineage(self):
		sublin_num  = 3
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		self.ssm_list = [ssm1, ssm2, ssm3]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_aux_dc()
		opt.vars_dssm_infl_cnv_same_lineage()

		# function to test
		opt.constraint_dssm_infl_cnv_same_lineage(self.ssm_list)

		entries = 12 + 12

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["L"] * entries
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = ["constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_0_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_0_2",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_1_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_1_2",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_2_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_a_p1_2_2",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_0_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_0_2",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_1_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_1_2",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_2_1",
			"constraint_dssm_infl_cnv_same_lineage_cn_change_b_p1_2_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_0_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_0_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_1_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_1_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_2_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_a_p1_2_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_0_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_0_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_1_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_1_2",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_2_1",
			"constraint_dssm_infl_cnv_same_lineage_ssm_b_p1_2_2",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		rows = [
			["dssm_infl_cnv_same_lineage_a_p1_0_1", "dc_a_p1_binary_1_1"],
			["dssm_infl_cnv_same_lineage_a_p1_0_2", "dc_a_p1_binary_1_2"],
			["dssm_infl_cnv_same_lineage_a_p1_1_1", "dc_a_p1_binary_0_1"],
			["dssm_infl_cnv_same_lineage_a_p1_1_2", "dc_a_p1_binary_0_2"],
			["dssm_infl_cnv_same_lineage_a_p1_2_1", "dc_a_p1_binary_0_1"],
			["dssm_infl_cnv_same_lineage_a_p1_2_2", "dc_a_p1_binary_0_2"],
			["dssm_infl_cnv_same_lineage_b_p1_0_1", "dc_b_p1_binary_1_1"],
			["dssm_infl_cnv_same_lineage_b_p1_0_2", "dc_b_p1_binary_1_2"],
			["dssm_infl_cnv_same_lineage_b_p1_1_1", "dc_b_p1_binary_0_1"],
			["dssm_infl_cnv_same_lineage_b_p1_1_2", "dc_b_p1_binary_0_2"],
			["dssm_infl_cnv_same_lineage_b_p1_2_1", "dc_b_p1_binary_0_1"],
			["dssm_infl_cnv_same_lineage_b_p1_2_2", "dc_b_p1_binary_0_2"],
			["dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_a_0_1"],
			["dssm_infl_cnv_same_lineage_a_p1_0_2", "dssm_a_0_2"],
			["dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_a_1_1"],
			["dssm_infl_cnv_same_lineage_a_p1_1_2", "dssm_a_1_2"],
			["dssm_infl_cnv_same_lineage_a_p1_2_1", "dssm_a_2_1"],
			["dssm_infl_cnv_same_lineage_a_p1_2_2", "dssm_a_2_2"],
			["dssm_infl_cnv_same_lineage_b_p1_0_1", "dssm_b_0_1"],
			["dssm_infl_cnv_same_lineage_b_p1_0_2", "dssm_b_0_2"],
			["dssm_infl_cnv_same_lineage_b_p1_1_1", "dssm_b_1_1"],
			["dssm_infl_cnv_same_lineage_b_p1_1_2", "dssm_b_1_2"],
			["dssm_infl_cnv_same_lineage_b_p1_2_1", "dssm_b_2_1"],
			["dssm_infl_cnv_same_lineage_b_p1_2_2", "dssm_b_2_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), rows)


	def test_constraint_remove_ssm_symmetry(self):

		# only simple CN changes are allowed
		# set up variables 
		sublin_num  = 2
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		self.ssm_list = [ssm1, ssm2, ssm3]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_aux_dc()
		opt.vars_z()
		opt.vars_dc_ancestral()
		opt.vars_dssm_infl_cnv_same_lineage()

		# function to test
		opt.constraint_remove_ssm_symmetry(self.ssm_list)

		entries1 = 3
		entries3 = 6

		# rhs
		rhs = [0.0] * entries1 + [1.0] * entries3
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * (entries1 + entries3) 
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_a_0_1", "dssm_b_0_1", "dc_a_m1_binary_1_1", "dc_b_m1_binary_1_1",
			"dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"], 
			["dssm_a_1_1", "dssm_b_1_1", "dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"],
			["dssm_a_2_1", "dssm_b_2_1", "dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_2_1", "dssm_infl_cnv_same_lineage_b_p1_2_1"],
			["dssm_0_1", "dssm_a_0_1", "dc_a_m1_binary_1_1"], 
			["dssm_1_1", "dssm_a_1_1", "dc_a_m1_binary_0_1"], 
			["dssm_2_1", "dssm_a_2_1", "dc_a_m1_binary_0_1"], 
			["dssm_0_1", "dssm_b_0_1", "dc_b_m1_binary_1_1"], 
			["dssm_1_1", "dssm_b_1_1", "dc_b_m1_binary_0_1"], 
			["dssm_2_1", "dssm_b_2_1", "dc_b_m1_binary_0_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]] * 3 + [[1.0, 1.0, 1.0]] * 6
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		####################################
		# more complicated CN changes are allowed, no variables dssm, no constraint #20.2
		# set up variables 
		sublin_num  = 2
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		self.ssm_list = [ssm1, ssm2, ssm3]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list, simple_CN_changes=False) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_aux_dc()
		opt.vars_z()
		opt.vars_dc_ancestral()
		opt.vars_dssm_infl_cnv_same_lineage()

		# function to test
		opt.constraint_remove_ssm_symmetry(self.ssm_list)

		entries1 = 3
		entries3 = 6

		# rhs
		rhs = [0.0] * entries1 + [1.0] * entries3
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * (entries1 + entries3) 
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_1"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dssm_b_0_1", "dc_a_m1_binary_1_1", "dc_b_m1_binary_1_1",
			"dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"],
			["dssm_b_1_1", "dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"],
			["dssm_b_2_1", "dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_2_1", "dssm_infl_cnv_same_lineage_b_p1_2_1"],
			["dssm_a_0_1", "dc_a_m1_binary_1_1"], 
			["dssm_a_1_1", "dc_a_m1_binary_0_1"], 
			["dssm_a_2_1", "dc_a_m1_binary_0_1"], 
			["dssm_b_0_1", "dc_b_m1_binary_1_1"], 
			["dssm_b_1_1", "dc_b_m1_binary_0_1"], 
			["dssm_b_2_1", "dc_b_m1_binary_0_1"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, -1.0, -1.0, -1.0, -1.0]] * 3 + [[1.0, 1.0]] * 6
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		################################################################################
		# only simple CN changes are allowed
		# set up variables
		sublin_num  = 4
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		self.ssm_list = [ssm1, ssm2, ssm3]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_aux_dc()
		opt.vars_z()
		opt.vars_dc_descendant()
		opt.vars_dc_ancestral()
		opt.vars_dssm_infl_cnv_same_lineage()

		# function to test
		opt.constraint_remove_ssm_symmetry(self.ssm_list)

		entries1 = 9
		entries2 = 12
		entries3 = 18

		# rhs
		rhs = [0.0] * entries1 + [1.0] * entries2 + [1.0] * entries3
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * (entries1 + entries2 + entries3)
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_3",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_3",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_3",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_0_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_0_2",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_1_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_1_2",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_2_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_2_2",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_0_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_0_2",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_1_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_1_2",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_2_1",
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_2_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_3",
			]	
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			# c20.1 constraint_remove_ssm_symmetry_no_CN_change_unphased
			["dssm_a_0_1", "dssm_b_0_1",
			"dc_descendant_a_p1_1_1_2", "dc_descendant_a_p1_1_1_3",
			"dc_descendant_b_p1_1_1_2", "dc_descendant_b_p1_1_1_3",
			"dc_descendant_a_m1_1_1_2", "dc_descendant_a_m1_1_1_3",
			"dc_descendant_b_m1_1_1_2", "dc_descendant_b_m1_1_1_3",
			"dc_a_m1_binary_1_1", "dc_b_m1_binary_1_1",
			"dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"],
			["dssm_a_1_1", "dssm_b_1_1",
			"dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3",
			"dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3",
			"dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"],
			["dssm_a_2_1", "dssm_b_2_1",
			"dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3",
			"dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3",
			"dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_2_1", "dssm_infl_cnv_same_lineage_b_p1_2_1"],
			["dssm_a_0_2", "dssm_b_0_2",
			"dc_descendant_a_p1_1_2_3",
			"dc_descendant_b_p1_1_2_3",
			"dc_descendant_a_m1_1_2_3",
			"dc_descendant_b_m1_1_2_3",
			"dc_ancestral_a_m1_1_2_1",
			"dc_ancestral_b_m1_1_2_1",
			"dc_a_m1_binary_1_2", "dc_b_m1_binary_1_2",
			"dssm_infl_cnv_same_lineage_a_p1_0_2", "dssm_infl_cnv_same_lineage_b_p1_0_2"],
			["dssm_a_1_2", "dssm_b_1_2",
			"dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",
			"dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",
			"dc_ancestral_a_m1_0_2_1",
			"dc_ancestral_b_m1_0_2_1",
			"dc_a_m1_binary_0_2", "dc_b_m1_binary_0_2",
			"dssm_infl_cnv_same_lineage_a_p1_1_2", "dssm_infl_cnv_same_lineage_b_p1_1_2"],
			["dssm_a_2_2", "dssm_b_2_2",
			"dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",
			"dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",
			"dc_ancestral_a_m1_0_2_1",
			"dc_ancestral_b_m1_0_2_1",
			"dc_a_m1_binary_0_2", "dc_b_m1_binary_0_2",
			"dssm_infl_cnv_same_lineage_a_p1_2_2", "dssm_infl_cnv_same_lineage_b_p1_2_2"],
			["dssm_a_0_3", "dssm_b_0_3",
			"dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2",
			"dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2",
			"dc_a_m1_binary_1_3", "dc_b_m1_binary_1_3",
			"dssm_infl_cnv_same_lineage_a_p1_0_3", "dssm_infl_cnv_same_lineage_b_p1_0_3"],
			["dssm_a_1_3", "dssm_b_1_3",
			"dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_a_m1_binary_0_3", "dc_b_m1_binary_0_3",
			"dssm_infl_cnv_same_lineage_a_p1_1_3", "dssm_infl_cnv_same_lineage_b_p1_1_3"],
			["dssm_a_2_3", "dssm_b_2_3",
			"dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_a_m1_binary_0_3", "dc_b_m1_binary_0_3",
			"dssm_infl_cnv_same_lineage_a_p1_2_3", "dssm_infl_cnv_same_lineage_b_p1_2_3"],
			# here more constraints for other values of j, number ssms
			# c20.2 constraint_remove_ssm_symmetry_uneffected_not_unphased
			["dssm_0_1", "dc_descendant_a_p1_1_1_2", "dc_descendant_a_p1_1_1_3",
			"dc_descendant_b_p1_1_1_2", "dc_descendant_b_p1_1_1_3"],
			["dssm_0_2", "dc_descendant_a_p1_1_2_3",
			"dc_descendant_b_p1_1_2_3",],
			["dssm_1_1", "dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3"],
			["dssm_1_2", "dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",], 
			["dssm_2_1", "dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3"],
			["dssm_2_2", "dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",], 
			["dssm_0_1", "dc_descendant_a_m1_1_1_2", "dc_descendant_a_m1_1_1_3",
			"dc_descendant_b_m1_1_1_2", "dc_descendant_b_m1_1_1_3"],
			["dssm_0_2", "dc_descendant_a_m1_1_2_3",
			"dc_descendant_b_m1_1_2_3",], 
			["dssm_1_1", "dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3"],
			["dssm_1_2", "dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",], 
			["dssm_2_1", "dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3"],
			["dssm_2_2", "dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",], 
			# c20.3 constraint_remove_ssm_symmetry_chromatid_loss
			["dssm_0_1", "dssm_a_0_1", "dc_a_m1_binary_1_1"], 
			["dssm_1_1", "dssm_a_1_1", "dc_a_m1_binary_0_1"], 
			["dssm_2_1", "dssm_a_2_1", "dc_a_m1_binary_0_1"], 
			["dssm_0_2", "dssm_a_0_2", "dc_ancestral_a_m1_1_2_1", "dc_a_m1_binary_1_2"],
			["dssm_1_2", "dssm_a_1_2", "dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_2"],
			["dssm_2_2", "dssm_a_2_2", "dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_2"],
			["dssm_0_3", "dssm_a_0_3", "dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2",
			"dc_a_m1_binary_1_3"],
			["dssm_1_3", "dssm_a_1_3", "dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_a_m1_binary_0_3"], 
			["dssm_2_3", "dssm_a_2_3", "dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_a_m1_binary_0_3"],
			["dssm_0_1", "dssm_b_0_1", "dc_b_m1_binary_1_1"], 
			["dssm_1_1", "dssm_b_1_1", "dc_b_m1_binary_0_1"], 
			["dssm_2_1", "dssm_b_2_1", "dc_b_m1_binary_0_1"], 
			["dssm_0_2", "dssm_b_0_2", "dc_ancestral_b_m1_1_2_1", "dc_b_m1_binary_1_2"],
			["dssm_1_2", "dssm_b_1_2", "dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_2"],
			["dssm_2_2", "dssm_b_2_2", "dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_2"],
			["dssm_0_3", "dssm_b_0_3", "dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2",
			"dc_b_m1_binary_1_3"], 
			["dssm_1_3", "dssm_b_1_3", "dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_b_m1_binary_0_3"],
			["dssm_2_3", "dssm_b_2_3", "dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_b_m1_binary_0_3"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)
		
		rows = [
			# c20.1 constraint_remove_ssm_symmetry_no_CN_change_unphased
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
			-1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
			-1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
			-1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			# here more constraints for other values of j, number ssms
			# c20.2 constraint_remove_ssm_symmetry_uneffected_not_unphased
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			# c20.3 constraint_remove_ssm_symmetry_chromatid_loss
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0, 1.0],
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)
		
		################################################################################
		# more complicated CN changes are allowed, variable dssm doesn't exist, 
		#	constraint 20.2 not used
		# set up variables
		sublin_num  = 4
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		self.ssm_list = [ssm1, ssm2, ssm3]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list, simple_CN_changes=False) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()
		opt.vars_aux_dc()
		opt.vars_z()
		opt.vars_dc_descendant()
		opt.vars_dc_ancestral()
		opt.vars_dssm_infl_cnv_same_lineage()

		# function to test
		opt.constraint_remove_ssm_symmetry(self.ssm_list)

		entries1 = 9
		entries2 = 0
		entries3 = 18

		# rhs
		rhs = [0.0] * entries1 + [1.0] * entries2 + [1.0] * entries3
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * (entries1 + entries2 + entries3)
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_1",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_2",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_0_3",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_1_3",
			"constraint_remove_ssm_symmetry_no_CN_change_unphased_2_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_0_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_1_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_a_2_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_1",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_2",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_0_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_1_3",
			"constraint_remove_ssm_symmetry_chromatid_loss_b_2_3",
			]	
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			# c20.1 constraint_remove_ssm_symmetry_no_CN_change_unphased
			["dssm_b_0_1",
			"dc_descendant_a_p1_1_1_2", "dc_descendant_a_p1_1_1_3",
			"dc_descendant_b_p1_1_1_2", "dc_descendant_b_p1_1_1_3",
			"dc_descendant_a_m1_1_1_2", "dc_descendant_a_m1_1_1_3",
			"dc_descendant_b_m1_1_1_2", "dc_descendant_b_m1_1_1_3",
			"dc_a_m1_binary_1_1", "dc_b_m1_binary_1_1",
			"dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_1"],
			["dssm_b_1_1",
			"dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3",
			"dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3",
			"dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_1"],
			["dssm_b_2_1",
			"dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3",
			"dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3",
			"dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3",
			"dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3",
			"dc_a_m1_binary_0_1", "dc_b_m1_binary_0_1",
			"dssm_infl_cnv_same_lineage_a_p1_2_1", "dssm_infl_cnv_same_lineage_b_p1_2_1"],
			["dssm_b_0_2",
			"dc_descendant_a_p1_1_2_3",
			"dc_descendant_b_p1_1_2_3",
			"dc_descendant_a_m1_1_2_3",
			"dc_descendant_b_m1_1_2_3",
			"dc_ancestral_a_m1_1_2_1",
			"dc_ancestral_b_m1_1_2_1",
			"dc_a_m1_binary_1_2", "dc_b_m1_binary_1_2",
			"dssm_infl_cnv_same_lineage_a_p1_0_2", "dssm_infl_cnv_same_lineage_b_p1_0_2"],
			["dssm_b_1_2",
			"dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",
			"dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",
			"dc_ancestral_a_m1_0_2_1",
			"dc_ancestral_b_m1_0_2_1",
			"dc_a_m1_binary_0_2", "dc_b_m1_binary_0_2",
			"dssm_infl_cnv_same_lineage_a_p1_1_2", "dssm_infl_cnv_same_lineage_b_p1_1_2"],
			["dssm_b_2_2",
			"dc_descendant_a_p1_0_2_3",
			"dc_descendant_b_p1_0_2_3",
			"dc_descendant_a_m1_0_2_3",
			"dc_descendant_b_m1_0_2_3",
			"dc_ancestral_a_m1_0_2_1",
			"dc_ancestral_b_m1_0_2_1",
			"dc_a_m1_binary_0_2", "dc_b_m1_binary_0_2",
			"dssm_infl_cnv_same_lineage_a_p1_2_2", "dssm_infl_cnv_same_lineage_b_p1_2_2"],
			["dssm_b_0_3",
			"dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2",
			"dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2",
			"dc_a_m1_binary_1_3", "dc_b_m1_binary_1_3",
			"dssm_infl_cnv_same_lineage_a_p1_0_3", "dssm_infl_cnv_same_lineage_b_p1_0_3"],
			["dssm_b_1_3",
			"dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_a_m1_binary_0_3", "dc_b_m1_binary_0_3",
			"dssm_infl_cnv_same_lineage_a_p1_1_3", "dssm_infl_cnv_same_lineage_b_p1_1_3"], 
			["dssm_b_2_3",
			"dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_a_m1_binary_0_3", "dc_b_m1_binary_0_3",
			"dssm_infl_cnv_same_lineage_a_p1_2_3", "dssm_infl_cnv_same_lineage_b_p1_2_3"], 
			# c20.3 constraint_remove_ssm_symmetry_chromatid_loss
			["dssm_a_0_1", "dc_a_m1_binary_1_1"],
			["dssm_a_1_1", "dc_a_m1_binary_0_1"],
			["dssm_a_2_1", "dc_a_m1_binary_0_1"],
			["dssm_a_0_2", "dc_ancestral_a_m1_1_2_1", "dc_a_m1_binary_1_2"],
			["dssm_a_1_2", "dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_2"],
			["dssm_a_2_2", "dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_2"],
			["dssm_a_0_3", "dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2",
			"dc_a_m1_binary_1_3"],
			["dssm_a_1_3", "dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_a_m1_binary_0_3"], 
			["dssm_a_2_3", "dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2",
			"dc_a_m1_binary_0_3"],
			["dssm_b_0_1", "dc_b_m1_binary_1_1"],
			["dssm_b_1_1", "dc_b_m1_binary_0_1"],
			["dssm_b_2_1", "dc_b_m1_binary_0_1"],
			["dssm_b_0_2", "dc_ancestral_b_m1_1_2_1", "dc_b_m1_binary_1_2"],
			["dssm_b_1_2", "dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_2"],
			["dssm_b_2_2", "dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_2"],
			["dssm_b_0_3", "dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2",
			"dc_b_m1_binary_1_3"],
			["dssm_b_1_3", "dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_b_m1_binary_0_3"], 
			["dssm_b_2_3", "dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2",
			"dc_b_m1_binary_0_3"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [
			# c20.1 constraint_remove_ssm_symmetry_no_CN_change_unphased
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
			# c20.3 constraint_remove_ssm_symmetry_chromatid_loss
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, 1.0],
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)


	def test_constraint_ssm_isa(self):

		# only simple CN changes are allowed
		# set up variables 
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()

		# function to test  
		opt.constraint_ssm_isa()

		entries = self.ssm_num
	
		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["ssm_isa_0", "ssm_isa_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [[opt.my_colnames_dssm[0][0], opt.my_colnames_dssm[0][1], opt.my_colnames_dssm[0][2],
			opt.my_colnames_dssm_a[0][0], opt.my_colnames_dssm_a[0][1], opt.my_colnames_dssm_a[0][2],
			opt.my_colnames_dssm_b[0][0], opt.my_colnames_dssm_b[0][1], opt.my_colnames_dssm_b[0][2]], 
			[opt.my_colnames_dssm[1][0], opt.my_colnames_dssm[1][1], opt.my_colnames_dssm[1][2],
			opt.my_colnames_dssm_a[1][0], opt.my_colnames_dssm_a[1][1], opt.my_colnames_dssm_a[1][2],
			opt.my_colnames_dssm_b[1][0], opt.my_colnames_dssm_b[1][1], opt.my_colnames_dssm_b[1][2]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		# more complictad CN changes are allowed, no variable dssm
		# set up variables 
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list,
			simple_CN_changes=False) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()

		# function to test  
		opt.constraint_ssm_isa()

		entries = self.ssm_num
	
		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["ssm_isa_0", "ssm_isa_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [[opt.my_colnames_dssm_a[0][0], opt.my_colnames_dssm_a[0][1], opt.my_colnames_dssm_a[0][2],
			opt.my_colnames_dssm_b[0][0], opt.my_colnames_dssm_b[0][1], opt.my_colnames_dssm_b[0][2]], 
			[opt.my_colnames_dssm_a[1][0], opt.my_colnames_dssm_a[1][1], opt.my_colnames_dssm_a[1][2],
			opt.my_colnames_dssm_b[1][0], opt.my_colnames_dssm_b[1][1], opt.my_colnames_dssm_b[1][2]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_no_ssm_normal_lineage(self):

		# only simple CN changes allowed
		# set up variables 
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()

		# function to test  
		opt.constraint_no_ssm_normal_lineage()

		entries = self.ssm_num * 3

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["no_ssm_normal_lin_0", "no_ssm_normal_lin_1", "no_ssm_a_normal_lin_0", "no_ssm_a_normal_lin_1",
			"no_ssm_b_normal_lin_0", "no_ssm_b_normal_lin_1"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [[opt.my_colnames_dssm[0][0]], [opt.my_colnames_dssm[1][0]],
			[opt.my_colnames_dssm_a[0][0]],[opt.my_colnames_dssm_a[1][0]],
			[opt.my_colnames_dssm_b[0][0]],[opt.my_colnames_dssm_b[1][0]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 6
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

		################################
		# more complicated CN changes allowed, no variable dssm
		# set up variables 
		sublin_num  = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list, 
			simple_CN_changes=False) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_ssm_matrices()

		# function to test  
		opt.constraint_no_ssm_normal_lineage()

		entries = self.ssm_num * 2

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["no_ssm_a_normal_lin_0", "no_ssm_a_normal_lin_1",
			"no_ssm_b_normal_lin_0", "no_ssm_b_normal_lin_1"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [[opt.my_colnames_dssm_a[0][0]], [opt.my_colnames_dssm_a[1][0]],
			[opt.my_colnames_dssm_b[0][0]], [opt.my_colnames_dssm_b[1][0]],]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0]] * 4
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_snp_value_spline_frequency(self):
		
		# set up variables 
		sublin_num = 3
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num
		snp1 = snp_ssm.SNP()
		snp1.seg_index = 1
		snp2 = snp_ssm.SNP()
		snp2.seg_index = 0
		self.snp_list = [snp1, snp2]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		#self.snp_spl_list = [self.snp_spl_1, self.seg_spl_1]
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_vars_mut_splines()
		opt.vars_three_snp_matrices()
		opt.vars_aux_snp_linear()
		
		# function to test 
		opt.constraint_snp_value_spline_frequency(self.snp_list)

		# rhs
		rhs = [1.0] * opt.snp_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * opt.snp_num 
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["snp_frequency_0", "snp_frequency_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		value_snp1 = ([opt.snp_splines[0].get_knots()[i] * opt.cn[snp1.seg_index]
			for i in xrange(len(opt.snp_splines[0].get_knots()))]
			+ [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
		value_snp2 = ([opt.snp_splines[1].get_knots()[i] * opt.cn[snp2.seg_index] 
			for i in xrange(len(opt.snp_splines[1].get_knots()))] 
			+ [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
		rows = [[opt.my_colnames_snp[0] 
			+ opt.my_colnames_snp_w_cnv_a_p1[0] + opt.my_colnames_snp_w_cnv_b_p1[0]
			+ opt.my_colnames_snp_w_cnv_a_m1[0] + opt.my_colnames_snp_w_cnv_b_m1[0], value_snp1],
			[opt.my_colnames_snp[1] 
			+ opt.my_colnames_snp_w_cnv_a_p1[1] + opt.my_colnames_snp_w_cnv_b_p1[1]
			+ opt.my_colnames_snp_w_cnv_a_m1[1] + opt.my_colnames_snp_w_cnv_b_m1[1], value_snp2]]
		self.assertListEqual(opt.my_rows, rows)

	def test_constraint_aux_snp_w_cnv(self):
		
		# set up variables 
		sublin_num = 3
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 
		snp1 = snp_ssm.SNP()
		snp1.seg_index = 1
		snp2 = snp_ssm.SNP()
		snp2.seg_index = 0
		self.snp_list = [snp1, snp2]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_snp_matrices()
		opt.vars_aux_dc()
		opt.vars_phi()
		opt.vars_aux_snp_linear()

		# function to test
		opt.constraint_aux_snp_w_cnv(self.snp_list)

		# rhs
		rhs = [0.0] * 3 * opt.snp_aux_linear_variables_num + [-2.0] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * 3 * opt.snp_aux_linear_variables_num + ["G"] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["snv_w_cnv_a_p1_phi_0_1", "snv_w_cnv_a_p1_phi_0_2",
			"snv_w_cnv_a_p1_phi_1_1", "snv_w_cnv_a_p1_phi_1_2", 
			"snv_w_cnv_b_p1_phi_0_1", "snv_w_cnv_b_p1_phi_0_2",
			"snv_w_cnv_b_p1_phi_1_1", "snv_w_cnv_b_p1_phi_1_2",
			"snv_w_cnv_a_m1_phi_0_1", "snv_w_cnv_a_m1_phi_0_2",
			"snv_w_cnv_a_m1_phi_1_1", "snv_w_cnv_a_m1_phi_1_2", 
			"snv_w_cnv_b_m1_phi_0_1", "snv_w_cnv_b_m1_phi_0_2",
			"snv_w_cnv_b_m1_phi_1_1", "snv_w_cnv_b_m1_phi_1_2",
			"snv_w_cnv_a_p1_dsnp_a_0_1", "snv_w_cnv_a_p1_dsnp_a_0_2",
			"snv_w_cnv_a_p1_dsnp_a_1_1", "snv_w_cnv_a_p1_dsnp_a_1_2",
			"snv_w_cnv_b_p1_dsnp_b_0_1", "snv_w_cnv_b_p1_dsnp_b_0_2",
			"snv_w_cnv_b_p1_dsnp_b_1_1", "snv_w_cnv_b_p1_dsnp_b_1_2",
			"snv_w_cnv_a_m1_dsnp_a_0_1", "snv_w_cnv_a_m1_dsnp_a_0_2",
			"snv_w_cnv_a_m1_dsnp_a_1_1", "snv_w_cnv_a_m1_dsnp_a_1_2",
			"snv_w_cnv_b_m1_dsnp_b_0_1", "snv_w_cnv_b_m1_dsnp_b_0_2",
			"snv_w_cnv_b_m1_dsnp_b_1_1", "snv_w_cnv_b_m1_dsnp_b_1_2",
			"snv_w_cnv_a_p1_dc_a_0_1", "snv_w_cnv_a_p1_dc_a_0_2",
			"snv_w_cnv_a_p1_dc_a_1_1", "snv_w_cnv_a_p1_dc_a_1_2", 
			"snv_w_cnv_b_p1_dc_b_0_1", "snv_w_cnv_b_p1_dc_b_0_2",
			"snv_w_cnv_b_p1_dc_b_1_1", "snv_w_cnv_b_p1_dc_b_1_2", 
			"snv_w_cnv_a_m1_dc_a_0_1", "snv_w_cnv_a_m1_dc_a_0_2",
			"snv_w_cnv_a_m1_dc_a_1_1", "snv_w_cnv_a_m1_dc_a_1_2", 
			"snv_w_cnv_b_m1_dc_b_0_1", "snv_w_cnv_b_m1_dc_b_0_2",
			"snv_w_cnv_b_m1_dc_b_1_1", "snv_w_cnv_b_m1_dc_b_1_2", 
			"snv_cnv_a_p1_together_0_1", "snv_cnv_a_p1_together_0_2",
			"snv_cnv_a_p1_together_1_1", "snv_cnv_a_p1_together_1_2",
			"snv_cnv_b_p1_together_0_1", "snv_cnv_b_p1_together_0_2",
			"snv_cnv_b_p1_together_1_1", "snv_cnv_b_p1_together_1_2",
			"snv_cnv_a_m1_together_0_1", "snv_cnv_a_m1_together_0_2",
			"snv_cnv_a_m1_together_1_1", "snv_cnv_a_m1_together_1_2",
			"snv_cnv_b_m1_together_0_1", "snv_cnv_b_m1_together_0_2",
			"snv_cnv_b_m1_together_1_1", "snv_cnv_b_m1_together_1_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			[[opt.my_colnames_snp_w_cnv_a_p1[0][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][0], opt.my_phis[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][1], opt.my_phis[2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][0], opt.my_colnames_dsnp_a[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][1], opt.my_colnames_dsnp_a[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][0], opt.my_colnames_dsnp_a[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][1], opt.my_colnames_dsnp_a[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][0], opt.my_colnames_dsnp_b[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][1], opt.my_colnames_dsnp_b[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][0], opt.my_colnames_dsnp_b[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][1], opt.my_colnames_dsnp_b[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][0], opt.my_colnames_dsnp_a[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][1], opt.my_colnames_dsnp_a[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][0], opt.my_colnames_dsnp_a[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][1], opt.my_colnames_dsnp_a[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][0], opt.my_colnames_dsnp_b[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][1], opt.my_colnames_dsnp_b[0]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][0], opt.my_colnames_dsnp_b[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][1], opt.my_colnames_dsnp_b[1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][0], opt.my_colnames_dc_a_p1_binary[snp1.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][1], opt.my_colnames_dc_a_p1_binary[snp1.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][0], opt.my_colnames_dc_a_p1_binary[snp2.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][1], opt.my_colnames_dc_a_p1_binary[snp2.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][0], opt.my_colnames_dc_b_p1_binary[snp1.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][1], opt.my_colnames_dc_b_p1_binary[snp1.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][0], opt.my_colnames_dc_b_p1_binary[snp2.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][1], opt.my_colnames_dc_b_p1_binary[snp2.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][0], opt.my_colnames_dc_a_m1_binary[snp1.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][1], opt.my_colnames_dc_a_m1_binary[snp1.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][0], opt.my_colnames_dc_a_m1_binary[snp2.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][1], opt.my_colnames_dc_a_m1_binary[snp2.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][0], opt.my_colnames_dc_b_m1_binary[snp1.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][1], opt.my_colnames_dc_b_m1_binary[snp1.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][0], opt.my_colnames_dc_b_m1_binary[snp2.seg_index][1]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][1], opt.my_colnames_dc_b_m1_binary[snp2.seg_index][2]], [1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][0], opt.my_phis[1], opt.my_colnames_dsnp_a[0], 
			opt.my_colnames_dc_a_p1_binary[snp1.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[0][1], opt.my_phis[2], opt.my_colnames_dsnp_a[0], 
			opt.my_colnames_dc_a_p1_binary[snp1.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][0], opt.my_phis[1], opt.my_colnames_dsnp_a[1], 
			opt.my_colnames_dc_a_p1_binary[snp2.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_p1[1][1], opt.my_phis[2], opt.my_colnames_dsnp_a[1], 
			opt.my_colnames_dc_a_p1_binary[snp2.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][0], opt.my_phis[1], opt.my_colnames_dsnp_b[0], 
			opt.my_colnames_dc_b_p1_binary[snp1.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[0][1], opt.my_phis[2], opt.my_colnames_dsnp_b[0], 
			opt.my_colnames_dc_b_p1_binary[snp1.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][0], opt.my_phis[1], opt.my_colnames_dsnp_b[1], 
			opt.my_colnames_dc_b_p1_binary[snp2.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_p1[1][1], opt.my_phis[2], opt.my_colnames_dsnp_b[1], 
			opt.my_colnames_dc_b_p1_binary[snp2.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][0], opt.my_phis[1], opt.my_colnames_dsnp_a[0], 
			opt.my_colnames_dc_a_m1_binary[snp1.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[0][1], opt.my_phis[2], opt.my_colnames_dsnp_a[0], 
			opt.my_colnames_dc_a_m1_binary[snp1.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][0], opt.my_phis[1], opt.my_colnames_dsnp_a[1], 
			opt.my_colnames_dc_a_m1_binary[snp2.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_a_m1[1][1], opt.my_phis[2], opt.my_colnames_dsnp_a[1], 
			opt.my_colnames_dc_a_m1_binary[snp2.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][0], opt.my_phis[1], opt.my_colnames_dsnp_b[0], 
			opt.my_colnames_dc_b_m1_binary[snp1.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[0][1], opt.my_phis[2], opt.my_colnames_dsnp_b[0], 
			opt.my_colnames_dc_b_m1_binary[snp1.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][0], opt.my_phis[1], opt.my_colnames_dsnp_b[1], 
			opt.my_colnames_dc_b_m1_binary[snp2.seg_index][1]], [1.0, -1.0, -1.0, -1.0]],
			[[opt.my_colnames_snp_w_cnv_b_m1[1][1], opt.my_phis[2], opt.my_colnames_dsnp_b[1], 
			opt.my_colnames_dc_b_m1_binary[snp2.seg_index][2]], [1.0, -1.0, -1.0, -1.0]],
			]
		self.assertListEqual(opt.my_rows, rows)

	def test_constraint_remove_snp_symmetry(self):
		
		# set up variables 
		sublin_num = 3
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 
		snp1 = snp_ssm.SNP()
		snp1.seg_index = 1
		snp2 = snp_ssm.SNP()
		snp2.seg_index = 0
		self.snp_list = [snp1, snp2]
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_snp_matrices()
		opt.vars_aux_dc()

		# function to test
		opt.constraint_remove_snp_symmetry(self.snp_list)

		entries = self.snp_num * cons.PHASE_NUMBER
		entries2 = self.snp_num * opt.cnv_state_num

		# rhs
		rhs = [0.0] * entries + [1.0] * entries2
		self.assertListEqual(opt.my_rhs, rhs)

		# senses 
		s = ["L"] * entries + ["L"] * entries2
		self.assertListEqual(opt.my_sense, s) 

		# rownames
		rownames = ["constraint_remove_snp_symmetry_no_CN_change_unphased_a_0",
			"constraint_remove_snp_symmetry_no_CN_change_unphased_a_1",
			"constraint_remove_snp_symmetry_no_CN_change_unphased_b_0",
			"constraint_remove_snp_symmetry_no_CN_change_unphased_b_1",
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_p1_0",
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_p1_1",
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_m1_0",
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_m1_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [
			["dsnp_a_0", "dc_a_p1_binary_1_0", "dc_a_p1_binary_1_1", "dc_a_p1_binary_1_2",
			"dc_b_p1_binary_1_0", "dc_b_p1_binary_1_1", "dc_b_p1_binary_1_2",
			"dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2",
			"dc_b_m1_binary_1_0", "dc_b_m1_binary_1_1", "dc_b_m1_binary_1_2"],
			["dsnp_a_1", "dc_a_p1_binary_0_0", "dc_a_p1_binary_0_1", "dc_a_p1_binary_0_2",
			"dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2",
			"dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2",
			"dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dsnp_b_0", "dc_a_p1_binary_1_0", "dc_a_p1_binary_1_1", "dc_a_p1_binary_1_2",
			"dc_b_p1_binary_1_0", "dc_b_p1_binary_1_1", "dc_b_p1_binary_1_2",
			"dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2",
			"dc_b_m1_binary_1_0", "dc_b_m1_binary_1_1", "dc_b_m1_binary_1_2"],
			["dsnp_b_1", "dc_a_p1_binary_0_0", "dc_a_p1_binary_0_1", "dc_a_p1_binary_0_2",
			"dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2",
			"dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2",
			"dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dsnp_0", "dc_a_p1_binary_1_0", "dc_a_p1_binary_1_1", "dc_a_p1_binary_1_2",
			"dc_b_p1_binary_1_0", "dc_b_p1_binary_1_1", "dc_b_p1_binary_1_2"],
			["dsnp_1", "dc_a_p1_binary_0_0", "dc_a_p1_binary_0_1", "dc_a_p1_binary_0_2",
			"dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2"],
			["dsnp_0", "dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2",
			"dc_b_m1_binary_1_0", "dc_b_m1_binary_1_1", "dc_b_m1_binary_1_2"],
			["dsnp_1", "dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2",
			"dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)
		
		rows = [
			[1.0] + [-1] * opt.cnv_state_num * cons.PHASE_NUMBER * sublin_num,
			[1.0] + [-1] * opt.cnv_state_num * cons.PHASE_NUMBER * sublin_num,
			[1.0] + [-1] * opt.cnv_state_num * cons.PHASE_NUMBER * sublin_num,
			[1.0] + [-1] * opt.cnv_state_num * cons.PHASE_NUMBER * sublin_num,
			[1.0] + [1.0] * sublin_num * cons.PHASE_NUMBER,
			[1.0] + [1.0] * sublin_num * cons.PHASE_NUMBER,
			[1.0] + [1.0] * sublin_num * cons.PHASE_NUMBER,
			[1.0] + [1.0] * sublin_num * cons.PHASE_NUMBER
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows)
		

	def test_constraint_snp_row_one_entry(self):
		
		# set up variables
		sublin_num = 1
		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_snp_matrices()

		# function to test
		opt.constraint_snp_row_one_entry()

		# rhs
		rhs = [1.0] * opt.snp_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * opt.snp_num 
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["snp_row_0", "snp_row_1"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows = [
			[opt.my_colnames_dsnp[0], opt.my_colnames_dsnp_a[0], opt.my_colnames_dsnp_b[0]],
			[opt.my_colnames_dsnp[1], opt.my_colnames_dsnp_a[1], opt.my_colnames_dsnp_b[1]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows)

		rows = [[1.0, 1.0, 1.0]] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), rows)

	def test_constraint_dc_descendant(self):
		
		# set up variables, only 2 lineages
		sublin_num = 2
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.constraint_dc_descendant()
		
		self.assertEqual(len(opt.my_rhs), 0)

		# set up variables, 4 lineages
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list,
			self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_dc_descendant()
		opt.vars_aux_dc()
		opt.vars_z()

		# function to test 
		opt.constraint_dc_descendant()

		entries = 24

		# rhs
		rhs = [0.0] * entries * 2 + [-1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries * 2 + ["G"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"dc_descendant_le_dc_binary_a_p1_0_1_2", "dc_descendant_le_dc_binary_a_p1_0_1_3",
			"dc_descendant_le_dc_binary_a_p1_0_2_3",
			"dc_descendant_le_dc_binary_a_p1_1_1_2", "dc_descendant_le_dc_binary_a_p1_1_1_3",
			"dc_descendant_le_dc_binary_a_p1_1_2_3",
			"dc_descendant_le_dc_binary_b_p1_0_1_2", "dc_descendant_le_dc_binary_b_p1_0_1_3",
			"dc_descendant_le_dc_binary_b_p1_0_2_3",
			"dc_descendant_le_dc_binary_b_p1_1_1_2", "dc_descendant_le_dc_binary_b_p1_1_1_3",
			"dc_descendant_le_dc_binary_b_p1_1_2_3",
			"dc_descendant_le_dc_binary_a_m1_0_1_2", "dc_descendant_le_dc_binary_a_m1_0_1_3",
			"dc_descendant_le_dc_binary_a_m1_0_2_3",
			"dc_descendant_le_dc_binary_a_m1_1_1_2", "dc_descendant_le_dc_binary_a_m1_1_1_3",
			"dc_descendant_le_dc_binary_a_m1_1_2_3",
			"dc_descendant_le_dc_binary_b_m1_0_1_2", "dc_descendant_le_dc_binary_b_m1_0_1_3",
			"dc_descendant_le_dc_binary_b_m1_0_2_3",
			"dc_descendant_le_dc_binary_b_m1_1_1_2", "dc_descendant_le_dc_binary_b_m1_1_1_3",
			"dc_descendant_le_dc_binary_b_m1_1_2_3",
			"dc_descendant_le_z_a_p1_0_1_2", "dc_descendant_le_z_a_p1_0_1_3",
			"dc_descendant_le_z_a_p1_0_2_3",
			"dc_descendant_le_z_a_p1_1_1_2", "dc_descendant_le_z_a_p1_1_1_3",
			"dc_descendant_le_z_a_p1_1_2_3",
			"dc_descendant_le_z_b_p1_0_1_2", "dc_descendant_le_z_b_p1_0_1_3",
			"dc_descendant_le_z_b_p1_0_2_3",
			"dc_descendant_le_z_b_p1_1_1_2", "dc_descendant_le_z_b_p1_1_1_3",
			"dc_descendant_le_z_b_p1_1_2_3",
			"dc_descendant_le_z_a_m1_0_1_2", "dc_descendant_le_z_a_m1_0_1_3",
			"dc_descendant_le_z_a_m1_0_2_3",
			"dc_descendant_le_z_a_m1_1_1_2", "dc_descendant_le_z_a_m1_1_1_3",
			"dc_descendant_le_z_a_m1_1_2_3",
			"dc_descendant_le_z_b_m1_0_1_2", "dc_descendant_le_z_b_m1_0_1_3",
			"dc_descendant_le_z_b_m1_0_2_3",
			"dc_descendant_le_z_b_m1_1_1_2", "dc_descendant_le_z_b_m1_1_1_3",
			"dc_descendant_le_z_b_m1_1_2_3",
			"dc_descendant_ge_dc_binary_z_a_p1_0_1_2", "dc_descendant_ge_dc_binary_z_a_p1_0_1_3",
			"dc_descendant_ge_dc_binary_z_a_p1_0_2_3",
			"dc_descendant_ge_dc_binary_z_a_p1_1_1_2", "dc_descendant_ge_dc_binary_z_a_p1_1_1_3",
			"dc_descendant_ge_dc_binary_z_a_p1_1_2_3",
			"dc_descendant_ge_dc_binary_z_b_p1_0_1_2", "dc_descendant_ge_dc_binary_z_b_p1_0_1_3",
			"dc_descendant_ge_dc_binary_z_b_p1_0_2_3",
			"dc_descendant_ge_dc_binary_z_b_p1_1_1_2", "dc_descendant_ge_dc_binary_z_b_p1_1_1_3",
			"dc_descendant_ge_dc_binary_z_b_p1_1_2_3",
			"dc_descendant_ge_dc_binary_z_a_m1_0_1_2", "dc_descendant_ge_dc_binary_z_a_m1_0_1_3",
			"dc_descendant_ge_dc_binary_z_a_m1_0_2_3",
			"dc_descendant_ge_dc_binary_z_a_m1_1_1_2", "dc_descendant_ge_dc_binary_z_a_m1_1_1_3",
			"dc_descendant_ge_dc_binary_z_a_m1_1_2_3",
			"dc_descendant_ge_dc_binary_z_b_m1_0_1_2", "dc_descendant_ge_dc_binary_z_b_m1_0_1_3",
			"dc_descendant_ge_dc_binary_z_b_m1_0_2_3",
			"dc_descendant_ge_dc_binary_z_b_m1_1_1_2", "dc_descendant_ge_dc_binary_z_b_m1_1_1_3",
			"dc_descendant_ge_dc_binary_z_b_m1_1_2_3",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		#rows
		rows_vars = [
			["dc_descendant_a_p1_0_1_2", "dc_a_p1_binary_0_2"], 
			["dc_descendant_a_p1_0_1_3", "dc_a_p1_binary_0_3"], 
			["dc_descendant_a_p1_0_2_3", "dc_a_p1_binary_0_3"], 
			["dc_descendant_a_p1_1_1_2", "dc_a_p1_binary_1_2"], 
			["dc_descendant_a_p1_1_1_3", "dc_a_p1_binary_1_3"], 
			["dc_descendant_a_p1_1_2_3", "dc_a_p1_binary_1_3"], 
			["dc_descendant_b_p1_0_1_2", "dc_b_p1_binary_0_2"], 
			["dc_descendant_b_p1_0_1_3", "dc_b_p1_binary_0_3"], 
			["dc_descendant_b_p1_0_2_3", "dc_b_p1_binary_0_3"], 
			["dc_descendant_b_p1_1_1_2", "dc_b_p1_binary_1_2"], 
			["dc_descendant_b_p1_1_1_3", "dc_b_p1_binary_1_3"], 
			["dc_descendant_b_p1_1_2_3", "dc_b_p1_binary_1_3"], 
			["dc_descendant_a_m1_0_1_2", "dc_a_m1_binary_0_2"], 
			["dc_descendant_a_m1_0_1_3", "dc_a_m1_binary_0_3"], 
			["dc_descendant_a_m1_0_2_3", "dc_a_m1_binary_0_3"], 
			["dc_descendant_a_m1_1_1_2", "dc_a_m1_binary_1_2"], 
			["dc_descendant_a_m1_1_1_3", "dc_a_m1_binary_1_3"], 
			["dc_descendant_a_m1_1_2_3", "dc_a_m1_binary_1_3"], 
			["dc_descendant_b_m1_0_1_2", "dc_b_m1_binary_0_2"], 
			["dc_descendant_b_m1_0_1_3", "dc_b_m1_binary_0_3"], 
			["dc_descendant_b_m1_0_2_3", "dc_b_m1_binary_0_3"], 
			["dc_descendant_b_m1_1_1_2", "dc_b_m1_binary_1_2"], 
			["dc_descendant_b_m1_1_1_3", "dc_b_m1_binary_1_3"], 
			["dc_descendant_b_m1_1_2_3", "dc_b_m1_binary_1_3"], 
			["dc_descendant_a_p1_0_1_2", "z_1_2"], 
			["dc_descendant_a_p1_0_1_3", "z_1_3"], 
			["dc_descendant_a_p1_0_2_3", "z_2_3"], 
			["dc_descendant_a_p1_1_1_2", "z_1_2"], 
			["dc_descendant_a_p1_1_1_3", "z_1_3"], 
			["dc_descendant_a_p1_1_2_3", "z_2_3"], 
			["dc_descendant_b_p1_0_1_2", "z_1_2"], 
			["dc_descendant_b_p1_0_1_3", "z_1_3"], 
			["dc_descendant_b_p1_0_2_3", "z_2_3"], 
			["dc_descendant_b_p1_1_1_2", "z_1_2"], 
			["dc_descendant_b_p1_1_1_3", "z_1_3"], 
			["dc_descendant_b_p1_1_2_3", "z_2_3"], 
			["dc_descendant_a_m1_0_1_2", "z_1_2"], 
			["dc_descendant_a_m1_0_1_3", "z_1_3"], 
			["dc_descendant_a_m1_0_2_3", "z_2_3"], 
			["dc_descendant_a_m1_1_1_2", "z_1_2"], 
			["dc_descendant_a_m1_1_1_3", "z_1_3"], 
			["dc_descendant_a_m1_1_2_3", "z_2_3"], 
			["dc_descendant_b_m1_0_1_2", "z_1_2"], 
			["dc_descendant_b_m1_0_1_3", "z_1_3"], 
			["dc_descendant_b_m1_0_2_3", "z_2_3"], 
			["dc_descendant_b_m1_1_1_2", "z_1_2"], 
			["dc_descendant_b_m1_1_1_3", "z_1_3"], 
			["dc_descendant_b_m1_1_2_3", "z_2_3"], 
			["dc_descendant_a_p1_0_1_2", "dc_a_p1_binary_0_2", "z_1_2"], 
			["dc_descendant_a_p1_0_1_3", "dc_a_p1_binary_0_3", "z_1_3"], 
			["dc_descendant_a_p1_0_2_3", "dc_a_p1_binary_0_3", "z_2_3"], 
			["dc_descendant_a_p1_1_1_2", "dc_a_p1_binary_1_2", "z_1_2"], 
			["dc_descendant_a_p1_1_1_3", "dc_a_p1_binary_1_3", "z_1_3"], 
			["dc_descendant_a_p1_1_2_3", "dc_a_p1_binary_1_3", "z_2_3"], 
			["dc_descendant_b_p1_0_1_2", "dc_b_p1_binary_0_2", "z_1_2"], 
			["dc_descendant_b_p1_0_1_3", "dc_b_p1_binary_0_3", "z_1_3"], 
			["dc_descendant_b_p1_0_2_3", "dc_b_p1_binary_0_3", "z_2_3"], 
			["dc_descendant_b_p1_1_1_2", "dc_b_p1_binary_1_2", "z_1_2"], 
			["dc_descendant_b_p1_1_1_3", "dc_b_p1_binary_1_3", "z_1_3"], 
			["dc_descendant_b_p1_1_2_3", "dc_b_p1_binary_1_3", "z_2_3"], 
			["dc_descendant_a_m1_0_1_2", "dc_a_m1_binary_0_2", "z_1_2"], 
			["dc_descendant_a_m1_0_1_3", "dc_a_m1_binary_0_3", "z_1_3"], 
			["dc_descendant_a_m1_0_2_3", "dc_a_m1_binary_0_3", "z_2_3"], 
			["dc_descendant_a_m1_1_1_2", "dc_a_m1_binary_1_2", "z_1_2"], 
			["dc_descendant_a_m1_1_1_3", "dc_a_m1_binary_1_3", "z_1_3"], 
			["dc_descendant_a_m1_1_2_3", "dc_a_m1_binary_1_3", "z_2_3"], 
			["dc_descendant_b_m1_0_1_2", "dc_b_m1_binary_0_2", "z_1_2"], 
			["dc_descendant_b_m1_0_1_3", "dc_b_m1_binary_0_3", "z_1_3"], 
			["dc_descendant_b_m1_0_2_3", "dc_b_m1_binary_0_3", "z_2_3"], 
			["dc_descendant_b_m1_1_1_2", "dc_b_m1_binary_1_2", "z_1_2"], 
			["dc_descendant_b_m1_1_1_3", "dc_b_m1_binary_1_3", "z_1_3"], 
			["dc_descendant_b_m1_1_2_3", "dc_b_m1_binary_1_3", "z_2_3"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0, -1.0]] * (24 * 2) + [[1.0, -1.0, -1.0]] * 24
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_dc_ancestral(self):
		
		# set up variables, only 2 lineages
		sublin_num = 2
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.constraint_dc_ancestral()
		
		self.assertEqual(len(opt.my_rhs), 0)

		# set up variables, 4 lineages
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list,
			self.ssm_spl_list) 
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_dc_ancestral()
		opt.vars_aux_dc()
		opt.vars_z()

		# function to test 
		opt.constraint_dc_ancestral()

		entries = 12

		# rhs
		rhs = [0.0] * entries * 2 + [-1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries * 2 + ["G"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"dc_ancestral_le_dc_binary_a_m1_0_2_1", "dc_ancestral_le_dc_binary_a_m1_0_3_1",
			"dc_ancestral_le_dc_binary_a_m1_0_3_2",
			"dc_ancestral_le_dc_binary_a_m1_1_2_1", "dc_ancestral_le_dc_binary_a_m1_1_3_1",
			"dc_ancestral_le_dc_binary_a_m1_1_3_2",
			"dc_ancestral_le_dc_binary_b_m1_0_2_1", "dc_ancestral_le_dc_binary_b_m1_0_3_1",
			"dc_ancestral_le_dc_binary_b_m1_0_3_2",
			"dc_ancestral_le_dc_binary_b_m1_1_2_1", "dc_ancestral_le_dc_binary_b_m1_1_3_1",
			"dc_ancestral_le_dc_binary_b_m1_1_3_2",

			"dc_ancestral_le_z_a_m1_0_2_1", "dc_ancestral_le_z_a_m1_0_3_1",
			"dc_ancestral_le_z_a_m1_0_3_2",
			"dc_ancestral_le_z_a_m1_1_2_1", "dc_ancestral_le_z_a_m1_1_3_1",
			"dc_ancestral_le_z_a_m1_1_3_2",
			"dc_ancestral_le_z_b_m1_0_2_1", "dc_ancestral_le_z_b_m1_0_3_1",
			"dc_ancestral_le_z_b_m1_0_3_2",
			"dc_ancestral_le_z_b_m1_1_2_1", "dc_ancestral_le_z_b_m1_1_3_1",
			"dc_ancestral_le_z_b_m1_1_3_2",

			"dc_ancestral_ge_dc_binary_z_a_m1_0_2_1", "dc_ancestral_ge_dc_binary_z_a_m1_0_3_1",
			"dc_ancestral_ge_dc_binary_z_a_m1_0_3_2",
			"dc_ancestral_ge_dc_binary_z_a_m1_1_2_1", "dc_ancestral_ge_dc_binary_z_a_m1_1_3_1",
			"dc_ancestral_ge_dc_binary_z_a_m1_1_3_2",
			"dc_ancestral_ge_dc_binary_z_b_m1_0_2_1", "dc_ancestral_ge_dc_binary_z_b_m1_0_3_1",
			"dc_ancestral_ge_dc_binary_z_b_m1_0_3_2",
			"dc_ancestral_ge_dc_binary_z_b_m1_1_2_1", "dc_ancestral_ge_dc_binary_z_b_m1_1_3_1",
			"dc_ancestral_ge_dc_binary_z_b_m1_1_3_2",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		#rows
		rows_vars = [
			["dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_1"], 
			["dc_ancestral_a_m1_0_3_1", "dc_a_m1_binary_0_1"], 
			["dc_ancestral_a_m1_0_3_2", "dc_a_m1_binary_0_2"], 
			["dc_ancestral_a_m1_1_2_1", "dc_a_m1_binary_1_1"], 
			["dc_ancestral_a_m1_1_3_1", "dc_a_m1_binary_1_1"], 
			["dc_ancestral_a_m1_1_3_2", "dc_a_m1_binary_1_2"], 
			["dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_1"], 
			["dc_ancestral_b_m1_0_3_1", "dc_b_m1_binary_0_1"], 
			["dc_ancestral_b_m1_0_3_2", "dc_b_m1_binary_0_2"], 
			["dc_ancestral_b_m1_1_2_1", "dc_b_m1_binary_1_1"], 
			["dc_ancestral_b_m1_1_3_1", "dc_b_m1_binary_1_1"], 
			["dc_ancestral_b_m1_1_3_2", "dc_b_m1_binary_1_2"], 
			["dc_ancestral_a_m1_0_2_1", "z_1_2"], 
			["dc_ancestral_a_m1_0_3_1", "z_1_3"], 
			["dc_ancestral_a_m1_0_3_2", "z_2_3"], 
			["dc_ancestral_a_m1_1_2_1", "z_1_2"], 
			["dc_ancestral_a_m1_1_3_1", "z_1_3"], 
			["dc_ancestral_a_m1_1_3_2", "z_2_3"], 
			["dc_ancestral_b_m1_0_2_1", "z_1_2"], 
			["dc_ancestral_b_m1_0_3_1", "z_1_3"], 
			["dc_ancestral_b_m1_0_3_2", "z_2_3"], 
			["dc_ancestral_b_m1_1_2_1", "z_1_2"], 
			["dc_ancestral_b_m1_1_3_1", "z_1_3"], 
			["dc_ancestral_b_m1_1_3_2", "z_2_3"], 
			["dc_ancestral_a_m1_0_2_1", "dc_a_m1_binary_0_1", "z_1_2"], 
			["dc_ancestral_a_m1_0_3_1", "dc_a_m1_binary_0_1", "z_1_3"], 
			["dc_ancestral_a_m1_0_3_2", "dc_a_m1_binary_0_2", "z_2_3"], 
			["dc_ancestral_a_m1_1_2_1", "dc_a_m1_binary_1_1", "z_1_2"], 
			["dc_ancestral_a_m1_1_3_1", "dc_a_m1_binary_1_1", "z_1_3"], 
			["dc_ancestral_a_m1_1_3_2", "dc_a_m1_binary_1_2", "z_2_3"], 
			["dc_ancestral_b_m1_0_2_1", "dc_b_m1_binary_0_1", "z_1_2"], 
			["dc_ancestral_b_m1_0_3_1", "dc_b_m1_binary_0_1", "z_1_3"], 
			["dc_ancestral_b_m1_0_3_2", "dc_b_m1_binary_0_2", "z_2_3"], 
			["dc_ancestral_b_m1_1_2_1", "dc_b_m1_binary_1_1", "z_1_2"], 
			["dc_ancestral_b_m1_1_3_1", "dc_b_m1_binary_1_1", "z_1_3"], 
			["dc_ancestral_b_m1_1_3_2", "dc_b_m1_binary_1_2", "z_2_3"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0, -1.0]] * 24 + [[1.0, -1.0, -1.0]] * 12
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_z_transitity(self):
		# set up variables
		sublin_num = 5
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_z()
		opt.vars_z_trans()

		# function to test
		opt.constraint_z_transitivity()

		entries = 4

		# rhs
		rhs = ([-1] * entries + [0] * entries + [0] * entries + [-1] * entries + [0] * entries + [0] * entries
			+ [0] * entries + [0] * entries)
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = (["G"] * entries + ["L"] * entries + ["L"] * entries + ["G"] * entries + ["L"] * entries + ["L"] * entries
			+ ["G"] * entries + ["G"] * entries)
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["constraint_z_trans_i_1_2_3_both_z", "constraint_z_trans_i_1_2_4_both_z",
			"constraint_z_trans_i_1_3_4_both_z", "constraint_z_trans_i_2_3_4_both_z",
			"constraint_z_trans_i_1_2_3_z_1_2", "constraint_z_trans_i_1_2_4_z_1_2",
			"constraint_z_trans_i_1_3_4_z_1_3", "constraint_z_trans_i_2_3_4_z_2_3",
			"constraint_z_trans_i_1_2_3_z_2_3", "constraint_z_trans_i_1_2_4_z_2_4",
			"constraint_z_trans_i_1_3_4_z_3_4", "constraint_z_trans_i_2_3_4_z_3_4",
			"constraint_z_trans_c_1_2_3_both_z", "constraint_z_trans_c_1_2_4_both_z",
			"constraint_z_trans_c_1_3_4_both_z", "constraint_z_trans_c_2_3_4_both_z",
			"constraint_z_trans_c_1_2_3_z_2_3", "constraint_z_trans_c_1_2_4_z_2_4",
			"constraint_z_trans_c_1_3_4_z_3_4", "constraint_z_trans_c_2_3_4_z_3_4",
			"constraint_z_trans_c_1_2_3_z_1_3", "constraint_z_trans_c_1_2_4_z_1_4",
			"constraint_z_trans_c_1_3_4_z_1_4", "constraint_z_trans_c_2_3_4_z_2_4",
			"constraint_z_trans_i_z_1_3_2", "constraint_z_trans_i_z_1_4_2",
			"constraint_z_trans_i_z_1_4_3", "constraint_z_trans_i_z_2_4_3",
			"constraint_z_trans_c_z_1_2_3", "constraint_z_trans_c_z_1_2_4",
			"constraint_z_trans_c_z_1_3_4", "constraint_z_trans_c_z_2_3_4"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["z_trans_i_1_2_3", "z_1_2", "z_2_3"],
			["z_trans_i_1_2_4", "z_1_2", "z_2_4"],
			["z_trans_i_1_3_4", "z_1_3", "z_3_4"],
			["z_trans_i_2_3_4", "z_2_3", "z_3_4"],
			["z_trans_i_1_2_3", "z_1_2"],
			["z_trans_i_1_2_4", "z_1_2"],
			["z_trans_i_1_3_4", "z_1_3"],
			["z_trans_i_2_3_4", "z_2_3"],
			["z_trans_i_1_2_3", "z_2_3"],
			["z_trans_i_1_2_4", "z_2_4"],
			["z_trans_i_1_3_4", "z_3_4"],
			["z_trans_i_2_3_4", "z_3_4"],
			["z_trans_c_1_2_3", "z_2_3", "z_1_3"],
			["z_trans_c_1_2_4", "z_2_4", "z_1_4"],
			["z_trans_c_1_3_4", "z_3_4", "z_1_4"],
			["z_trans_c_2_3_4", "z_3_4", "z_2_4"],
			["z_trans_c_1_2_3", "z_2_3"],
			["z_trans_c_1_2_4", "z_2_4"],
			["z_trans_c_1_3_4", "z_3_4"],
			["z_trans_c_2_3_4", "z_3_4"],
			["z_trans_c_1_2_3", "z_1_3"],
			["z_trans_c_1_2_4", "z_1_4"],
			["z_trans_c_1_3_4", "z_1_4"],
			["z_trans_c_2_3_4", "z_2_4"],
			["z_1_3", "z_trans_i_1_2_3"],
			["z_1_4", "z_trans_i_1_2_4"],
			["z_1_4", "z_trans_i_1_3_4"],
			["z_2_4", "z_trans_i_2_3_4"],
			["z_1_2", "z_trans_c_1_2_3"],
			["z_1_2", "z_trans_c_1_2_4"],
			["z_1_3", "z_trans_c_1_3_4"],
			["z_2_3", "z_trans_c_2_3_4"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = ([[1.0, -1.0, -1.0]] * entries + [[1.0, -1.0]] * entries + [[1.0, -1.0]] * entries
			+ [[1.0, -1.0, -1.0]] * entries + [[1.0, -1.0]] * entries + [[1.0, -1.0]] * entries
			+ [[1.0, -1.0]] * entries + [[1.0, -1.0]] * entries)
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_remove_symmetry_z_matrix(self):

		# set up variables
		sublin_num = 4
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_ssm_aux_2_cn()
		opt.vars_z()
		opt.vars_dssm_infl_cnv()
		opt.vars_z_trans()
		opt.vars_ldr_active()

		# function to test
		opt.constraint_remove_symmetry_z_matrix()

		entries = 3

		# rhs
		rhs = [0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["remove_symmetry_z_matrix_1_2", "remove_symmetry_z_matrix_1_3", "remove_symmetry_z_matrix_2_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["z_1_2", "dssm_infl_cnv_a_p1_0_1_2", "dssm_infl_cnv_a_p1_1_1_2",
			"dssm_infl_cnv_b_p1_0_1_2", "dssm_infl_cnv_b_p1_1_1_2",
			"dssm_infl_cnv_a_m1_0_1_2", "dssm_infl_cnv_a_m1_1_1_2",
			"dssm_infl_cnv_b_m1_0_1_2", "dssm_infl_cnv_b_m1_1_1_2",
			"z_trans_c_1_2_3", "LDR_active_1_2"],
			["z_1_3", "dssm_infl_cnv_a_p1_0_1_3", "dssm_infl_cnv_a_p1_1_1_3",
			"dssm_infl_cnv_b_p1_0_1_3", "dssm_infl_cnv_b_p1_1_1_3",
			"dssm_infl_cnv_a_m1_0_1_3", "dssm_infl_cnv_a_m1_1_1_3",
			"dssm_infl_cnv_b_m1_0_1_3", "dssm_infl_cnv_b_m1_1_1_3",
			"z_trans_i_1_2_3", "LDR_active_1_3"],
			["z_2_3", "dssm_infl_cnv_a_p1_0_2_3", "dssm_infl_cnv_a_p1_1_2_3",
			"dssm_infl_cnv_b_p1_0_2_3", "dssm_infl_cnv_b_p1_1_2_3",
			"dssm_infl_cnv_a_m1_0_2_3", "dssm_infl_cnv_a_m1_1_2_3",
			"dssm_infl_cnv_b_m1_0_2_3", "dssm_infl_cnv_b_m1_1_2_3", "LDR_active_2_3"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0] + [-1.0] * 10] + [[1.0] + [-1.0] * 10] + [[1.0] + [-1.0] * 9]
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)
		
		# set up variables with 5 lineages
		sublin_num = 5
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_ssm_aux_2_cn()
		opt.vars_z()
		opt.vars_dssm_infl_cnv()
		opt.vars_z_trans()
		opt.vars_ldr_active()

		# function to test
		opt.constraint_remove_symmetry_z_matrix()

		entries = 6

		# rhs
		rhs = [0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["remove_symmetry_z_matrix_1_2", "remove_symmetry_z_matrix_1_3", "remove_symmetry_z_matrix_1_4",
			"remove_symmetry_z_matrix_2_3", "remove_symmetry_z_matrix_2_4", "remove_symmetry_z_matrix_3_4"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["z_1_2", "dssm_infl_cnv_a_p1_0_1_2", "dssm_infl_cnv_a_p1_1_1_2",
			"dssm_infl_cnv_b_p1_0_1_2", "dssm_infl_cnv_b_p1_1_1_2",
			"dssm_infl_cnv_a_m1_0_1_2", "dssm_infl_cnv_a_m1_1_1_2",
			"dssm_infl_cnv_b_m1_0_1_2", "dssm_infl_cnv_b_m1_1_1_2",
			"z_trans_c_1_2_3", "z_trans_c_1_2_4", "LDR_active_1_2"],
			["z_1_3", "dssm_infl_cnv_a_p1_0_1_3", "dssm_infl_cnv_a_p1_1_1_3",
			"dssm_infl_cnv_b_p1_0_1_3", "dssm_infl_cnv_b_p1_1_1_3",
			"dssm_infl_cnv_a_m1_0_1_3", "dssm_infl_cnv_a_m1_1_1_3",
			"dssm_infl_cnv_b_m1_0_1_3", "dssm_infl_cnv_b_m1_1_1_3",
			"z_trans_c_1_3_4", "z_trans_i_1_2_3", "LDR_active_1_3"],
			["z_1_4", "dssm_infl_cnv_a_p1_0_1_4", "dssm_infl_cnv_a_p1_1_1_4",
			"dssm_infl_cnv_b_p1_0_1_4", "dssm_infl_cnv_b_p1_1_1_4",
			"dssm_infl_cnv_a_m1_0_1_4", "dssm_infl_cnv_a_m1_1_1_4",
			"dssm_infl_cnv_b_m1_0_1_4", "dssm_infl_cnv_b_m1_1_1_4",
			"z_trans_i_1_2_4", "z_trans_i_1_3_4", "LDR_active_1_4"],
			["z_2_3", "dssm_infl_cnv_a_p1_0_2_3", "dssm_infl_cnv_a_p1_1_2_3",
			"dssm_infl_cnv_b_p1_0_2_3", "dssm_infl_cnv_b_p1_1_2_3",
			"dssm_infl_cnv_a_m1_0_2_3", "dssm_infl_cnv_a_m1_1_2_3",
			"dssm_infl_cnv_b_m1_0_2_3", "dssm_infl_cnv_b_m1_1_2_3",
			"z_trans_c_2_3_4", "LDR_active_2_3"],
			["z_2_4", "dssm_infl_cnv_a_p1_0_2_4", "dssm_infl_cnv_a_p1_1_2_4",
			"dssm_infl_cnv_b_p1_0_2_4", "dssm_infl_cnv_b_p1_1_2_4",
			"dssm_infl_cnv_a_m1_0_2_4", "dssm_infl_cnv_a_m1_1_2_4",
			"dssm_infl_cnv_b_m1_0_2_4", "dssm_infl_cnv_b_m1_1_2_4",
			"z_trans_i_2_3_4", "LDR_active_2_4"],
			["z_3_4", "dssm_infl_cnv_a_p1_0_3_4", "dssm_infl_cnv_a_p1_1_3_4",
			"dssm_infl_cnv_b_p1_0_3_4", "dssm_infl_cnv_b_p1_1_3_4",
			"dssm_infl_cnv_a_m1_0_3_4", "dssm_infl_cnv_a_m1_1_3_4",
			"dssm_infl_cnv_b_m1_0_3_4", "dssm_infl_cnv_b_m1_1_3_4", "LDR_active_3_4"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = ([[1.0] + [-1.0] * 11] + [[1.0] + [-1.0] * 11] + [[1.0] + [-1.0] * 11] + [[1.0] + [-1.0] * 10] 
			+ [[1.0] + [-1.0] * 10] + [[1.0] + [-1.0] * 9])
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_dssm_infl_cnv(self):
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		# first to SSMs belong to segment 0, other to 1
		self.ssm_num = 3
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 1
		self.ssm_list = [ssm1, ssm2, ssm3]
		
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_aux_dc()
		opt.vars_three_ssm_matrices()
		opt.vars_dssm_infl_cnv()

		# function to test
		opt.constraint_dssm_infl_cnv(self.ssm_list)

		entries = 3 * self.ssm_num * cons.PHASE_NUMBER * opt.cnv_state_num

		# rhs
		rhs = [0.0] * entries + [0.0] * entries + [-1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["L"] * entries + ["L"] * entries + ["G"] * entries
		self.assertListEqual(opt.my_sense, senses)

		# rownames
		rownames = [
			"dssm_infl_cnv_le_dssm_a_p1_0_1_2", "dssm_infl_cnv_le_dssm_a_p1_0_1_3", 
			"dssm_infl_cnv_le_dssm_a_p1_0_2_3",
			"dssm_infl_cnv_le_dssm_a_p1_1_1_2", "dssm_infl_cnv_le_dssm_a_p1_1_1_3", 
			"dssm_infl_cnv_le_dssm_a_p1_1_2_3",
			"dssm_infl_cnv_le_dssm_a_p1_2_1_2", "dssm_infl_cnv_le_dssm_a_p1_2_1_3", 
			"dssm_infl_cnv_le_dssm_a_p1_2_2_3",
			"dssm_infl_cnv_le_dssm_b_p1_0_1_2", "dssm_infl_cnv_le_dssm_b_p1_0_1_3", 
			"dssm_infl_cnv_le_dssm_b_p1_0_2_3",
			"dssm_infl_cnv_le_dssm_b_p1_1_1_2", "dssm_infl_cnv_le_dssm_b_p1_1_1_3", 
			"dssm_infl_cnv_le_dssm_b_p1_1_2_3",
			"dssm_infl_cnv_le_dssm_b_p1_2_1_2", "dssm_infl_cnv_le_dssm_b_p1_2_1_3", 
			"dssm_infl_cnv_le_dssm_b_p1_2_2_3",
			"dssm_infl_cnv_le_dssm_a_m1_0_1_2", "dssm_infl_cnv_le_dssm_a_m1_0_1_3", 
			"dssm_infl_cnv_le_dssm_a_m1_0_2_3",
			"dssm_infl_cnv_le_dssm_a_m1_1_1_2", "dssm_infl_cnv_le_dssm_a_m1_1_1_3", 
			"dssm_infl_cnv_le_dssm_a_m1_1_2_3",
			"dssm_infl_cnv_le_dssm_a_m1_2_1_2", "dssm_infl_cnv_le_dssm_a_m1_2_1_3", 
			"dssm_infl_cnv_le_dssm_a_m1_2_2_3",
			"dssm_infl_cnv_le_dssm_b_m1_0_1_2", "dssm_infl_cnv_le_dssm_b_m1_0_1_3", 
			"dssm_infl_cnv_le_dssm_b_m1_0_2_3",
			"dssm_infl_cnv_le_dssm_b_m1_1_1_2", "dssm_infl_cnv_le_dssm_b_m1_1_1_3", 
			"dssm_infl_cnv_le_dssm_b_m1_1_2_3",
			"dssm_infl_cnv_le_dssm_b_m1_2_1_2", "dssm_infl_cnv_le_dssm_b_m1_2_1_3", 
			"dssm_infl_cnv_le_dssm_b_m1_2_2_3",

			"dssm_infl_cnv_le_dc_binary_a_p1_0_1_2", "dssm_infl_cnv_le_dc_binary_a_p1_0_1_3",
			"dssm_infl_cnv_le_dc_binary_a_p1_0_2_3",
			"dssm_infl_cnv_le_dc_binary_a_p1_1_1_2", "dssm_infl_cnv_le_dc_binary_a_p1_1_1_3",
			"dssm_infl_cnv_le_dc_binary_a_p1_1_2_3",
			"dssm_infl_cnv_le_dc_binary_a_p1_2_1_2", "dssm_infl_cnv_le_dc_binary_a_p1_2_1_3",
			"dssm_infl_cnv_le_dc_binary_a_p1_2_2_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_0_1_2", "dssm_infl_cnv_le_dc_binary_b_p1_0_1_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_0_2_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_1_1_2", "dssm_infl_cnv_le_dc_binary_b_p1_1_1_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_1_2_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_2_1_2", "dssm_infl_cnv_le_dc_binary_b_p1_2_1_3",
			"dssm_infl_cnv_le_dc_binary_b_p1_2_2_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_0_1_2", "dssm_infl_cnv_le_dc_binary_a_m1_0_1_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_0_2_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_1_1_2", "dssm_infl_cnv_le_dc_binary_a_m1_1_1_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_1_2_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_2_1_2", "dssm_infl_cnv_le_dc_binary_a_m1_2_1_3",
			"dssm_infl_cnv_le_dc_binary_a_m1_2_2_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_0_1_2", "dssm_infl_cnv_le_dc_binary_b_m1_0_1_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_0_2_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_1_1_2", "dssm_infl_cnv_le_dc_binary_b_m1_1_1_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_1_2_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_2_1_2", "dssm_infl_cnv_le_dc_binary_b_m1_2_1_3",
			"dssm_infl_cnv_le_dc_binary_b_m1_2_2_3",

			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_0_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_0_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_0_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_1_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_1_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_1_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_2_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_2_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1_2_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_0_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_0_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_0_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_1_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_1_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_1_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_2_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_2_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1_2_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_0_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_0_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_0_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_1_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_1_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_1_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_2_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_2_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1_2_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_0_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_0_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_0_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_1_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_1_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_1_2_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_2_1_2", 
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_2_1_3",
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1_2_2_3"
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dssm_infl_cnv_a_p1_0_1_2", "dssm_a_0_1"],
			["dssm_infl_cnv_a_p1_0_1_3", "dssm_a_0_1"],
			["dssm_infl_cnv_a_p1_0_2_3", "dssm_a_0_2"],
			["dssm_infl_cnv_a_p1_1_1_2", "dssm_a_1_1"],
			["dssm_infl_cnv_a_p1_1_1_3", "dssm_a_1_1"],
			["dssm_infl_cnv_a_p1_1_2_3", "dssm_a_1_2"],
			["dssm_infl_cnv_a_p1_2_1_2", "dssm_a_2_1"],
			["dssm_infl_cnv_a_p1_2_1_3", "dssm_a_2_1"],
			["dssm_infl_cnv_a_p1_2_2_3", "dssm_a_2_2"],
			["dssm_infl_cnv_b_p1_0_1_2", "dssm_b_0_1"],
			["dssm_infl_cnv_b_p1_0_1_3", "dssm_b_0_1"],
			["dssm_infl_cnv_b_p1_0_2_3", "dssm_b_0_2"],
			["dssm_infl_cnv_b_p1_1_1_2", "dssm_b_1_1"],
			["dssm_infl_cnv_b_p1_1_1_3", "dssm_b_1_1"],
			["dssm_infl_cnv_b_p1_1_2_3", "dssm_b_1_2"],
			["dssm_infl_cnv_b_p1_2_1_2", "dssm_b_2_1"],
			["dssm_infl_cnv_b_p1_2_1_3", "dssm_b_2_1"],
			["dssm_infl_cnv_b_p1_2_2_3", "dssm_b_2_2"],
			["dssm_infl_cnv_a_m1_0_1_2", "dssm_a_0_1"],
			["dssm_infl_cnv_a_m1_0_1_3", "dssm_a_0_1"],
			["dssm_infl_cnv_a_m1_0_2_3", "dssm_a_0_2"],
			["dssm_infl_cnv_a_m1_1_1_2", "dssm_a_1_1"],
			["dssm_infl_cnv_a_m1_1_1_3", "dssm_a_1_1"],
			["dssm_infl_cnv_a_m1_1_2_3", "dssm_a_1_2"],
			["dssm_infl_cnv_a_m1_2_1_2", "dssm_a_2_1"],
			["dssm_infl_cnv_a_m1_2_1_3", "dssm_a_2_1"],
			["dssm_infl_cnv_a_m1_2_2_3", "dssm_a_2_2"],
			["dssm_infl_cnv_b_m1_0_1_2", "dssm_b_0_1"],
			["dssm_infl_cnv_b_m1_0_1_3", "dssm_b_0_1"],
			["dssm_infl_cnv_b_m1_0_2_3", "dssm_b_0_2"],
			["dssm_infl_cnv_b_m1_1_1_2", "dssm_b_1_1"],
			["dssm_infl_cnv_b_m1_1_1_3", "dssm_b_1_1"],
			["dssm_infl_cnv_b_m1_1_2_3", "dssm_b_1_2"],
			["dssm_infl_cnv_b_m1_2_1_2", "dssm_b_2_1"],
			["dssm_infl_cnv_b_m1_2_1_3", "dssm_b_2_1"],
			["dssm_infl_cnv_b_m1_2_2_3", "dssm_b_2_2"],

			["dssm_infl_cnv_a_p1_0_1_2", "dc_a_p1_binary_0_2"], 
			["dssm_infl_cnv_a_p1_0_1_3", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_0_2_3", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_1_1_2", "dc_a_p1_binary_0_2"], 
			["dssm_infl_cnv_a_p1_1_1_3", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_1_2_3", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_2_1_2", "dc_a_p1_binary_1_2"], 
			["dssm_infl_cnv_a_p1_2_1_3", "dc_a_p1_binary_1_3"], 
			["dssm_infl_cnv_a_p1_2_2_3", "dc_a_p1_binary_1_3"], 
			["dssm_infl_cnv_b_p1_0_1_2", "dc_b_p1_binary_0_2"], 
			["dssm_infl_cnv_b_p1_0_1_3", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_0_2_3", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_1_1_2", "dc_b_p1_binary_0_2"], 
			["dssm_infl_cnv_b_p1_1_1_3", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_1_2_3", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_2_1_2", "dc_b_p1_binary_1_2"], 
			["dssm_infl_cnv_b_p1_2_1_3", "dc_b_p1_binary_1_3"], 
			["dssm_infl_cnv_b_p1_2_2_3", "dc_b_p1_binary_1_3"], 
			["dssm_infl_cnv_a_m1_0_1_2", "dc_a_m1_binary_0_2"], 
			["dssm_infl_cnv_a_m1_0_1_3", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_0_2_3", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_1_1_2", "dc_a_m1_binary_0_2"], 
			["dssm_infl_cnv_a_m1_1_1_3", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_1_2_3", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_2_1_2", "dc_a_m1_binary_1_2"], 
			["dssm_infl_cnv_a_m1_2_1_3", "dc_a_m1_binary_1_3"], 
			["dssm_infl_cnv_a_m1_2_2_3", "dc_a_m1_binary_1_3"], 
			["dssm_infl_cnv_b_m1_0_1_2", "dc_b_m1_binary_0_2"], 
			["dssm_infl_cnv_b_m1_0_1_3", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_0_2_3", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_1_1_2", "dc_b_m1_binary_0_2"], 
			["dssm_infl_cnv_b_m1_1_1_3", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_1_2_3", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_2_1_2", "dc_b_m1_binary_1_2"], 
			["dssm_infl_cnv_b_m1_2_1_3", "dc_b_m1_binary_1_3"], 
			["dssm_infl_cnv_b_m1_2_2_3", "dc_b_m1_binary_1_3"], 

			["dssm_infl_cnv_a_p1_0_1_2", "dssm_a_0_1", "dc_a_p1_binary_0_2"], 
			["dssm_infl_cnv_a_p1_0_1_3", "dssm_a_0_1", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_0_2_3", "dssm_a_0_2", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_1_1_2", "dssm_a_1_1", "dc_a_p1_binary_0_2"], 
			["dssm_infl_cnv_a_p1_1_1_3", "dssm_a_1_1", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_1_2_3", "dssm_a_1_2", "dc_a_p1_binary_0_3"], 
			["dssm_infl_cnv_a_p1_2_1_2", "dssm_a_2_1", "dc_a_p1_binary_1_2"], 
			["dssm_infl_cnv_a_p1_2_1_3", "dssm_a_2_1", "dc_a_p1_binary_1_3"], 
			["dssm_infl_cnv_a_p1_2_2_3", "dssm_a_2_2", "dc_a_p1_binary_1_3"], 
			["dssm_infl_cnv_b_p1_0_1_2", "dssm_b_0_1", "dc_b_p1_binary_0_2"], 
			["dssm_infl_cnv_b_p1_0_1_3", "dssm_b_0_1", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_0_2_3", "dssm_b_0_2", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_1_1_2", "dssm_b_1_1", "dc_b_p1_binary_0_2"], 
			["dssm_infl_cnv_b_p1_1_1_3", "dssm_b_1_1", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_1_2_3", "dssm_b_1_2", "dc_b_p1_binary_0_3"], 
			["dssm_infl_cnv_b_p1_2_1_2", "dssm_b_2_1", "dc_b_p1_binary_1_2"], 
			["dssm_infl_cnv_b_p1_2_1_3", "dssm_b_2_1", "dc_b_p1_binary_1_3"], 
			["dssm_infl_cnv_b_p1_2_2_3", "dssm_b_2_2", "dc_b_p1_binary_1_3"], 
			["dssm_infl_cnv_a_m1_0_1_2", "dssm_a_0_1", "dc_a_m1_binary_0_2"], 
			["dssm_infl_cnv_a_m1_0_1_3", "dssm_a_0_1", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_0_2_3", "dssm_a_0_2", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_1_1_2", "dssm_a_1_1", "dc_a_m1_binary_0_2"], 
			["dssm_infl_cnv_a_m1_1_1_3", "dssm_a_1_1", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_1_2_3", "dssm_a_1_2", "dc_a_m1_binary_0_3"], 
			["dssm_infl_cnv_a_m1_2_1_2", "dssm_a_2_1", "dc_a_m1_binary_1_2"], 
			["dssm_infl_cnv_a_m1_2_1_3", "dssm_a_2_1", "dc_a_m1_binary_1_3"], 
			["dssm_infl_cnv_a_m1_2_2_3", "dssm_a_2_2", "dc_a_m1_binary_1_3"], 
			["dssm_infl_cnv_b_m1_0_1_2", "dssm_b_0_1", "dc_b_m1_binary_0_2"], 
			["dssm_infl_cnv_b_m1_0_1_3", "dssm_b_0_1", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_0_2_3", "dssm_b_0_2", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_1_1_2", "dssm_b_1_1", "dc_b_m1_binary_0_2"], 
			["dssm_infl_cnv_b_m1_1_1_3", "dssm_b_1_1", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_1_2_3", "dssm_b_1_2", "dc_b_m1_binary_0_3"], 
			["dssm_infl_cnv_b_m1_2_1_2", "dssm_b_2_1", "dc_b_m1_binary_1_2"], 
			["dssm_infl_cnv_b_m1_2_1_3", "dssm_b_2_1", "dc_b_m1_binary_1_3"], 
			["dssm_infl_cnv_b_m1_2_2_3", "dssm_b_2_2", "dc_b_m1_binary_1_3"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)
	
		rows_values = [[1.0, -1.0]] * (36 * 2) + [[1.0, -1.0, -1.0]] * 36
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)
	
	def test_constraint_CN_direction(self):
		sublin_num = 4
		self.seg_num = 3
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list = [spl_seg_1, spl_seg_1, spl_seg_1]
		seg1 = segment.Segment_allele_specific(1, 1, 1, 0.1, 0.25, 1.5, 0.25)
		seg2 = segment.Segment_allele_specific(1, 1, 1, 2.1, 0.25, 0.5, 0.25)
		seg3 = segment.Segment_allele_specific(1, 1, 1, 1.0, 0.25, 1.0, 0.25)
		self.seg_list = [seg1, seg2, seg3]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list, seg_splines_B=seg_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test
		opt.constraint_CN_direction(self.seg_list)

		entries = 12

		# rhs 
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * entries
		self.assertListEqual(opt.my_sense, s)

		rownames = [
			"CN_direction_no_gain_A_0_1", "CN_direction_no_gain_A_0_2", "CN_direction_no_gain_A_0_3",
			"CN_direction_no_loss_A_1_1", "CN_direction_no_loss_A_1_2", "CN_direction_no_loss_A_1_3",
			"CN_direction_no_loss_B_0_1", "CN_direction_no_loss_B_0_2", "CN_direction_no_loss_B_0_3",
			"CN_direction_no_gain_B_1_1", "CN_direction_no_gain_B_1_2", "CN_direction_no_gain_B_1_3",
			]
		self.assertListEqual(opt.my_rownames, rownames)


		rows_vars = [
			["dc_a_p1_binary_0_1"], ["dc_a_p1_binary_0_2"], ["dc_a_p1_binary_0_3"],
			["dc_a_m1_binary_1_1"], ["dc_a_m1_binary_1_2"], ["dc_a_m1_binary_1_3"],
			["dc_b_m1_binary_0_1"], ["dc_b_m1_binary_0_2"], ["dc_b_m1_binary_0_3"],
			["dc_b_p1_binary_1_1"], ["dc_b_p1_binary_1_2"], ["dc_b_p1_binary_1_3"]
			]
		self.assertEqual(self.get_opt_rows_vars(opt), rows_vars)
		
		rows_values = [[1.0] for _ in xrange(entries)]
		self.assertEqual(self.get_opt_rows_values(opt), rows_values)

		

	def test_constraint_only_gains_losses_LOH(self):

		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test
		opt.constraint_only_gains_losses_LOH()

		entries = 3 * 2 * 2 * 4
		
		# rhs 
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"only_gains_losses_LOH_a_a_0_1_2", "only_gains_losses_LOH_a_a_0_1_3",
			"only_gains_losses_LOH_a_a_0_2_1", "only_gains_losses_LOH_a_a_0_2_3",
			"only_gains_losses_LOH_a_a_0_3_1", "only_gains_losses_LOH_a_a_0_3_2",
			"only_gains_losses_LOH_a_a_1_1_2", "only_gains_losses_LOH_a_a_1_1_3",
			"only_gains_losses_LOH_a_a_1_2_1", "only_gains_losses_LOH_a_a_1_2_3",
			"only_gains_losses_LOH_a_a_1_3_1", "only_gains_losses_LOH_a_a_1_3_2",
			"only_gains_losses_LOH_a_b_0_1_2", "only_gains_losses_LOH_a_b_0_1_3",
			"only_gains_losses_LOH_a_b_0_2_1", "only_gains_losses_LOH_a_b_0_2_3",
			"only_gains_losses_LOH_a_b_0_3_1", "only_gains_losses_LOH_a_b_0_3_2",
			"only_gains_losses_LOH_a_b_1_1_2", "only_gains_losses_LOH_a_b_1_1_3",
			"only_gains_losses_LOH_a_b_1_2_1", "only_gains_losses_LOH_a_b_1_2_3",
			"only_gains_losses_LOH_a_b_1_3_1", "only_gains_losses_LOH_a_b_1_3_2",
			"only_gains_losses_LOH_b_a_0_1_2", "only_gains_losses_LOH_b_a_0_1_3",
			"only_gains_losses_LOH_b_a_0_2_1", "only_gains_losses_LOH_b_a_0_2_3",
			"only_gains_losses_LOH_b_a_0_3_1", "only_gains_losses_LOH_b_a_0_3_2",
			"only_gains_losses_LOH_b_a_1_1_2", "only_gains_losses_LOH_b_a_1_1_3",
			"only_gains_losses_LOH_b_a_1_2_1", "only_gains_losses_LOH_b_a_1_2_3",
			"only_gains_losses_LOH_b_a_1_3_1", "only_gains_losses_LOH_b_a_1_3_2",
			"only_gains_losses_LOH_b_b_0_1_2", "only_gains_losses_LOH_b_b_0_1_3",
			"only_gains_losses_LOH_b_b_0_2_1", "only_gains_losses_LOH_b_b_0_2_3",
			"only_gains_losses_LOH_b_b_0_3_1", "only_gains_losses_LOH_b_b_0_3_2",
			"only_gains_losses_LOH_b_b_1_1_2", "only_gains_losses_LOH_b_b_1_1_3",
			"only_gains_losses_LOH_b_b_1_2_1", "only_gains_losses_LOH_b_b_1_2_3",
			"only_gains_losses_LOH_b_b_1_3_1", "only_gains_losses_LOH_b_b_1_3_2",
			]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_a_p1_binary_0_1", "dc_a_m1_binary_0_2"],
			["dc_a_p1_binary_0_1", "dc_a_m1_binary_0_3"],
			["dc_a_p1_binary_0_2", "dc_a_m1_binary_0_1"],
			["dc_a_p1_binary_0_2", "dc_a_m1_binary_0_3"],
			["dc_a_p1_binary_0_3", "dc_a_m1_binary_0_1"],
			["dc_a_p1_binary_0_3", "dc_a_m1_binary_0_2"],
			["dc_a_p1_binary_1_1", "dc_a_m1_binary_1_2"],
			["dc_a_p1_binary_1_1", "dc_a_m1_binary_1_3"],
			["dc_a_p1_binary_1_2", "dc_a_m1_binary_1_1"],
			["dc_a_p1_binary_1_2", "dc_a_m1_binary_1_3"],
			["dc_a_p1_binary_1_3", "dc_a_m1_binary_1_1"],
			["dc_a_p1_binary_1_3", "dc_a_m1_binary_1_2"],
			["dc_a_p1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dc_a_p1_binary_0_1", "dc_b_m1_binary_0_3"],
			["dc_a_p1_binary_0_2", "dc_b_m1_binary_0_1"],
			["dc_a_p1_binary_0_2", "dc_b_m1_binary_0_3"],
			["dc_a_p1_binary_0_3", "dc_b_m1_binary_0_1"],
			["dc_a_p1_binary_0_3", "dc_b_m1_binary_0_2"],
			["dc_a_p1_binary_1_1", "dc_b_m1_binary_1_2"],
			["dc_a_p1_binary_1_1", "dc_b_m1_binary_1_3"],
			["dc_a_p1_binary_1_2", "dc_b_m1_binary_1_1"],
			["dc_a_p1_binary_1_2", "dc_b_m1_binary_1_3"],
			["dc_a_p1_binary_1_3", "dc_b_m1_binary_1_1"],
			["dc_a_p1_binary_1_3", "dc_b_m1_binary_1_2"],
			["dc_b_p1_binary_0_1", "dc_a_m1_binary_0_2"],
			["dc_b_p1_binary_0_1", "dc_a_m1_binary_0_3"],
			["dc_b_p1_binary_0_2", "dc_a_m1_binary_0_1"],
			["dc_b_p1_binary_0_2", "dc_a_m1_binary_0_3"],
			["dc_b_p1_binary_0_3", "dc_a_m1_binary_0_1"],
			["dc_b_p1_binary_0_3", "dc_a_m1_binary_0_2"],
			["dc_b_p1_binary_1_1", "dc_a_m1_binary_1_2"],
			["dc_b_p1_binary_1_1", "dc_a_m1_binary_1_3"],
			["dc_b_p1_binary_1_2", "dc_a_m1_binary_1_1"],
			["dc_b_p1_binary_1_2", "dc_a_m1_binary_1_3"],
			["dc_b_p1_binary_1_3", "dc_a_m1_binary_1_1"],
			["dc_b_p1_binary_1_3", "dc_a_m1_binary_1_2"],
			["dc_b_p1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dc_b_p1_binary_0_1", "dc_b_m1_binary_0_3"],
			["dc_b_p1_binary_0_2", "dc_b_m1_binary_0_1"],
			["dc_b_p1_binary_0_2", "dc_b_m1_binary_0_3"],
			["dc_b_p1_binary_0_3", "dc_b_m1_binary_0_1"],
			["dc_b_p1_binary_0_3", "dc_b_m1_binary_0_2"],
			["dc_b_p1_binary_1_1", "dc_b_m1_binary_1_2"],
			["dc_b_p1_binary_1_1", "dc_b_m1_binary_1_3"],
			["dc_b_p1_binary_1_2", "dc_b_m1_binary_1_1"],
			["dc_b_p1_binary_1_2", "dc_b_m1_binary_1_3"],
			["dc_b_p1_binary_1_3", "dc_b_m1_binary_1_1"],
			["dc_b_p1_binary_1_3", "dc_b_m1_binary_1_2"],
			]
		self.assertEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			[1.0, 1.0],
			]
		self.assertEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_no_gains_or_losses_of_lost_chromatids(self):
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()
		opt.vars_dc_ancestral()

		# function to test
		opt.constraint_no_gains_or_losses_of_lost_chromatids()

		entries = 2 * 2 * 2

		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)
	
		# rownames 
		rownames = ["no_gains_or_losses_of_lost_chromatids_a_0_2", "no_gains_or_losses_of_lost_chromatids_a_0_3",
			"no_gains_or_losses_of_lost_chromatids_a_1_2", "no_gains_or_losses_of_lost_chromatids_a_1_3",
			"no_gains_or_losses_of_lost_chromatids_b_0_2", "no_gains_or_losses_of_lost_chromatids_b_0_3",
			"no_gains_or_losses_of_lost_chromatids_b_1_2", "no_gains_or_losses_of_lost_chromatids_b_1_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_ancestral_a_m1_0_2_1", "dc_a_p1_binary_0_2", "dc_a_m1_binary_0_2"],
			["dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2", "dc_a_p1_binary_0_3", "dc_a_m1_binary_0_3"], 
			["dc_ancestral_a_m1_1_2_1", "dc_a_p1_binary_1_2", "dc_a_m1_binary_1_2"],
			["dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2", "dc_a_p1_binary_1_3", "dc_a_m1_binary_1_3"], 
			["dc_ancestral_b_m1_0_2_1", "dc_b_p1_binary_0_2", "dc_b_m1_binary_0_2"],
			["dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2", "dc_b_p1_binary_0_3", "dc_b_m1_binary_0_3"], 
			["dc_ancestral_b_m1_1_2_1", "dc_b_p1_binary_1_2", "dc_b_m1_binary_1_2"],
			["dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2", "dc_b_p1_binary_1_3", "dc_b_m1_binary_1_3"], 
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)
	
		rows_values = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]] * 4
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)
	
	def test_constraint_no_gains_of_lost_chromatids(self):
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()
		opt.vars_dc_ancestral()

		# function to test
		opt.constraint_no_gains_of_lost_chromatids()

		entries = 2 * 2 * 2

		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)
	
		# rownames 
		rownames = ["no_gains_of_lost_chromatids_a_0_2", "no_gains_of_lost_chromatids_a_0_3",
			"no_gains_of_lost_chromatids_a_1_2", "no_gains_of_lost_chromatids_a_1_3",
			"no_gains_of_lost_chromatids_b_0_2", "no_gains_of_lost_chromatids_b_0_3",
			"no_gains_of_lost_chromatids_b_1_2", "no_gains_of_lost_chromatids_b_1_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_ancestral_a_m1_0_2_1", "dc_a_p1_binary_0_2"],
			["dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2", "dc_a_p1_binary_0_3"],
			["dc_ancestral_a_m1_1_2_1", "dc_a_p1_binary_1_2"],
			["dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2", "dc_a_p1_binary_1_3"],
			["dc_ancestral_b_m1_0_2_1", "dc_b_p1_binary_0_2"],
			["dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2", "dc_b_p1_binary_0_3"],
			["dc_ancestral_b_m1_1_2_1", "dc_b_p1_binary_1_2"],
			["dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2", "dc_b_p1_binary_1_3"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)
		
		rows_values = [[1.0, 1.0], [1.0, 1.0, 1.0]] * 4
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)
		
	
	def test_constraint_loss_per_chromatid(self):
		
		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test
		opt.constraint_loss_per_chromatid()

		entries = 4

		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)
		
		# rownames
		rownames = ["loss_per_chromatid_a_0", "loss_per_chromatid_a_1", "loss_per_chromatid_b_0", 
			"loss_per_chromatid_b_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2"],
			["dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2"],
			["dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dc_b_m1_binary_1_0", "dc_b_m1_binary_1_1", "dc_b_m1_binary_1_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0, 1.0, 1.0]] * 4
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraint_no_simultaneous_loss_and_gain(self):

		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test 
		opt.constraint_no_simultaneous_loss_and_gain()

		entries = 2 * 2 * 2

		#rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)
		
		# rownames
		rownames = ["no_simultaneous_loss_and_gain_a_0_1","no_simultaneous_loss_and_gain_a_0_2",
			"no_simultaneous_loss_and_gain_a_1_1","no_simultaneous_loss_and_gain_a_1_2",
			"no_simultaneous_loss_and_gain_b_0_1","no_simultaneous_loss_and_gain_b_0_2",
			"no_simultaneous_loss_and_gain_b_1_1","no_simultaneous_loss_and_gain_b_1_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_a_p1_binary_0_1", "dc_a_m1_binary_0_1"],
			["dc_a_p1_binary_0_2", "dc_a_m1_binary_0_2"],
			["dc_a_p1_binary_1_1", "dc_a_m1_binary_1_1"],
			["dc_a_p1_binary_1_2", "dc_a_m1_binary_1_2"],
			["dc_b_p1_binary_0_1", "dc_b_m1_binary_0_1"],
			["dc_b_p1_binary_0_2", "dc_b_m1_binary_0_2"],
			["dc_b_p1_binary_1_1", "dc_b_m1_binary_1_1"],
			["dc_b_p1_binary_1_2", "dc_b_m1_binary_1_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0, 1.0]] * 8
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

		


	def test_constraint_max_cn_changes(self):

		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		max_cn_changes = 4.0
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list,
			simple_CN_changes=False, max_x_CN_changes=max_cn_changes)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test 
		opt.constraint_max_cn_changes()

		# rhs
		rhs = [max_cn_changes] * self.seg_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * self.seg_num
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["max_cn_changes_0", "max_cn_changes_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["dc_a_p1_binary_0_0", "dc_a_p1_binary_0_1", "dc_a_p1_binary_0_2", 
			"dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2",
			"dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2",
			"dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"],
			["dc_a_p1_binary_1_0", "dc_a_p1_binary_1_1", "dc_a_p1_binary_1_2",
			"dc_b_p1_binary_1_0", "dc_b_p1_binary_1_1", "dc_b_p1_binary_1_2",
			"dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2",
			"dc_b_m1_binary_1_0", "dc_b_m1_binary_1_1", "dc_b_m1_binary_1_2"],
			]
		self.assertEqual(self.get_opt_rows_vars(opt), rows_vars)	

		rows_values = [[1.0] * 12] * 2
		self.assertEqual(self.get_opt_rows_values(opt), rows_values)	

	
	def test_constraint_no_cnv_in_lineage_0(self):

		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		# function to test 
		opt.constraint_no_cnv_in_lineage_0()

		# rhs
		rhs = [0.0] * self.seg_num
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * self.seg_num
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["no_CNV_in_first_lineage_0", "no_CNV_in_first_lineage_1"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows_vars = [[opt.my_colnames_dc_a_p1_binary[0][0], 
			opt.my_colnames_dc_b_p1_binary[0][0],
			opt.my_colnames_dc_a_m1_binary[0][0], opt.my_colnames_dc_b_m1_binary[0][0]],
			[opt.my_colnames_dc_a_p1_binary[1][0], opt.my_colnames_dc_b_p1_binary[1][0],
			opt.my_colnames_dc_a_m1_binary[1][0], opt.my_colnames_dc_b_m1_binary[1][0]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0] * 4] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)


	def test_constraint_delta_c_allowed_entries_per_row(self):
		
		# set up variables
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_aux_dc()

		entries_2 = (opt.sublin_num - 1) * opt.seg_num * opt.cnv_state_num

		# function to test
		opt.constraint_delta_c_allowed_entries_per_row()

		#rhs
		rhs = [1.0] * (opt.seg_num * opt.cnv_state_num * 2 + entries_2)
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * (opt.seg_num * opt.cnv_state_num *  2 + entries_2)
		self.assertListEqual(opt.my_sense, s)  

		# rownames
		rownames = ["dc_row_same_state_p1_0", "dc_row_same_state_p1_1", "dc_row_same_state_m1_0", 
			"dc_row_same_state_m1_1",
			"dc_row_same_chromatid_a_0", "dc_row_same_chromatid_a_1", "dc_row_same_chromatid_b_0", 
			"dc_row_same_chromatid_b_1",
			"dc_row_different_a_p1_0_1", "dc_row_different_a_p1_0_2", "dc_row_different_a_p1_0_3",
			"dc_row_different_a_p1_1_1", "dc_row_different_a_p1_1_2", "dc_row_different_a_p1_1_3",
			"dc_row_different_b_p1_0_1", "dc_row_different_b_p1_0_2", "dc_row_different_b_p1_0_3",
			"dc_row_different_b_p1_1_1", "dc_row_different_b_p1_1_2", "dc_row_different_b_p1_1_3"]			
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows_vars = [
			# dc_row_same_state
			[opt.my_colnames_dc_a_p1_binary[0][0], opt.my_colnames_dc_a_p1_binary[0][1], 
			opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_a_p1_binary[0][3],
			opt.my_colnames_dc_b_p1_binary[0][0],
			opt.my_colnames_dc_b_p1_binary[0][1], opt.my_colnames_dc_b_p1_binary[0][2],
			opt.my_colnames_dc_b_p1_binary[0][3]],
			[opt.my_colnames_dc_a_p1_binary[1][0], opt.my_colnames_dc_a_p1_binary[1][1], 
			opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_a_p1_binary[1][3],
			opt.my_colnames_dc_b_p1_binary[1][0],
			opt.my_colnames_dc_b_p1_binary[1][1], opt.my_colnames_dc_b_p1_binary[1][2],
			opt.my_colnames_dc_b_p1_binary[1][3]],
			[opt.my_colnames_dc_a_m1_binary[0][0], opt.my_colnames_dc_a_m1_binary[0][1], 
			opt.my_colnames_dc_a_m1_binary[0][2], opt.my_colnames_dc_a_m1_binary[0][3],
			opt.my_colnames_dc_b_m1_binary[0][0],
			opt.my_colnames_dc_b_m1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
			opt.my_colnames_dc_b_m1_binary[0][3]],
			[opt.my_colnames_dc_a_m1_binary[1][0], opt.my_colnames_dc_a_m1_binary[1][1], 
			opt.my_colnames_dc_a_m1_binary[1][2], opt.my_colnames_dc_a_m1_binary[1][3],
			opt.my_colnames_dc_b_m1_binary[1][0],
			opt.my_colnames_dc_b_m1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
			opt.my_colnames_dc_b_m1_binary[1][3]],
			# dc_row_same_chromatid
			[opt.my_colnames_dc_a_p1_binary[0][0], opt.my_colnames_dc_a_p1_binary[0][1], 
			opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_a_p1_binary[0][3],
			opt.my_colnames_dc_a_m1_binary[0][0],
			opt.my_colnames_dc_a_m1_binary[0][1], opt.my_colnames_dc_a_m1_binary[0][2],
			opt.my_colnames_dc_a_m1_binary[0][3]],
			[opt.my_colnames_dc_a_p1_binary[1][0], opt.my_colnames_dc_a_p1_binary[1][1], 
			opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_a_p1_binary[1][3],
			opt.my_colnames_dc_a_m1_binary[1][0],
			opt.my_colnames_dc_a_m1_binary[1][1], opt.my_colnames_dc_a_m1_binary[1][2],
			opt.my_colnames_dc_a_m1_binary[1][3]],
			[opt.my_colnames_dc_b_p1_binary[0][0], opt.my_colnames_dc_b_p1_binary[0][1], 
			opt.my_colnames_dc_b_p1_binary[0][2], opt.my_colnames_dc_b_p1_binary[0][3],
			opt.my_colnames_dc_b_m1_binary[0][0],
			opt.my_colnames_dc_b_m1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
			opt.my_colnames_dc_b_m1_binary[0][3]],
			[opt.my_colnames_dc_b_p1_binary[1][0], opt.my_colnames_dc_b_p1_binary[1][1], 
			opt.my_colnames_dc_b_p1_binary[1][2], opt.my_colnames_dc_b_p1_binary[1][3],
			opt.my_colnames_dc_b_m1_binary[1][0],
			opt.my_colnames_dc_b_m1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
			opt.my_colnames_dc_b_m1_binary[1][3]],
			# dc_row_different
			[opt.my_colnames_dc_a_p1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
			opt.my_colnames_dc_b_m1_binary[0][3]],
			[opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_b_m1_binary[0][1],
			opt.my_colnames_dc_b_m1_binary[0][3]],
			[opt.my_colnames_dc_a_p1_binary[0][3], opt.my_colnames_dc_b_m1_binary[0][1],
			opt.my_colnames_dc_b_m1_binary[0][2]],
			[opt.my_colnames_dc_a_p1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
			opt.my_colnames_dc_b_m1_binary[1][3]], 
			[opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_b_m1_binary[1][1],
			opt.my_colnames_dc_b_m1_binary[1][3]], 
			[opt.my_colnames_dc_a_p1_binary[1][3], opt.my_colnames_dc_b_m1_binary[1][1],
			opt.my_colnames_dc_b_m1_binary[1][2]], 
			[opt.my_colnames_dc_b_p1_binary[0][1], opt.my_colnames_dc_a_m1_binary[0][2],
			opt.my_colnames_dc_a_m1_binary[0][3]], 
			[opt.my_colnames_dc_b_p1_binary[0][2], opt.my_colnames_dc_a_m1_binary[0][1],
			opt.my_colnames_dc_a_m1_binary[0][3]], 
			[opt.my_colnames_dc_b_p1_binary[0][3], opt.my_colnames_dc_a_m1_binary[0][1],
			opt.my_colnames_dc_a_m1_binary[0][2]], 
			[opt.my_colnames_dc_b_p1_binary[1][1], opt.my_colnames_dc_a_m1_binary[1][2],
			opt.my_colnames_dc_a_m1_binary[1][3]], 
			[opt.my_colnames_dc_b_p1_binary[1][2], opt.my_colnames_dc_a_m1_binary[1][1],
			opt.my_colnames_dc_a_m1_binary[1][3]], 
			[opt.my_colnames_dc_b_p1_binary[1][3], opt.my_colnames_dc_a_m1_binary[1][1],
			opt.my_colnames_dc_a_m1_binary[1][2]],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = (
			# dc_row_same_state
			[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]] * 8 +
			# dc_row_different
			[[1.0, 1.0, 1.0]] * 12)
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

		#rows = [
		#	# dc_row_same_state
		#	[[opt.my_colnames_dc_a_p1_binary[0][0], opt.my_colnames_dc_a_p1_binary[0][1], 
		#	opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_a_p1_binary[0][3],
		#	opt.my_colnames_dc_b_p1_binary[0][0],
		#	opt.my_colnames_dc_b_p1_binary[0][1], opt.my_colnames_dc_b_p1_binary[0][2],
		#	opt.my_colnames_dc_b_p1_binary[0][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[1][0], opt.my_colnames_dc_a_p1_binary[1][1], 
		#	opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_a_p1_binary[1][3],
		#	opt.my_colnames_dc_b_p1_binary[1][0],
		#	opt.my_colnames_dc_b_p1_binary[1][1], opt.my_colnames_dc_b_p1_binary[1][2],
		#	opt.my_colnames_dc_b_p1_binary[1][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_m1_binary[0][0], opt.my_colnames_dc_a_m1_binary[0][1], 
		#	opt.my_colnames_dc_a_m1_binary[0][2], opt.my_colnames_dc_a_m1_binary[0][3],
		#	opt.my_colnames_dc_b_m1_binary[0][0],
		#	opt.my_colnames_dc_b_m1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
		#	opt.my_colnames_dc_b_m1_binary[0][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_m1_binary[1][0], opt.my_colnames_dc_a_m1_binary[1][1], 
		#	opt.my_colnames_dc_a_m1_binary[1][2], opt.my_colnames_dc_a_m1_binary[1][3],
		#	opt.my_colnames_dc_b_m1_binary[1][0],
		#	opt.my_colnames_dc_b_m1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
		#	opt.my_colnames_dc_b_m1_binary[1][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	# dc_row_same_chromatid
		#	[[opt.my_colnames_dc_a_p1_binary[0][0], opt.my_colnames_dc_a_p1_binary[0][1], 
		#	opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_a_p1_binary[0][3],
		#	opt.my_colnames_dc_a_m1_binary[0][0],
		#	opt.my_colnames_dc_a_m1_binary[0][1], opt.my_colnames_dc_a_m1_binary[0][2],
		#	opt.my_colnames_dc_a_m1_binary[0][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[1][0], opt.my_colnames_dc_a_p1_binary[1][1], 
		#	opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_a_p1_binary[1][3],
		#	opt.my_colnames_dc_a_m1_binary[1][0],
		#	opt.my_colnames_dc_a_m1_binary[1][1], opt.my_colnames_dc_a_m1_binary[1][2],
		#	opt.my_colnames_dc_a_m1_binary[1][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[0][0], opt.my_colnames_dc_b_p1_binary[0][1], 
		#	opt.my_colnames_dc_b_p1_binary[0][2], opt.my_colnames_dc_b_p1_binary[0][3],
		#	opt.my_colnames_dc_b_m1_binary[0][0],
		#	opt.my_colnames_dc_b_m1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
		#	opt.my_colnames_dc_b_m1_binary[0][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[1][0], opt.my_colnames_dc_b_p1_binary[1][1], 
		#	opt.my_colnames_dc_b_p1_binary[1][2], opt.my_colnames_dc_b_p1_binary[1][3],
		#	opt.my_colnames_dc_b_m1_binary[1][0],
		#	opt.my_colnames_dc_b_m1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
		#	opt.my_colnames_dc_b_m1_binary[1][3]],
		#	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
		#	# dc_row_different
		#	[[opt.my_colnames_dc_a_p1_binary[0][1], opt.my_colnames_dc_b_m1_binary[0][2],
		#	opt.my_colnames_dc_b_m1_binary[0][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[0][2], opt.my_colnames_dc_b_m1_binary[0][1],
		#	opt.my_colnames_dc_b_m1_binary[0][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[0][3], opt.my_colnames_dc_b_m1_binary[0][1],
		#	opt.my_colnames_dc_b_m1_binary[0][2]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[1][1], opt.my_colnames_dc_b_m1_binary[1][2],
		#	opt.my_colnames_dc_b_m1_binary[1][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[1][2], opt.my_colnames_dc_b_m1_binary[1][1],
		#	opt.my_colnames_dc_b_m1_binary[1][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_a_p1_binary[1][3], opt.my_colnames_dc_b_m1_binary[1][1],
		#	opt.my_colnames_dc_b_m1_binary[1][2]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[0][1], opt.my_colnames_dc_a_m1_binary[0][2],
		#	opt.my_colnames_dc_a_m1_binary[0][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[0][2], opt.my_colnames_dc_a_m1_binary[0][1],
		#	opt.my_colnames_dc_a_m1_binary[0][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[0][3], opt.my_colnames_dc_a_m1_binary[0][1],
		#	opt.my_colnames_dc_a_m1_binary[0][2]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[1][1], opt.my_colnames_dc_a_m1_binary[1][2],
		#	opt.my_colnames_dc_a_m1_binary[1][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[1][2], opt.my_colnames_dc_a_m1_binary[1][1],
		#	opt.my_colnames_dc_a_m1_binary[1][3]], [1.0, 1.0, 1.0]],
		#	[[opt.my_colnames_dc_b_p1_binary[1][3], opt.my_colnames_dc_a_m1_binary[1][1],
		#	opt.my_colnames_dc_a_m1_binary[1][2]], [1.0, 1.0, 1.0]]
		#	]
		#self.assertListEqual(opt.my_rows, rows)

	def test_constraint_major_minor_cn(self):
		sublin_num = 3
		self.seg_num = 2
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list = [spl_seg_1, spl_seg_1]
		my_seg = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		self.seg_list = [my_seg, my_seg]

		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list, seg_splines_B=seg_spl_list)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.vars_aux_dc()  
		opt.vars_phi()

		# function to test
		opt.constraint_major_minor_cn()

		# rhs
		rhs = [0.0] * self.seg_num
		self.assertEqual(rhs, opt.my_rhs)

		# senses
		senses = ["G"] * self.seg_num
		self.assertEqual(senses, opt.my_sense)

		# rownames
		rownames = ["major_minor_cn_0", "major_minor_cn_1"]
		self.assertEqual(rownames, opt.my_rownames)

		# rows
		rows_vars = [
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2",
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2",
			"dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2",
			"dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"],
			["dc_a_p1_float_1_0", "dc_a_p1_float_1_1", "dc_a_p1_float_1_2",
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1", "dc_a_m1_float_1_2",
			"dc_b_p1_float_1_0", "dc_b_p1_float_1_1", "dc_b_p1_float_1_2",
			"dc_b_m1_float_1_0", "dc_b_m1_float_1_1", "dc_b_m1_float_1_2"],
			]
		self.assertEqual(rows_vars, self.get_opt_rows_vars(opt))


		rows_values = [[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]
		self.assertEqual(rows_values, self.get_opt_rows_values(opt))

	def test_constraint_remove_cn_symmetry(self):

		# set up variables
		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_phi()
		opt.vars_aux_dc()

		# function to test
		opt.constraint_remove_cn_symmetry()

		# rhs
		rhs = [0.0] * self.seg_num * 2
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * self.seg_num * 2
		self.assertListEqual(opt.my_sense, s) 

		# rownames
		rownames = ["remove_cn_symmetry_b_p1_zero_0", "remove_cn_symmetry_b_p1_zero_1",
			"remove_cn_symmetry_a_m1_zero_0", "remove_cn_symmetry_a_m1_zero_1"]
		self.assertListEqual(opt.my_rownames, rownames) 

		# rows
		rows_vars = [
			["dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2"],
			["dc_b_p1_binary_1_0", "dc_b_p1_binary_1_1", "dc_b_p1_binary_1_2"],
			["dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2"],
			["dc_a_m1_binary_1_0", "dc_a_m1_binary_1_1", "dc_a_m1_binary_1_2"]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = [[1.0, 1.0, 1.0]] * 4
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)


	def test_constraints_aux_cnv_dc_matrices(self):
		
		# set up variables
		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_phi()
		opt.vars_aux_dc()

		# function to test
		opt.constraints_aux_cnv_dc_matrices()

		entries = opt.cnv_aux_all_states_one_type_entry_num

		# rhs
		rhs = [0.0] * entries + [-1.0] * entries + [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries + ["G"] * entries + ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["dc_a_p1_float_less_c_0_0", "dc_a_p1_float_less_c_0_1", "dc_a_p1_float_less_c_0_2",
			"dc_a_p1_float_less_c_1_0", "dc_a_p1_float_less_c_1_1", "dc_a_p1_float_less_c_1_2",
			"dc_b_p1_float_less_c_0_0", "dc_b_p1_float_less_c_0_1", "dc_b_p1_float_less_c_0_2",
			"dc_b_p1_float_less_c_1_0", "dc_b_p1_float_less_c_1_1", "dc_b_p1_float_less_c_1_2",
			"dc_a_m1_float_less_c_0_0", "dc_a_m1_float_less_c_0_1", "dc_a_m1_float_less_c_0_2",
			"dc_a_m1_float_less_c_1_0", "dc_a_m1_float_less_c_1_1", "dc_a_m1_float_less_c_1_2",
			"dc_b_m1_float_less_c_0_0", "dc_b_m1_float_less_c_0_1", "dc_b_m1_float_less_c_0_2",
			"dc_b_m1_float_less_c_1_0", "dc_b_m1_float_less_c_1_1", "dc_b_m1_float_less_c_1_2",
			"dc_a_p1_float_greater_than_0_0", "dc_a_p1_float_greater_than_0_1", "dc_a_p1_float_greater_than_0_2",
			"dc_a_p1_float_greater_than_1_0", "dc_a_p1_float_greater_than_1_1", "dc_a_p1_float_greater_than_1_2",
			"dc_b_p1_float_greater_than_0_0", "dc_b_p1_float_greater_than_0_1", "dc_b_p1_float_greater_than_0_2",
			"dc_b_p1_float_greater_than_1_0", "dc_b_p1_float_greater_than_1_1", "dc_b_p1_float_greater_than_1_2",
			"dc_a_m1_float_greater_than_0_0", "dc_a_m1_float_greater_than_0_1", "dc_a_m1_float_greater_than_0_2",
			"dc_a_m1_float_greater_than_1_0", "dc_a_m1_float_greater_than_1_1", "dc_a_m1_float_greater_than_1_2",
			"dc_b_m1_float_greater_than_0_0", "dc_b_m1_float_greater_than_0_1", "dc_b_m1_float_greater_than_0_2",
			"dc_b_m1_float_greater_than_1_0", "dc_b_m1_float_greater_than_1_1", "dc_b_m1_float_greater_than_1_2",
			"dc_a_p1_float_smaller_phi_0_0", "dc_a_p1_float_smaller_phi_0_1", "dc_a_p1_float_smaller_phi_0_2",
			"dc_a_p1_float_smaller_phi_1_0", "dc_a_p1_float_smaller_phi_1_1", "dc_a_p1_float_smaller_phi_1_2",
			"dc_b_p1_float_smaller_phi_0_0", "dc_b_p1_float_smaller_phi_0_1", "dc_b_p1_float_smaller_phi_0_2",
			"dc_b_p1_float_smaller_phi_1_0", "dc_b_p1_float_smaller_phi_1_1", "dc_b_p1_float_smaller_phi_1_2",
			"dc_a_m1_float_smaller_phi_0_0", "dc_a_m1_float_smaller_phi_0_1", "dc_a_m1_float_smaller_phi_0_2",
			"dc_a_m1_float_smaller_phi_1_0", "dc_a_m1_float_smaller_phi_1_1", "dc_a_m1_float_smaller_phi_1_2",
			"dc_b_m1_float_smaller_phi_0_0", "dc_b_m1_float_smaller_phi_0_1", "dc_b_m1_float_smaller_phi_0_2",
			"dc_b_m1_float_smaller_phi_1_0", "dc_b_m1_float_smaller_phi_1_1", "dc_b_m1_float_smaller_phi_1_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			[opt.my_colnames_dc_a_p1_float[0][0], opt.my_colnames_dc_a_p1_binary[0][0]],
			[opt.my_colnames_dc_a_p1_float[0][1], opt.my_colnames_dc_a_p1_binary[0][1]],
			[opt.my_colnames_dc_a_p1_float[0][2], opt.my_colnames_dc_a_p1_binary[0][2]],
			[opt.my_colnames_dc_a_p1_float[1][0], opt.my_colnames_dc_a_p1_binary[1][0]],
			[opt.my_colnames_dc_a_p1_float[1][1], opt.my_colnames_dc_a_p1_binary[1][1]],
			[opt.my_colnames_dc_a_p1_float[1][2], opt.my_colnames_dc_a_p1_binary[1][2]],
			[opt.my_colnames_dc_b_p1_float[0][0], opt.my_colnames_dc_b_p1_binary[0][0]],
			[opt.my_colnames_dc_b_p1_float[0][1], opt.my_colnames_dc_b_p1_binary[0][1]],
			[opt.my_colnames_dc_b_p1_float[0][2], opt.my_colnames_dc_b_p1_binary[0][2]],
			[opt.my_colnames_dc_b_p1_float[1][0], opt.my_colnames_dc_b_p1_binary[1][0]],
			[opt.my_colnames_dc_b_p1_float[1][1], opt.my_colnames_dc_b_p1_binary[1][1]],
			[opt.my_colnames_dc_b_p1_float[1][2], opt.my_colnames_dc_b_p1_binary[1][2]],
			[opt.my_colnames_dc_a_m1_float[0][0], opt.my_colnames_dc_a_m1_binary[0][0]],
			[opt.my_colnames_dc_a_m1_float[0][1], opt.my_colnames_dc_a_m1_binary[0][1]],
			[opt.my_colnames_dc_a_m1_float[0][2], opt.my_colnames_dc_a_m1_binary[0][2]],
			[opt.my_colnames_dc_a_m1_float[1][0], opt.my_colnames_dc_a_m1_binary[1][0]],
			[opt.my_colnames_dc_a_m1_float[1][1], opt.my_colnames_dc_a_m1_binary[1][1]],
			[opt.my_colnames_dc_a_m1_float[1][2], opt.my_colnames_dc_a_m1_binary[1][2]],
			[opt.my_colnames_dc_b_m1_float[0][0], opt.my_colnames_dc_b_m1_binary[0][0]],
			[opt.my_colnames_dc_b_m1_float[0][1], opt.my_colnames_dc_b_m1_binary[0][1]],
			[opt.my_colnames_dc_b_m1_float[0][2], opt.my_colnames_dc_b_m1_binary[0][2]],
			[opt.my_colnames_dc_b_m1_float[1][0], opt.my_colnames_dc_b_m1_binary[1][0]],
			[opt.my_colnames_dc_b_m1_float[1][1], opt.my_colnames_dc_b_m1_binary[1][1]],
			[opt.my_colnames_dc_b_m1_float[1][2], opt.my_colnames_dc_b_m1_binary[1][2]],
			[opt.my_colnames_dc_a_p1_float[0][0], opt.my_phis[0], opt.my_colnames_dc_a_p1_binary[0][0]],
			[opt.my_colnames_dc_a_p1_float[0][1], opt.my_phis[1], opt.my_colnames_dc_a_p1_binary[0][1]],
			[opt.my_colnames_dc_a_p1_float[0][2], opt.my_phis[2], opt.my_colnames_dc_a_p1_binary[0][2]],
			[opt.my_colnames_dc_a_p1_float[1][0], opt.my_phis[0], opt.my_colnames_dc_a_p1_binary[1][0]],
			[opt.my_colnames_dc_a_p1_float[1][1], opt.my_phis[1], opt.my_colnames_dc_a_p1_binary[1][1]],
			[opt.my_colnames_dc_a_p1_float[1][2], opt.my_phis[2], opt.my_colnames_dc_a_p1_binary[1][2]],
			[opt.my_colnames_dc_b_p1_float[0][0], opt.my_phis[0], opt.my_colnames_dc_b_p1_binary[0][0]],
			[opt.my_colnames_dc_b_p1_float[0][1], opt.my_phis[1], opt.my_colnames_dc_b_p1_binary[0][1]],
			[opt.my_colnames_dc_b_p1_float[0][2], opt.my_phis[2], opt.my_colnames_dc_b_p1_binary[0][2]],
			[opt.my_colnames_dc_b_p1_float[1][0], opt.my_phis[0], opt.my_colnames_dc_b_p1_binary[1][0]],
			[opt.my_colnames_dc_b_p1_float[1][1], opt.my_phis[1], opt.my_colnames_dc_b_p1_binary[1][1]],
			[opt.my_colnames_dc_b_p1_float[1][2], opt.my_phis[2], opt.my_colnames_dc_b_p1_binary[1][2]],
			[opt.my_colnames_dc_a_m1_float[0][0], opt.my_phis[0], opt.my_colnames_dc_a_m1_binary[0][0]],
			[opt.my_colnames_dc_a_m1_float[0][1], opt.my_phis[1], opt.my_colnames_dc_a_m1_binary[0][1]],
			[opt.my_colnames_dc_a_m1_float[0][2], opt.my_phis[2], opt.my_colnames_dc_a_m1_binary[0][2]],
			[opt.my_colnames_dc_a_m1_float[1][0], opt.my_phis[0], opt.my_colnames_dc_a_m1_binary[1][0]],
			[opt.my_colnames_dc_a_m1_float[1][1], opt.my_phis[1], opt.my_colnames_dc_a_m1_binary[1][1]],
			[opt.my_colnames_dc_a_m1_float[1][2], opt.my_phis[2], opt.my_colnames_dc_a_m1_binary[1][2]],
			[opt.my_colnames_dc_b_m1_float[0][0], opt.my_phis[0], opt.my_colnames_dc_b_m1_binary[0][0]],
			[opt.my_colnames_dc_b_m1_float[0][1], opt.my_phis[1], opt.my_colnames_dc_b_m1_binary[0][1]],
			[opt.my_colnames_dc_b_m1_float[0][2], opt.my_phis[2], opt.my_colnames_dc_b_m1_binary[0][2]],
			[opt.my_colnames_dc_b_m1_float[1][0], opt.my_phis[0], opt.my_colnames_dc_b_m1_binary[1][0]],
			[opt.my_colnames_dc_b_m1_float[1][1], opt.my_phis[1], opt.my_colnames_dc_b_m1_binary[1][1]],
			[opt.my_colnames_dc_b_m1_float[1][2], opt.my_phis[2], opt.my_colnames_dc_b_m1_binary[1][2]],
			[opt.my_colnames_dc_a_p1_float[0][0], opt.my_phis[0]],
			[opt.my_colnames_dc_a_p1_float[0][1], opt.my_phis[1]],
			[opt.my_colnames_dc_a_p1_float[0][2], opt.my_phis[2]],
			[opt.my_colnames_dc_a_p1_float[1][0], opt.my_phis[0]],
			[opt.my_colnames_dc_a_p1_float[1][1], opt.my_phis[1]],
			[opt.my_colnames_dc_a_p1_float[1][2], opt.my_phis[2]],
			[opt.my_colnames_dc_b_p1_float[0][0], opt.my_phis[0]],
			[opt.my_colnames_dc_b_p1_float[0][1], opt.my_phis[1]],
			[opt.my_colnames_dc_b_p1_float[0][2], opt.my_phis[2]],
			[opt.my_colnames_dc_b_p1_float[1][0], opt.my_phis[0]],
			[opt.my_colnames_dc_b_p1_float[1][1], opt.my_phis[1]],
			[opt.my_colnames_dc_b_p1_float[1][2], opt.my_phis[2]],
			[opt.my_colnames_dc_a_m1_float[0][0], opt.my_phis[0]],
			[opt.my_colnames_dc_a_m1_float[0][1], opt.my_phis[1]],
			[opt.my_colnames_dc_a_m1_float[0][2], opt.my_phis[2]],
			[opt.my_colnames_dc_a_m1_float[1][0], opt.my_phis[0]],
			[opt.my_colnames_dc_a_m1_float[1][1], opt.my_phis[1]],
			[opt.my_colnames_dc_a_m1_float[1][2], opt.my_phis[2]],
			[opt.my_colnames_dc_b_m1_float[0][0], opt.my_phis[0]],
			[opt.my_colnames_dc_b_m1_float[0][1], opt.my_phis[1]],
			[opt.my_colnames_dc_b_m1_float[0][2], opt.my_phis[2]],
			[opt.my_colnames_dc_b_m1_float[1][0], opt.my_phis[0]],
			[opt.my_colnames_dc_b_m1_float[1][1], opt.my_phis[1]],
			[opt.my_colnames_dc_b_m1_float[1][2], opt.my_phis[2]],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		rows_values = ([[1.0, -1.0]] * 24 +
			[[1.0, -1.0, -1.0]] * 24 +
			[[1.0, -1.0]] * 24)
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_constraints_phi_0_no_mutations(self):

		# for simple CN changes
		sublin_num = 3
		self.seg_num = 2
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list_A = [spl_seg_1, spl_seg_1]
		seg_spl_list_B = [spl_seg_1, spl_seg_1]
		seg_1 = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		self.seg_list = [seg_1, seg_1]
		self.ssm_num = 4
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list_A, seg_splines_B=seg_spl_list_B)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_aux_dc()
		opt.vars_three_ssm_matrices()

		# function to test
		opt.constraints_phi_0_no_mutations()

		entries = 2 * 2 * 4 + 2 * 4 * 3

		# rhs
		rhs = [-0.9999] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["G"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"phi_1_0_no_mutations_dc_a_p1_0", "phi_1_0_no_mutations_dc_a_p1_1",
			"phi_2_0_no_mutations_dc_a_p1_0", "phi_2_0_no_mutations_dc_a_p1_1",
			"phi_1_0_no_mutations_dc_b_p1_0", "phi_1_0_no_mutations_dc_b_p1_1",
			"phi_2_0_no_mutations_dc_b_p1_0", "phi_2_0_no_mutations_dc_b_p1_1",
			"phi_1_0_no_mutations_dc_a_m1_0", "phi_1_0_no_mutations_dc_a_m1_1",
			"phi_2_0_no_mutations_dc_a_m1_0", "phi_2_0_no_mutations_dc_a_m1_1",
			"phi_1_0_no_mutations_dc_b_m1_0", "phi_1_0_no_mutations_dc_b_m1_1",
			"phi_2_0_no_mutations_dc_b_m1_0", "phi_2_0_no_mutations_dc_b_m1_1",
			"phi_1_0_no_mutations_ssm_unphased_0", "phi_1_0_no_mutations_ssm_unphased_1",
			"phi_1_0_no_mutations_ssm_unphased_2", "phi_1_0_no_mutations_ssm_unphased_3",
			"phi_2_0_no_mutations_ssm_unphased_0", "phi_2_0_no_mutations_ssm_unphased_1",
			"phi_2_0_no_mutations_ssm_unphased_2", "phi_2_0_no_mutations_ssm_unphased_3",
			"phi_1_0_no_mutations_ssm_a_0", "phi_1_0_no_mutations_ssm_a_1",
			"phi_1_0_no_mutations_ssm_a_2", "phi_1_0_no_mutations_ssm_a_3",
			"phi_2_0_no_mutations_ssm_a_0", "phi_2_0_no_mutations_ssm_a_1",
			"phi_2_0_no_mutations_ssm_a_2", "phi_2_0_no_mutations_ssm_a_3",
			"phi_1_0_no_mutations_ssm_b_0", "phi_1_0_no_mutations_ssm_b_1",
			"phi_1_0_no_mutations_ssm_b_2", "phi_1_0_no_mutations_ssm_b_3",
			"phi_2_0_no_mutations_ssm_b_0", "phi_2_0_no_mutations_ssm_b_1",
			"phi_2_0_no_mutations_ssm_b_2", "phi_2_0_no_mutations_ssm_b_3",
			]
		self.assertListEqual(opt.my_rownames, rownames)	

		# rows
		rows_vars = [
			["phi_1", "dc_a_p1_binary_0_1"], ["phi_1", "dc_a_p1_binary_1_1"],
			["phi_2", "dc_a_p1_binary_0_2"], ["phi_2", "dc_a_p1_binary_1_2"],
			["phi_1", "dc_b_p1_binary_0_1"], ["phi_1", "dc_b_p1_binary_1_1"],
			["phi_2", "dc_b_p1_binary_0_2"], ["phi_2", "dc_b_p1_binary_1_2"],
			["phi_1", "dc_a_m1_binary_0_1"], ["phi_1", "dc_a_m1_binary_1_1"],
			["phi_2", "dc_a_m1_binary_0_2"], ["phi_2", "dc_a_m1_binary_1_2"],
			["phi_1", "dc_b_m1_binary_0_1"], ["phi_1", "dc_b_m1_binary_1_1"],
			["phi_2", "dc_b_m1_binary_0_2"], ["phi_2", "dc_b_m1_binary_1_2"],
			["phi_1", "dssm_0_1"], ["phi_1", "dssm_1_1"],
			["phi_1", "dssm_2_1"], ["phi_1", "dssm_3_1"],
			["phi_2", "dssm_0_2"], ["phi_2", "dssm_1_2"],
			["phi_2", "dssm_2_2"], ["phi_2", "dssm_3_2"],
			["phi_1", "dssm_a_0_1"], ["phi_1", "dssm_a_1_1"],
			["phi_1", "dssm_a_2_1"], ["phi_1", "dssm_a_3_1"],
			["phi_2", "dssm_a_0_2"], ["phi_2", "dssm_a_1_2"],
			["phi_2", "dssm_a_2_2"], ["phi_2", "dssm_a_3_2"],
			["phi_1", "dssm_b_0_1"], ["phi_1", "dssm_b_1_1"],
			["phi_1", "dssm_b_2_1"], ["phi_1", "dssm_b_3_1"],
			["phi_2", "dssm_b_0_2"], ["phi_2", "dssm_b_1_2"],
			["phi_2", "dssm_b_2_2"], ["phi_2", "dssm_b_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)
		row_values = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), row_values)

		# for non-simple CN changes
		sublin_num = 3
		self.seg_num = 2
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list_A = [spl_seg_1, spl_seg_1]
		seg_spl_list_B = [spl_seg_1, spl_seg_1]
		seg_1 = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		self.seg_list = [seg_1, seg_1]
		self.ssm_num = 4
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list_A, seg_splines_B=seg_spl_list_B)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.simple_CN_changes = False
		opt.vars_phi()
		opt.vars_aux_dc()
		opt.vars_three_ssm_matrices()

		# function to test
		opt.constraints_phi_0_no_mutations()

		entries = 2 * 2 * 4 + 2 * 4 * 2

		# rhs
		rhs = [-0.9999] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["G"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = [
			"phi_1_0_no_mutations_dc_a_p1_0", "phi_1_0_no_mutations_dc_a_p1_1",
			"phi_2_0_no_mutations_dc_a_p1_0", "phi_2_0_no_mutations_dc_a_p1_1",
			"phi_1_0_no_mutations_dc_b_p1_0", "phi_1_0_no_mutations_dc_b_p1_1",
			"phi_2_0_no_mutations_dc_b_p1_0", "phi_2_0_no_mutations_dc_b_p1_1",
			"phi_1_0_no_mutations_dc_a_m1_0", "phi_1_0_no_mutations_dc_a_m1_1",
			"phi_2_0_no_mutations_dc_a_m1_0", "phi_2_0_no_mutations_dc_a_m1_1",
			"phi_1_0_no_mutations_dc_b_m1_0", "phi_1_0_no_mutations_dc_b_m1_1",
			"phi_2_0_no_mutations_dc_b_m1_0", "phi_2_0_no_mutations_dc_b_m1_1",
			"phi_1_0_no_mutations_ssm_a_0", "phi_1_0_no_mutations_ssm_a_1",
			"phi_1_0_no_mutations_ssm_a_2", "phi_1_0_no_mutations_ssm_a_3",
			"phi_2_0_no_mutations_ssm_a_0", "phi_2_0_no_mutations_ssm_a_1",
			"phi_2_0_no_mutations_ssm_a_2", "phi_2_0_no_mutations_ssm_a_3",
			"phi_1_0_no_mutations_ssm_b_0", "phi_1_0_no_mutations_ssm_b_1",
			"phi_1_0_no_mutations_ssm_b_2", "phi_1_0_no_mutations_ssm_b_3",
			"phi_2_0_no_mutations_ssm_b_0", "phi_2_0_no_mutations_ssm_b_1",
			"phi_2_0_no_mutations_ssm_b_2", "phi_2_0_no_mutations_ssm_b_3",
			]
		self.assertListEqual(opt.my_rownames, rownames)	

		# rows
		rows_vars = [
			["phi_1", "dc_a_p1_binary_0_1"], ["phi_1", "dc_a_p1_binary_1_1"],
			["phi_2", "dc_a_p1_binary_0_2"], ["phi_2", "dc_a_p1_binary_1_2"],
			["phi_1", "dc_b_p1_binary_0_1"], ["phi_1", "dc_b_p1_binary_1_1"],
			["phi_2", "dc_b_p1_binary_0_2"], ["phi_2", "dc_b_p1_binary_1_2"],
			["phi_1", "dc_a_m1_binary_0_1"], ["phi_1", "dc_a_m1_binary_1_1"],
			["phi_2", "dc_a_m1_binary_0_2"], ["phi_2", "dc_a_m1_binary_1_2"],
			["phi_1", "dc_b_m1_binary_0_1"], ["phi_1", "dc_b_m1_binary_1_1"],
			["phi_2", "dc_b_m1_binary_0_2"], ["phi_2", "dc_b_m1_binary_1_2"],
			["phi_1", "dssm_a_0_1"], ["phi_1", "dssm_a_1_1"],
			["phi_1", "dssm_a_2_1"], ["phi_1", "dssm_a_3_1"],
			["phi_2", "dssm_a_0_2"], ["phi_2", "dssm_a_1_2"],
			["phi_2", "dssm_a_2_2"], ["phi_2", "dssm_a_3_2"],
			["phi_1", "dssm_b_0_1"], ["phi_1", "dssm_b_1_1"],
			["phi_1", "dssm_b_2_1"], ["phi_1", "dssm_b_3_1"],
			["phi_2", "dssm_b_0_2"], ["phi_2", "dssm_b_1_2"],
			["phi_2", "dssm_b_2_2"], ["phi_2", "dssm_b_3_2"],
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)
		row_values = [[1.0, -1.0]] * entries
		self.assertListEqual(self.get_opt_rows_values(opt), row_values)


	def test_constraints_phi_mutation_number(self):
		# for simple CN changes
		sublin_num = 3
		self.seg_num = 3
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list_A = [spl_seg_1, spl_seg_1, spl_seg_1]
		seg_spl_list_B = [spl_seg_1, spl_seg_1, spl_seg_1]
		seg_1 = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		self.seg_list = [seg_1, seg_1, seg_1]
		self.ssm_num = 4
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list_A, seg_splines_B=seg_spl_list_B)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.vars_phi()
		opt.vars_aux_dc()
		opt.vars_three_ssm_matrices()

		# function to test
		opt.constraints_phi_mutation_number()

		entries = 2

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["phi_mutation_number_1", "phi_mutation_number_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["phi_1", "dc_a_p1_binary_0_1", "dc_a_p1_binary_1_1", "dc_a_p1_binary_2_1",
			"dc_b_p1_binary_0_1", "dc_b_p1_binary_1_1", "dc_b_p1_binary_2_1",
			"dc_a_m1_binary_0_1", "dc_a_m1_binary_1_1", "dc_a_m1_binary_2_1",
			"dc_b_m1_binary_0_1", "dc_b_m1_binary_1_1", "dc_b_m1_binary_2_1",
			"dssm_0_1", "dssm_1_1", "dssm_2_1", "dssm_3_1",
			"dssm_a_0_1", "dssm_a_1_1", "dssm_a_2_1", "dssm_a_3_1",
			"dssm_b_0_1", "dssm_b_1_1", "dssm_b_2_1", "dssm_b_3_1"],
			["phi_2", "dc_a_p1_binary_0_2", "dc_a_p1_binary_1_2", "dc_a_p1_binary_2_2",
			"dc_b_p1_binary_0_2", "dc_b_p1_binary_1_2", "dc_b_p1_binary_2_2",
			"dc_a_m1_binary_0_2", "dc_a_m1_binary_1_2", "dc_a_m1_binary_2_2",
			"dc_b_m1_binary_0_2", "dc_b_m1_binary_1_2", "dc_b_m1_binary_2_2",
			"dssm_0_2", "dssm_1_2", "dssm_2_2", "dssm_3_2",
			"dssm_a_0_2", "dssm_a_1_2", "dssm_a_2_2", "dssm_a_3_2",
			"dssm_b_0_2", "dssm_b_1_2", "dssm_b_2_2", "dssm_b_3_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		row_values = [[1.0] + [-1.0] * ((3*4) + (4*3))] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), row_values)

		# for non-simple CN changes
		sublin_num = 3
		self.seg_num = 3
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		seg_spl_list_A = [spl_seg_1, spl_seg_1, spl_seg_1]
		seg_spl_list_B = [spl_seg_1, spl_seg_1, spl_seg_1]
		seg_1 = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		self.seg_list = [seg_1, seg_1, seg_1]
		self.ssm_num = 4
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list_A, seg_splines_B=seg_spl_list_B)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.simple_CN_changes = False
		opt.vars_phi()
		opt.vars_aux_dc()
		opt.vars_three_ssm_matrices()

		# function to test
		opt.constraints_phi_mutation_number()

		entries = 2

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["L"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["phi_mutation_number_1", "phi_mutation_number_2"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [
			["phi_1", "dc_a_p1_binary_0_1", "dc_a_p1_binary_1_1", "dc_a_p1_binary_2_1",
			"dc_b_p1_binary_0_1", "dc_b_p1_binary_1_1", "dc_b_p1_binary_2_1",
			"dc_a_m1_binary_0_1", "dc_a_m1_binary_1_1", "dc_a_m1_binary_2_1",
			"dc_b_m1_binary_0_1", "dc_b_m1_binary_1_1", "dc_b_m1_binary_2_1",
			"dssm_a_0_1", "dssm_a_1_1", "dssm_a_2_1", "dssm_a_3_1",
			"dssm_b_0_1", "dssm_b_1_1", "dssm_b_2_1", "dssm_b_3_1"],
			["phi_2", "dc_a_p1_binary_0_2", "dc_a_p1_binary_1_2", "dc_a_p1_binary_2_2",
			"dc_b_p1_binary_0_2", "dc_b_p1_binary_1_2", "dc_b_p1_binary_2_2",
			"dc_a_m1_binary_0_2", "dc_a_m1_binary_1_2", "dc_a_m1_binary_2_2",
			"dc_b_m1_binary_0_2", "dc_b_m1_binary_1_2", "dc_b_m1_binary_2_2",
			"dssm_a_0_2", "dssm_a_1_2", "dssm_a_2_2", "dssm_a_3_2",
			"dssm_b_0_2", "dssm_b_1_2", "dssm_b_2_2", "dssm_b_3_2"]
			]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars)

		row_values = [[1.0] + [-1.0] * ((3*4) + (4*2))] * 2
		self.assertListEqual(self.get_opt_rows_values(opt), row_values)

	def test_constraint_phi_k(self):
		
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()

		# function to test
		opt.constraint_phi_k()

		entries = sublin_num - 1

		# rhs
		rhs = [0.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["G"] * entries
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["phi_0_greater_equal_phi_1", "phi_1_greater_equal_phi_2", "phi_2_greater_equal_phi_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows_vars = [[opt.my_phis[0], opt.my_phis[1]],
			[opt.my_phis[1], opt.my_phis[2]],
			[opt.my_phis[2], opt.my_phis[3]]]
		self.assertListEqual(self.get_opt_rows_vars(opt), rows_vars) 

		rows_values = [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values) 

		#rows = [[[opt.my_phis[1], opt.my_phis[2]], [1.0, -1.0]],
		#	[[opt.my_phis[2], opt.my_phis[3]], [1.0, -1.0]]]
		#self.assertListEqual(opt.my_rows, rows) 

	def test_constraint_cnv_value_spline_cn_allele_specific(self):

		sublin_num = 3
		self.seg_num = 2
		spl_seg_1 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		spl_seg_2 = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 5.3,1000,
			log_pdf.log_normal, 0.3, 0.125)[0]
		seg_spl_list_A = [spl_seg_1, spl_seg_2]
		seg_spl_list_B = [spl_seg_2, spl_seg_1]
		seg_1 = segment.Segment_allele_specific(1, 1, 1, 1.3, 0.25, 0.3, 0.125)
		seg_2 = segment.Segment_allele_specific(1, 1, 1, 0.3, 0.125, 1.3, 0.25)
		self.seg_list = [seg_1, seg_2]
		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list, allele_specific=True,
			seg_splines_A=seg_spl_list_A, seg_splines_B=seg_spl_list_B)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, self.seg_list)
		opt.vars_aux_vars_mut_splines()
		opt.vars_aux_dc()  

		# functio to test
		opt.constraint_cnv_value_spline_cn_allele_specific(self.seg_list)

		# rhs
		rhs = [1, 1, 1, 1] 
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * self.seg_num * 2
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["seg_A_0", "seg_A_1", "seg_B_0", "seg_B_1",]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		opt_rows_vars = self.get_opt_rows_vars(opt)
		rows_vars = [
			opt.my_colnames_seg_A[0] + 
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2", 
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2"],
			opt.my_colnames_seg_A[1] + 
			["dc_a_p1_float_1_0", "dc_a_p1_float_1_1", "dc_a_p1_float_1_2", 
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1", "dc_a_m1_float_1_2"],
			opt.my_colnames_seg_B[0] + 
			["dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2", 
			"dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"],
			opt.my_colnames_seg_B[1] + 
			["dc_b_p1_float_1_0", "dc_b_p1_float_1_1", "dc_b_p1_float_1_2", 
			"dc_b_m1_float_1_0", "dc_b_m1_float_1_1", "dc_b_m1_float_1_2"]
			]
		self.assertListEqual(opt_rows_vars, rows_vars) 

		rows_values = [
			opt.seg_splines_A[0].get_knots().tolist() + [-1] * sublin_num
			+ [1] * sublin_num,
			opt.seg_splines_A[1].get_knots().tolist() + [-1] * sublin_num
			+ [1] * sublin_num,
			opt.seg_splines_B[0].get_knots().tolist() + [-1] * sublin_num
			+ [1] * sublin_num,
			opt.seg_splines_B[1].get_knots().tolist() + [-1] * sublin_num
			+ [1] * sublin_num,
			]
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values) 

	def test_constraint_cnv_value_spline_mean(self):
		
		sublin_num = 3
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1, self.ssm_spl_1]
		seg = segment.Segment(1, 1, 1, 1, 1)
		self.seg_list = [seg] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_aux_vars_mut_splines()
		opt.vars_aux_dc()  

		# functio to test
		opt.constraint_cnv_value_spline_mean(self.seg_list)

		# rhs
		rhs = [2 * self.seg_list[0].hm, 2 * self.seg_list[1].hm]
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		s = ["E"] * self.seg_num
		self.assertListEqual(opt.my_sense, s)

		# rownames
		rownames = ["cnv_mean_0", "cnv_mean_1"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows

		opt_rows_vars = self.get_opt_rows_vars(opt)
			
		rows_vars = [opt.my_colnames_seg[0] + 
			["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", 
			"dc_a_p1_float_0_2", "dc_b_p1_float_0_0", "dc_b_p1_float_0_1", 
			"dc_b_p1_float_0_2",
			"dc_a_m1_float_0_0", "dc_a_m1_float_0_1", 
			"dc_a_m1_float_0_2", "dc_b_m1_float_0_0", "dc_b_m1_float_0_1", 
			"dc_b_m1_float_0_2"],
			opt.my_colnames_seg[1] + 
			["dc_a_p1_float_1_0", "dc_a_p1_float_1_1",
			"dc_a_p1_float_1_2", "dc_b_p1_float_1_0", "dc_b_p1_float_1_1", 
			"dc_b_p1_float_1_2",
			"dc_a_m1_float_1_0", "dc_a_m1_float_1_1",
			"dc_a_m1_float_1_2", "dc_b_m1_float_1_0", "dc_b_m1_float_1_1", 
			"dc_b_m1_float_1_2"]]
		self.assertListEqual(opt_rows_vars, rows_vars)

		rows_values = [ 
			opt.seg_splines[0].get_knots().tolist() + [-self.seg_list[0].hm] * sublin_num * 
			opt.aux_matrices_cnv_linear_types_num + [self.seg_list[0].hm] * sublin_num 
			* opt.aux_matrices_cnv_linear_types_num,
			opt.seg_splines[1].get_knots().tolist() + [-self.seg_list[1].hm] * sublin_num * 
			opt.aux_matrices_cnv_linear_types_num + [self.seg_list[1].hm] * sublin_num 
			* opt.aux_matrices_cnv_linear_types_num]
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values) 

	def get_opt_rows_vars(self, opt):
		opt_rows_vars = []
		for i in range(len(opt.my_rows)):
			opt_rows_vars.append([opt.my_colnames[j] for j in opt.my_rows[i][0]])
		return opt_rows_vars
	
	def get_opt_rows_values(self, opt):
		return [opt.my_rows[i][1] for i in range(len(opt.my_rows))]


	def test_constraint_aux_vars_spline_sum_one(self):
		
		# independant from lineage, only 1
		# different numbers of segs, snps, ssms to test whether loops are right
		# mixed collection of splines to test loops indices
		sublin_num = 1
		self.seg_num = 2
		self.snp_num = 3
		self.ssm_num = 4
		self.seg_spl_list = [self.seg_spl_1, self.snp_spl_1] 
		self.snp_spl_list = [self.snp_spl_1, self.seg_spl_1, self.ssm_spl_1]
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_aux_vars_mut_splines()

		# function to test
		opt.constraint_aux_vars_spline_sum_one()
	
		# check values
		entries = self.seg_num + self.snp_num + self.ssm_num

		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, senses)
		
		# rownames
		rownames = ["seg_sum_a_0", "seg_sum_a_1", "snp_sum_a_0", "snp_sum_a_1", "snp_sum_a_2", "ssm_sum_a_0",
			"ssm_sum_a_1", "ssm_sum_a_2", "ssm_sum_a_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		rows = [[opt.my_colnames_seg[0], [1.0] * len(self.seg_spl_list[0].get_coeffs())],
			[opt.my_colnames_seg[1], [1.0] * len(self.seg_spl_list[1].get_coeffs())],
			[opt.my_colnames_snp[0], [1.0] * len(self.snp_spl_list[0].get_coeffs())],
			[opt.my_colnames_snp[1], [1.0] * len(self.snp_spl_list[1].get_coeffs())],
			[opt.my_colnames_snp[2], [1.0] * len(self.snp_spl_list[2].get_coeffs())],
			[opt.my_colnames_ssm[0], [1.0] * len(self.ssm_spl_list[0].get_coeffs())],
			[opt.my_colnames_ssm[1], [1.0] * len(self.ssm_spl_list[1].get_coeffs())],
			[opt.my_colnames_ssm[2], [1.0] * len(self.ssm_spl_list[2].get_coeffs())],
			[opt.my_colnames_ssm[3], [1.0] * len(self.ssm_spl_list[3].get_coeffs())]]
		self.assertListEqual(opt.my_rows, rows)

		# test for case with allele-specific CNs
		# independant from lineage, only 1
		# different numbers of segs, snps, ssms to test whether loops are right
		# mixed collection of splines to test loops indices
		sublin_num = 1
		self.seg_num = 2
		self.ssm_num = 4
		seg_spl_1 = log_pdf.find_piecewise_linear(10, 0, 5, 100, log_pdf.log_normal,
			2.3, 0.25)[0]
		seg_spl_2 = log_pdf.find_piecewise_linear(10, 0, 5, 100, log_pdf.log_normal,
			1.3, 0.125)[0]
		self.seg_A_spl_list = [seg_spl_1, seg_spl_2] 
		self.seg_B_spl_list = [seg_spl_2, seg_spl_1] 
		self.ssm_spl_list = [self.ssm_spl_1, self.seg_spl_1, self.seg_spl_1, self.snp_spl_1]

		seg_1 = segment.Segment_allele_specific(1, 1, 2, 2.3, 0.25, 1.3, 0.125)
		seg_2 = segment.Segment_allele_specific(1, 3, 4, 1.3, 0.125, 2.3, 0.25)
		seg_list = [seg_1, seg_2]

		opt = optimization.Optimization_with_CPLEX([], [], self.ssm_spl_list,
			allele_specific=True, seg_splines_A=self.seg_A_spl_list, 
			seg_splines_B=self.seg_B_spl_list)
		opt.set_other_parameter(sublin_num, [], self.ssm_list, seg_list)
		opt.vars_aux_vars_mut_splines()

		# function to test
		opt.constraint_aux_vars_spline_sum_one()
	
		# check values
		entries = self.seg_num * 2 + self.ssm_num

		# rhs
		rhs = [1.0] * entries
		self.assertListEqual(opt.my_rhs, rhs)

		# senses
		senses = ["E"] * entries
		self.assertListEqual(opt.my_sense, senses)
		
		# rownames
		rownames = ["seg_sum_A_a_0", "seg_sum_A_a_1", "seg_sum_B_a_0", "seg_sum_B_a_1", 
			"ssm_sum_a_0", "ssm_sum_a_1", "ssm_sum_a_2", "ssm_sum_a_3"]
		self.assertListEqual(opt.my_rownames, rownames)

		# rows
		opt_rows_vars = self.get_opt_rows_vars(opt)
		rows_vars = [opt.my_colnames_seg_A[0], opt.my_colnames_seg_A[1],
			opt.my_colnames_seg_B[0], opt.my_colnames_seg_B[1],
			opt.my_colnames_ssm[0], opt.my_colnames_ssm[1],
			opt.my_colnames_ssm[2], opt.my_colnames_ssm[3]]
		self.assertListEqual(opt_rows_vars, rows_vars)

		rows_values = [[1.0] * len(self.seg_A_spl_list[0].get_coeffs()),
			[1.0] * len(self.seg_A_spl_list[1].get_coeffs()),
			[1.0] * len(self.seg_B_spl_list[0].get_coeffs()),
			[1.0] * len(self.seg_B_spl_list[1].get_coeffs()),
			[1.0] * len(self.ssm_spl_list[0].get_coeffs()),
			[1.0] * len(self.ssm_spl_list[1].get_coeffs()),
			[1.0] * len(self.ssm_spl_list[2].get_coeffs()),
			[1.0] * len(self.ssm_spl_list[3].get_coeffs())]
		self.assertListEqual(self.get_opt_rows_values(opt), rows_values)

	def test_vars_ldr_inactive(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_ldr_inactive()

		entry_num = 3

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colnames_ldr_inactive = ["LDR_inactive_1_2", "LDR_inactive_1_3", "LDR_inactive_2_3"]
		self.assertListEqual(opt.my_colnames, colnames_ldr_inactive)

		# indices
		my_colnames_ldr_inactive_index_friendly_form = [
			[None, None, None, None],
			[None, None, 0, 1],
			[None, None, None, 2]
			]
		self.assertEqual(opt.my_colnames_ldr_inactive_index_friendly_form, my_colnames_ldr_inactive_index_friendly_form)

	def test_vars_chf_m_pf_LDRi(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_chf_m_pf_LDRi()

		entry_num = 3

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [0.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [-1.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colnames = ["chf_m_pf_LDRi_1_2", "chf_m_pf_LDRi_1_3", "chf_m_pf_LDRi_2_3"]
		self.assertListEqual(opt.my_colnames, colnames)

		# indices
		my_indices = [
			[None, None, None, None],
			[None, None, 0, 1],
			[None, None, None, 2]
			]
		self.assertEqual(opt.my_colnames_chf_m_pf_LDRi_index_friendly_form, my_indices)

	def test_vars_chf_m_pf_LDRa(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_chf_m_pf_LDRa()

		entry_num = 3

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colnames = ["chf_m_pf_LDRa_1_2", "chf_m_pf_LDRa_1_3", "chf_m_pf_LDRa_2_3"]
		self.assertListEqual(opt.my_colnames, colnames)

		# indices
		my_indices = [
			[None, None, None, None],
			[None, None, 0, 1],
			[None, None, None, 2]
			]
		self.assertEqual(opt.my_colnames_chf_m_pf_LDRa_index_friendly_form, my_indices)

	def test_vars_child_freq_minus_par_freq(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_child_freq_minus_par_freq()

		entry_num = 3

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [-1.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colnames_child_freq_minus_par_freq = [
			"child_freq_minus_par_freq_1_2", "child_freq_minus_par_freq_1_3", "child_freq_minus_par_freq_2_3"]
		self.assertListEqual(opt.my_colnames, colnames_child_freq_minus_par_freq)

		# indices
		child_freq_minus_par_freq_index_friendly_form = [
			[None, None, None, None],
			[None, None, 0, 1],
			[None, None, None, 2]
			]
		self.assertEqual(opt.my_colnames_child_freq_minus_par_freq_index_friendly_form, 
			child_freq_minus_par_freq_index_friendly_form)

	def test_vars_ldr_active(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_ldr_active()

		entry_num = 3

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colnames_ldr_active = ["LDR_active_1_2", "LDR_active_1_3", "LDR_active_2_3"]
		self.assertListEqual(opt.my_colnames, colnames_ldr_active)

		# indices
		my_colnames_ldr_active_index_friendly_form = [
			[None, None, None, None],
			[None, None, 0, 1],
			[None, None, None, 2]
			]
		self.assertEqual(opt.my_colnames_ldr_active_index_friendly_form, my_colnames_ldr_active_index_friendly_form)


	def test_vars_sibling_freq(self):
		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_sibling_freq()

		entry_num = 8

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		colnames_sibling_freq = [
			"sibling_0_1_2_freq", "sibling_0_1_3_freq", "sibling_0_2_1_freq", "sibling_0_2_3_freq", 
			"sibling_0_3_1_freq", "sibling_0_3_2_freq", "sibling_1_2_3_freq", "sibling_1_3_2_freq"
			]
		self.assertListEqual(opt.my_colnames_sibling_freq, colnames_sibling_freq)
		self.assertListEqual(opt.my_colnames, colnames_sibling_freq)

		# indices
		self.assertEqual(opt.sibling_freq_start_index, 0)
		self.assertTrue(opt.my_colnames_sibling_freq_index.any() == np.array([i for i in xrange(entry_num)]).any())

		my_colnames_sibling_freq_index_friendly_form = [
			[[None, None, None, None],
			[None, None, 0, 1],
			[None, 2, None, 3],
			[None, 4, 5, None]],

			[[None, None, None, None],
			[None, None, None, None],
			[None, None, None, 6],
			[None, None, 7, None]]
			]
		self.assertEqual(opt.my_colnames_sibling_freq_index_friendly_form, my_colnames_sibling_freq_index_friendly_form)
			

		

	def test_vars_parent_freq(self):

		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_parent_freq()
		
		entry_num = 6

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num 
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		colnames_parent_freq = [
			["parent_0_1_freq", "parent_0_2_freq", "parent_0_3_freq"],
			["parent_1_2_freq", "parent_1_3_freq"],
			["parent_2_3_freq"]
			]
		self.assertListEqual(opt.my_colnames_parent_freq, colnames_parent_freq)
		self.assertListEqual(opt.my_colnames, ["parent_0_1_freq", "parent_0_2_freq", "parent_0_3_freq", 
			"parent_1_2_freq", "parent_1_3_freq", "parent_2_3_freq"])

		# indices
		self.assertEqual(opt.parent_freq_start_index, 0)
		self.assertTrue(opt.my_colnames_parent_freq_index.any() == np.array([i for i in xrange(6)]).any())

		my_colnames_parent_freq_index_friendly_form = [
			[None, 0, 1, 2],
			[None, None, 3, 4],
			[None, None, None, 5],
			[None, None, None, None]
			]
		self.assertEqual(opt.my_colnames_parent_freq_index_friendly_form, 
			my_colnames_parent_freq_index_friendly_form)

	def test_vars_child_freq(self):

		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_child_freq()
		
		entry_num = 6

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num 
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		colnames_children_freq = [
			["child_0_1_freq", "child_0_2_freq", "child_0_3_freq"],
			["child_1_2_freq", "child_1_3_freq"],
			["child_2_3_freq"]
			]
		self.assertListEqual(opt.my_colnames_children_freq, colnames_children_freq)
		self.assertListEqual(opt.my_colnames, ["child_0_1_freq", "child_0_2_freq", "child_0_3_freq", 
			"child_1_2_freq", "child_1_3_freq", "child_2_3_freq"])

		# indices
		self.assertEqual(opt.children_freq_start_index, 0)
		self.assertTrue(opt.my_colnames_children_freq_index.any() == np.array([i for i in xrange(6)]).any())

		my_colnames_children_freq_index_friendly_form = [
			[None, 0, 1, 2],
			[None, None, 3, 4],
			[None, None, None, 5],
			[None, None, None, None]
			]
		self.assertEqual(opt.my_colnames_children_freq_index_friendly_form, 
			my_colnames_children_freq_index_friendly_form)

	def test_vars_child(self):

		# do for 4 lineages
		sublin_num = 4
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_child()

		entry_num = 6

		# obj
		obj = [0.0] * entry_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entry_num 
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entry_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entry_num
		self.assertListEqual(opt.my_ctype, t)

		colnames_children = [
			["child_0_1", "child_0_2", "child_0_3"],
			["child_1_2", "child_1_3"],
			["child_2_3"]
			]
		self.assertListEqual(opt.my_colnames_children, colnames_children)
		self.assertListEqual(opt.my_colnames, ["child_0_1", "child_0_2", "child_0_3", "child_1_2", "child_1_3",
			"child_2_3"])

		# indices
		self.assertEqual(opt.children_start_index, 0)
		self.assertTrue(opt.my_colnames_children_index.any() == np.array([i for i in xrange(6)]).any())

		my_colnames_children_index_friendly_form = [
			[None, 0, 1, 2],
			[None, None, 3, 4],
			[None, None, None, 5],
			[None, None, None, None]
			]
		self.assertEqual(opt.my_colnames_children_index_friendly_form, my_colnames_children_index_friendly_form)

	def test_vars_dc_descendant(self):
		
		# set up variables, only 2 sublineages
		sublin_num = 2
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_dc_descendant()

		self.assertEqual(len(opt.my_rhs), 0)

		# set up variables, 4 sublineages
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test 
		opt.vars_dc_descendant()

		entries = 3 * self.seg_num * cons.PHASE_NUMBER * opt.cnv_state_num

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries 
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries 
		self.assertListEqual(opt.my_ctype, t)

		colnames_a_p1 = [
			[["dc_descendant_a_p1_0_1_2", "dc_descendant_a_p1_0_1_3"], 
			["dc_descendant_a_p1_0_2_3"]],
			[["dc_descendant_a_p1_1_1_2", "dc_descendant_a_p1_1_1_3"],
			["dc_descendant_a_p1_1_2_3"]]
			]
		colnames_b_p1 = [
			[["dc_descendant_b_p1_0_1_2", "dc_descendant_b_p1_0_1_3"], 
			["dc_descendant_b_p1_0_2_3"]],
			[["dc_descendant_b_p1_1_1_2", "dc_descendant_b_p1_1_1_3"],
			["dc_descendant_b_p1_1_2_3"]]
			]
		colnames_a_m1 = [
			[["dc_descendant_a_m1_0_1_2", "dc_descendant_a_m1_0_1_3"], 
			["dc_descendant_a_m1_0_2_3"]],
			[["dc_descendant_a_m1_1_1_2", "dc_descendant_a_m1_1_1_3"],
			["dc_descendant_a_m1_1_2_3"]]
			]
		colnames_b_m1 = [
			[["dc_descendant_b_m1_0_1_2", "dc_descendant_b_m1_0_1_3"], 
			["dc_descendant_b_m1_0_2_3"]],
			[["dc_descendant_b_m1_1_1_2", "dc_descendant_b_m1_1_1_3"],
			["dc_descendant_b_m1_1_2_3"]]
			]
		self.assertListEqual(opt.my_colnames_dc_descendant_a_p1, colnames_a_p1)
		self.assertListEqual(opt.my_colnames_dc_descendant_b_p1, colnames_b_p1)
		self.assertListEqual(opt.my_colnames_dc_descendant_a_m1, colnames_a_m1)
		self.assertListEqual(opt.my_colnames_dc_descendant_b_m1, colnames_b_m1)
		self.assertListEqual(opt.my_colnames, opt.flatten_3d_list(colnames_a_p1) +
			opt.flatten_3d_list(colnames_b_p1) + opt.flatten_3d_list(colnames_a_m1)
			+ opt.flatten_3d_list(colnames_b_m1))


	def test_create_dc_ancestral_indices(self):
		# set up variables, 4 sublineages
		sublin_num = 5
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_dc_ancestral()

		# function to test
		start_index = 5
		dc_ancestral_indices = opt.create_dc_ancestral_indices(start_index)

		my_list = [[[5], [6, 7], [8, 9, 10]],
			[[11], [12, 13], [14, 15, 16]]]
		self.assertEqual(dc_ancestral_indices, my_list)
		
	
	def test_vars_dc_ancestral(self):
		
		# set up variables, only 2 sublineages
		sublin_num = 2
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_dc_ancestral()

		self.assertEqual(len(opt.my_rhs), 0)

		# set up variables, 4 sublineages
		sublin_num = 4
		self.seg_num = 2
		self.seg_spl_list = [self.seg_spl_1] * self.seg_num
		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)

		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test 
		opt.vars_dc_ancestral()

		entries = 3 * self.seg_num * cons.PHASE_NUMBER

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries 
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries 
		self.assertListEqual(opt.my_ctype, t)

		colnames_a_m1 = [
			[["dc_ancestral_a_m1_0_2_1"],
			["dc_ancestral_a_m1_0_3_1", "dc_ancestral_a_m1_0_3_2"]],
			[["dc_ancestral_a_m1_1_2_1"],
			["dc_ancestral_a_m1_1_3_1", "dc_ancestral_a_m1_1_3_2"]]
			]
		colnames_b_m1 = [
			[["dc_ancestral_b_m1_0_2_1"],
			["dc_ancestral_b_m1_0_3_1", "dc_ancestral_b_m1_0_3_2"]],
			[["dc_ancestral_b_m1_1_2_1"],
			["dc_ancestral_b_m1_1_3_1", "dc_ancestral_b_m1_1_3_2"]]
			]
		self.assertListEqual(opt.my_colnames_dc_ancestral_a_m1, colnames_a_m1)
		self.assertListEqual(opt.my_colnames_dc_ancestral_b_m1, colnames_b_m1)
		self.assertListEqual(opt.my_colnames, opt.flatten_3d_list(colnames_a_m1)
			+ opt.flatten_3d_list(colnames_b_m1))


	def test_vars_dssm_infl_cnv(self):
	
		# only 2 sublineages
		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_dssm_infl_cnv()

		entries = 0

		# when sublin_num <= 2 nothing should happen
		self.assertListEqual(opt.my_obj, [])
		
		# 4 sublineages
		sublin_num = 4
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_dssm_infl_cnv()

		entries = 3 * self.ssm_num * cons.PHASE_NUMBER * opt.cnv_state_num

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		colnames_a_p1 = [[["dssm_infl_cnv_a_p1_0_1_2", "dssm_infl_cnv_a_p1_0_1_3"], ["dssm_infl_cnv_a_p1_0_2_3"]],
			[["dssm_infl_cnv_a_p1_1_1_2", "dssm_infl_cnv_a_p1_1_1_3"], ["dssm_infl_cnv_a_p1_1_2_3"]]]
		colnames_b_p1 = [[["dssm_infl_cnv_b_p1_0_1_2", "dssm_infl_cnv_b_p1_0_1_3"], ["dssm_infl_cnv_b_p1_0_2_3"]],
			[["dssm_infl_cnv_b_p1_1_1_2", "dssm_infl_cnv_b_p1_1_1_3"], ["dssm_infl_cnv_b_p1_1_2_3"]]]
		colnames_a_m1 = [[["dssm_infl_cnv_a_m1_0_1_2", "dssm_infl_cnv_a_m1_0_1_3"], ["dssm_infl_cnv_a_m1_0_2_3"]],
			[["dssm_infl_cnv_a_m1_1_1_2", "dssm_infl_cnv_a_m1_1_1_3"], ["dssm_infl_cnv_a_m1_1_2_3"]]]
		colnames_b_m1 = [[["dssm_infl_cnv_b_m1_0_1_2", "dssm_infl_cnv_b_m1_0_1_3"], ["dssm_infl_cnv_b_m1_0_2_3"]],
			[["dssm_infl_cnv_b_m1_1_1_2", "dssm_infl_cnv_b_m1_1_1_3"], ["dssm_infl_cnv_b_m1_1_2_3"]]]
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_a_p1, colnames_a_p1)
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_b_p1, colnames_b_p1)
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_a_m1, colnames_a_m1)
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_b_m1, colnames_b_m1)
		self.assertListEqual(opt.my_colnames, opt.flatten_3d_list(colnames_a_p1) + opt.flatten_3d_list(colnames_b_p1)
			+ opt.flatten_3d_list(colnames_a_m1) + opt.flatten_3d_list(colnames_b_m1))

	def test_vars_ssm_aux_2_cn(self):

		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_ssm_aux_2_cn()

		entries = opt.ssm_num * opt.sublin_num * opt.sublin_num * cons.PHASE_NUMBER * opt.cnv_state_num

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		colname_a_p1 = [[["dssm_aux_2_cn_a_p1_0_0_0", "dssm_aux_2_cn_a_p1_0_0_1"], 
			["dssm_aux_2_cn_a_p1_0_1_0", "dssm_aux_2_cn_a_p1_0_1_1"]],
			[["dssm_aux_2_cn_a_p1_1_0_0", "dssm_aux_2_cn_a_p1_1_0_1"],
			["dssm_aux_2_cn_a_p1_1_1_0", "dssm_aux_2_cn_a_p1_1_1_1"]]]
		self.assertListEqual(opt.my_colnames_dssm_aux_2_cn_a_p1, colname_a_p1)
		colname_b_p1 = [[["dssm_aux_2_cn_b_p1_0_0_0", "dssm_aux_2_cn_b_p1_0_0_1"], 
			["dssm_aux_2_cn_b_p1_0_1_0", "dssm_aux_2_cn_b_p1_0_1_1"]],
			[["dssm_aux_2_cn_b_p1_1_0_0", "dssm_aux_2_cn_b_p1_1_0_1"],
			["dssm_aux_2_cn_b_p1_1_1_0", "dssm_aux_2_cn_b_p1_1_1_1"]]]
		self.assertListEqual(opt.my_colnames_dssm_aux_2_cn_b_p1, colname_b_p1)
		colname_a_m1 = [[["dssm_aux_2_cn_a_m1_0_0_0", "dssm_aux_2_cn_a_m1_0_0_1"], 
			["dssm_aux_2_cn_a_m1_0_1_0", "dssm_aux_2_cn_a_m1_0_1_1"]],
			[["dssm_aux_2_cn_a_m1_1_0_0", "dssm_aux_2_cn_a_m1_1_0_1"],
			["dssm_aux_2_cn_a_m1_1_1_0", "dssm_aux_2_cn_a_m1_1_1_1"]]]
		self.assertListEqual(opt.my_colnames_dssm_aux_2_cn_a_m1, colname_a_m1)
		colname_b_m1 = [[["dssm_aux_2_cn_b_m1_0_0_0", "dssm_aux_2_cn_b_m1_0_0_1"], 
			["dssm_aux_2_cn_b_m1_0_1_0", "dssm_aux_2_cn_b_m1_0_1_1"]],
			[["dssm_aux_2_cn_b_m1_1_0_0", "dssm_aux_2_cn_b_m1_1_0_1"],
			["dssm_aux_2_cn_b_m1_1_1_0", "dssm_aux_2_cn_b_m1_1_1_1"]]]
		self.assertListEqual(opt.my_colnames_dssm_aux_2_cn_b_m1, colname_b_m1)
		self.assertListEqual(opt.my_colnames, opt.flatten_3d_list(colname_a_p1) + opt.flatten_3d_list(colname_b_p1)
			+ opt.flatten_3d_list(colname_a_m1) + opt.flatten_3d_list(colname_b_m1))

	def test_vars_ssm_aux_15_cn(self):

		sublin_num = 3

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_ssm_aux_15_cn()

		entries = 8

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		colname_a_p1 = [["dssm_aux_15_cn_a_p1_0_1", "dssm_aux_15_cn_a_p1_0_2"], 
			["dssm_aux_15_cn_a_p1_1_1", "dssm_aux_15_cn_a_p1_1_2"]]
		self.assertListEqual(opt.my_colnames_dssm_aux_15_cn_a_p1, colname_a_p1)
		colname_b_p1 = [["dssm_aux_15_cn_b_p1_0_1", "dssm_aux_15_cn_b_p1_0_2"], 
			["dssm_aux_15_cn_b_p1_1_1", "dssm_aux_15_cn_b_p1_1_2"]]
		self.assertListEqual(opt.my_colnames_dssm_aux_15_cn_b_p1, colname_b_p1)
		self.assertListEqual(opt.my_colnames, opt.flatten_list(colname_a_p1) 
			+ opt.flatten_list(colname_b_p1))


	def test_vars_dssm_infl_cnv_same_lineage(self):

		sublin_num = 3
		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, 
			self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		# function to test
		opt.vars_dssm_infl_cnv_same_lineage()

		entries = 8

		#obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		namesA = [["dssm_infl_cnv_same_lineage_a_p1_0_1", "dssm_infl_cnv_same_lineage_a_p1_0_2"],
			["dssm_infl_cnv_same_lineage_a_p1_1_1", "dssm_infl_cnv_same_lineage_a_p1_1_2"]]
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_same_lineage_a, namesA)
		namesB = [["dssm_infl_cnv_same_lineage_b_p1_0_1", "dssm_infl_cnv_same_lineage_b_p1_0_2"],
			["dssm_infl_cnv_same_lineage_b_p1_1_1", "dssm_infl_cnv_same_lineage_b_p1_1_2"]]
		self.assertListEqual(opt.my_colnames_dssm_infl_cnv_same_lineage_b, namesB)
		self.assertListEqual(opt.my_colnames, opt.flatten_list(namesA) +
			opt.flatten_list(namesB))
	


	def test_vars_ssm_aux_1_cn(self):
		
		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
	
		opt.vars_ssm_aux_1_cn()

		entries = opt.ssm_num * opt.sublin_num

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		dssm = [["dssm_aux_1_cn_0_0", "dssm_aux_1_cn_0_1"], ["dssm_aux_1_cn_1_0", "dssm_aux_1_cn_1_1"]]
		self.assertListEqual(opt.my_colnames_dssm_aux_1_cn, dssm)
		self.assertListEqual(opt.my_colnames, opt.flatten_list(dssm))



	def test_vars_three_ssm_matrices(self):

		# for simple CN changes, with dssm
		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()

		entries = opt.ssm_num * opt.sublin_num * cons.SNP_SSM_PHASING_MATRIX_NUM

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		col_dssm = [["dssm_0_0", "dssm_0_1"], ["dssm_1_0", "dssm_1_1"]]
		col_dssm_a = [["dssm_a_0_0", "dssm_a_0_1"], ["dssm_a_1_0", "dssm_a_1_1"]]
		col_dssm_b = [["dssm_b_0_0", "dssm_b_0_1"], ["dssm_b_1_0", "dssm_b_1_1"]]
		colname = opt.flatten_list(col_dssm) + opt.flatten_list(col_dssm_a) + opt.flatten_list(col_dssm_b)
		self.assertListEqual(opt.my_colnames_dssm, col_dssm)
		self.assertListEqual(opt.my_colnames_dssm_a, col_dssm_a)
		self.assertListEqual(opt.my_colnames_dssm_b, col_dssm_b)
		self.assertListEqual(opt.my_colnames, colname)

		# test start indeces of variables in my_colnames
		self.assertEqual(opt.dssm_start_index, 0)
		self.assertEqual(opt.my_colnames[opt.dssm_start_index + sublin_num * self.ssm_num + 1], "dssm_a_0_1")
		# add another time
		opt.vars_three_ssm_matrices()
		self.assertEqual(opt.dssm_start_index, sublin_num * self.ssm_num * cons.SNP_SSM_PHASING_MATRIX_NUM)
		
		###############################################
		# for more complicated CN changes, without dssm
		sublin_num = 2

		self.ssm_num = 2
		self.ssm_spl_list = [self.ssm_spl_1] * self.ssm_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list,
			simple_CN_changes=False)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_three_ssm_matrices()

		entries = opt.ssm_num * opt.sublin_num * cons.PHASE_NUMBER

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# names
		col_dssm = []
		col_dssm_a = [["dssm_a_0_0", "dssm_a_0_1"], ["dssm_a_1_0", "dssm_a_1_1"]]
		col_dssm_b = [["dssm_b_0_0", "dssm_b_0_1"], ["dssm_b_1_0", "dssm_b_1_1"]]
		colname = opt.flatten_list(col_dssm) + opt.flatten_list(col_dssm_a) + opt.flatten_list(col_dssm_b)
		self.assertListEqual(opt.my_colnames_dssm, col_dssm)
		self.assertListEqual(opt.my_colnames_dssm_a, col_dssm_a)
		self.assertListEqual(opt.my_colnames_dssm_b, col_dssm_b)
		self.assertListEqual(opt.my_colnames, colname)

		# test start indeces of variables in my_colnames
		self.assertEqual(opt.dssm_start_index, 0)
		self.assertEqual(opt.my_colnames[opt.dssm_start_index + sublin_num * self.ssm_num + 1], "dssm_b_0_1")
		# add another time
		opt.vars_three_ssm_matrices()
		self.assertEqual(opt.dssm_start_index, sublin_num * self.ssm_num * cons.PHASE_NUMBER)

	#def test_vars_snp_cn(self):

	#	sublin_num = 1

	#	self.snp_num = 2
	#	self.snp_spl_list = [self.snp_spl_1] * self.snp_num 

	#	opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
	#	opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

	#	opt.vars_snp_cn()

	#	# obj
	#	obj = [0.0] * self.snp_num
	#	self.assertListEqual(opt.my_obj, obj)

	#	# ub
	#	ub = [cplex.infinity] * self.snp_num 
	#	self.assertListEqual(opt.my_ub, ub)

	#	# lb
	#	lb = [0.0] * self.snp_num
	#	self.assertListEqual(opt.my_lb, lb)

	#	# types
	#	t = ["C"] * self.snp_num
	#	self.assertListEqual(opt.my_ctype, t)

	#	# names
	#	colnames = ["snp_cn_0", "snp_cn_1"]
	#	self.assertListEqual(opt.my_colnames, colnames)
	#	self.assertListEqual(opt.my_colnames_snp_cn, colnames)


	def test_vars_aux_snp_linear(self):

		sublin_num = 3

		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		
		opt.vars_aux_snp_linear()

		# obj
		obj = [0.0] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * opt.snp_aux_linear_variables_num
		self.assertListEqual(opt.my_ctype, t)

		# names
		snp_w_cnv_a_p1 = [["snp_w_cnv_a_p1_0_1", "snp_w_cnv_a_p1_0_2"], ["snp_w_cnv_a_p1_1_1",
			"snp_w_cnv_a_p1_1_2"]]
		snp_w_cnv_b_p1 = [["snp_w_cnv_b_p1_0_1", "snp_w_cnv_b_p1_0_2"], ["snp_w_cnv_b_p1_1_1",
			"snp_w_cnv_b_p1_1_2"]]
		snp_w_cnv_a_m1 = [["snp_w_cnv_a_m1_0_1", "snp_w_cnv_a_m1_0_2"], ["snp_w_cnv_a_m1_1_1",
			"snp_w_cnv_a_m1_1_2"]]
		snp_w_cnv_b_m1 = [["snp_w_cnv_b_m1_0_1", "snp_w_cnv_b_m1_0_2"], ["snp_w_cnv_b_m1_1_1",
			"snp_w_cnv_b_m1_1_2"]]
		colnames = (opt.flatten_list(snp_w_cnv_a_p1) + opt.flatten_list(snp_w_cnv_b_p1) 
			+ opt.flatten_list(snp_w_cnv_a_m1) + opt.flatten_list(snp_w_cnv_b_m1))
		self.assertListEqual(opt.my_colnames_snp_w_cnv_a_p1, snp_w_cnv_a_p1)
		self.assertListEqual(opt.my_colnames_snp_w_cnv_b_p1, snp_w_cnv_b_p1)
		self.assertListEqual(opt.my_colnames_snp_w_cnv_a_m1, snp_w_cnv_a_m1)
		self.assertListEqual(opt.my_colnames_snp_w_cnv_b_m1, snp_w_cnv_b_m1)
		self.assertListEqual(opt.my_colnames, colnames)


	def test_vars_three_snp_matrices(self):

		sublin_num = 1

		self.snp_num = 2
		self.snp_spl_list = [self.snp_spl_1] * self.snp_num 

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_three_snp_matrices()

		entries = opt.delta_snp_entry_num * cons.SNP_SSM_PHASING_MATRIX_NUM

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		name_dsnp = ["dsnp_0", "dsnp_1"]
		name_dsnp_a = ["dsnp_a_0", "dsnp_a_1"]
		name_dsnp_b = ["dsnp_b_0", "dsnp_b_1"]
		colname = name_dsnp + name_dsnp_a + name_dsnp_b
		self.assertListEqual(opt.my_colnames_dsnp, name_dsnp)
		self.assertListEqual(opt.my_colnames_dsnp_a, name_dsnp_a)
		self.assertListEqual(opt.my_colnames_dsnp_b, name_dsnp_b)
		self.assertListEqual(opt.my_colnames, colname)

		# test start index in my_colnames
		self.assertEqual(opt.dsnp_start_index, 0)
		self.assertEqual(opt.my_colnames[opt.dsnp_start_index + self.snp_num], "dsnp_a_0")
		# add one more
		opt.vars_three_snp_matrices()
		self.assertEqual(opt.dsnp_start_index, self.snp_num * cons.SNP_SSM_PHASING_MATRIX_NUM)

	def test_vars_z_trans(self):
		# 4 lineages
		sublin_num = 5

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_z_trans()

		entries = 4 * 2

		# obj
		obj = [-0.00001] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		names_3d_i = [[["z_trans_i_1_2_3", "z_trans_i_1_2_4"], ["z_trans_i_1_3_4"]], [["z_trans_i_2_3_4"]]]
		names_3d_c = [[["z_trans_c_1_2_3", "z_trans_c_1_2_4"], ["z_trans_c_1_3_4"]], [["z_trans_c_2_3_4"]]]
		flat_lists = ["z_trans_i_1_2_3", "z_trans_i_1_2_4", "z_trans_i_1_3_4", "z_trans_i_2_3_4",
			"z_trans_c_1_2_3", "z_trans_c_1_2_4", "z_trans_c_1_3_4", "z_trans_c_2_3_4"]
		self.assertEqual(names_3d_i, opt.my_colnames_z_trans_i)
		self.assertEqual(names_3d_c, opt.my_colnames_z_trans_c)
		self.assertEqual(flat_lists, opt.my_colnames)

		# indices
		indices_3d_i = [[[0, 1], [2]], [[3]]]
		indices_3d_c = [[[4, 5], [6]], [[7]]]
		self.assertEqual(indices_3d_i, opt.z_trans_i_index)
		self.assertEqual(indices_3d_c, opt.z_trans_c_index)
		self.assertEqual(opt.z_trans_i_index_start, 0)
		self.assertEqual(opt.z_trans_c_index_start, 4)


		# 3 lineages
		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_z_trans()

		entries = 0

		# obj
		obj = [-0.00000005] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		names_3d_i = []
		names_3d_c = []
		flat_lists = []
		self.assertEqual(names_3d_i, opt.my_colnames_z_trans_i)
		self.assertEqual(names_3d_c, opt.my_colnames_z_trans_c)
		self.assertEqual(flat_lists, opt.my_colnames)

		# indices
		indices_3d_i = []
		indices_3d_c = []
		self.assertEqual(indices_3d_i, opt.z_trans_i_index)
		self.assertEqual(indices_3d_c, opt.z_trans_c_index)
		self.assertEqual(opt.z_trans_i_index_start, 0)
		self.assertEqual(opt.z_trans_c_index_start, 0)

	def test_vars_z(self):

		sublin_num = 2

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_z()

		entries = sublin_num * sublin_num

		# obj
		obj = [0.0] * entries
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * entries
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * entries
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["B"] * entries
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		names_2d = [["z_0_0", "z_0_1"], ["z_1_0", "z_1_1"]]
		names_flat = opt.flatten_list(names_2d)
		self.assertListEqual(opt.my_colnames, names_flat)
		self.assertListEqual(opt.my_colnames_z, names_2d)

		# test start index in my_colnames
		self.assertEqual(opt.z_index_start, 0)
		# insert one more
		opt.vars_z()
		self.assertEqual(opt.z_index_start, sublin_num*sublin_num)

	def test_vars_aux_dc(self):

		sublin_num = 3

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)

		opt.vars_aux_dc()

		variables_num = opt.cnv_aux_all_states_one_type_entry_num * opt.aux_matrices_cnv_linear_types_num
		
		# obj
		obj = [0.0] * variables_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * variables_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * variables_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = (["B"] * opt.cnv_aux_all_states_one_type_entry_num + 
			["C"] * opt.cnv_aux_all_states_one_type_entry_num)
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		colname_a_p1_binary = [["dc_a_p1_binary_0_0", "dc_a_p1_binary_0_1", "dc_a_p1_binary_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_a_p1_binary, colname_a_p1_binary)
		colname_b_p1_binary = [["dc_b_p1_binary_0_0", "dc_b_p1_binary_0_1", "dc_b_p1_binary_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_b_p1_binary, colname_b_p1_binary)
		colname_a_m1_binary = [["dc_a_m1_binary_0_0", "dc_a_m1_binary_0_1", "dc_a_m1_binary_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_a_m1_binary, colname_a_m1_binary)
		colname_b_m1_binary = [["dc_b_m1_binary_0_0", "dc_b_m1_binary_0_1", "dc_b_m1_binary_0_2"]] 
		self.assertListEqual(opt.my_colnames_dc_b_m1_binary, colname_b_m1_binary)

		colname_a_p1_float = [["dc_a_p1_float_0_0", "dc_a_p1_float_0_1", "dc_a_p1_float_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_a_p1_float, colname_a_p1_float)
		colname_b_p1_float = [["dc_b_p1_float_0_0", "dc_b_p1_float_0_1", "dc_b_p1_float_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_b_p1_float, colname_b_p1_float)
		colname_a_m1_float = [["dc_a_m1_float_0_0", "dc_a_m1_float_0_1", "dc_a_m1_float_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_a_m1_float, colname_a_m1_float)
		colname_b_m1_float = [["dc_b_m1_float_0_0", "dc_b_m1_float_0_1", "dc_b_m1_float_0_2"]]
		self.assertListEqual(opt.my_colnames_dc_b_m1_float, colname_b_m1_float)


		colname = (opt.flatten_list(colname_a_p1_binary) + opt.flatten_list(colname_b_p1_binary) +
			opt.flatten_list(colname_a_m1_binary) + opt.flatten_list(colname_b_m1_binary) +
			opt.flatten_list(colname_a_p1_float) + opt.flatten_list(colname_b_p1_float) +
			opt.flatten_list(colname_a_m1_float) + opt.flatten_list(colname_b_m1_float))
		self.assertListEqual(opt.my_colnames, colname)

		# test indices in list my_colnames
		self.assertEqual(opt.dc_binary_index_start_p1, 0)
		self.assertEqual(opt.my_colnames[opt.dc_binary_index_start_p1 + sublin_num], "dc_b_p1_binary_0_0")
		self.assertEqual(opt.dc_binary_index_start_m1, self.seg_num * sublin_num * cons.PHASE_NUMBER)
		self.assertEqual(opt.my_colnames[opt.dc_binary_index_start_m1], "dc_a_m1_binary_0_0")
		# insert again to test if index is increased
		opt.vars_aux_dc()
		self.assertEqual(opt.dc_binary_index_start_p1, opt.cnv_aux_all_states_one_type_entry_num * 
			opt.aux_matrices_cnv_linear_types_num)
		self.assertEqual(opt.dc_binary_index_start_m1, opt.dc_binary_index_start_p1 + (self.seg_num *
			sublin_num * cons.PHASE_NUMBER))

	def test_vars_phi(self):
		sublin_num = 5

		opt = optimization.Optimization_with_CPLEX(self.seg_spl_list, self.snp_spl_list, self.ssm_spl_list)
		opt.set_other_parameter(sublin_num, self.snp_list, self.ssm_list, self.seg_list)
		opt.vars_phi()

		# test objective
		obj = [0.0] * sublin_num
		self.assertListEqual(opt.my_obj, obj)

		# ub
		ub = [1.0] * sublin_num
		self.assertListEqual(opt.my_ub, ub)

		# lb
		lb = [0.0] * sublin_num
		self.assertListEqual(opt.my_lb, lb)

		# types
		t = ["C"] * sublin_num
		self.assertListEqual(opt.my_ctype, t)

		# colnames
		col = []
		for i in range(sublin_num):
			col.append("phi_{0}".format(i))
		self.assertListEqual(opt.my_phis, col)
		self.assertListEqual(opt.my_colnames, col)

		# test index value
		self.assertEqual(opt.phi_start_index, 0)
		# insert phis again to see if indexing works
		opt.vars_phi()
		self.assertEqual(opt.phi_start_index, sublin_num)

	def print_variable_values(self, obj, ub, lb, t, col):
		print ""
		print obj
		print ub
		print lb
		print t
		print col

	def test_vars_aux_vars_mut_splines(self):
		# create three different splines and lists
		s1 = log_pdf.find_piecewise_linear(self.number_spline_points, 300,400,1000,
			log_pdf.neg_binomial, 340,1000)[0]
		s2 = log_pdf.find_piecewise_linear(self.number_spline_points, 0.5,0.7,1000,
			log_pdf.beta_binomial,220,340,1000)[0]
		s3 = log_pdf.find_piecewise_linear(self.number_spline_points, 0.6,0.65,1000,
			log_pdf.beta_binomial,220,340,1000)[0]
		spl_seq = [s1] * 2
		spl_snp = [s2] * 3
		spl_ssm = [s3] * 4
		spline_list = [spl_seq, spl_snp, spl_ssm]


		opt = optimization.Optimization_with_CPLEX(spl_seq, spl_snp, spl_ssm)

		opt.vars_aux_vars_mut_splines()

		# test objective
		obj = []
		coeff_num = 0
		for elem in spline_list:
			for spl in elem:
				obj.append(spl.get_coeffs().tolist())
				coeff_num = coeff_num + len(spl.get_coeffs())
		obj = [item for sublist in obj for item in sublist]
		self.assertListEqual(opt.my_obj, obj)

		# test upper bound
		ub = [1.0] * coeff_num
		self.assertListEqual(opt.my_ub, ub)

		# test lower bound
		lb = [0.0] * coeff_num
		self.assertListEqual(opt.my_lb, lb)

		# test types
		t = ["C"] * coeff_num
		self.assertListEqual(opt.my_ctype, t)

		# test colnames
		col_names = []
		col_seg = []
		col_snp = []
		col_ssm = []
		for i, elem in enumerate(spline_list):
			mut = "seg"
			if i == 1:
				mut = "snp"
			elif i == 2:
				mut = "ssm"
			for k, spl in enumerate(elem):
				tmp_col = []
				for j in range(len(spl.get_coeffs())):
					name = "{0}_a_{1}_{2}".format(mut, k, j)
					col_names.append(name)
					tmp_col.append(name)
				if i == 0:
					col_seg.append(tmp_col)
				elif i == 1:
					col_snp.append(tmp_col)
				elif i == 2:
					col_ssm.append(tmp_col)
					
		self.assertListEqual(opt.my_colnames, col_names)
		self.assertListEqual(opt.my_colnames_seg, col_seg)
		self.assertListEqual(opt.my_colnames_snp, col_snp)
		self.assertListEqual(opt.my_colnames_ssm, col_ssm)

		# variables in case with allele specific CNs
		# create three different splines and lists
		spl_seg_A = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 6.3,1000,
			log_pdf.log_normal, 1.3, 0.25)[0]
		spl_seg_B = log_pdf.find_piecewise_linear(self.number_spline_points, 0, 5.3,1000,
			log_pdf.log_normal, 0.3, 0.25)[0]
		spl_ssm = log_pdf.find_piecewise_linear(self.number_spline_points, 0.6,0.65,1000,
			log_pdf.beta_binomial,220,340,1000)[0]
		spl_seg_A_list = [spl_seg_A]
		spl_seg_B_list = [spl_seg_B]
		spl_ssm_list = [spl_ssm]
		spline_list = [spl_seg_A_list, spl_seg_B_list, spl_ssm_list]

		opt = optimization.Optimization_with_CPLEX([], [], spl_ssm_list, allele_specific=True,
			seg_splines_A=spl_seg_A_list, seg_splines_B=spl_seg_B_list)

		opt.vars_aux_vars_mut_splines()

		# test objective
		obj = []
		coeff_num = 0
		for elem in spline_list:
			for spl in elem:
				obj.append(spl.get_coeffs().tolist())
				coeff_num = coeff_num + len(spl.get_coeffs())
		obj = [item for sublist in obj for item in sublist]
		self.assertListEqual(opt.my_obj, obj)

		# test upper bound
		ub = [1.0] * coeff_num
		self.assertListEqual(opt.my_ub, ub)

		# test lower bound
		lb = [0.0] * coeff_num
		self.assertListEqual(opt.my_lb, lb)

		# test types
		t = ["C"] * coeff_num
		self.assertListEqual(opt.my_ctype, t)

		# test colnames
		col_names = []
		col_seg_A = []
		col_seg_B = []
		col_ssm = []
		for i, elem in enumerate(spline_list):
			mut = "seg_A"
			if i == 1:
				mut = "seg_B"
			elif i == 2:
				mut = "ssm"
			for k, spl in enumerate(elem):
				tmp_col = []
				for j in range(len(spl.get_coeffs())):
					name = "{0}_a_{1}_{2}".format(mut, k, j)
					col_names.append(name)
					tmp_col.append(name)
				if i == 0:
					col_seg_A.append(tmp_col)
				elif i == 1:
					col_seg_B.append(tmp_col)
				elif i == 2:
					col_ssm.append(tmp_col)
					
		self.assertListEqual(opt.my_colnames, col_names)
		self.assertListEqual(opt.my_colnames_seg_A, col_seg_A)
		self.assertListEqual(opt.my_colnames_seg_B, col_seg_B)
		self.assertListEqual(opt.my_colnames_ssm, col_ssm)

		


def suite():
	return unittest.TestLoader().loadTestsFromTestCase(OptimizationTest)

