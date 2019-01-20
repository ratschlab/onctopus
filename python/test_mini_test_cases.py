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
import json
from cplex.exceptions import CplexError
import constants as cons
import exceptions_onctopus as eo

class MiniTestCasesTest(unittest.TestCase):
	
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

	def test_1_1_only_CNV_p1(self):
		input_seg = "testdata/unittests/mini_test_cases/1_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True)

		lineages = oio.read_result_file(out_results)
		
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_1_1_only_CNV_p1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/1_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_1_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)
		
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_1_2_only_CNV_m1(self):
		input_seg = "testdata/unittests/mini_test_cases/1_2_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_2"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads,
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry,
			self.strategy_file, self.workmem, self.workdir, self.treememory,
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_1_2_only_CNV_m1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/1_2_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_2_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)
		
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
	
	def test_1_3_only_CNV_0(self):
		input_seg = "testdata/unittests/mini_test_cases/1_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_3"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))

	def test_1_3_only_CNV_0__allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/1_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		fixed_phi = "testdata/unittests/mini_test_cases/1_3_fixed_phi"
		out_results = "testdata/unittests/mini_test_cases/result_1_3_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True, fixed_phi_file=fixed_phi)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))


	def test_1_4_CNV_p1m1_2SNPs(self):
		input_seg = "testdata/unittests/mini_test_cases/1_4_seg"
		input_snp = "testdata/unittests/mini_test_cases/1_4_snp"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_4"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		snp2 = snp_ssm.SNP()
		snp2.chr = 1
		snp2.pos = 1
		self.assertEqual(lineages[0].snps_a[0], snp1)
		self.assertEqual(lineages[0].snps_b[0], snp2)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_1_4_CNV_p1m1_2SNPs_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/1_4_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_4_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_1_5_avg_CN_fixed_all(self):
		input_seg = "testdata/unittests/mini_test_cases/1_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_5"
		fixed_avg_cn = "testdata/unittests/mini_test_cases/1_5_fixed_copy_numbers"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, 
			fixed_avg_cn_file=fixed_avg_cn, 
			test_run=True)

		lineages = oio.read_result_file(out_results)
			
		self.assertEqual(model.compute_average_cn(lineages, 3)[0], 2)
		self.assertEqual(model.compute_average_cn(lineages, 3)[1], 2.5)
		self.assertEqual(model.compute_average_cn(lineages, 3)[2], 1.5)

	def test_1_5_avg_CN_fixed_all_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/1_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_1_5_as"
		fixed_avg_cn = "testdata/unittests/mini_test_cases/1_5_fixed_copy_numbers"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, 
			fixed_avg_cn_file=fixed_avg_cn, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)
			
		self.assertEqual(model.compute_average_cn(lineages, 3)[0], 2)
		self.assertAlmostEqual(model.compute_average_cn(lineages, 3)[1], 2.5)
		self.assertAlmostEqual(model.compute_average_cn(lineages, 3)[2], 1.5)

	def test_2_1_only_CNV_1_lineage1(self):
		input_seg = "testdata/unittests/mini_test_cases/2_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_1"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, 
			fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_2_1_only_CNV_1_lineage1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_1_as"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_2_2_only_CNV_1_lineage1_z_0(self):
		input_seg = "testdata/unittests/mini_test_cases/2_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_2"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_2_2_only_CNV_1_lineage1_z_0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_2_as"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_2_3_only_CNV_1_lineage2(self):
		input_seg = "testdata/unittests/mini_test_cases/2_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_3"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_2_3_only_CNV_1_lineage2_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_3_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_2_4_only_CNV_1_lineage2_z_0(self):
		input_seg = "testdata/unittests/mini_test_cases/2_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_4"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_2_4_only_CNV_1_lineage2_z_0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_2_5_only_CNV_m1_lineage1_z_1(self):
		input_seg = "testdata/unittests/mini_test_cases/2_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_5"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_5_only_CNV_m1_lineage1_z_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_5_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_6_only_CNV_m1_lineage1_z_0(self):
		input_seg = "testdata/unittests/mini_test_cases/2_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_6"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_6_only_CNV_m1_lineage1_z_0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_6_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_7_only_CNV_m1_lineage1_z_1(self):
		input_seg = "testdata/unittests/mini_test_cases/2_7_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_7"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_7_only_CNV_m1_lineage1_z_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_7_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_7_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_8_only_CNV_m1_lineage1_z_0(self):
		input_seg = "testdata/unittests/mini_test_cases/2_7_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_8"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_2_8_only_CNV_m1_lineage1_z_0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/2_7_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_2_8_as"
		fixed_phi = "testdata/unittests/mini_test_cases/2_3_fixed_phis"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_3_1_unphased_SNP(self):
		input_seg = "testdata/unittests/mini_test_cases/3_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/3_1_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_3_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps[0], snp1)

	def test_3_2_A_SNP_CNV_p1(self):
		input_seg = "testdata/unittests/mini_test_cases/3_2_seg"
		input_snp = "testdata/unittests/mini_test_cases/3_2_snp"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_3_2"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_a[0], snp1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_3_3_A_SNP_CNV_m1(self):
		input_seg = "testdata/unittests/mini_test_cases/3_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/3_3_snp"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_3_3"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_a[0], snp1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_3_4_B_SNP_CNV_m1(self):
		input_seg = "testdata/unittests/mini_test_cases/3_4_seg"
		input_snp = "testdata/unittests/mini_test_cases/3_4_snp"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_3_4"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_b[0], snp1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_3_5_B_SNP_CNV_p1(self):
		input_seg = "testdata/unittests/mini_test_cases/3_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/3_5_snp"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_3_5"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_b[0], snp1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)

	def test_4_1_unphased_SSM_CNV_0_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_1"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = (
			main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, test_run=True))

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertEqual(lineages[1].ssms[0].phase, cons.UNPHASED)
		self.assertEqual(lineages[1].ssms[0].lineage, 1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

		self.assertEqual(my_lineages[1].ssms[0].phase, cons.UNPHASED)
		self.assertEqual(my_lineages[1].ssms[0].lineage, 1)

	def test_4_1_unphased_SSM_CNV_0_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_1_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_4_2_unphased_SSM_CNV_0_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_2"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_4_2_unphased_SSM_CNV_0_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_2_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_4_3_unphased_SSM_CNV_0_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_3"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[2].freq, 0.3)

	def test_4_3_unphased_SSM_CNV_0_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_3_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True, 
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[2].freq, 0.3)

	def test_4_4_unphased_SSM_CNV_0_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_3_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_4"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_4_unphased_SSM_CNV_0_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_5_A_SSM_CNV_p1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_5"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_5_A_SSM_CNV_p1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_5_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_6_B_SSM_CNV_p1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_6_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_6"
		fixed_phi = "testdata/unittests/mini_test_cases/4_6_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)

	def test_4_6_B_SSM_CNV_p1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_6_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_6_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)

	def test_4_7_unphased_SSM_CNV_p1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_7_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_7"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_7_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_7_unphased_SSM_CNV_p1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_7_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_7_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_8_unphased_SSM_2_CNV_p1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_8"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_8_unphased_SSM_2_CNV_p1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_8_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_9_unphased_SSM_2_CNV_p1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_9"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_9_unphased_SSM_2_CNV_p1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_5_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_9_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_10_unphased_SSM_1_CNV_p1_1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_10"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_4_10_unphased_SSM_1_CNV_p1_1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_10_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_4_11_unphased_SSM_1_CNV_p1_1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_11"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_4_11_unphased_SSM_1_CNV_p1_1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_11_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_11_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			fixed_phi_file=fixed_phi)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_4_12_unphased_SSM_2_CNV_p1_1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_12_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_12"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# better approximation of freq of lineage 2 is not possible
		# because beta binomial function only approximates binomial function
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_4_12_unphased_SSM_2_CNV_p1_1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_12_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_12_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# better approximation of freq of lineage 2 is not possible
		# because beta binomial function only approximates binomial function
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_4_13_unphased_SSM_2_CNV_p1_1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_12_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_13"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# better approximation of freq of lineage 2 is not possible
		# because beta binomial function only approximates binomial function
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_4_13_unphased_SSM_2_CNV_p1_1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_12_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_13_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# better approximation of freq of lineage 2 is not possible
		# because beta binomial function only approximates binomial function
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_4_14_A_SSM_1_CNV_m1_2_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_14_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_14"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_14_A_SSM_1_CNV_m1_2_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_14_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_14_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_15_B_SSM_1_CNV_m1_2_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_15_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_15"
		fixed_phi = "testdata/unittests/mini_test_cases/4_15_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)

	def test_4_15_B_SSM_1_CNV_m1_2_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_15_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_15_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_15_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)

	def test_4_16_unphased_SSM_1_CNV_m1_2_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_14_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_16"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_16_unphased_SSM_1_CNV_m1_2_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_14_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_14_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_16_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_4_17_phased_SSM_1_A_CNV_m1_1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_17_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_17"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_4_17_phased_SSM_1_A_CNV_m1_1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_17_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_17_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_4_18_phased_SSM_1_A_CNV_m1_1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_17_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_18"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_4_18_phased_SSM_1_A_CNV_m1_1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_17_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_18_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_4_19_phased_SSM_2_A_CNV_m1_2_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_19_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_19_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_19"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_19_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_19_phased_SSM_2_A_CNV_m1_2_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_19_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_19_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_19_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_19_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_20_phased_SSM_2_A_CNV_m1_2_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_19_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_19_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_20"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_20_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_20_phased_SSM_2_A_CNV_m1_2_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_19_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_19_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_20_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_20_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_21_phased_SSM_2_A_CNV_m1_1_z1(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_21_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_21"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_21_phased_SSM_2_A_CNV_m1_1_z1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_21_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_21_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_1"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_4_22_unphased_SSM_2_CNV_m1_1_z0(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/4_21_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_22"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_4_22_unphased_SSM_2_CNV_m1_1_z0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/4_17_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/4_21_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_4_22_as"
		fixed_z = "testdata/unittests/mini_test_cases/4_fixed_z_3_lin_0"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_z_matrix_file=fixed_z,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_5_1_z_0_nothing(self):
		input_seg = "testdata/unittests/mini_test_cases/5_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].sublins, [])
		self.assertEqual(len(lineages), 1)

	def test_5_1_z_0_nothing_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_1_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].sublins, [])
		self.assertEqual(len(lineages), 1)

	def test_5_2_z_0_CNV_p1_1_SSM_2(self):
		input_seg = "testdata/unittests/mini_test_cases/5_2_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/5_2_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_2"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# doesn't work more precisely becaue of approximation of function with beta
		# binomial distribution
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_5_2_z_0_CNV_p1_1_SSM_2_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_2_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/5_2_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_2_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		# doesn't work more precisely becaue of approximation of function with beta
		# binomial distribution
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.01)

	def test_5_3_z_0_CNV_p1_1_SSM_1(self):
		input_seg = "testdata/unittests/mini_test_cases/5_2_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/5_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_3"
		fixed_phi = "testdata/unittests/mini_test_cases/5_3_as_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertEqual(len(lineages), 2)

	def test_5_3_z_0_CNV_p1_1_SSM_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_2_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/5_3_ssm"
		num = 3
		fixed_phi = "testdata/unittests/mini_test_cases/5_3_as_fixed_phi"
		out_results = "testdata/unittests/mini_test_cases/result_5_3_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(len(lineages), 2)

	def test_5_4_z_0_CNV_p1_2_SSM_1(self):
		input_seg = "testdata/unittests/mini_test_cases/5_4_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/5_4_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_4"
		fixed_phi = "testdata/unittests/mini_test_cases/5_4_as_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, fixed_phi_file=fixed_phi)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_5_4_z_0_CNV_p1_2_SSM_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_4_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/5_4_ssm_as"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/5_4_as_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertAlmostEqual(lineages[1].freq, 0.6, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.3, delta=0.001)

	def test_5_5_z_1_CNV_p1_2_SSM_1(self):
		input_seg = "testdata/unittests/mini_test_cases/5_4_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/5_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_5"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_5_5_z_1_CNV_p1_2_SSM_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_4_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/5_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_5_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_2_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_5_6_z_1_CNV_m1_2_SSM_1(self):
		input_seg = "testdata/unittests/mini_test_cases/5_6_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/5_6_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_6"
		fixed_phi = "testdata/unittests/mini_test_cases/4_15_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_5_6_z_1_CNV_m1_2_SSM_1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/5_6_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/5_6_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_5_6_as"
		fixed_phi = "testdata/unittests/mini_test_cases/4_15_fixed_phi"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_6_1_phis(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/6_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_1"
		cn_weight = 0.00001
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points, cn_weight=cn_weight,
			test_run=True)

		lineages = oio.read_result_file(out_results)
		
		# no CNV
		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		self.assertEqual(lineages[1].sublins, [])
		self.assertEqual(lineages[2].sublins, [])
		# assignment of SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		# frequencies
		self.assertAlmostEqual(lineages[1].freq, 0.6, places=3)
		self.assertAlmostEqual(lineages[2].freq, 0.3, places=2)

	def test_6_1_phis_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/6_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_1_as"
		
		# do optimization
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, 
			input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True, lineage_divergence_rule=True)

		lineages = oio.read_result_file(out_results)

		# no CNV
		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		self.assertEqual(lineages[1].sublins, [])
		self.assertEqual(lineages[2].sublins, [])
		# assignment of SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		# frequencies
		self.assertAlmostEqual(lineages[1].freq, 0.6, places=3)
		self.assertAlmostEqual(lineages[2].freq, 0.3, places=2)

	def test_6_2_phis_sum(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_2"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		fixed_phi = "testdata/unittests/mini_test_cases/6_2_fixed_phi"
		
		# optimization can't be done
		with self.assertRaises(main.optimization.cplex.exceptions.CplexSolverError):
			main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
				self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect,
				self.cplex_log_file, self.number_spline_points,
				fixed_phi_file=fixed_phi,
				fixed_z_matrix_file=fixed_z, test_run=True, 
				lineage_divergence_rule=True)


	def test_6_2_phis_sum_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_2_as"
		fixed_z = "testdata/unittests/mini_test_cases/2_1_fixed_z_0"
		fixed_phi = "testdata/unittests/mini_test_cases/6_2_fixed_phi"
		
		# optimization can't be done
		with self.assertRaises(main.optimization.cplex.exceptions.CplexSolverError):
			main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
				self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect,
				self.cplex_log_file, self.number_spline_points,
				fixed_phi_file=fixed_phi,
				fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
				lineage_divergence_rule=True)

	def test_6_3_phis_sum_z_unfixed(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_3"
		fixed_phi = "testdata/unittests/mini_test_cases/6_2_fixed_phi"
		
		# optimization can be done
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
			self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True)

		lineages = oio.read_result_file(out_results) 
		
		# lin 2 is a descendant of lin 1 because of lineage divergence rule
		self.assertEqual(lineages[1].sublins, [2])
		# frequencies are as assigned
		self.assertEqual(lineages[1].freq, 0.8)
		self.assertEqual(lineages[2].freq, 0.3)

	def test_6_3_phis_sum_z_unfixed_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_3_as"
		fixed_phi = "testdata/unittests/mini_test_cases/6_2_fixed_phi"
		
		# optimization can be done
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
			self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results) 
		
		# lin 2 is a descendant of lin 1 because of lineage divergence rule
		self.assertEqual(lineages[1].sublins, [2])
		# frequencies are as assigned
		self.assertEqual(lineages[1].freq, 0.8)
		self.assertEqual(lineages[2].freq, 0.3)

	def test_6_4_freq_0_no_muts_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/6_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/6_4_fixed_phi"
		fixed_ssm = "testdata/unittests/mini_test_cases/6_4_fixed_ssm"
		
		# optimization can be done
		with self.assertRaises(CplexError):
			main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
				self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect,
				self.cplex_log_file, self.number_spline_points,
				fixed_phi_file=fixed_phi, fixed_ssm_file=fixed_ssm,
				test_run=True, allele_specific=True)

	def test_6_5_freq_0_remove_lineage_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/6_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_5_as"
		fixed_phi = "testdata/unittests/mini_test_cases/6_4_fixed_phi"
		fixed_ssm = "testdata/unittests/mini_test_cases/6_4_fixed_ssm"
		
		# optimization can be done
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
			self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(len(lineages), 2)

	def test_6_6_freq_0_remove_lineage_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/6_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/6_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_6_6_as"
		fixed_phi = "testdata/unittests/mini_test_cases/6_4_fixed_phi"
		fixed_ssm = "testdata/unittests/mini_test_cases/6_4_fixed_ssm"
		
		# optimization can be done
		with self.assertRaises(eo.ZMatrixNotNone):
			main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, 
				self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect,
				self.cplex_log_file, self.number_spline_points,
				fixed_phi_file=fixed_phi, z_matrix_list=[],
				test_run=True, allele_specific=True)

	def test_7_1_CNV_0(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_1_fixed_cnv"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_1"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_cnv_file=fixed_cnv_file, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_7_1_CNV_0_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_1_fixed_cnv"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_1_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_cnv_file=fixed_cnv_file, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	# to specifically test the fixation of a segment via indices
	def test_7_1_CNV_0_allele_specific_2(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_index = [0]
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_1_as_2"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			 test_run=True, allele_specific=True, normal_seg_indices=fixed_cnv_index)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_7_2_CNV_p1(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_2_fixed_cnv"
		fixed_phi = "testdata/unittests/mini_test_cases/7_2_fixed_phi"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_2"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			fixed_cnv_file=fixed_cnv_file, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
	
	def test_7_2_CNV_p1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg_as"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_2_fixed_cnv"
		fixed_phi = "testdata/unittests/mini_test_cases/7_2_fixed_phi"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_2_as"
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi,
			fixed_cnv_file=fixed_cnv_file, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)

	def test_7_3_CNV_m1(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_3_fixed_cnv"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_3"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_cnv_file=fixed_cnv_file, test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_7_3_CNV_m1_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/7_seg_as"
		input_snp = "testdata/unittests/mini_test_cases/no_snps"
		input_ssm = "testdata/unittests/mini_test_cases/7_ssm"
		fixed_cnv_file = "testdata/unittests/mini_test_cases/7_3_fixed_cnv"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_7_3_as"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_cnv_file=fixed_cnv_file, test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_8_1_all_variables(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg"
		input_snp = "testdata/unittests/mini_test_cases/8_snp"
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_8_1"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)
		snp1 = snp_ssm.SSM()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_a[0], snp1)
		snp2 = snp_ssm.SSM()
		snp2.chr = 1
		snp2.pos = 1
		self.assertEqual(lineages[0].snps_b[0], snp2)

	def test_8_1_all_variables_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_8_1_as"

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)

	def test_9_1_warm_start_optimal(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg"
		input_snp = "testdata/unittests/mini_test_cases/8_snp"
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_9_1"
		fixed_z = "testdata/unittests/mini_test_cases/9_1_fixed_z_1"
		
		# get optimal result for warm start
		lineages = oio.read_result_file("testdata/unittests/mini_test_cases/result_8_1")
		values_dc_binary = model.create_fixed_CNV_data(lineages, 2)[0]
		values_dsnp = model.create_fixed_SNPs_data(lineages)[0]
		values_dssm = model.create_fixed_SSMs_data(lineages)[0]

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, warm_start_dc_binary=values_dc_binary, warm_start_dsnp=values_dsnp,
			warm_start_dssm=values_dssm, fixed_z_matrix_file=fixed_z)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)
		snp1 = snp_ssm.SSM()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_a[0], snp1)
		snp2 = snp_ssm.SSM()
		snp2.chr = 1
		snp2.pos = 1
		self.assertEqual(lineages[0].snps_b[0], snp2)

	def test_9_1_warm_start_optimal_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_9_1_as"
		fixed_z = "testdata/unittests/mini_test_cases/9_1_fixed_z_1"
		
		# get optimal result for warm start
		lineages = oio.read_result_file("testdata/unittests/mini_test_cases/result_8_1_as")
		values_dc_binary = model.create_fixed_CNV_data(lineages, 2)[0]
		values_dssm = model.create_fixed_SSMs_data(lineages)[0]

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, warm_start_dc_binary=values_dc_binary,
			warm_start_dssm=values_dssm, fixed_z_matrix_file=fixed_z, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)

	def test_9_2_warm_start_optimal(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg"
		input_snp = "testdata/unittests/mini_test_cases/8_snp"
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_9_2"
		fixed_z = "testdata/unittests/mini_test_cases/9_1_fixed_z_1"
		
		# get optimal result for warm start
		lineages = oio.read_result_file("testdata/unittests/mini_test_cases/result_8_1")
		values_dc_binary = model.create_fixed_CNV_data(lineages, 2)[0]
		values_dsnp = model.create_fixed_SNPs_data(lineages)[0]
		values_dssm = model.create_fixed_SSMs_data(lineages)[0]
		# change optimal result
		values_dssm[1][0][1] = 0
		values_dssm[1][0][2] = 1
		values_dc_binary[3][0][2] = 0

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, warm_start_dc_binary=values_dc_binary, warm_start_dsnp=values_dsnp,
			warm_start_dssm=values_dssm, fixed_z_matrix_file=fixed_z)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)
		snp1 = snp_ssm.SSM()
		snp1.chr = 1
		snp1.pos = 0
		self.assertEqual(lineages[0].snps_a[0], snp1)
		snp2 = snp_ssm.SSM()
		snp2.chr = 1
		snp2.pos = 1
		self.assertEqual(lineages[0].snps_b[0], snp2)

	def test_9_2_warm_start_optimal_allele_specific(self):
		input_seg = "testdata/unittests/mini_test_cases/8_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_9_2_as"
		fixed_z = "testdata/unittests/mini_test_cases/9_1_fixed_z_1"
		
		# get optimal result for warm start
		lineages = oio.read_result_file("testdata/unittests/mini_test_cases/result_8_1_as")
		values_dc_binary = model.create_fixed_CNV_data(lineages, 2)[0]
		values_dssm = model.create_fixed_SSMs_data(lineages)[0]
		# change optimal result
		values_dssm[1][0][1] = 0
		values_dssm[1][0][2] = 1
		values_dc_binary[3][0][2] = 0

		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			test_run=True, warm_start_dc_binary=values_dc_binary,
			warm_start_dssm=values_dssm, fixed_z_matrix_file=fixed_z, allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)

	def test_10_1_comb_1_2_gains_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_1_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms[1], ssm3)

	def test_10_2_comb_1_2_gains_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_2_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_2_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_2_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms_b[0], ssm3)

	def test_10_3_comb_1_2_gains_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_3_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms_a[0], ssm3)

	def test_10_4_comb_1_2_gains_3_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms[1], ssm3)

	def test_10_5_comb_1_2_gains_1_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_5_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 1
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms[1], ssm3)

	def test_10_6_comb_1_1_loss_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_6_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_6_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_6_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms_a[0], ssm2)

	def test_10_7_comb_1_2_losses_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_7_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_7_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_7_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_10_7_b_comb_1_2_losses_2_x_z_0(self):
		input_seg = "testdata/unittests/mini_test_cases/10_7_b_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_7_b_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_7_b_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_13_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_13_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_10_8_comb_1_1_LOH_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_8_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_8_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_8_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)

	def test_10_9_comb_1_1_gain_1_loss_2_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_9_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_9_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_9_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b, [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms[1], ssm3)
		
	def test_10_10_comb_1_2_gains_1_loss_3_x(self):
		input_seg = "testdata/unittests/mini_test_cases/10_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_10_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms[0], ssm2)

	def test_10_12_comb_4_2_losses_diff_lins(self):
		input_seg = "testdata/unittests/mini_test_cases/10_12_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_12_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_12_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)

	def test_10_13_comb_4_4_losses_diff_lins(self):
		input_seg = "testdata/unittests/mini_test_cases/10_13_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_13_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_10_13_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_13_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_13_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			fixed_z_matrix_file=fixed_z, test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_10_14_comb_4_2_losses_2_lins(self):
		input_seg = "testdata/unittests/mini_test_cases/10_14_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_14_ssm"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_10_14_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_14_fixed_phi"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, 
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))

	# not possible anymore with constraints 9.91 and 9.92
	#def test_10_15_comb_5_2_losses_1_gain(self):
	#	input_seg = "testdata/unittests/mini_test_cases/10_15_seg_as"
	#	input_snp = None
	#	input_ssm = "testdata/unittests/mini_test_cases/10_15_ssm"
	#	num = 3
	#	out_results = "testdata/unittests/mini_test_cases/result_10_15_as"
	#	fixed_phi = "testdata/unittests/mini_test_cases/10_13_fixed_phi"
	#	fixed_z = "testdata/unittests/mini_test_cases/10_13_fixed_z"
	#	simple_CN_changes = False
	#	max_x_CN_changes = -1
	#	only_one_loss = False
	#	only_gains_losses_LOH = False
	#	
	#	# do optimization
	#	main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
	#		self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
	#		self.strategy_file, self.workmem, self.workdir, self.treememory, 
	#		self.emphasis_memory, self.nodeselect,
	#		self.cplex_log_file, self.number_spline_points,
	#		fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
	#		test_run=True, allele_specific=True,
	#		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
	#		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
	#		review_ambiguous_relations=False)

	#	lineages = oio.read_result_file(out_results)

	#	self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
	#	self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
	#	self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
	#	ssm1 = snp_ssm.SSM()
	#	ssm1.chr = 1
	#	ssm1.pos = 0
	#	self.assertEqual(lineages[2].ssms[0], ssm1)
	#	self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))

	def test_10_16_extend_objective_weights(self):
		# no weights
		input_seg = "testdata/unittests/mini_test_cases/10_16_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_10_16_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_16_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_16_fixed_z"
		fixed_seg = "testdata/unittests/mini_test_cases/10_16_fixed_seg_1change"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = False

		# do optimization for one CN change
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			fixed_cnv_file=fixed_seg, write_output_to_disk=False,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		objective_1change = cplex_obj.my_prob.solution.get_objective_value()
		
		# do optimization for one CN change
		fixed_seg = "testdata/unittests/mini_test_cases/10_16_fixed_seg_2changes"
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			fixed_cnv_file=fixed_seg, write_output_to_disk=False,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		objective_2changes = cplex_obj.my_prob.solution.get_objective_value()

		# objectives are the same if no CN weights are used
		self.assertEqual(objective_1change, objective_2changes)

		#################################################

		# with weights
		cn_weight = 0.000001

		# do optimization for one CN change
		fixed_seg = "testdata/unittests/mini_test_cases/10_16_fixed_seg_1change"
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			fixed_cnv_file=fixed_seg, write_output_to_disk=False,
			test_run=True, allele_specific=True, cn_weight=cn_weight,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		objective_1change = cplex_obj.my_prob.solution.get_objective_value()
		
		# do optimization for one CN change
		fixed_seg = "testdata/unittests/mini_test_cases/10_16_fixed_seg_2changes"
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			fixed_cnv_file=fixed_seg, write_output_to_disk=False,
			test_run=True, allele_specific=True, cn_weight=cn_weight,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		objective_2changes = cplex_obj.my_prob.solution.get_objective_value()

		# objectives are the same if no CN weights are used
		self.assertTrue(objective_1change >  objective_2changes)

	def test_10_17_extend_objective_high_weight(self):
		# no weights
		input_seg = "testdata/unittests/mini_test_cases/10_16_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_10_17_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_16_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_16_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = False
		cn_weight = 99999999999

		# do optimization for one CN change
		main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True, cn_weight=cn_weight,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))

	def test_10_18_extend_objective_low_weight(self):
		# no weights
		input_seg = "testdata/unittests/mini_test_cases/10_16_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_10_18_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_16_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_16_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = False
		cn_weight = 0.000001

		# do optimization for one CN change
		main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True, cn_weight=cn_weight,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_10_19_extend_objective_with_SSM(self):
		# no weights
		input_seg = "testdata/unittests/mini_test_cases/10_16_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/10_19_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_10_19_as"
		fixed_phi = "testdata/unittests/mini_test_cases/10_19_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/10_16_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = -1
		only_one_loss = False
		only_gains_losses_LOH = False
		cn_weight = 0.000001

		# do optimization for one CN change
		main.go_onctopus(input_seg, input_snp, input_ssm, num, 
			out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True, cn_weight=cn_weight,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[3].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))

	def test_11_01_no_change(self):
		input_seg = "testdata/unittests/mini_test_cases/11_01_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_01_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_01_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_01_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].cnvs_a[0], cnv.CNV(0, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms[0], ssm1)

	def test_11_02_dc_des_p1_gain_A(self):
		input_seg = "testdata/unittests/mini_test_cases/11_02_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_02_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_02_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_01_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[3].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[3].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms[0], ssm3)

	def test_11_03_dc_des_p1_2_gains(self):
		input_seg = "testdata/unittests/mini_test_cases/11_03_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_03_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_03_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_01_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[3].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms_b[0], ssm2)

	def test_11_04_dc_des_p1_2_gains_equally(self):
		input_seg = "testdata/unittests/mini_test_cases/11_04_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_04_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_04_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_01_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[3].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[3].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_11_05_dc_des_m1_1_loss(self):
		input_seg = "testdata/unittests/mini_test_cases/11_05_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_05_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_05_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_05_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[3].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_b[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms[0], ssm2)

	def test_11_06_dc_des_m1_3_losses(self):
		input_seg = "testdata/unittests/mini_test_cases/11_06_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_06_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_06_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_05_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[3].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[3].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_11_07_dc_anc_m1_1_loss(self):
		input_seg = "testdata/unittests/mini_test_cases/11_07_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_07_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_07_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_05_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[3].ssms_a[0], ssm1)

	def test_11_08_m1_1_loss(self):
		input_seg = "testdata/unittests/mini_test_cases/11_08_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_08_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_08_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_05_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_a[0], ssm1)

	def test_11_09_multiple_segments(self):
		input_seg = "testdata/unittests/mini_test_cases/11_09_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_09_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_11_09_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_05_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_01_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[3].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(-1, 1, 1, 11, 20))
		self.assertEqual(lineages[3].cnvs_a[0], cnv.CNV(-1, 1, 1, 11, 20))
		self.assertEqual(lineages[3].cnvs_b[1], cnv.CNV(-1, 1, 1, 11, 20))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_b[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 11
		self.assertEqual(lineages[2].ssms_a[0], ssm3)

	def test_11_10_3_lins(self):
		input_seg = "testdata/unittests/mini_test_cases/11_10_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_10_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_11_10_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_10_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_10_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_b[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[2].ssms_a[0], ssm2)
	
	def test_11_11_4_lins(self):
		input_seg = "testdata/unittests/mini_test_cases/11_11_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/11_11_ssm"
		num = 5
		out_results = "testdata/unittests/mini_test_cases/result_11_11_as"
		fixed_phi = "testdata/unittests/mini_test_cases/11_11_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/11_11_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 3
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(-1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)

	# not possible with constraint that lineages with frequency of 0 are not allowed to have mutations
	# 
	#def test_11_12_phased_to_B(self):
	#	input_seg = "testdata/unittests/mini_test_cases/11_12_seg_as"
	#	input_snp = None
	#	input_ssm = "testdata/unittests/mini_test_cases/11_12_ssm"
	#	num = 2
	#	out_results = "testdata/unittests/mini_test_cases/result_11_12_as"
	#	fixed_phi = "testdata/unittests/mini_test_cases/11_12_fixed_phi"
	#	fixed_cnv = "testdata/unittests/mini_test_cases/11_12_fixed_seg_as"
	#	simple_CN_changes = False
	#	max_x_CN_changes = 2
	#	only_one_loss = False
	#	only_gains_losses_LOH = True
	#	
	#	# do optimization
	#	main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
	#		self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
	#		self.strategy_file, self.workmem, self.workdir, self.treememory, 
	#		self.emphasis_memory, self.nodeselect,
	#		self.cplex_log_file, self.number_spline_points,
	#		fixed_phi_file=fixed_phi, fixed_cnv_file=fixed_cnv,
	#		test_run=True, allele_specific=True,
	#		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
	#		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

	#	lineages = oio.read_result_file(out_results)

	#	self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(-1, 0, 1, 0, 10))
	#	ssm1 = snp_ssm.SSM()
	#	ssm1.chr = 1
	#	ssm1.pos = 0
	#	self.assertEqual(lineages[1].ssms_b[0], ssm1)

	def test_12_1_clustered_ssms(self):
		input_seg = "testdata/unittests/mini_test_cases/12_0_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/12_0_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_12_1_as"
		fixed_phi = [0.8, 0.56]

		ssm_indices_per_cluster = [[[0, 1, 2]]]

		# read and prepare input files for allele-specific option
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg,
			[], input_ssm, True)
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, [], ssm_list, 50, True)
		# create cplex object and optimize, simple_CN_changes option
		cplex_obj = optimization.Optimization_with_CPLEX([], [],
			ssm_splines, True, seg_splines_A, seg_splines_B)
		cplex_obj.set_other_parameter(num, [], ssm_list, seg_list)
		cplex_obj.create_variables()
		cplex_obj.create_standard_constraints([], ssm_list, seg_list, fixed_phi=fixed_phi)
		cplex_obj.create_fixed_variables_constraints(ssm_indices_per_cluster=ssm_indices_per_cluster,
			fixed_phi=fixed_phi)
		cplex_obj.cplex_log_file = None
		cplex_obj.start_CPLEX()
		# write result to file
		my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, [], ssm_list, seg_list)
		oio.write_lineages_to_result_file(out_results, my_lineages, True)

		lineages = oio.read_result_file(out_results)
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[2].ssms[1], ssm2)
		self.assertEqual(lineages[2].ssms[2], ssm3)

	def test_12_2_clustered_ssms(self):
		input_seg = "testdata/unittests/mini_test_cases/12_0_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/12_0_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_12_2_as"
		fixed_phi = [0.8, 0.56]

		ssm_indices_per_cluster = [[[0, 1, 2]]]

		# read and prepare input files for allele-specific option
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg,
			[], input_ssm, True)
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, [], ssm_list, 50, True)
		# create cplex object and optimize, non-simple_CN_changes option
		cplex_obj = optimization.Optimization_with_CPLEX([], [],
			ssm_splines, True, seg_splines_A, seg_splines_B, simple_CN_changes=False)
		cplex_obj.set_other_parameter(num, [], ssm_list, seg_list)
		cplex_obj.create_variables()
		cplex_obj.create_standard_constraints([], ssm_list, seg_list, fixed_phi=fixed_phi)
		cplex_obj.create_fixed_variables_constraints(ssm_indices_per_cluster=ssm_indices_per_cluster,
			fixed_phi=fixed_phi)
		cplex_obj.cplex_log_file = None
		cplex_obj.start_CPLEX()
		# write result to file
		my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, [], ssm_list, seg_list)
		oio.write_lineages_to_result_file(out_results, my_lineages, True)

		lineages = oio.read_result_file(out_results)
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms[0], ssm1)
		self.assertEqual(lineages[2].ssms[1], ssm2)
		self.assertEqual(lineages[2].ssms[2], ssm3)

	def test_13_1_rule_of_thumb(self):
		input_seg = "testdata/unittests/mini_test_cases/13_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/13_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_1_as"
		fixed_phi = "testdata/unittests/mini_test_cases/13_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/13_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertTrue(lineages[1].ssms_a[0].infl_cnv_same_lin)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms[0], ssm2)
		self.assertFalse(lineages[1].ssms[0].infl_cnv_same_lin)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms_a[1], ssm3)
		self.assertTrue(lineages[1].ssms_a[1].infl_cnv_same_lin)

	def test_13_1_rule_of_thumb_simple(self):
		input_seg = "testdata/unittests/mini_test_cases/13_1_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/13_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_1_as_simple"
		fixed_phi = "testdata/unittests/mini_test_cases/13_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/13_1_fixed_z"
		simple_CN_changes = True
		max_x_CN_changes = -1
		only_one_loss = True
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		self.assertTrue(lineages[1].ssms_a[0].infl_cnv_same_lin)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms[0], ssm2)
		self.assertFalse(lineages[1].ssms[0].infl_cnv_same_lin)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[1].ssms_a[1], ssm3)
		self.assertTrue(lineages[1].ssms_a[1].infl_cnv_same_lin)

	def test_13_2_rule_of_thumb(self):
		input_seg = "testdata/unittests/mini_test_cases/13_2_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/13_2_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_2_as"
		fixed_phi = "testdata/unittests/mini_test_cases/13_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/13_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[1].ssms_a[0], ssm1)
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		self.assertEqual(lineages[1].ssms_b[0], ssm2)
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		self.assertEqual(lineages[2].ssms_a[0], ssm3)
		ssm4 = snp_ssm.SSM()
		ssm4.chr = 1
		ssm4.pos = 3
		self.assertEqual(lineages[2].ssms[0], ssm4)
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 1
		ssm5.pos = 4
		self.assertEqual(lineages[1].ssms_a[1], ssm5)

	def test_13_3_rule_of_thumb(self):
		input_seg = "testdata/unittests/mini_test_cases/13_3_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/13_3_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_3_as"
		fixed_phi = "testdata/unittests/mini_test_cases/13_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/13_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[1].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		try:
			self.assertEqual(lineages[1].ssms_a[0], ssm1)
			self.assertTrue(lineages[1].ssms_a[1].infl_cnv_same_lin)
		except IndexError:
			pass
		try:
			self.assertEqual(lineages[1].ssms_b[0], ssm1)
			self.assertTrue(lineages[1].ssms_b[1].infl_cnv_same_lin)
		except IndexError:
			pass


	def test_13_4_rule_of_thumb(self):
		input_seg = "testdata/unittests/mini_test_cases/13_4_seg_as"
		input_snp = None
		input_ssm = "testdata/unittests/mini_test_cases/13_4_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_4_as"
		fixed_phi = "testdata/unittests/mini_test_cases/13_1_fixed_phi"
		fixed_z = "testdata/unittests/mini_test_cases/13_1_fixed_z"
		simple_CN_changes = False
		max_x_CN_changes = 2
		only_one_loss = False
		only_gains_losses_LOH = True
		
		# do optimization
		main.go_onctopus(input_seg, input_snp, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect,
			self.cplex_log_file, self.number_spline_points,
			fixed_phi_file=fixed_phi, fixed_z_matrix_file=fixed_z,
			test_run=True, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes, 
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].cnvs_a[0], cnv.CNV(1, 0, 1, 0, 10))
		self.assertEqual(lineages[2].cnvs_b[0], cnv.CNV(1, 0, 1, 0, 10))
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		self.assertEqual(lineages[2].ssms_b[0], ssm1)
		self.assertTrue(lineages[2].ssms_b[0].infl_cnv_same_lin)

	def test_13_5_rule_of_thumb(self):
		input_seg = "testdata/unittests/mini_test_cases/13_5_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/13_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_5_as"
		fixed_phi = [0.8, 0.6]

		ssm_indices_per_cluster = [[[0, 1, 2, 3, 4]]]

		# read and prepare input files for allele-specific option
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg,
			[], input_ssm, True)
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, [], ssm_list, 50, True)
		# create cplex object and optimize, non-simple_CN_changes option
		cplex_obj = optimization.Optimization_with_CPLEX([], [],
			ssm_splines, True, seg_splines_A, seg_splines_B, simple_CN_changes=False)
		cplex_obj.set_other_parameter(num, [], ssm_list, seg_list)
		cplex_obj.create_variables()
		cplex_obj.create_standard_constraints([], ssm_list, seg_list, fixed_phi=fixed_phi)
		cplex_obj.create_fixed_variables_constraints(ssm_indices_per_cluster=ssm_indices_per_cluster,
			fixed_phi=fixed_phi)
		cplex_obj.cplex_log_file = None
		cplex_obj.start_CPLEX()
		# write result to file
		my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, [], ssm_list, seg_list)
		oio.write_lineages_to_result_file(out_results, my_lineages, True)

		lineages = oio.read_result_file(out_results)
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		ssm4 = snp_ssm.SSM()
		ssm4.chr = 1
		ssm4.pos = 3
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 1
		ssm5.pos = 4
		self.assertEqual(lineages[2].ssms_a[0], ssm1)
		self.assertTrue(lineages[2].ssms_a[0].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[1], ssm2)
		self.assertTrue(lineages[2].ssms_a[1].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[2], ssm3)
		self.assertTrue(lineages[2].ssms_a[2].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[3], ssm4)
		self.assertTrue(lineages[2].ssms_a[3].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[4], ssm5)
		self.assertTrue(lineages[2].ssms_a[4].infl_cnv_same_lin)

	def test_13_5_rule_of_thumb_simple(self):
		input_seg = "testdata/unittests/mini_test_cases/13_5_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/13_5_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_13_5_as_simple"
		fixed_phi = [0.8, 0.6]

		ssm_indices_per_cluster = [[[0, 1, 2, 3, 4]]]

		# read and prepare input files for allele-specific option
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg,
			[], input_ssm, True)
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, [], ssm_list, 50, True)
		# create cplex object and optimize, non-simple_CN_changes option
		cplex_obj = optimization.Optimization_with_CPLEX([], [],
			ssm_splines, True, seg_splines_A, seg_splines_B, simple_CN_changes=True)
		cplex_obj.set_other_parameter(num, [], ssm_list, seg_list)
		cplex_obj.create_variables()
		cplex_obj.create_standard_constraints([], ssm_list, seg_list, fixed_phi=fixed_phi)
		cplex_obj.create_fixed_variables_constraints(ssm_indices_per_cluster=ssm_indices_per_cluster,
			fixed_phi=fixed_phi)
		cplex_obj.cplex_log_file = None
		cplex_obj.start_CPLEX()
		# write result to file
		my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, [], ssm_list, seg_list)
		oio.write_lineages_to_result_file(out_results, my_lineages, True)

		lineages = oio.read_result_file(out_results)
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 1
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 2
		ssm4 = snp_ssm.SSM()
		ssm4.chr = 1
		ssm4.pos = 3
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 1
		ssm5.pos = 4
		self.assertEqual(lineages[2].ssms_a[0], ssm1)
		self.assertTrue(lineages[2].ssms_a[0].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[1], ssm2)
		self.assertTrue(lineages[2].ssms_a[1].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[2], ssm3)
		self.assertTrue(lineages[2].ssms_a[2].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[3], ssm4)
		self.assertTrue(lineages[2].ssms_a[3].infl_cnv_same_lin)
		self.assertEqual(lineages[2].ssms_a[4], ssm5)
		self.assertTrue(lineages[2].ssms_a[4].infl_cnv_same_lin)

	# test 14 in test_mini_test_cases_2.py

	def test_15_1_matrix_transitivity_case_i(self):
		input_seg = "testdata/unittests/mini_test_cases/15_1_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/15_1_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_15_1_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/15_1_fixed_phi"
		fixed_cnv = "testdata/unittests/mini_test_cases/15_1_fixed_cnv"

		# do optimization
		main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_phi_file=fixed_phi_file, fixed_cnv_file=fixed_cnv,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2, 3])
		self.assertEqual(lineages[2].sublins, [3])
		self.assertEqual(len(lineages[1].ssms_a), 1)
		self.assertEqual(len(lineages[2].ssms_a), 1)

	def test_15_2_matrix_transitivity_case_c(self):
		input_seg = "testdata/unittests/mini_test_cases/15_2_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/15_2_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_15_2_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/15_1_fixed_phi"
		fixed_cnv = "testdata/unittests/mini_test_cases/15_2_fixed_cnv"

		# do optimization
		main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_phi_file=fixed_phi_file, fixed_cnv_file=fixed_cnv,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2, 3])
		self.assertEqual(lineages[2].sublins, [3])
		self.assertEqual(len(lineages[1].ssms_a), 1)
		self.assertEqual(len(lineages[2].ssms_b), 1)

	def test_15_3_matrix_transitivity_control(self):
		input_seg = "testdata/unittests/mini_test_cases/15_3_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/15_3_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_15_3_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/15_3_fixed_phi"
		fixed_cnv = "testdata/unittests/mini_test_cases/15_2_fixed_cnv"

		# do optimization
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, None, 
			input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_phi_file=fixed_phi_file, fixed_cnv_file=fixed_cnv,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [])
		self.assertEqual(lineages[2].sublins, [3])
		self.assertEqual(len(lineages[2].ssms_a), 1)
		self.assertEqual(len(lineages[2].ssms_b), 1)

	def test_15_4_matrix_transitivity_conflict(self):
		input_seg = "testdata/unittests/mini_test_cases/15_2_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/15_2_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_15_4_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/15_1_fixed_phi"
		fixed_cnv = "testdata/unittests/mini_test_cases/15_2_fixed_cnv"
		fixed_z = "testdata/unittests/mini_test_cases/15_4_fixed_z"

		# do optimization
		with self.assertRaises(CplexError):
			main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect, 
				self.cplex_log_file, self.number_spline_points, test_run=True,
				fixed_phi_file=fixed_phi_file, fixed_cnv_file=fixed_cnv,
				fixed_z_matrix_file=fixed_z,
				allele_specific=True, simple_CN_changes=False)

	def test_15_5_matrix_transitivity_conflict(self):
		input_seg = "testdata/unittests/mini_test_cases/15_2_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/15_2_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_15_5_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/15_1_fixed_phi"
		fixed_cnv = "testdata/unittests/mini_test_cases/15_2_fixed_cnv"
		fixed_z = "testdata/unittests/mini_test_cases/15_5_fixed_z"

		# do optimization
		with self.assertRaises(CplexError):
			main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect, 
				self.cplex_log_file, self.number_spline_points, test_run=True,
				fixed_phi_file=fixed_phi_file, fixed_cnv_file=fixed_cnv,
				fixed_z_matrix_file=fixed_z,
				allele_specific=True, simple_CN_changes=False)

	# if this test throws an error, look at function 'create_standard_constraints' in 'optimization.py'
	# code needs to be activated for test
	@unittest.expectedFailure
	def test_16_right_children(self):
		input_seg = "testdata/unittests/mini_test_cases/16_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 7
		out_results = "testdata/unittests/mini_test_cases/result_16_as"
		normal_seg_indices = [0]
		fixed_z = "testdata/unittests/mini_test_cases/16_fixed_z"
		fixed_phi = "testdata/unittests/mini_test_cases/16_fixed_phi"

		# do optimization
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(input_seg, None, 
			input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_z_matrix_file=fixed_z, fixed_phi_file=fixed_phi,
			normal_seg_indices=normal_seg_indices,
			allele_specific=True, simple_CN_changes=False,
			lineage_divergence_rule=False)

		# values children should have
		right_children = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]
		first_child = "child_0_1"
		last_child = "child_5_6"
		
		start_index = cplex_obj.children_start_index
		stop_index = cplex_obj.children_start_index + len(cplex_obj.my_colnames_children_index) - 1

		self.assertEqual(cplex_obj.my_colnames[start_index], first_child)
		self.assertEqual(cplex_obj.my_colnames[stop_index], last_child)
		self.assertEqual(cplex_obj.my_prob.solution.get_values()[start_index:stop_index+1], right_children)

		# values child_freqs should have
		right_child_freqs = [0.9, 0, 0.7, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.6, 0, 0.4, 0, 0, 0, 0.5, 0, 0]
		first_child_freq = "child_0_1_freq"
		last_child_freq = "child_5_6_freq"

		start_freq_index = cplex_obj.children_freq_start_index
		stop_freq_index = cplex_obj.children_freq_start_index + len(cplex_obj.my_colnames_children_freq_index) - 1

		self.assertEqual(cplex_obj.my_colnames[start_freq_index], first_child_freq)
		self.assertEqual(cplex_obj.my_colnames[stop_freq_index], last_child_freq)
		self.assertEqual(cplex_obj.my_prob.solution.get_values()[start_freq_index:stop_freq_index+1],
			right_child_freqs)

		# values parent_freq should have
		right_parent_freqs = [1.0, 0, 1.0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0.8, 0, 0.8, 0, 0, 0, 0.6, 0, 0]
		first_parent_freq = "parent_0_1_freq"
		last_parent_freq = "parent_5_6_freq"

		start_parent_freq_index = cplex_obj.parent_freq_start_index
		stop_parent_freq_index = cplex_obj.parent_freq_start_index + len(cplex_obj.my_colnames_parent_freq_index) - 1

		self.assertEqual(cplex_obj.my_colnames[start_parent_freq_index], first_parent_freq)
		self.assertEqual(cplex_obj.my_colnames[stop_parent_freq_index], last_parent_freq)
		self.assertEqual(cplex_obj.my_prob.solution.get_values()[start_parent_freq_index:stop_parent_freq_index+1],
			right_parent_freqs)

		# values sibling_freq should have
		right_sibling_freq = [
			0, 0.7, 0, 0, 0,
			0, 0, 0, 0, 0,
			0.9, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0,
			0, 0, 0.4,
			0, 0, 0,
			0, 0.6, 0,
			0, 0,
			0, 0,
			0, 0,
			0,
			0
			]
		first_sibling_freq = "sibling_0_1_2_freq"
		last_sibling_freq = "sibling_4_6_5_freq"

		start_sibling_freq_index = cplex_obj.sibling_freq_start_index
		stop_sibling_freq_index = cplex_obj.sibling_freq_start_index + len(cplex_obj.my_colnames_sibling_freq) - 1

		self.assertEqual(cplex_obj.my_colnames[start_sibling_freq_index], first_sibling_freq)
		self.assertEqual(cplex_obj.my_colnames[stop_sibling_freq_index], last_sibling_freq)
		self.assertEqual(cplex_obj.my_prob.solution.get_values()[start_sibling_freq_index:stop_sibling_freq_index+1],
			right_sibling_freq)

	def test_17_superSSMs(self):
		input_seg = "testdata/unittests/mini_test_cases/17_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/17_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_17_as"
		reconstruction_1 = "testdata/unittests/mini_test_cases/result_17_as.reconstruction_1"
		reconstructed_matrix = "testdata/unittests/mini_test_cases/result_17_as_1.zmatrix"
		fixed_cnv = "testdata/unittests/mini_test_cases/17_fixed_cnv"
		fixed_z = "testdata/unittests/mini_test_cases/17_fixed_z_0"

		cplex_log_file = "testdata/unittests/mini_test_cases/17.cplex.log"

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			cplex_log_file, self.number_spline_points, test_run=True,
			fixed_cnv_file=fixed_cnv, fixed_z_matrix_file=fixed_z,
			allele_specific=True, simple_CN_changes=False,
			use_super_SSMs=True, review_ambiguous_relations=True, cluster_SSM=True,
			lineage_divergence_rule=True)

		lineages = oio.read_result_file(out_results)
		lineages_2 = oio.read_result_file(reconstruction_1)
		reconstructed_matrix = json.load(open(reconstructed_matrix))

		self.assertEqual(len(z_matrix_list), 1)
		self.assertEqual(lineages, lineages_2)
		self.assertEqual(len(lineages[1].ssms), 3)
		self.assertEqual(len(lineages[2].ssms), 2)
		self.assertEqual(reconstructed_matrix[1][2], 0)

	def test_18_1_Z_fixed(self):
		input_seg = "testdata/unittests/mini_test_cases/18_1_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/18_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_18_1_as"
		fixed_z = "testdata/unittests/mini_test_cases/18_1_fixed_z"

		# do optimization
		main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_z_matrix_file=fixed_z,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[1].sublins, [2])
		self.assertEqual(len(lineages[1].ssms), 1)
		self.assertEqual(len(lineages[2].ssms), 1)
		self.assertEqual(len(lineages[0].cnvs_a), 1)
		self.assertAlmostEqual(lineages[1].freq, 0.8, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.4, delta=0.001)

	def test_18_2_LDR(self):
		input_seg = "testdata/unittests/mini_test_cases/18_1_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/18_1_ssm"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_18_2_as"

		# do optimization
		main.go_onctopus(input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].sublins, [1,2])
		self.assertEqual(lineages[1].sublins, [2])
		self.assertEqual(lineages[2].sublins, [])
		self.assertEqual(len(lineages[1].ssms), 1)
		self.assertEqual(len(lineages[2].ssms), 1)
		self.assertEqual(len(lineages[0].cnvs_a), 1)
		self.assertAlmostEqual(lineages[1].freq, 0.8, delta=0.001)
		self.assertAlmostEqual(lineages[2].freq, 0.4, delta=0.001)

	def test_18_3_infl(self):
		input_seg = "testdata/unittests/mini_test_cases/18_3_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/18_3_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_18_3_as"
		fixed_ssms = "testdata/unittests/mini_test_cases/18_3_fixed_ssm"

		# do optimization
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_ssm_file=fixed_ssms,
			allele_specific=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(my_lineages, lineages)
		self.assertEqual(lineages[0].sublins, [1,2,3])
		self.assertEqual(lineages[1].sublins, [2,3])
		self.assertEqual(lineages[2].sublins, [3])
		self.assertEqual(len(lineages[1].ssms_a), 1)
		self.assertEqual(len(lineages[2].ssms_a), 1)
		self.assertEqual(len(lineages[3].ssms), 1)
		self.assertEqual(len(lineages[3].cnvs_a), 1)
		self.assertAlmostEqual(lineages[1].freq, 0.4, delta=0.01)
		self.assertAlmostEqual(lineages[2].freq, 0.2, delta=0.05)
		self.assertAlmostEqual(lineages[3].freq, 0.1, delta=0.05)

	# for this test, the checking part in the initialization of the optimization object has to be
	# disabeled and a breakpoint has to be set in add_sublins_from_CPLEX in the model to see
	# whether the set lineage relations are removed
	@unittest.skip("change optimization.py for this test")
	def test_18_4_infl(self):
		input_seg = "testdata/unittests/mini_test_cases/18_1_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/18_3_ssm"
		num = 4
		out_results = "testdata/unittests/mini_test_cases/result_18_4_as"
		z_trans_weight = -100
		normal_seg_indices = [0]
		fixed_phi = "testdata/unittests/mini_test_cases/18_4_fixed_phi"

		# do optimization
		my_lineages, cplex_obj, z_matrix_list, new_lineage_list = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True, z_trans_weight=z_trans_weight,
			normal_seg_indices=normal_seg_indices, fixed_phi_file=fixed_phi)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(lineages[0].sublins, [1,2,3])
		self.assertEqual(lineages[1].sublins, [])
		self.assertEqual(lineages[2].sublins, [])
		self.assertEqual(lineages[3].sublins, [])
		self.assertEqual(len(lineages[1].ssms), 1)
		self.assertEqual(len(lineages[2].ssms), 1)
		self.assertEqual(len(lineages[3].ssms), 1)
		self.assertEqual(len(lineages[0].cnvs_a), 1)

	def test_19_LDR(self):
		input_seg = "testdata/unittests/mini_test_cases/19_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/no_ssms"
		num = 5
		out_results = "testdata/unittests/mini_test_cases/result_19_as"
		reconstructed_matrix = "testdata/unittests/mini_test_cases/result_19_as_1.zmatrix"
		fixed_phi = "testdata/unittests/mini_test_cases/19_fixed_phi"

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_phi_file=fixed_phi,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True)

		self.assertEqual(len(z_matrix_list), 8)

	def test_20_new_fixation(self):
		input_seg = "testdata/unittests/mini_test_cases/20_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/20_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_20_as"
		fixed_phi = "testdata/unittests/mini_test_cases/20_fixed_phi"
		fixed_seg = "testdata/unittests/mini_test_cases/20_fixed_segs"
		fixed_ssm = "testdata/unittests/mini_test_cases/20_fixed_ssms"

		with open(fixed_seg, "r") as f:
			fixed_seg_list_new = json.load(f)
		with open(fixed_ssm, "r") as f:
			fixed_ssm_list_new = json.load(f)

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			fixed_phi_file=fixed_phi,
			fixed_cnv_list_new=fixed_seg_list_new, fixed_ssm_list_new=fixed_ssm_list_new,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True)

		lineages = oio.read_result_file(out_results)

		self.assertEqual(len(lineages[1].ssms_b), 1)
		self.assertEqual(len(lineages[2].cnvs_b), 1)

	def test_21_1_clustering_without_SSMs(self):
		input_seg = "testdata/unittests/mini_test_cases/21_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/21_1_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_21_as"

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True, cluster_SSM=True, use_super_SSMs=True)

		lineages = oio.read_result_file(out_results)

		# result doesn't matter, it's just important that no error occures becaue segment with index 1
		# doesn't have any SSMs

	def test_22_combining_multiple_segments(self):
		input_seg = "testdata/unittests/mini_test_cases/22_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/22_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_22_as"
		out_results_true = "testdata/unittests/mini_test_cases/result_true_22_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/22_fixed_phi"
		normal_seg_indices = [1,4]

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True, fixed_phi_file=fixed_phi_file,
			normal_seg_indices=normal_seg_indices, combine_normal_segments=True)

		lineages = oio.read_result_file(out_results)
		true_lineages = oio.read_result_file(out_results_true)
	
		self.assertEqual(lineages, true_lineages)

	def test_23_phi_seg_compability(self):
		input_seg = "testdata/unittests/mini_test_cases/23_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/23_ssms"
		num = 2
		out_results = "testdata/unittests/mini_test_cases/result_23_as"
		fixed_phi_file = "testdata/unittests/mini_test_cases/23_fixed_phi"

		# do optimization
		with self.assertRaises(eo.FixPhiIncompatibleException):
			(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
				input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
				self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
				self.strategy_file, self.workmem, self.workdir, self.treememory, 
				self.emphasis_memory, self.nodeselect, 
				self.cplex_log_file, self.number_spline_points, test_run=True,
				allele_specific=True, simple_CN_changes=False,
				review_ambiguous_relations=True, 
				lineage_divergence_rule=True, fixed_phi_file=fixed_phi_file)

	def test_24_warm_start(self):
		input_seg = "testdata/unittests/mini_test_cases/24_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/24_ssms"
		num = 3
		out_results = "testdata/unittests/mini_test_cases/result_24_as"
		warm_phi_file = "testdata/unittests/mini_test_cases/24_warm_phi"
		warm_ssm_file = "testdata/unittests/mini_test_cases/24_warm_ssm"
		warm_cnv_file = "testdata/unittests/mini_test_cases/24_warm_cnv"
		warm_Z_file = "testdata/unittests/mini_test_cases/24_warm_Z"

		# do optimization
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True, warm_start_z_matrix_file=warm_Z_file,
			warm_start_dc_binary_file=warm_cnv_file, warm_start_dssm_file=warm_ssm_file,
			warm_start_freqs_file=warm_phi_file)
	
		# nothing can be tested here in terms of asserts
		# look in logging file and see, whether all for warm start files are being used

	def test_24_1_warm_start(self):
		input_seg = "testdata/unittests/mini_test_cases/24_seg_as"
		input_ssm = "testdata/unittests/mini_test_cases/24_ssms"
		num = 3
		in_results = "testdata/unittests/mini_test_cases/result_24_as"
		out_results = "testdata/unittests/mini_test_cases/result_24_1_as"

		# do optimization
		self.cplex_log_file = "testdata/unittests/mini_test_cases/24_1.cplex.log"
		(my_lineages, cplex_obj, z_matrix_list, new_lineage_list) = main.go_onctopus(
			input_seg, None, input_ssm, num, out_results, self.time, self.threads, 
			self.probing, self.emph_switch, self.coef_reduc, self.mipgap, self.symmetry, 
			self.strategy_file, self.workmem, self.workdir, self.treememory, 
			self.emphasis_memory, self.nodeselect, 
			self.cplex_log_file, self.number_spline_points, test_run=True,
			allele_specific=True, simple_CN_changes=False,
			review_ambiguous_relations=True, 
			lineage_divergence_rule=True, warm_start_solution=in_results)

		MIP_start_used = False
		with open(self.cplex_log_file, "r") as f:
			for line in f:
				if "MIP starts provided solutions" in line:
					MIP_start_used = True
					break
		self.assertTrue(MIP_start_used)
	
		
		self.cplex_log_file = None

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(MiniTestCasesTest)
