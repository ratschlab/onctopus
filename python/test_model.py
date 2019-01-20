import unittest
import model
import snp_ssm
import segment
import cnv
import exceptions_onctopus as eo
import log_pdf
import onctopus_io as oio
import numpy as np
import lineage
import constants as cons
from StringIO import StringIO
import numpy as np
import main
import lineage
import clustered_mutation_optimization as clusmod
import copy

class ModelTest(unittest.TestCase):

	def test_change_unnecessary_phasing(self):

		lin_num = 5

		CNVs_0 = {}
		CNVs_0[cons.GAIN] = {}
		CNVs_0[cons.GAIN][cons.A] = {}
		CNVs_0[cons.GAIN][cons.A][3] = "something"
		CNVs_1 = {}
		CNVs_1[cons.LOSS] = {}
		CNVs_1[cons.LOSS][cons.B] = {}
		CNVs_1[cons.LOSS][cons.B][1] = "something"
		CNVs_2 = {}
		CNVs_2[cons.GAIN] = {}
		CNVs_2[cons.GAIN][cons.A] = {}
		CNVs_2[cons.GAIN][cons.A][4] = "something"
		CNVs_3 = {}
		CNVs_3[cons.GAIN] = {}
		CNVs_3[cons.GAIN][cons.A] = {}
		CNVs_3[cons.GAIN][cons.A][2] = "something"
		CNVs_4 = {}
		CNVs = [CNVs_0, CNVs_1, CNVs_2, CNVs_3, CNVs_4]

		present_ssms = [
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, True, True, True, True], 
			[False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			]

		ssm_infl_cnv_same_lineage = [
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, True], [False, False, False, False, False]],
			[[False, False, True, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			]

		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 0, 1],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]
			]

		seg_num = 5

		# function to test
		model.change_unnecessary_phasing(lin_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num)

		self.assertEqual(present_ssms[0], [[False, True, False, False, False], [False, True, False, False, False],
			[False, False, True, True, True]])
		self.assertEqual(present_ssms[1], [[False, True, True, True, True], [False, True, True, True, True],
			[False, False, False, False, False]])
		self.assertEqual(present_ssms[2], [[False, False, False, False, False], [False, True, True, False, False],
			[False, False, False, True, True]])
		self.assertEqual(present_ssms[3], [[False, True, True, False, False], [False, True, False, False, False],
			[False, False, True, True, True]])
		self.assertEqual(present_ssms[4], [[False, False, False, False, False], [False, False, False, False, False],
			[False, True, True, True, True]])

	def test_is_CN_gain_in_k(self):
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][1] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][2] = "something"
		CNVs[cons.GAIN][cons.B][3] = "something"
		ssm_infl_cnv_same_lin_i = [
			[False, True, False, True],
			[False, False, True, False]
			]

		self.assertEqual(model.is_CN_gain_in_k(1, CNVs, ssm_infl_cnv_same_lin_i), (True, False))
		self.assertEqual(model.is_CN_gain_in_k(2, CNVs, ssm_infl_cnv_same_lin_i), (False, True))
		self.assertEqual(model.is_CN_gain_in_k(3, CNVs, ssm_infl_cnv_same_lin_i), (False, False))

	def test_do_descendants_have_CN_change(self):
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][4] = "something"

		self.assertEqual(model.do_descendants_have_CN_change([0, 1], CNVs), True)
		self.assertEqual(model.do_descendants_have_CN_change([0, 2], CNVs), True)
		self.assertEqual(model.do_descendants_have_CN_change([0, 3], CNVs), True)
		self.assertEqual(model.do_descendants_have_CN_change([0, 4], CNVs), True)
		self.assertEqual(model.do_descendants_have_CN_change([0, 5], CNVs), False)

	def test_do_ancestors_have_CN_loss(self):

		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"

		self.assertEqual(model.do_ancestors_have_CN_loss([0, 1], CNVs), True)
		self.assertEqual(model.do_ancestors_have_CN_loss([0, 2], CNVs), True)
		self.assertEqual(model.do_ancestors_have_CN_loss([3], CNVs), False)

	# tests used to test ambiguity in simulated datasets
	def test_get_all_possible_z_matrices_with_lineages(self):

		# simulated datasets, 2 SSMs in lineage 1 phased to A and B, although they are not influenced by any
		#	CN change, CN gain in lineage 2 an A, only one segment
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm2.seg_index = 0
		cnv1 = cnv.CNV(1, 0, 1, 1, 10)
		lin0 = lineage.Lineage([1,2], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [ssm1], [ssm2])
		lin2 = lineage.Lineage([], 0.3, [cnv1], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]


		my_lineages, z_matrix_list, new_lineage_list, lin_div_rule_feasibility = (
			model.get_all_possible_z_matrices_with_lineages(my_lins, 1))

		self.assertEqual(z_matrix_list[0][1][2], 0)

		# simulated datasets, 2 SSMs in lineage 1 both unphased
		#	CN gain in lineage 2 an A, only one segment
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm2.seg_index = 0
		cnv1 = cnv.CNV(1, 0, 1, 1, 10)
		lin0 = lineage.Lineage([1,2], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.5, [], [], [], [], [], [ssm1, ssm2], [], [])
		lin2 = lineage.Lineage([], 0.3, [cnv1], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]


		my_lineages, z_matrix_list, new_lineage_list, lin_div_rule_feasibility = (
			model.get_all_possible_z_matrices_with_lineages(my_lins, 1))

		self.assertEqual(z_matrix_list[0][1][2], 0)

	def test_assign_lin_index_to_ssms(self):

		ssm1 = snp_ssm.SSM()
		ssm2 = snp_ssm.SSM()
		ssm3 = snp_ssm.SSM()
		ssm4 = snp_ssm.SSM()
		lin0 = lineage.Lineage([1], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0, [], [], [], [], [], [ssm1], [], [])
		lin2 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [ssm3])
		lin3 = lineage.Lineage([], 0, [], [], [], [], [], [], [ssm2, ssm4], [])
		my_lins = [lin0, lin1, lin2, lin3]

		model.assign_lin_index_to_ssms(my_lins)

		self.assertEqual(ssm1.lineage, 1)
		self.assertEqual(ssm2.lineage, 3)
		self.assertEqual(ssm3.lineage, 2)
		self.assertEqual(ssm4.lineage, 3)

	def test_shorten_lineages(self):

		# Z-matrix is not None
		with self.assertRaises(eo.ZMatrixNotNone):
			model.shorten_lineages([], True)

		# lineage with frequency of 0 has mutations
		ssm1 = snp_ssm.SSM()
		lin0 = lineage.Lineage([1], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0, [], [], [], [], [], [ssm1], [], [])
		my_lins = [lin0, lin1]

		with self.assertRaises(eo.LineageWith0FreqMutations):
			model.shorten_lineages(my_lins, None)

		# last lineage gets removed
		ssm1 = snp_ssm.SSM()
		lin0 = lineage.Lineage([1, 2], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2], 0.5, [], [], [], [], [], [ssm1], [], [])
		lin2 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]

		model.shorten_lineages(my_lins, None)

		self.assertEqual(len(my_lins), 2)
		self.assertEqual(my_lins[0], lin0)
		self.assertEqual(my_lins[1], lin1)

		# all but first lineage get removed
		lin0 = lineage.Lineage([1, 2], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2], 0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]

		model.shorten_lineages(my_lins, None)

		self.assertEqual(len(my_lins), 1)
		self.assertEqual(my_lins[0], lin0)

	def test_shorten_sublineages(self):
		# lineages
		# following sublineage lists don't make sense! Just are like this so that function can be tested properly.
		lin0 = lineage.Lineage([1, 2, 3, 4], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([1, 4, 5, 5], 1, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([4, 6, 6, 6], 1, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]

		# index to remove is at end in lin 0 and in middle of lin 1 and at beginning of lin 2
		model.shorten_sublineages(my_lins, 4)

		self.assertEqual(lin0.sublins, [1, 2, 3])
		self.assertEqual(lin1.sublins, [1, 5, 5])
		self.assertEqual(lin2.sublins, [6, 6, 6])

	def test_count_ssms_in_lineage_segments_with_cn_gains_phase(self):
		# no CN changes but SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm_list = [ssm1]
		cnv_list = []

		self.assertEqual(model.count_ssms_in_lineage_segments_with_cn_gains_phase(ssm_list, cnv_list), 0)

		# no SSMs but CN changes
		ssm_list = []
		cnv1 = cnv.CNV(1, 0, 1, 1, 1)
		cnv_list = [cnv1]

		self.assertEqual(model.count_ssms_in_lineage_segments_with_cn_gains_phase(ssm_list, cnv_list), 0)

		# CN gain on same segment than SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm_list = [ssm1]
		cnv1 = cnv.CNV(1, 1, 1, 1, 1)
		cnv_list = [cnv1]

		self.assertEqual(model.count_ssms_in_lineage_segments_with_cn_gains_phase(ssm_list, cnv_list), 1)

		# SSMs with lower index, then gain, then SSMs with higher index, then loss in same segment
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 5
		ssm_list = [ssm1, ssm2]
		cnv1 = cnv.CNV(1, 2, 1, 1, 1)
		cnv2 = cnv.CNV(-1, 5, 1, 1, 1)
		cnv_list = [cnv1, cnv2]

		self.assertEqual(model.count_ssms_in_lineage_segments_with_cn_gains_phase(ssm_list, cnv_list), 0)

		# CNVs, SSMs with higher index, then CN gain on same segment, then more SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 10
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 15
		ssm_list = [ssm1, ssm2]
		cnv1 = cnv.CNV(1, 2, 1, 1, 1)
		cnv2 = cnv.CNV(1, 3, 1, 1, 1)
		cnv3 = cnv.CNV(1, 10, 1, 1, 1)
		cnv_list = [cnv1, cnv2, cnv3]

		self.assertEqual(model.count_ssms_in_lineage_segments_with_cn_gains_phase(ssm_list, cnv_list), 1)

	def test_count_lineages_based_on_freq(self):
		my_lin0 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		my_lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		my_lin2 = lineage.Lineage([], cons.EPSILON_FREQUENCY - 0.000000000001, [], [], [], [], [], [], [], [])
		my_lin3 = lineage.Lineage([], cons.EPSILON_FREQUENCY - 0.5 * cons.EPSILON_FREQUENCY, [], [], [], [], [], 
			[], [], [])
		lins = [my_lin0, my_lin1, my_lin2, my_lin3]

		self.assertEqual(model.count_lineages_based_on_freq(lins), 3)

	def test_get_number_of_different_segments_and_CNAs(self):
		cna_free = cnv.CNV(0, 1, 1, 10, 19)
		cna_1 = cnv.CNV(1, 0, 1, 1, 9)
		cna_2 = cnv.CNV(-1, 2, 1, 20, 29)
		cna_3 = cnv.CNV(-1, 2, 1, 20, 29)

		my_lin0 = lineage.Lineage([], 1.0, [cna_free], [], [], [], [], [], [], [])
		my_lin1 = lineage.Lineage([], 1.0, [cna_1], [cna_3], [], [], [], [], [], [])
		my_lin2 = lineage.Lineage([], 1.0, [], [cna_2], [], [], [], [], [], [])
		lins = [my_lin0, my_lin1, my_lin2]

		self.assertEqual(model.get_number_of_different_segments_and_CNAs(lins, 3), (1, 2, 3, 1))

	def test_get_segments_w_CNAs(self):
		# segments: 0, 1, 2, 3, 4
		# free segs: 1, 3
		cna_free1 = cnv.CNV(0, 1, 1, 10, 19)
		cna_free3 = cnv.CNV(0, 3, 1, 30, 39)
		my_lin0 = lineage.Lineage([], 1.0, [cna_free1, cna_free3], [], [], [], [], [], [], [])

		self.assertEqual(model.get_segments_w_CNAs([my_lin0], 5), [0, 2, 4])

		# segments: 0, 1, 2, 3, 4, 5
		# free segs: 0, 2, 3, 5
		cna_free0 = cnv.CNV(0, 0, 1, 1, 9)
		cna_free2 = cnv.CNV(0, 2, 1, 21, 29)
		cna_free3 = cnv.CNV(0, 3, 1, 30, 39)
		cna_free5 = cnv.CNV(0, 5, 1, 50, 59)
		my_lin0 = lineage.Lineage([], 1.0, [cna_free0, cna_free2, cna_free3, cna_free5], [], [], [], [], [], [], [])

		self.assertEqual(model.get_segments_w_CNAs([my_lin0], 6), [1, 4])

	def test_get_number_SSMs_on_segs_w_CNAs(self):
		# segments: 0, 1, 2, 3
		# CNA-free segments: 1, 3
		# SSMs: on all
		cna_free1 = cnv.CNV(0, 1, 1, 10, 19)
		cna_free3 = cnv.CNV(0, 3, 1, 30, 39)
		ssm_0 = snp_ssm.SSM()
		ssm_0.seg_index = 0
		ssm_1 = snp_ssm.SSM()
		ssm_1.seg_index = 1
		ssm_1_1 = snp_ssm.SSM()
		ssm_1_1.seg_index = 1
		ssm_2 = snp_ssm.SSM()
		ssm_2.seg_index = 2
		ssm_2_1 = snp_ssm.SSM()
		ssm_2_1.seg_index = 2
		ssm_2_2 = snp_ssm.SSM()
		ssm_2_2.seg_index = 2
		ssm_2_3 = snp_ssm.SSM()
		ssm_2_3.seg_index = 2
		ssm_3 = snp_ssm.SSM()
		ssm_3.seg_index = 3
		lin_0 = lineage.Lineage([], 1.0, [cna_free1, cna_free3], [], [], [], [], [], [], [])
		lin_1 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm_0, ssm_1, ssm_1_1], [], [])
		lin_2 = lineage.Lineage([], 0.5, [], [], [], [], [], [ssm_2, ssm_2_1], [ssm_2_2], [])
		lin_3 = lineage.Lineage([], 0.4, [], [], [], [], [], [ssm_3], [], [ssm_2_3])
		my_lins = [lin_0, lin_1, lin_2, lin_3]

		self.assertEqual(model.get_number_SSMs_on_segs_w_CNAs(my_lins, 4), 5)
		
		# segments: 0, 1, 2, 3
		# CNA-free segments: 1, 2
		# SSMs: on all
		cna_free1 = cnv.CNV(0, 1, 1, 10, 19)
		cna_free2 = cnv.CNV(0, 2, 1, 30, 39)
		ssm_0 = snp_ssm.SSM()
		ssm_0.seg_index = 0
		ssm_1 = snp_ssm.SSM()
		ssm_1.seg_index = 1
		ssm_1_1 = snp_ssm.SSM()
		ssm_1_1.seg_index = 1
		ssm_2 = snp_ssm.SSM()
		ssm_2.seg_index = 2
		ssm_2_1 = snp_ssm.SSM()
		ssm_2_1.seg_index = 2
		ssm_2_2 = snp_ssm.SSM()
		ssm_2_2.seg_index = 2
		ssm_2_3 = snp_ssm.SSM()
		ssm_2_3.seg_index = 2
		ssm_3 = snp_ssm.SSM()
		ssm_3.seg_index = 3
		lin_0 = lineage.Lineage([], 1.0, [cna_free1, cna_free2], [], [], [], [], [], [], [])
		lin_1 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm_0, ssm_1, ssm_1_1], [], [])
		lin_2 = lineage.Lineage([], 0.5, [], [], [], [], [], [ssm_2, ssm_2_1, ssm_2_2], [], [])
		lin_3 = lineage.Lineage([], 0.4, [], [], [], [], [], [ssm_3, ssm_2_3], [], [])
		my_lins = [lin_0, lin_1, lin_2, lin_3]

		self.assertEqual(model.get_number_SSMs_on_segs_w_CNAs(my_lins, 4), 2)
		

	def test_decombine_normal_segments_and_ssms(self):

		# seg_list
		seg_n_0 = segment.Segment_allele_specific(1, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_n_0.index = 0
		seg_c_1 = segment.Segment_allele_specific(2, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_1.index = 1
		seg_n_2 = segment.Segment_allele_specific(3, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_n_2.index = 2
		seg_c_3 = segment.Segment_allele_specific(4, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_c_3.index = 3
		seg_list = [seg_n_0, seg_c_1, seg_n_2, seg_c_3]

		# new_seg_list
		seg_n_0_new = segment.Segment_allele_specific(-1, 1, 40, 1.0, 0.25, 1.0, 0.25)
		seg_n_0_new.index = 0
		seg_c_1_new = segment.Segment_allele_specific(2, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_1_new.index = 1
		seg_c_3_new = segment.Segment_allele_specific(4, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_c_3_new.index = 2
		new_seg_list = [seg_n_0_new, seg_c_1_new, seg_c_3_new]

		# ssm_list
		sn0 = snp_ssm.SSM()
		sn0.chr = 1
		sn0.pos = 1
		sn0.seg_index = 0
		sc1 = snp_ssm.SSM()
		sc1.chr = 2
		sc1.pos = 1
		sc1.seg_index = 1
		sc2 = snp_ssm.SSM()
		sc2.chr = 2
		sc2.pos = 2
		sc2.seg_index = 1
		sn3 = snp_ssm.SSM()
		sn3.chr = 3
		sn3.pos = 1
		sn3.seg_index = 3
		sn4 = snp_ssm.SSM()
		sn4.chr = 3
		sn4.pos = 2
		sn4.seg_index = 3
		sc5 = snp_ssm.SSM()
		sc5.chr = 4
		sc5.pos = 1
		sc5.seg_index = 3
		ssm_list = [sn0, sc1, sc2, sn3, sn4, sc5]

		# ssm_normal
		ssm_normal = [sn0, sn3, sn4]

		# ssm_normal_changed
		sn0_new = snp_ssm.SSM()
		sn0_new.chr = -1
		sn0_new.pos = 1
		sn0_new.seg_index = 0
		sn3_new = snp_ssm.SSM()
		sn3_new.chr = -1
		sn3_new.pos = 2
		sn3_new.seg_index = 0
		sn4_new = snp_ssm.SSM()
		sn4_new.chr = -1
		sn4_new.pos = 3
		sn4_new.seg_index = 0
		ssm_normal_changed = [sn0_new, sn3_new, sn4_new]

		new_ssm_list = [sn0_new, sn3_new, sn4_new, sc1, sc2, sc5]

		# cnvs
		cnv0 = cnv.CNV(0, 0, -1, 1, 40)
		cnv1 = cnv.CNV(1, 1, 2, 1, 20)
		cnv4 = cnv.CNV(0, 2, 4, 1, 20)

		# cnvs original
		cnv0_ori = cnv.CNV(0, 0, 1, 1, 20)
		cnv1_ori = cnv.CNV(1, 1, 2, 1, 20)
		cnv2_ori = cnv.CNV(0, 2, 3, 1, 20)
		cnv4_ori = cnv.CNV(0, 3, 4, 1, 20)

		# lineages
		lin0 = lineage.Lineage([1], 1, [cnv0, cnv4], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1, [], [cnv1], [], [], [], [sn0_new, sc2], [sn3_new, sc1], [sn4_new, sc5])
		my_lineages = [lin0, lin1]

		new_normal_seg_indices = [0]
		normal_seg_indices = [0, 2]
		original_seg_index = [-1, 1, 3, -1]

		# to test
		(returned_seg_list, returned_new_ssm_list, returned_normal_seg_indices, returned_new_seg_list, returned_new_ssm_list_copied, returned_new_normal_seg_indices, returned_ssm_normal_changed_copied, returned_ssm_normal) = (model.decombine_normal_segments_and_ssms(my_lineages, new_seg_list, new_ssm_list, 
			new_normal_seg_indices, seg_list, ssm_list, ssm_normal_changed, ssm_normal, normal_seg_indices,
			original_seg_index))

		# lineage contains now originial SSMs and CNVs with correct indices
		self.assertEqual(my_lineages[0].cnvs_a, [cnv0_ori, cnv2_ori, cnv4_ori])
		self.assertEqual([my_cnv.seg_index for my_cnv in my_lineages[0].cnvs_a], [0, 2, 3])
		self.assertEqual(my_lineages[1].cnvs_b[0], cnv1_ori)
		self.assertEqual(my_lineages[1].cnvs_b[0].seg_index, 1)
		self.assertEqual(my_lineages[1].ssms, [sn0, sc2])
		self.assertEqual(my_lineages[1].ssms_a, [sc1, sn3])
		self.assertEqual(my_lineages[1].ssms_b, [sn4, sc5])

	def test_combine_normal_segments_and_ssms(self):

		# segments
		seg_n_0 = segment.Segment_allele_specific(1, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_n_0.index = 0
		seg_n_2 = segment.Segment_allele_specific(3, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_n_2.index = 2
		seg_c_1 = segment.Segment_allele_specific(2, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_1.index = 1
		# SSMs
		sn0 = snp_ssm.SSM()
		sn0.chr = 1
		sn0.pos = 1
		sn0.variant_count = 1
		sn0.ref_count = 1
		sn0.seg_index = 0
		sn2 = snp_ssm.SSM()
		sn2.chr = 3
		sn2.pos = 1
		sn2.variant_count = 1
		sn2.ref_count = 1
		sn2.seg_index = 2
		sc1 = snp_ssm.SSM()
		sc1.chr = 2
		sc1.pos = 1
		sc1.variant_count = 1
		sc1.ref_count = 1
		sc1.seg_index = 1

		seg_list = [seg_n_0, seg_c_1, seg_n_2]
		ssm_list = [sn0, sc1, sn2]
		normal_seg_indices = [0, 2]

		# to test
		(new_seg_list, new_ssm_list, new_normal_seg_indices, returned_seg_list, returned_ssm_list, ssm_normal_changed, ssm_normal, returned_normal_seg_indices, original_seg_index) = model.combine_normal_segments_and_ssms(
			seg_list, ssm_list, normal_seg_indices)

		self.assertEqual(new_normal_seg_indices, [0])
		self.assertEqual(returned_normal_seg_indices, normal_seg_indices)
		self.assertEqual(len(new_seg_list), 2)
		self.assertEqual(len(new_ssm_list), 3)
		self.assertEqual(returned_seg_list, seg_list)
		self.assertEqual(returned_ssm_list, ssm_list)

	def test_combine_normal_ssms_to_new_list(self):

		# create SSMs
		sn0 = snp_ssm.SSM()
		sn0.chr = 1
		sn0.pos = 1
		sn0.variant_count = 1
		sn0.seg_index = 0
		sn4 = snp_ssm.SSM()
		sn4.chr = 4
		sn4.pos = 1
		sn4.variant_count = 4
		sn4.seg_index = 3
		sn5 = snp_ssm.SSM()
		sn5.chr = 4
		sn5.pos = 2
		sn5.variant_count = 5
		sn5.seg_index = 3
		sn6 = snp_ssm.SSM()
		sn6.chr = 5
		sn6.pos = 1
		sn6.variant_count = 6
		sn6.seg_index = 4
		sn8 = snp_ssm.SSM()
		sn8.chr = 7
		sn8.pos = 1
		sn8.variant_count = 8
		sn8.seg_index = 6
		sc1 = snp_ssm.SSM()
		sc1.chr = 2
		sc1.pos = 1
		sc1.variant_count = 1
		sc1.seg_index = 1
		sc2 = snp_ssm.SSM()
		sc2.chr = 2
		sc2.pos = 2
		sc2.variant_count = 2
		sc2.seg_index = 1
		sc3 = snp_ssm.SSM()
		sc3.chr = 3
		sc3.pos = 1
		sc3.variant_count = 3
		sc3.seg_index = 2
		sc7 = snp_ssm.SSM()
		sc7.chr = 6
		sc7.pos = 1
		sc7.variant_count = 7
		sc7.seg_index = 5

		# create segments
		seg_n_0 = segment.Segment_allele_specific(-1, 1, 20, 1.0, 0.25, 1.0, 0.25)
		seg_n_0.index = 0
		seg_c_1 = segment.Segment_allele_specific(2, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_1.index = 1
		seg_c_2 = segment.Segment_allele_specific(3, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_1.index = 2
		seg_c_5 = segment.Segment_allele_specific(6, 1, 20, 2.0, 0.25, 1.0, 0.25)
		seg_c_5.index = 3
		new_seg_list = [seg_n_0, seg_c_1, seg_c_2, seg_c_5]
		
		ssm_list = [sn0, sc1, sc2, sc3, sn4, sn5, sn6, sc7, sn8]
		normal_seg_indices = [0, 3, 4, 6]

		# to test
		new_ssm_list, ssm_normal_changed, ssm_normal = model.combine_normal_ssms_to_new_list(
			ssm_list, normal_seg_indices, new_seg_list)

		# list ssm_normal contains normal SSMs without changes
		self.assertEqual(ssm_normal, [sn0, sn4, sn5, sn6, sn8])
		# SSMs from list ssm_normal_changed have correct positions
		self.assertEqual([ssm.pos for ssm in ssm_normal_changed], [1, 2, 3, 4, 5])
		# correct segment indices in new list
		self.assertEqual([ssm.seg_index for ssm in new_ssm_list], [0, 0, 0, 0, 0, 1, 1, 2, 3])
		# correct order of all SSMs in new list
		self.assertEqual([ssm.variant_count for ssm in new_ssm_list], [1, 4, 5, 6, 8, 1, 2, 3, 7])
		self.assertEqual(len(ssm_normal_changed), len(ssm_normal))

	def test_combine_normal_segments_to_new_list(self):

		# segments
		sn0 = segment.Segment_allele_specific(1, 1, 10, 1.0, 0.25, 1.0, 0.25)
		sn0.index = 0
		sn2 = segment.Segment_allele_specific(3, 1, 10, 1.0, 0.25, 1.0, 0.25)
		sn2.index = 2
		sn3 = segment.Segment_allele_specific(4, 1, 10, 1.0, 0.25, 1.0, 0.25)
		sn3.index = 3
		sn6 = segment.Segment_allele_specific(7, 1, 20, 1.0, 0.25, 1.0, 0.25)
		sn6.index = 6
		sc1 = segment.Segment_allele_specific(2, 1, 10, 2.0, 0.25, 1.0, 0.25)
		sc1.index = 1
		sc4 = segment.Segment_allele_specific(5, 1, 10, 2.0, 0.25, 1.0, 0.25)
		sc4.index = 4
		sc5 = segment.Segment_allele_specific(6, 1, 10, 2.0, 0.25, 1.0, 0.25)
		sc5.index = 5
		
		seg_list = [sn0, sc1, sn2, sn3, sc4, sc5, sn6]
		seg_list_copy = copy.deepcopy(seg_list)
		normal_seg_indices = [0, 2, 3, 6]
		original_seg_index_true = [-1, 1, 4, 5, -1, -1, -1]

		# to test
		new_seg_list, original_seg_index = model.combine_normal_segments_to_new_list(seg_list, normal_seg_indices)

		self.assertEqual(len(new_seg_list), 4)
		self.assertEqual(seg_list, seg_list_copy)
		self.assertEqual([0, 1, 2, 3], [seg.index for seg in new_seg_list])
		self.assertEqual(new_seg_list[0].chr, -1)
		self.assertEqual(new_seg_list[0].start, 1)
		self.assertEqual(new_seg_list[0].end, 50)
		self.assertEqual(original_seg_index, original_seg_index_true)

	def test_find_absent_adrs_necessary_sumrule(self):
		#####################
		# 1) K=4
		# lineages
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([3], 0.5, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.3, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		z_matrix_ori = copy.deepcopy(z_matrix)
		zero_count = 1
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# function to test
		model.find_absent_adrs_necessary_sumrule(zmcos, my_lins)

		self.assertTrue(np.array(z_matrix_ori).all() == zmcos[0].z_matrix.all())

		#####################
		# 1) K=4
		# lineages
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		z_matrix_ori = copy.deepcopy(z_matrix)
		zero_count = 1
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# function to test
		model.find_absent_adrs_necessary_sumrule(zmcos, my_lins)

		self.assertTrue(np.array(z_matrix_ori).all() == zmcos[0].z_matrix.all())

		#####################
		# 1) K=4
		# lineages
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([3], 0.5, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.3, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		zero_count = 1
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# function to test
		model.find_absent_adrs_necessary_sumrule(zmcos, my_lins)

		self.assertEqual(len(zmcos), 1)
		self.assertEqual(zmcos[0].z_matrix[1][2], -1)
		
	def test_get_all_matrices_fulfilling_LDR(self):

		#####################
		# 1) K=3, linear phylogeny, k=0 only has one child, done
		# lineages
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		self.assertTrue((z_matrix == new_zmcos[0].z_matrix).all())

		#####################
		# 2) K=3, branching phylogeny, k=0 has two children, but LDR is fulfilled, done
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		self.assertTrue((z_matrix == new_zmcos[0].z_matrix).all())

		#####################
		# 3) K=3, branching phylogeny, k=0 has two children, LDR is violated but not possible that 2 is child of 1
		# no solution
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# change Z-matrix
		z_matrix_list[0][1][2] = -1
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(0, len(new_zmcos))

		#####################
		# 4) K=3, branching phylogeny, k=0 has two children, LDR is violated, if 2 is child of 1, LDR is fulfilled
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		z_matrix[1][2] = 1
		self.assertTrue((z_matrix == new_zmcos[0].z_matrix).all())

		#####################
		# 5) K=4, branching phylogeny, k=0 has three children, LDR is violated, only fulfilled if 2 is child of 1,
		# and 3 of 2
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		z_matrix[1][2] = 1
		z_matrix[1][3] = 1
		z_matrix[2][3] = 1
		self.assertTrue((z_matrix == new_zmcos[0].z_matrix).all())

		#####################
		# 6) K=4, branching phylogeny, k=0 has three children, as soon as it has only 2, LDR is fulfilled
		# with SSM phasing
		ssm1_u = snp_ssm.SNP_SSM()
		ssm1_u.phase = cons.UNPHASED
		ssm1_u.seg_index = 0
		ssm2_u = snp_ssm.SNP_SSM()
		ssm2_u.phase = cons.UNPHASED
		ssm2_u.seg_index = 0
		gain3_a = cnv.CNV(-1, 0, 1, 1, 1)
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm1_u], [], [])
		lin2 = lineage.Lineage([], 0.2, [], [], [], [], [], [ssm2_u], [], [])
		lin3 = lineage.Lineage([], 0.1, [gain3_a], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(3, len(new_zmcos))
		self.assertEqual(new_zmcos[2].z_matrix[1][2], 1)
		self.assertEqual(new_zmcos[1].z_matrix[1][3], 1)
		self.assertEqual(new_zmcos[0].z_matrix[2][3], 1)
		self.assertTrue(new_zmcos[1].present_ssms[0][cons.B][1])
		self.assertTrue(new_zmcos[0].present_ssms[0][cons.B][2])

		#####################
		# 7) K=4, branching phylogeny with 2 starting matrices
		# example shows, that all matrices from input are checked
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# change Z-matrix in list and create a second of everything
		new_z_matrix = copy.deepcopy(z_matrix_list[0])
		z_matrix_list[0][1][2] = -1
		new_z_matrix[2][3] = -1
		z_matrix_list.append(new_z_matrix)
		present_ssms_list.append(copy.deepcopy(present_ssms_list[0]))
		triplets_list.append(copy.deepcopy(triplets_list[0]))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(4, len(new_zmcos))
		self.assertEqual(new_zmcos[0].z_matrix[1][2], -1)
		self.assertEqual(new_zmcos[0].z_matrix[1][3], -1)
		self.assertEqual(new_zmcos[0].z_matrix[2][3], 1)
		self.assertEqual(new_zmcos[1].z_matrix[1][2], -1)
		self.assertEqual(new_zmcos[1].z_matrix[1][3], 1)
		self.assertEqual(new_zmcos[1].z_matrix[2][3], -1)
		self.assertEqual(new_zmcos[2].z_matrix[1][2], 0)
		self.assertEqual(new_zmcos[2].z_matrix[1][3], 1)
		self.assertEqual(new_zmcos[2].z_matrix[2][3], -1)
		self.assertEqual(new_zmcos[3].z_matrix[1][2], 1)
		self.assertEqual(new_zmcos[3].z_matrix[1][3], 0)
		self.assertEqual(new_zmcos[3].z_matrix[2][3], -1)

		#####################
		# 8) LDR is violated, 2 needs to become a child of 1, 2 has onw child, this must become a descendant of 1
		# through transitivity
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		z_matrix_list[0][1][2] = 1
		z_matrix_list[0][1][3] = 1
		self.assertTrue((z_matrix_list[0]==new_zmcos[0].z_matrix).all())

		#####################
		# 9) LDR is violated, needs to be resolved twice
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3, 4, 5], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([4, 5], 0.6, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.4, [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3, lin4, lin5]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1, 1, 1], [-1, -1, 0, 0, 0, 0], [-1, -1, -1, 1, 1, 1], [-1, -1, -1, -1, 1, 1],
			[-1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		z_matrix_list[0][1][2] = 1
		z_matrix_list[0][1][3] = 1
		z_matrix_list[0][1][4] = 1
		z_matrix_list[0][1][5] = 1
		z_matrix_list[0][4][5] = 1
		self.assertTrue((z_matrix_list[0]==new_zmcos[0].z_matrix).all())

		#####################
		# 10) first, LDR is only violated for k=0, than 2 becomes a child of 1, then LDR is violated for 1,
		# then 4 becomes a child of 2, then LDR is violated for 2, then 4 becomes a child of 3
		# with SSM phasing
		ssm1_u = snp_ssm.SNP_SSM()
		ssm1_u.phase = cons.UNPHASED
		ssm1_u.seg_index = 0
		ssm2_a = snp_ssm.SNP_SSM()
		ssm2_a.phase = cons.A
		ssm2_a.seg_index = 0
		ssm2_b = snp_ssm.SNP_SSM()
		ssm2_b.phase = cons.B
		ssm2_b.seg_index = 0
		ssm3_b = snp_ssm.SNP_SSM()
		ssm3_b.phase = cons.B
		ssm3_b.seg_index = 0
		ssm4_u = snp_ssm.SNP_SSM()
		ssm4_u.phase = cons.UNPHASED
		ssm4_u.seg_index = 0
		loss3_b = cnv.CNV(-1, 0, 1, 1, 1)
		lin0 = lineage.Lineage([1, 2, 3, 4], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([4], 0.9, [], [], [], [], [], [ssm1_u], [], [])
		lin2 = lineage.Lineage([3], 0.8, [], [], [], [], [], [], [ssm2_a], [ssm2_b])
		lin3 = lineage.Lineage([], 0.7, [], [loss3_b], [], [], [], [], [], [ssm3_b])
		lin4 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm4_u], [], [])
		my_lins = [lin0, lin1, lin2, lin3, lin4]
		seg_num = 1
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 0, 0, 1], [-1, -1, -1, 1, 0], [-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]
		zero_count = 0
		# triplets
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		# original Z-matrix and other stuff
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
			model.post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, 
			triplet_xsy, get_CNVs=True))
		# Z_Matrix_Co object
		zmcos = model.create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)

		# to test
		new_zmcos = model.get_all_matrices_fulfilling_LDR(zmcos, my_lins)
		self.assertEqual(1, len(new_zmcos))
		z_matrix_list[0][1][2] = 1
		z_matrix_list[0][1][3] = 1
		z_matrix_list[0][2][4] = 1
		z_matrix_list[0][3][4] = 1
		self.assertTrue((z_matrix_list[0]==new_zmcos[0].z_matrix).all())
		self.assertFalse(new_zmcos[0].present_ssms[0][cons.UNPHASED][1])
		self.assertTrue(new_zmcos[0].present_ssms[0][cons.A][1])
		self.assertFalse(new_zmcos[0].present_ssms[0][cons.UNPHASED][4])
		self.assertTrue(new_zmcos[0].present_ssms[0][cons.A][4])


	def test_get_all_possible_children_combinations(self):

		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 0, -1, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]
			]
		self.assertEqual([(1, 2), (1, 4), (2, 4), (3, 4)], model.get_all_possible_children_combinations(
			z_matrix, [1, 2, 3, 4]))

		z_matrix[1][2] = 1
		with self.assertRaises(eo.MyException):
			model.get_all_possible_children_combinations(z_matrix, [1, 2, 3, 4])

	def test_get_children(self):

		z_matrix = [
			[-1, 1, 1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 0, 1, 1],
			[-1, -1, -1, 1, 0, 1, 0],
			[-1, -1, -1, -1, 0, 1, 0],
			[-1, -1, -1, -1, -1, 0, 0],
			[-1, -1, -1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1, -1, -1]
			]
		self.assertEqual([2,6], model.get_children(z_matrix, 1))
		self.assertEqual([1,4], model.get_children(z_matrix, 0))
		self.assertEqual([], model.get_children(z_matrix, 5))

	def test_get_0_number_in_z_matrix(self):

		z_matrix = [[-1, 1, 1 ,1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		self.assertEqual(2, model.get_0_number_in_z_matrix(z_matrix))

	def test_check_CN_influence(self):

		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		cnv1 = cnv.CNV("+1", 0, 1, 1, 10)
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], 0.3, [], [], [], [], [], [], [ssm1], [])
		lin2 = lineage.Lineage([3], 0.2, [cnv1], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		lineages = [lin0, lin1, lin2, lin3]

		z_matrix = model.get_Z_matrix(lineages)[0]
		self.assertEqual(z_matrix, [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]])

		model.check_CN_influence(z_matrix, lineages)
		self.assertEqual(z_matrix, [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]])

	def test_is_CN_influence_present(self):

		k = 1
		k_prime = 2

		# not present
		CN_changes_hash = {}
		CN_changes_hash[k_prime] = {}
		CN_changes_hash[k_prime][0] = [cons.A]
		CN_changes_hash[k_prime][1] = [cons.B]
		SSMs_hash = {}
		SSMs_hash[k] = {}
		SSMs_hash[k][0] = [cons.B]

		self.assertFalse(model.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))

		# for lineages that don't contain mutations
		self.assertFalse(model.is_CN_influence_present(3, 4, CN_changes_hash, SSMs_hash))
		self.assertFalse(model.is_CN_influence_present(k, 4, CN_changes_hash, SSMs_hash))

		# influence is present, on A
		CN_changes_hash = {}
		CN_changes_hash[k_prime] = {}
		CN_changes_hash[k_prime][0] = [cons.A]
		SSMs_hash = {}
		SSMs_hash[k] = {}
		SSMs_hash[k][0] = [cons.A]
		self.assertTrue(model.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))

		# influence is present, on B, but not on first segment
		CN_changes_hash = {}
		CN_changes_hash[k_prime] = {}
		CN_changes_hash[k_prime][0] = [cons.B]
		CN_changes_hash[k_prime][1] = [cons.A, cons.B]
		SSMs_hash = {}
		SSMs_hash[k] = {}
		SSMs_hash[k][0] = [cons.A]
		SSMs_hash[k][1] = [cons.B]
		self.assertTrue(model.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))


	def test_create_CN_changes_and_SSM_hash_for_LDR(self):

		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		cnv1 = cnv.CNV("+1", 0, 1, 1, 10)
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		cnv2 = cnv.CNV("+1", 2, 1, 1, 10)

		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.5, [cnv1, cnv2], [cnv1], [], [], [], [], [ssm1], [ssm1, ssm2])
		lin2 = lineage.Lineage([], 0.4, [], [cnv2], [], [], [], [], [ssm2], [])

		CN_changes, SSMs = model.create_CN_changes_and_SSM_hash_for_LDR([lin0, lin1, lin2])
		
		self.assertEqual(len(CN_changes), 2)
		self.assertEqual(CN_changes[1][0], [cons.A, cons.B])
		self.assertEqual(CN_changes[1][2], [cons.A])
		self.assertEqual(CN_changes[2][2], [cons.B])
		self.assertEqual(len(SSMs), 2)
		self.assertEqual(SSMs[1][0], [cons.A, cons.B])
		self.assertEqual(SSMs[1][1], [cons.B])
		self.assertEqual(SSMs[2][1], [cons.A])
		


	def test_add_SSM_appearence_to_hash(self):

		seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = seg_index
		SSMs = {}
		lin_index = 1
		phase = cons.A

		# add as first entry
		model.add_SSM_appearence_to_hash(SSMs, [ssm1], phase, lin_index)
		self.assertEqual(len(SSMs), 1)
		self.assertEqual(SSMs[lin_index][seg_index], [phase])

		# add something twice, not happens
		model.add_SSM_appearence_to_hash(SSMs, [ssm1], phase, lin_index)
		self.assertEqual(len(SSMs), 1)
		self.assertEqual(SSMs[lin_index][seg_index], [phase])

		# add second thing
		phase_2 = cons.B
		model.add_SSM_appearence_to_hash(SSMs, [ssm1], phase_2, lin_index)
		self.assertEqual(len(SSMs), 1)
		self.assertEqual(SSMs[lin_index][seg_index], [phase, phase_2])

	def test_add_CN_changes_to_hash(self):

		seg_index = 0
		cnv1 = cnv.CNV("+1", seg_index, 1, 1, 10)
		CN_changes = {}
		lin_index = 1
		phase = cons.A

		# add as first entry
		model.add_CN_changes_to_hash(CN_changes, [cnv1], phase, lin_index)
		self.assertEqual(len(CN_changes), 1)
		self.assertEqual(CN_changes[lin_index][seg_index], [phase])

		# add something twice, not possible
		with self.assertRaises(eo.MyException):
			model.add_CN_changes_to_hash(CN_changes, [cnv1], phase, lin_index)

		# add second thing
		phase_2 = cons.B
		model.add_CN_changes_to_hash(CN_changes, [cnv1], phase_2, lin_index)
		self.assertEqual(len(CN_changes), 1)
		self.assertEqual(CN_changes[lin_index][seg_index], [phase, phase_2])


	def test_combine_ssm_lists_with_different_info(self):

		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm1.variant_count = 10
		ssm1.ref_count = 20
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm2.variant_count = 20
		ssm2.ref_count = 30
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 2
		ssm3.pos = 1
		ssm3.variant_count = 40
		ssm3.ref_count = 50

		ssm4 = snp_ssm.SSM()
		ssm4.chr = 1
		ssm4.pos = 1
		ssm4.lineage = 2
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 1
		ssm5.pos = 2
		ssm5.lineage = 1
		ssm6 = snp_ssm.SSM()
		ssm6.chr = 2
		ssm6.pos = 1
		ssm6.lineage = 2

		# lists don't have the same length
		list_counts_1 = [ssm1]
		list_reconstruction = [ssm4, ssm5, ssm6]

		with self.assertRaises(eo.MyException):
			model.combine_ssm_lists_with_different_info(list_counts_1, list_reconstruction)

		# SSMs are not ordered properly
		list_counts_2 = [ssm1, ssm3, ssm2]
		with self.assertRaises(eo.MyException):
			model.combine_ssm_lists_with_different_info(list_counts_2, list_reconstruction)

		# everything works
		list_counts_3 = [ssm1, ssm2, ssm3]
		model.combine_ssm_lists_with_different_info(list_counts_3, list_reconstruction)
		self.assertTrue(ssm4.variant_count, 10)
		self.assertTrue(ssm5.variant_count, 20)
		self.assertTrue(ssm6.variant_count, 40)
		self.assertTrue(ssm4.ref_count, 20)
		self.assertTrue(ssm5.ref_count, 30)
		self.assertTrue(ssm6.ref_count, 50)
		

	def test_build_ssm_list_from_lineages(self):
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 2
		ssm3.pos = 1

		lin_0 = lineage.Lineage([1,2], 1.0, [], [], [], [], [], [], [], [])
		lin_1 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm3], [], [])
		lin_2 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [ssm2], [ssm1])
		lineages = [lin_0, lin_1, lin_2]

		ssm_list_true = [ssm1, ssm2, ssm3]
		ssm_list_build = model.build_ssm_list_from_lineages(lineages)

		self.assertEqual(ssm_list_true, ssm_list_true)

	def test_compare_SSMs_keep_the_ones_of_one_lineage_list(self):
		# lineage with SSMs to keep has SSMs in all phases, on different chromosomes
		# lineage with which it is compared is bigger, has more SSMs in all phases

		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 2
		ssm3.pos = 1
		ssm4 = snp_ssm.SSM()
		ssm4.chr = 3
		ssm4.pos = 4
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 3
		ssm5.pos = 5
		ssm6 = snp_ssm.SSM()
		ssm6.chr = 5
		ssm6.pos = 1
		lin_keep_0 = lineage.Lineage([1,2], 1.0, [], [], [], [], [], [], [], [])
		lin_keep_1 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm1], [ssm2, ssm3], [ssm4])
		lin_keep_2 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [ssm5], [ssm6])
		lin_keep = [lin_keep_0, lin_keep_1, lin_keep_2]

		ssm7 = snp_ssm.SSM()
		ssm7.chr = 1
		ssm7.pos = 3
		ssm8 = snp_ssm.SSM()
		ssm8.chr = 1
		ssm8.pos = 10
		ssm9 = snp_ssm.SSM()
		ssm9.chr = 2
		ssm9.pos = 2
		ssm10 = snp_ssm.SSM()
		ssm10.chr = 3
		ssm10.pos = 6
		ssm11 = snp_ssm.SSM()
		ssm11.chr = 7
		ssm11.pos = 5
		ssm12 = snp_ssm.SSM()
		ssm12.chr = 8
		ssm12.pos = 1
		lin_other_0 = lineage.Lineage([1,2], 1.0, [], [], [], [], [], [], [], [])
		lin_other_1 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm7, ssm8], [ssm3], [ssm2])
		lin_other_2 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm6], [ssm1, ssm12], [ssm5])
		lin_other_3 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm10, ssm11], [ssm9], [ssm4])
		lin_other = [lin_other_0, lin_other_1, lin_other_2, lin_other_3]

		lin_true_changed_0 = lineage.Lineage([1,2], 1.0, [], [], [], [], [], [], [], [])
		lin_true_changed_1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [ssm3], [ssm2])
		lin_true_changed_2 = lineage.Lineage([], 1.0, [], [], [], [], [], [ssm6], [ssm1], [ssm5])
		lin_true_changed_3 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [ssm4])
		lin_true_changed = [lin_true_changed_0, lin_true_changed_1, lin_true_changed_2, lin_true_changed_3]

		lin_changed = model.compare_SSMs_keep_the_ones_of_one_lineage_list(lin_other, lin_keep)

		self.assertEqual(lin_changed, lin_true_changed)



	def test_count_number_of_ssms_per_segment(self):

		# each segment has SSMs
		seg1 = segment.Segment_allele_specific(1, 1, 10, 1, 0.1, 1, 0.1)
		seg2 = segment.Segment_allele_specific(1, 11, 20, 1, 0.1, 1, 0.1)
		seg3 = segment.Segment_allele_specific(1, 21, 30, 1, 0.1, 1, 0.1)
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		seg_list = [seg1, seg2, seg3]
		ssm_list = [ssm1, ssm1, ssm1, ssm2, ssm3, ssm3]

		self.assertEqual(model.count_number_of_ssms_per_segment(seg_list, ssm_list), [3, 1, 2])

		# not all segments have SSMs
		seg1 = segment.Segment_allele_specific(1, 1, 10, 1, 0.1, 1, 0.1)
		seg2 = segment.Segment_allele_specific(1, 11, 20, 1, 0.1, 1, 0.1)
		seg3 = segment.Segment_allele_specific(1, 21, 30, 1, 0.1, 1, 0.1)
		seg4 = segment.Segment_allele_specific(1, 31, 40, 1, 0.1, 1, 0.1)
		seg5 = segment.Segment_allele_specific(1, 41, 50, 1, 0.1, 1, 0.1)
		seg6 = segment.Segment_allele_specific(1, 51, 60, 1, 0.1, 1, 0.1)
		seg7 = segment.Segment_allele_specific(1, 61, 70, 1, 0.1, 1, 0.1)
		seg8 = segment.Segment_allele_specific(1, 61, 70, 1, 0.1, 1, 0.1)
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 4
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 5
		seg_list = [seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8]
		ssm_list = [ssm1, ssm1, ssm2, ssm3, ssm3]

		self.assertEqual(model.count_number_of_ssms_per_segment(seg_list, ssm_list), [0, 2, 0, 0, 1, 2, 0, 0])



	def test_adapt_lineages_after_Z_matrix_update(self):
		
		# no forking, Z-matrix wasn't change after first round
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 3
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
		my_lineages = [lin0, lin1, lin2]
		z_matrix_fst_rnd = np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])
		z_matrix_list = [np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]

		my_lins, new_lineages_list = model.adapt_lineages_after_Z_matrix_update(my_lineages, 
			z_matrix_fst_rnd, z_matrix_list, 
			None, None)

		self.assertEqual(len(new_lineages_list), 1)
		self.assertEqual(new_lineages_list[0], my_lineages)
		self.assertEqual(my_lins, my_lineages)

		# Z-matrix was changed after first round
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 3
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		# ssm3 is unphased at beginning, segment 2
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
		my_lineages = [lin0, lin1, lin2]
		my_lin1 = copy.deepcopy(lin1)
		my_lin1.ssms = [ssm3]
		my_lin1.ssms_a = [ssm0, ssm1]
		my_lin1.sublins = [2]
		my_lin2 = copy.deepcopy(lin2)
		my_lin2.ssms = []
		my_lin2.ssms_b = [ssm2]
		my_right_lineages = [lin0, my_lin1, my_lin2]
		my_lineages = [lin0, lin1, lin2]
		my_lin2_1 = copy.deepcopy(lin1)
		my_lin2_1.ssms = [ssm3]
		my_lin2_1.ssms_b = [ssm0, ssm1, ssm4]
		my_lin2_1.sublins = [2]
		my_lin2_2 = copy.deepcopy(lin2)
		my_lin2_2.ssms = []
		my_lin2_2.ssms_a = [ssm2]
		my_right_lineages2 = [lin0, my_lin2_1, my_lin2_2]
		lin_num = 3
		seg_num = 4
		# [segment][lineage][A, B, unphased]
		origin_present_ssms = [[[False] * lin_num for _ in xrange(3)] for x in xrange(seg_num)]
		origin_present_ssms[0][cons.UNPHASED][1] = True
		origin_present_ssms[2][cons.UNPHASED][2] = True
		origin_present_ssms[2][cons.UNPHASED][1] = True
		origin_present_ssms[3][cons.B][1] = True
		current_ssms_list = copy.deepcopy(origin_present_ssms)
		current_ssms_list[0][cons.UNPHASED][1]= False
		current_ssms_list[0][cons.A][1] = True
		current_ssms_list[2][cons.UNPHASED][2] = False
		current_ssms_list[2][cons.B][2] = True
		current_ssms_list2 = copy.deepcopy(origin_present_ssms)
		current_ssms_list2[0][cons.UNPHASED][1] = False
		current_ssms_list2[0][cons.B][1] = True
		current_ssms_list2[2][cons.UNPHASED][2] = False
		current_ssms_list2[2][cons.A][2] = True
		present_ssms_list = [current_ssms_list, current_ssms_list2]
		i = 0
		present_ssms_list = [current_ssms_list, current_ssms_list2]
		z_matrix_list = [np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]), 
			np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]
		z_matrix_fst_rnd = [np.asarray([[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]])]

		my_lins, new_lineages_list = model.adapt_lineages_after_Z_matrix_update(my_lineages, 
			z_matrix_fst_rnd, z_matrix_list, origin_present_ssms, present_ssms_list)

		self.assertEqual(new_lineages_list[0], my_right_lineages)
		self.assertEqual(new_lineages_list[1], my_right_lineages2)
	

	def test_create_updates_lineages(self):
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 3
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
		my_lineages = [lin0, lin1, lin2]
		my_lin1 = copy.deepcopy(lin1)
		my_lin1.ssms = [ssm3]
		my_lin1.ssms_a = [ssm0, ssm1]
		my_lin1.sublins = [2]
		my_lin2 = copy.deepcopy(lin2)
		my_lin2.ssms = []
		my_lin2.ssms_b = [ssm2]
		my_right_lineages = [lin0, my_lin1, my_lin2]
		lin_num = 3
		seg_num = 4
		origin_present_ssms = [[[False] * lin_num for _ in xrange(3)] for x in xrange(seg_num)]
		origin_present_ssms[0][cons.UNPHASED][1] = True
		origin_present_ssms[2][cons.UNPHASED][2] = True
		origin_present_ssms[2][cons.UNPHASED][1] = True
		origin_present_ssms[3][cons.B][1] = True
		current_ssms_list = copy.deepcopy(origin_present_ssms)
		current_ssms_list[0][cons.UNPHASED][1] = False
		current_ssms_list[0][cons.A][1] = True
		current_ssms_list[2][cons.UNPHASED][2] = False
		current_ssms_list[2][cons.B][2] = True
		i = 5
		present_ssms_list = [[], [], [], [], [], current_ssms_list]
		z_matrix_list = [[], [], [], [], [], np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]

		new_lineages = model.create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, 
			present_ssms_list)

		self.assertEqual(new_lineages, my_right_lineages)
		self.assertEqual(my_lineages[1].sublins, [])

	def test_update_SSM_phasing_after_Z_matrix_update(self):

		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 3
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
		current_lineages = [lin0, lin1, lin2]
		my_lin1 = copy.deepcopy(lin1)
		my_lin1.ssms = [ssm3]
		my_lin1.ssms_a = [ssm0, ssm1]
		my_lin2 = copy.deepcopy(lin2)
		my_lin2.ssms = []
		my_lin2.ssms_b = [ssm2]
		my_right_lineages = [lin0, my_lin1, my_lin2]
		lin_num = 3
		seg_num = 4
		origin_present_ssms = [[[False] * lin_num for _ in xrange(3)] for x in xrange(seg_num)]
		origin_present_ssms[0][cons.UNPHASED][1] = True
		origin_present_ssms[2][cons.UNPHASED][2] = True
		origin_present_ssms[2][cons.UNPHASED][1] = True
		origin_present_ssms[3][cons.B][1] = True
		current_ssms_list = copy.deepcopy(origin_present_ssms)
		current_ssms_list[0][cons.UNPHASED][1] = False
		current_ssms_list[0][cons.A][1] = True
		current_ssms_list[2][cons.UNPHASED][2] = False
		current_ssms_list[2][cons.B][2] = True

		model.update_SSM_phasing_after_Z_matrix_update(current_lineages, origin_present_ssms, current_ssms_list)

		self.assertEqual(current_lineages, my_right_lineages)

	def test_get_updated_SSM_list(self):
		lin_index = 1
		phase = cons.B
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssms_per_segments = [[[[], []], [[], []], [[], []]], [[[], []], [[ssm0], [ssm1, ssm2]], [[], []]]]

		new_list = model.get_updated_SSM_list(lin_index, phase, ssms_per_segments)

		self.assertEqual(new_list, [ssm0, ssm1, ssm2])


	def test_move_SSMs_in_list_per_segment(self):
		seg_index = 1
		lin_index = 0
		new_phase = cons.A
		old_phase = cons.B
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 1
		ssm0.chr = 1
		ssm0.pos = 3
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm1.chr = 1
		ssm1.pos = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		ssm2.chr = 1
		ssm2.pos = 5
		ssms_per_segments = [[[[], [ssm0]], [[], [ssm1, ssm2]], [[], []]], [[[], []], [[], []], [[], []]]]

		model.move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, new_phase, old_phase)

		self.assertEqual(ssms_per_segments[lin_index][new_phase][seg_index], [ssm1, ssm0, ssm2])
		self.assertEqual(ssms_per_segments[lin_index][old_phase][seg_index], [])


	def test_get_ssms_per_segments(self):
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 3
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1], [], [ssm3, ssm4])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [ssm2], [])
		current_lineages = [lin0, lin1, lin2]
		seg_num = 4

		my_ssms_per_segments = [
			[[[], [], [], []], [[], [], [], []], [[], [], [], []]], 
			[[[], [], [], []], [[], [], [ssm3], [ssm4]], [[ssm0, ssm1], [], [], []]], 
			[[[], [], [ssm2], []], [[], [], [], []], [[], [], [], []]]
			]

		ssms_per_segments = model.get_ssms_per_segments(current_lineages, seg_num)

		self.assertEqual(ssms_per_segments, my_ssms_per_segments)
		

	def test_get_ssms_per_segments_lineage_phase(self):
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 2
		my_ssms = [ssm0, ssm1, ssm2]
		seg_num = 4

		ssms_per_segment_tmp = model.get_ssms_per_segments_lineage_phase(my_ssms, seg_num)

		self.assertEqual(ssms_per_segment_tmp, [[ssm0, ssm1], [], [ssm2], []])

	def test_update_sublineages_after_Z_matrix_update(self):
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		current_lineages = [lin0, lin1, lin2]
		lin_num = 3
		current_z_matrix = np.asarray([
			[-1, 1, 1],
			[-1, -1, 1],
			[-1, -1, -1]
			])

		model.update_sublineages_after_Z_matrix_update(current_lineages, current_z_matrix)

		self.assertEqual(lin0.sublins, [1, 2])
		self.assertEqual(lin1.sublins, [2])
		self.assertEqual(lin2.sublins, [])

	def test_activate_ancestor_descendant_relation(self):

		# k < k_prime
		z_matrix = [[-1, 1], [-1, -1]]
		k = 2
		k_prime = 1
		with self.assertRaises(eo.MyException):
			model.activate_ancestor_descendant_relation(z_matrix, k, k_prime, None, 0)

		# entry in Z-matrix not 0
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		k = 1
		k_prime = 2
		with self.assertRaises(eo.MyException):
			model.activate_ancestor_descendant_relation(z_matrix, k, k_prime, None, 0)

		# Z-matrix was not updated completely before use
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		k = 1
		k_prime = 2
		with self.assertRaises(eo.MyException):
			model.activate_ancestor_descendant_relation(z_matrix, k, k_prime, None, 0)

		# ancestor-descendant relation gets updated and phasing of SSMs is changed
		ssm1 = snp_ssm.SSM()
		ssm1.pos = 1
		ssm1.chr = 1
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.pos = 2
		ssm2.chr = 1
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.pos = 12
		ssm3.chr = 1
		ssm3.seg_index = 1
		cnv1 = cnv.CNV(1, 0, 1, 10, 1)
		lin1 = lineage.Lineage([1,2,3], 1, None, None, None, None, None, [], [] , [])
		lin2 = lineage.Lineage([2,3], 0.9, None, None, None, None, None, [], [ssm1], [])
		lin3 = lineage.Lineage([], 0.4, None, None, None, None, None, [ssm2, ssm3], [], [])
		lin4 = lineage.Lineage([], 0.3, [cnv1], None, None, None, None, [], [], [])
		my_lins = [lin1, lin2, lin3, lin4]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		z_matrix_updated = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		k = 2
		k_prime = 3
		new_lins = model.activate_ancestor_descendant_relation(z_matrix, k, k_prime, my_lins, 2)[1]

		self.assertEqual(z_matrix, z_matrix_updated)
		self.assertEqual(new_lins[2].ssms[0].pos, 12)
		self.assertEqual(new_lins[2].ssms_b[0].pos, 2)

		# a.-d. relation gets updated + more relations gets updated
		lin1 = lineage.Lineage([1,2,3,4], 1, None, None, None, None, None, [], [], [])
		lin2 = lineage.Lineage([2,3,4], 0.9, None, None, None, None, None, [], [], [])
		lin3 = lineage.Lineage([], 0.5, None, None, None, None, None, [], [], [])
		lin4 = lineage.Lineage([4], 0.4, None, None, None, None, None, [], [], [])
		lin5 = lineage.Lineage([], 0.3, None, None, None, None, None, [], [], [])
		my_lins = [lin1, lin2, lin3, lin4, lin5]
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 1, 1, 1], [-1, -1, -1, 0, 0], [-1, -1, -1, -1, 1],
			[-1, -1, -1, -1, -1]]
		z_matrix_updated = [[-1, 1, 1, 1, 1], [-1, -1, 1, 1, 1], [-1, -1, -1, 1, 1], [-1, -1, -1, -1, 1],
			[-1, -1, -1, -1, -1]]
		k = 2
		k_prime = 3
		model.activate_ancestor_descendant_relation(z_matrix, k, k_prime, my_lins, 1)

		self.assertEqual(z_matrix, z_matrix_updated)

	def test_has_SSMs_in_segment(self):

		lin_num = 3
		seg_num = 2
		present_ssms_1 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_1[0][cons.A][1] = True
		present_ssms_1[0][cons.UNPHASED][2] = True
		present_ssms_2 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_2[0][cons.UNPHASED][1] = True
		present_ssms_2[0][cons.B][2] = True
		present_ssms_list = [present_ssms_1, present_ssms_2]

		self.assertTrue(model.has_SSMs_in_segment(present_ssms_list, 1, 0))
		self.assertTrue(model.has_SSMs_in_segment(present_ssms_list, 2, 0))
		self.assertFalse(model.has_SSMs_in_segment(present_ssms_list, 1, 1))
		self.assertFalse(model.has_SSMs_in_segment(present_ssms_list, 2, 1))
		

	def test_fork_z_matrix(self):

		# from 2 Z-matrices to 2 Z-matrices, without new phasing
		lin_num = 5
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[0][3] = 1
		matrix_after_first_round[0][4] = 1
		matrix_after_first_round[1][2] = 0
		matrix_after_first_round[1][3] = 0
		matrix_after_first_round[1][4] = -1
		matrix_after_first_round[2][3] = 0
		matrix_after_first_round[2][4] = 0
		matrix_after_first_round[3][4] = -1

		z_matrix_1 = copy.deepcopy(matrix_after_first_round)
		z_matrix_1[1][3] = -1
		z_matrix_2 = copy.deepcopy(matrix_after_first_round)
		z_matrix_2[1][3] = 1
		z_matrix_list = [z_matrix_1, z_matrix_2]
		
		k = 1
		k_prime = 2
		k_prime_prime = 3
		hard_case = cons.HC_1_F

		triplet_xys_1 = {}
		triplet_ysx_1 = {}
		triplet_xsy_1 = {}
		model.update_triplet_hash(triplet_xys_1, 1, 2, 3)
		model.update_triplet_hash(triplet_xys_1, 2, 3, 4)
		model.update_triplet_hash(triplet_ysx_1, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx_1, 3, 4, 2)
		model.update_triplet_hash(triplet_xsy_1, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy_1, 2, 4, 3)
		triplet_xys_2 = copy.deepcopy(triplet_xys_1)
		triplet_ysx_2 = copy.deepcopy(triplet_ysx_1)
		triplet_xsy_2 = copy.deepcopy(triplet_xsy_1)
		triplets_list = [[triplet_xys_1, triplet_ysx_1, triplet_xsy_1], 
			[triplet_xys_2, triplet_ysx_2, triplet_xsy_2]]

		seg_num = 2
		present_ssms_1 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_1[0][cons.A][1] = True
		present_ssms_2 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_2[0][cons.A][1] = True
		present_ssms_2[1][cons.A][1] = True
		present_ssms_list = [present_ssms_1, present_ssms_2]

		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.B] = {}
		CNV_0[cons.LOSS][cons.B][2] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.B] = {}
		CNV_1[cons.LOSS][cons.B][3] = True
		CNVs = [CNV_0, CNV_0]

		# function to test
		model.fork_z_matrix(matrix_after_first_round, z_matrix_list, k, k_prime, k_prime_prime, hard_case,
			triplets_list, present_ssms_list, CNVs)

		self.assertEqual(len(z_matrix_list), 5)
		self.assertEqual(len(triplets_list), 5)
		self.assertEqual(len(present_ssms_list), 5)
		self.assertEqual(z_matrix_list[0][2][3], -1)
		self.assertEqual(z_matrix_list[0][1][3], -1)
		self.assertEqual(z_matrix_list[1][2][3], -1)
		self.assertEqual(z_matrix_list[1][1][3], -1)
		self.assertEqual(z_matrix_list[2][2][3], 1)
		self.assertEqual(z_matrix_list[2][1][2], -1)
		self.assertEqual(z_matrix_list[3][2][3], -1)
		self.assertEqual(z_matrix_list[3][1][3], 1)
		self.assertEqual(z_matrix_list[4][1][2], 0)
		self.assertEqual(len(triplets_list[0][0]), 1)


		# from 2 Z-matrices to 3 Z-matrices, with new phasing
		lin_num = 6
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[0][3] = 1
		matrix_after_first_round[0][4] = 1
		matrix_after_first_round[0][5] = 1
		matrix_after_first_round[1][2] = -1
		matrix_after_first_round[1][3] = 0
		matrix_after_first_round[1][4] = 0
		matrix_after_first_round[1][5] = 0
		matrix_after_first_round[2][3] = -1
		matrix_after_first_round[2][4] = 0
		matrix_after_first_round[2][5] = 0
		matrix_after_first_round[3][4] = 0
		matrix_after_first_round[3][5] = 0
		matrix_after_first_round[4][5] = 0

		z_matrix_1 = copy.deepcopy(matrix_after_first_round)
		z_matrix_1[1][3] = -1
		z_matrix_1[3][4] = -1
		z_matrix_2 = copy.deepcopy(matrix_after_first_round)
		z_matrix_2[1][3] = 1
		z_matrix_2[3][4] = 1
		z_matrix_list = [z_matrix_1, z_matrix_2]
		
		k = 3
		k_prime = 4
		k_prime_prime = 5
		hard_case = cons.HC_2_D

		triplet_xys_1 = {}
		triplet_ysx_1 = {}
		triplet_xsy_1 = {}
		model.update_triplet_hash(triplet_xys_1, 3, 4, 5)
		model.update_triplet_hash(triplet_ysx_1, 4, 5, 3)
		model.update_triplet_hash(triplet_xsy_1, 3, 5, 4)
		triplet_xys_2 = copy.deepcopy(triplet_xys_1)
		triplet_ysx_2 = copy.deepcopy(triplet_ysx_1)
		triplet_xsy_2 = copy.deepcopy(triplet_xsy_1)
		triplets_list = [[triplet_xys_1, triplet_ysx_1, triplet_xsy_1], 
			[triplet_xys_2, triplet_ysx_2, triplet_xsy_2]]

		seg_num = 1
		present_ssms_1 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_1[0][cons.UNPHASED][4] = True
		present_ssms_2 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_2[0][cons.UNPHASED][4] = True
		present_ssms_list = [present_ssms_1, present_ssms_2]

		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.B] = {}
		CNV_0[cons.LOSS][cons.B][5] = True
		CNVs = [CNV_0]

		# function to test
		model.fork_z_matrix(matrix_after_first_round, z_matrix_list, k, k_prime, k_prime_prime, hard_case,
			triplets_list, present_ssms_list, CNVs)

		self.assertEqual(len(z_matrix_list), 4)
		self.assertEqual(len(triplets_list), 4)
		self.assertEqual(len(present_ssms_list), 4)
		self.assertEqual(z_matrix_list[0][4][5], -1)
		self.assertEqual(z_matrix_list[1][3][5], 1)
		self.assertEqual(z_matrix_list[2][4][5], 1)
		self.assertEqual(z_matrix_list[3][3][4], 1)
		self.assertEqual(len(triplets_list[0][0]), 0)
		self.assertEqual(present_ssms_list[0][0][cons.UNPHASED][4], True)
		self.assertEqual(present_ssms_list[1][0][cons.UNPHASED][4], True)
		self.assertEqual(present_ssms_list[2][0][cons.UNPHASED][4], False)
		self.assertEqual(present_ssms_list[2][0][cons.A][4], True)
		self.assertEqual(present_ssms_list[3][0][cons.UNPHASED][4], True)


		# from 2 Z-matrices to 4 matrices
		# first matrix is not forked but stays like it is
		# second matrix is forked to all 4 cases
		#	because one of these matrices equals the first matrix, it is not stored in the list
		lin_num = 5
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[0][3] = 1
		matrix_after_first_round[0][4] = 1
		matrix_after_first_round[1][2] = 0
		matrix_after_first_round[1][3] = 0
		matrix_after_first_round[1][4] = 0
		matrix_after_first_round[2][3] = 0
		matrix_after_first_round[2][4] = 0
		matrix_after_first_round[3][4] = 0

		z_matrix_1 = copy.deepcopy(matrix_after_first_round)
		z_matrix_1[1][2] = 1
		z_matrix_1[1][3] = -1
		z_matrix_1[2][3] = -1
		z_matrix_2 = copy.deepcopy(matrix_after_first_round)
		z_matrix_list = [z_matrix_1, z_matrix_2]
		
		k = 1
		k_prime = 2
		k_prime_prime = 3
		hard_case = cons.HC_2_D

		triplet_xys_1 = {}
		triplet_ysx_1 = {}
		triplet_xsy_1 = {}
		triplet_xys_2 = copy.deepcopy(triplet_xys_1)
		triplet_ysx_2 = copy.deepcopy(triplet_ysx_1)
		triplet_xsy_2 = copy.deepcopy(triplet_xsy_1)
		model.update_triplet_hash(triplet_xys_2, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx_2, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy_2, 1, 3, 2)
		triplets_list = [[triplet_xys_1, triplet_ysx_1, triplet_xsy_1], 
			[triplet_xys_2, triplet_ysx_2, triplet_xsy_2]]

		seg_num = 1
		present_ssms_1 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_1[0][cons.A][1] = True
		present_ssms_2 = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms_2[0][cons.UNPHASED][1] = True
		present_ssms_list = [present_ssms_1, present_ssms_2]

		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.B] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.B][2] = True
		CNV_0[cons.LOSS][cons.A][3] = True
		CNVs = [CNV_0]

		# function to test
		model.fork_z_matrix(matrix_after_first_round, z_matrix_list, k, k_prime, k_prime_prime, hard_case,
			triplets_list, present_ssms_list, CNVs)

		self.assertEqual(len(z_matrix_list), 4)
		self.assertEqual(len(triplets_list), 4)
		self.assertEqual(len(present_ssms_list), 4)
		self.assertEqual(z_matrix_list[0][1][2], 1)
		self.assertEqual(z_matrix_list[0][1][3], -1)
		self.assertEqual(z_matrix_list[0][2][3], -1)
		self.assertEqual(z_matrix_list[1][1][2], -1)
		self.assertEqual(z_matrix_list[1][1][3], -1)
		self.assertEqual(z_matrix_list[1][2][3], -1)
		self.assertEqual(len(triplets_list[0][0]), 0)
		self.assertEqual(len(triplets_list[1][0]), 0)
		self.assertEqual(present_ssms_list[0][0][cons.A][1], True)
		self.assertEqual(present_ssms_list[1][0][cons.UNPHASED][1], True)


	def test_create_new_Z_matrix(self):

		x = -1
		y = -1
		s = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 0
		current_matrix[2][3] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 1
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[0][cons.UNPHASED][2] = True
		present_ssms[0][cons.UNPHASED][3] = True
		CNVs_tmp = {}
		CNVs_tmp[cons.LOSS] = {}
		CNVs_tmp[cons.LOSS][cons.A] = {}
		CNVs_tmp[cons.LOSS][cons.A][1] = True
		CNVs = [CNVs_tmp]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		new_matrix, new_triplet_xys, new_triplet_ysx, new_triplet_xsy, new_present_ssms = (
			model.create_new_Z_matrix(matrix_after_first_round, current_matrix, triplet_xys, triplet_ysx, 
				triplet_xsy, x, y, s, k, k_prime, k_prime_prime, present_ssms, CNVs))

		self.assertEqual(current_matrix[1][2], 0)
		self.assertEqual(new_matrix[1][2], -1)
		self.assertEqual(new_matrix[1][3], 1)
		self.assertEqual(new_matrix[2][3], -1)
		self.assertEqual(triplet_xys.keys(), [1])
		self.assertEqual(new_triplet_xys, {})
		self.assertEqual(new_triplet_ysx, {})
		self.assertEqual(new_triplet_xsy, {})
		self.assertEqual(present_ssms[0][cons.UNPHASED][3], True)
		self.assertEqual(new_present_ssms[0][cons.UNPHASED][3], False)
		self.assertEqual(new_present_ssms[0][cons.B][3], True)

	def test_update_new_Z_matrix(self):

		# value 0, no update necessary
		value = 0
		k = 1
		k_prime = 2
		k_prime_prime = 3
		changed_field = cons.X
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1

		self.assertTrue(model.update_new_Z_matrix(value, changed_field, k, k_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=None,
			present_ssms=None, CNVs=None, triplet_xys=None, triplet_ysx=None, triplet_xsy=None))

		# value already set, no update necessary
		value = -1
		i = 2
		i_prime = 3
		k = 1
		k_prime = 2
		k_prime_prime = 3
		changed_field = cons.X
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[i][i_prime] = -1

		self.assertTrue(model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=None,
			present_ssms=None, CNVs=None, triplet_xys=None, triplet_ysx=None, triplet_xsy=None))

		# update not possible because entry is different
		value = -1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.X
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[i][i_prime] = 1

		with self.assertRaises(eo.ZUpdateNotPossible):
			model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
				current_matrix, matrix_after_first_round=None,
				present_ssms=None, CNVs=None, triplet_xys=None, triplet_ysx=None, triplet_xsy=None)

		# phasing not possible
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 2
		i_prime = 3
		changed_field = cons.X
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[k][k_prime] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.A][i] = True
		CNV_tmp = {}
		CNV_tmp[cons.LOSS] = {}
		CNV_tmp[cons.LOSS][cons.A] = {}
		CNV_tmp[cons.LOSS][cons.A][i_prime] = True
		CNVs = [CNV_tmp]

		with self.assertRaises(eo.ZUpdateNotPossible):
			model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime, 
				current_matrix, matrix_after_first_round=matrix_after_first_round,
				present_ssms=present_ssms, CNVs=CNVs, triplet_xys=None, triplet_ysx=None, triplet_xsy=None)

		# update not possible
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = -1
		current_matrix[2][3] = 1
		current_matrix[1][3] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		CNV_tmp = {}
		CNVs = [CNV_tmp]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		with self.assertRaises(eo.ZUpdateNotPossible):
			model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime, 
				current_matrix, matrix_after_first_round=matrix_after_first_round,
				present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, 
				triplet_xsy=triplet_xsy)

		# phasing and update possible
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 1
		current_matrix[2][3] = 1
		current_matrix[1][3] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][i] = True
		CNV_tmp = {}
		CNV_tmp[cons.LOSS] = {}
		CNV_tmp[cons.LOSS][cons.A] = {}
		CNV_tmp[cons.LOSS][cons.A][i_prime] = True
		CNVs = [CNV_tmp]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][3], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][i], False)
		self.assertEqual(present_ssms[0][cons.B][i], True)

		# phasing not necessary and update possible because of no a.-d. relation
		value = -1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 4
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[2][3] = 0
		current_matrix[1][3] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][i] = True
		CNV_tmp = {}
		CNV_tmp[cons.LOSS] = {}
		CNV_tmp[cons.LOSS][cons.A] = {}
		CNV_tmp[cons.LOSS][cons.A][i_prime] = True
		CNVs = [CNV_tmp]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][3], -1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][i], True)

		# phasing in current lineages, and update which leads to phasing in other lineages as well
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 1
		current_matrix[1][3] = 0
		current_matrix[1][4] = 0
		current_matrix[2][3] = 1
		current_matrix[2][4] = 1
		current_matrix[3][4] = 1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.UNPHASED][4] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 4, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 3, 4)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][3], 1)
		self.assertEqual(current_matrix[1][4], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], False)
		self.assertEqual(present_ssms[1][cons.UNPHASED][3], False)
		self.assertEqual(present_ssms[2][cons.UNPHASED][4], False) 
		self.assertEqual(present_ssms[0][cons.B][1], True)
		self.assertEqual(present_ssms[1][cons.B][3], True)
		self.assertEqual(present_ssms[2][cons.B][4], True)
		
		# phasing in current lineages, and update which leads to phasing conflict
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 1
		current_matrix[1][3] = 0
		current_matrix[1][4] = 0
		current_matrix[2][3] = 1
		current_matrix[2][4] = 1
		current_matrix[3][4] = 1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.A][4] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 4, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 3, 4)

		with self.assertRaises(eo.ZUpdateNotPossible):
			model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
				current_matrix, matrix_after_first_round=matrix_after_first_round,
				present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
				triplet_xsy=triplet_xsy)

		# value leads to transitivity update, which leads to further phasing
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 0
		current_matrix[1][4] = -1
		current_matrix[2][3] = 1
		current_matrix[2][4] = -1
		current_matrix[3][4] = -1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.UNPHASED][2] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][3], 1)
		self.assertEqual(current_matrix[1][2], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], False)
		self.assertEqual(present_ssms[1][cons.UNPHASED][3], False)
		self.assertEqual(present_ssms[2][cons.UNPHASED][2], False) 
		self.assertEqual(present_ssms[0][cons.B][1], True)
		self.assertEqual(present_ssms[1][cons.B][3], True)
		self.assertEqual(present_ssms[2][cons.B][2], True)

		# value leads to transitivity update, which leads to phasing conflict
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 0
		current_matrix[1][4] = -1
		current_matrix[2][3] = 1
		current_matrix[2][4] = -1
		current_matrix[3][4] = -1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.A][2] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)

		with self.assertRaises(eo.ZUpdateNotPossible):
			model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
				current_matrix, matrix_after_first_round=matrix_after_first_round,
				present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
				triplet_xsy=triplet_xsy)

		# changed field is X, tripet update in S, transitivity update in X
		value = 1
		k = 1
		k_prime = 3
		k_prime_prime = 4
		i = 1
		i_prime = 3
		changed_field = cons.X
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 0
		current_matrix[1][4] = 0
		current_matrix[2][3] = 1
		current_matrix[2][4] = 1
		current_matrix[3][4] = 1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 4
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.UNPHASED][4] = True
		present_ssms[3][cons.UNPHASED][2] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNV_3 = {}
		CNV_3[cons.LOSS] = {}
		CNV_3[cons.LOSS][cons.A] = {}
		CNV_3[cons.LOSS][cons.A][1] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 4, 3)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][2], 1)
		self.assertEqual(current_matrix[1][3], 1)
		self.assertEqual(current_matrix[1][4], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], False)
		self.assertEqual(present_ssms[1][cons.UNPHASED][3], False)
		self.assertEqual(present_ssms[2][cons.UNPHASED][4], False) 
		self.assertEqual(present_ssms[3][cons.UNPHASED][2], False) 
		self.assertEqual(present_ssms[0][cons.B][1], True)
		self.assertEqual(present_ssms[1][cons.B][3], True)
		self.assertEqual(present_ssms[2][cons.B][4], True)
		self.assertEqual(present_ssms[3][cons.B][2], True)

		# changed field is Y, tripet update in S, transitivity update in Y
		value = 1
		k = 1
		k_prime = 3
		k_prime_prime = 4
		i = 3
		i_prime = 4
		changed_field = cons.Y
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 1
		current_matrix[1][4] = 0
		current_matrix[2][3] = -1
		current_matrix[2][4] = 0
		current_matrix[3][4] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 4
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][3] = True
		present_ssms[1][cons.UNPHASED][4] = True
		present_ssms[2][cons.UNPHASED][2] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][4] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][3] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][4] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_xys, 2, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 2)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 4, 3)
		model.update_triplet_hash(triplet_xsy, 2, 4, 3)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][2], 0)
		self.assertEqual(current_matrix[1][4], 1)
		self.assertEqual(current_matrix[2][4], -1)
		self.assertEqual(current_matrix[3][4], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][3], False)
		self.assertEqual(present_ssms[1][cons.UNPHASED][4], False)
		self.assertEqual(present_ssms[2][cons.UNPHASED][2], True) 
		self.assertEqual(present_ssms[0][cons.B][3], True)
		self.assertEqual(present_ssms[1][cons.B][4], True)

		# changed field is S, tripet update in X, transitivity update in S
		value = 1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 0
		current_matrix[1][3] = 0
		current_matrix[1][4] = 0
		current_matrix[2][3] = 1
		current_matrix[2][4] = 1
		current_matrix[3][4] = 1
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 4
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.UNPHASED][2] = True
		present_ssms[3][cons.UNPHASED][1] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][1] = True
		CNV_3 = {}
		CNV_3[cons.LOSS] = {}
		CNV_3[cons.LOSS][cons.A] = {}
		CNV_3[cons.LOSS][cons.A][4] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 4, 3)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][2], 1)
		self.assertEqual(current_matrix[1][3], 1)
		self.assertEqual(current_matrix[1][4], 1)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], False)
		self.assertEqual(present_ssms[1][cons.UNPHASED][3], False)
		self.assertEqual(present_ssms[2][cons.UNPHASED][2], False) 
		self.assertEqual(present_ssms[3][cons.UNPHASED][1], False) 
		self.assertEqual(present_ssms[0][cons.B][1], True)
		self.assertEqual(present_ssms[1][cons.B][3], True)
		self.assertEqual(present_ssms[2][cons.B][2], True)
		self.assertEqual(present_ssms[3][cons.B][1], True)

		# changed field is S with -1, tripet update in Y
		value = -1
		k = 1
		k_prime = 2
		k_prime_prime = 3
		i = 1
		i_prime = 3
		changed_field = cons.S
		lin_num = 5
		current_matrix = [[-1] * lin_num for _ in xrange(lin_num)]
		current_matrix[0][1] = 1
		current_matrix[0][2] = 1
		current_matrix[0][3] = 1
		current_matrix[1][2] = 1
		current_matrix[1][3] = 0
		current_matrix[1][4] = 0
		current_matrix[2][3] = 0
		current_matrix[2][4] = 0
		current_matrix[3][4] = 0
		matrix_after_first_round = copy.deepcopy(current_matrix)
		seg_num = 4
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][1] = True
		present_ssms[1][cons.UNPHASED][3] = True
		present_ssms[2][cons.UNPHASED][2] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][3] = True
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.A] = {}
		CNV_1[cons.LOSS][cons.A][1] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][4] = True
		CNVs = [CNV_0, CNV_1, CNV_2]
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_xys, 1, 3, 4)
		model.update_triplet_hash(triplet_xys, 2, 3, 4)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 1)
		model.update_triplet_hash(triplet_ysx, 3, 4, 2)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		model.update_triplet_hash(triplet_xsy, 1, 4, 3)
		model.update_triplet_hash(triplet_xsy, 2, 4, 3)

		model.update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime,
			current_matrix, matrix_after_first_round=matrix_after_first_round,
			present_ssms=present_ssms, CNVs=CNVs, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx,
			triplet_xsy=triplet_xsy)
		self.assertEqual(current_matrix[1][3], -1)
		self.assertEqual(current_matrix[2][3], -1)
		self.assertEqual(current_matrix[2][4], 0)
		self.assertEqual(current_matrix[1][4], 0)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], True)
		self.assertEqual(present_ssms[1][cons.UNPHASED][3], True)
		self.assertEqual(present_ssms[2][cons.UNPHASED][2], True) 

	def test_phasing_allows_relation(self):

		# ancestor-descendant relation between lineages given in matrix after first round
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 1
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		self.assertTrue(model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1))

		# ancestor-descendant relation between lineages between lineages is possible
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 0
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][k] = True
		present_ssms[1][cons.UNPHASED][k] = True
		present_ssms[2][cons.UNPHASED][k] = True
		present_ssms[3][cons.UNPHASED][k] = True
		present_ssms[0][cons.UNPHASED][k_prime] = True
		present_ssms[1][cons.UNPHASED][k_prime] = True
		present_ssms[2][cons.UNPHASED][k_prime] = True
		present_ssms[3][cons.UNPHASED][k_prime] = True
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		self.assertTrue(model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1))

		# ancestor-descendant relation between lineages not possible because of SSMs in k and
		# CNVs in k_prime in  A
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 0
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.A][k] = True
		present_ssms[1][cons.UNPHASED][k] = True
		present_ssms[2][cons.UNPHASED][k] = True
		present_ssms[3][cons.UNPHASED][k] = True
		present_ssms[0][cons.UNPHASED][k_prime] = True
		present_ssms[1][cons.UNPHASED][k_prime] = True
		present_ssms[2][cons.UNPHASED][k_prime] = True
		present_ssms[3][cons.UNPHASED][k_prime] = True
		CNV_0 = {}
		CNV_0[cons.LOSS] = {}
		CNV_0[cons.LOSS][cons.A] = {}
		CNV_0[cons.LOSS][cons.A][k_prime] = True
		CNV_1 = {}
		CNV_2 = {}
		CNV_3 = {}
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

		# ancestor-descendant relation between lineages not possible because of SSMs in k and
		# CNVs in k_prime in  B
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 0
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][k] = True
		present_ssms[1][cons.B][k] = True
		present_ssms[2][cons.UNPHASED][k] = True
		present_ssms[3][cons.UNPHASED][k] = True
		present_ssms[0][cons.UNPHASED][k_prime] = True
		present_ssms[1][cons.UNPHASED][k_prime] = True
		present_ssms[2][cons.UNPHASED][k_prime] = True
		present_ssms[3][cons.UNPHASED][k_prime] = True
		CNV_0 = {}
		CNV_1 = {}
		CNV_1[cons.LOSS] = {}
		CNV_1[cons.LOSS][cons.B] = {}
		CNV_1[cons.LOSS][cons.B][k_prime] = True
		CNV_2 = {}
		CNV_3 = {}
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

		# ancestor-descendant relation between lineages not possible because of SSMs in k_prime and
		# CNVs in k in  A
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 0
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][k] = True
		present_ssms[1][cons.UNPHASED][k] = True
		present_ssms[2][cons.UNPHASED][k] = True
		present_ssms[3][cons.UNPHASED][k] = True
		present_ssms[0][cons.UNPHASED][k_prime] = True
		present_ssms[1][cons.UNPHASED][k_prime] = True
		present_ssms[2][cons.A][k_prime] = True
		present_ssms[3][cons.UNPHASED][k_prime] = True
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][k] = True
		CNV_3 = {}
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

		# ancestor-descendant relation between lineages not possible because of SSMs in k_prime and
		# CNVs in k in  B
		k = 1
		k_prime = 2
		lin_num = 3
		seg_num = 4
		matrix_after_first_round = [[-1] * lin_num for _ in xrange(lin_num)]
		matrix_after_first_round[0][1] = 1
		matrix_after_first_round[0][2] = 1
		matrix_after_first_round[k][k_prime] = 0
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][k] = True
		present_ssms[1][cons.UNPHASED][k] = True
		present_ssms[2][cons.UNPHASED][k] = True
		present_ssms[3][cons.UNPHASED][k] = True
		present_ssms[0][cons.UNPHASED][k_prime] = True
		present_ssms[1][cons.UNPHASED][k_prime] = True
		present_ssms[2][cons.UNPHASED][k_prime] = True
		present_ssms[3][cons.B][k_prime] = True
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_3 = {}
		CNV_3[cons.LOSS] = {}
		CNV_3[cons.LOSS][cons.B] = {}
		CNV_3[cons.LOSS][cons.B][k] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

	def test_phasing_allows_relation_per_allele_lineage(self):

		# no phased SSMs
		lineage_ssm = 1
		lineage_cnv = 2
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		self.assertTrue(model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
			present_ssms, CNVs, phase, seg_index))

		# phased to other allele
		lineage_ssm = 1
		lineage_cnv = 2
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][model.other_phase(phase)][lineage_ssm] = True
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		self.assertTrue(model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
			present_ssms, CNVs, phase, seg_index))

		# no CN change in same phase as SSMs but in other
		lineage_ssm = 1
		lineage_cnv = 2
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][phase][lineage_ssm] = True
		CNVs = []
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_tmp = {}
		CNV_tmp[cons.GAIN] = {}
		CNV_tmp[cons.GAIN][model.other_phase(phase)] = {}
		CNV_tmp[cons.GAIN][model.other_phase(phase)][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

		self.assertTrue(model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
			present_ssms, CNVs, phase, seg_index))

		# lineage_ssm < lineage_cnv, with gain
		lineage_ssm = 1
		lineage_cnv = 2
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][phase][lineage_ssm] = True
		CNVs = []
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_tmp = {}
		CNV_tmp[cons.GAIN] = {}
		CNV_tmp[cons.GAIN][phase] = {}
		CNV_tmp[cons.GAIN][phase][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
				present_ssms, CNVs, phase, seg_index)

		# lineage_ssm < lineage_cnv, with loss
		lineage_ssm = 1
		lineage_cnv = 2
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][phase][lineage_ssm] = True
		CNVs = []
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_tmp = {}
		CNV_tmp[cons.LOSS] = {}
		CNV_tmp[cons.LOSS][phase] = {}
		CNV_tmp[cons.LOSS][phase][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
				present_ssms, CNVs, phase, seg_index)

		# lineage_ssm > lineage_cnv, with loss
		lineage_ssm = 2
		lineage_cnv = 1
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][phase][lineage_ssm] = True
		CNVs = []
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_tmp = {}
		CNV_tmp[cons.LOSS] = {}
		CNV_tmp[cons.LOSS][phase] = {}
		CNV_tmp[cons.LOSS][phase][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

		with self.assertRaises(eo.ADRelationNotPossible):
			model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
				present_ssms, CNVs, phase, seg_index)

		# lineage_ssm > lineage_cnv, with gain, ok
		lineage_ssm = 2
		lineage_cnv = 1
		lin_num = 4
		seg_num = 5
		seg_index = 3
		phase = cons.A
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][phase][lineage_ssm] = True
		CNVs = []
		CNV_0 = {}
		CNV_1 = {}
		CNV_2 = {}
		CNV_tmp = {}
		CNV_tmp[cons.GAIN] = {}
		CNV_tmp[cons.GAIN][phase] = {}
		CNV_tmp[cons.GAIN][phase][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

		self.assertTrue(model.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
			present_ssms, CNVs, phase, seg_index))

	def test_unphased_checking(self):

		# no unphased SSMs
		lineage_ssm = 2
		lineage_cnv = 3
		lin_num = 4
		seg_num = 5
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		model.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		self.assertEqual(my_present_ssms, present_ssms)

		# only one segment with unphased SSMs and no CNVs in this segment
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		seg_index = 0
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		model.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		self.assertEqual(my_present_ssms, present_ssms)

		# four segments with unphased SSMs
		# influence of gain A, gain B, loss A and loss B
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[0][cons.UNPHASED][lineage_ssm] = True
		present_ssms[1][cons.UNPHASED][lineage_ssm] = True
		present_ssms[2][cons.UNPHASED][lineage_ssm] = True
		present_ssms[3][cons.UNPHASED][lineage_ssm] = True
		CNV_0 = {}
		CNV_0[cons.GAIN] = {}
		CNV_0[cons.GAIN][cons.A] = {}
		CNV_0[cons.GAIN][cons.A][lineage_cnv] = True
		CNV_1 = {}
		CNV_1[cons.GAIN] = {}
		CNV_1[cons.GAIN][cons.B] = {}
		CNV_1[cons.GAIN][cons.B][lineage_cnv] = True
		CNV_2 = {}
		CNV_2[cons.LOSS] = {}
		CNV_2[cons.LOSS][cons.A] = {}
		CNV_2[cons.LOSS][cons.A][lineage_cnv] = True
		CNV_3 = {}
		CNV_3[cons.LOSS] = {}
		CNV_3[cons.LOSS][cons.B] = {}
		CNV_3[cons.LOSS][cons.B][lineage_cnv] = True
		CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

		model.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[0][cons.B][lineage_ssm] = True
		my_present_ssms[1][cons.A][lineage_ssm] = True
		my_present_ssms[2][cons.B][lineage_ssm] = True
		my_present_ssms[3][cons.A][lineage_ssm] = True
		self.assertEqual(my_present_ssms, present_ssms)

	def test_get_CNVs_of_lineage(self):
		lineage_cnv = 2
		lineage_ssm = 3
		seg_index = 0

		# no CNVs
		CNVs = []
		CNV_tmp = {}
		CNVs.append(CNV_tmp)

		with self.assertRaises(eo.no_CNVs):
			model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

		# one CN gain on A
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.GAIN
		phase = cons.A
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		loss_a, loss_b, gain_a, gain_b = model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		self.assertEqual((False, False, True, False), (loss_a, loss_b, gain_a, gain_b))

		# one CN gain on B
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.GAIN
		phase = cons.B
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		loss_a, loss_b, gain_a, gain_b = model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		self.assertEqual((False, False, False, True), (loss_a, loss_b, gain_a, gain_b))

		# one CN loss on A
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.LOSS
		phase = cons.A
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		loss_a, loss_b, gain_a, gain_b = model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		self.assertEqual((True, False, False, False), (loss_a, loss_b, gain_a, gain_b))

		# one CN loss on B
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.LOSS
		phase = cons.B
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		loss_a, loss_b, gain_a, gain_b = model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		self.assertEqual((False, True, False, False), (loss_a, loss_b, gain_a, gain_b))
	
		# two CN gains
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.GAIN
		phase = cons.A
		lineage_cnv = 3
		lineage_ssm = 2
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNV_tmp[mutation_type][model.other_phase(phase)] = {}
		CNV_tmp[mutation_type][model.other_phase(phase)][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		with self.assertRaises(eo.MyException):
			model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

		# two CN losses
		lineage_cnv = 2
		lineage_ssm = 3
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.LOSS
		phase = cons.A
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNV_tmp[mutation_type][model.other_phase(phase)] = {}
		CNV_tmp[mutation_type][model.other_phase(phase)][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		with self.assertRaises(eo.MyException):
			model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

		# loss and gain on A and B
		CNVs = []
		CNV_tmp = {}
		lineage_cnv = 3
		lineage_ssm = 2
		mutation_type = cons.GAIN
		mutation_type_2 = cons.LOSS
		phase = cons.A
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNV_tmp[mutation_type_2] = {}
		CNV_tmp[mutation_type_2][model.other_phase(phase)] = {}
		CNV_tmp[mutation_type_2][model.other_phase(phase)][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		with self.assertRaises(eo.MyException):
			model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

		# loss and gain on B and A
		lineage_cnv = 3
		lineage_ssm = 2
		CNVs = []
		CNV_tmp = {}
		mutation_type = cons.GAIN
		mutation_type_2 = cons.LOSS
		phase = cons.B
		CNV_tmp[mutation_type] = {}
		CNV_tmp[mutation_type][phase] = {}
		CNV_tmp[mutation_type][phase][lineage_cnv] = True
		CNV_tmp[mutation_type_2] = {}
		CNV_tmp[mutation_type_2][model.other_phase(phase)] = {}
		CNV_tmp[mutation_type_2][model.other_phase(phase)][lineage_cnv] = True
		CNVs.append(CNV_tmp)

		with self.assertRaises(eo.MyException):
			model.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

	def test_has_CNV_in_phase(self):
		seg_index = 0
		mutation_type = cons.GAIN
		phase = cons.A
		CNVs = []
		CNVs_tmp = {}
		CNVs_tmp[mutation_type] = {}
		CNVs_tmp[mutation_type][phase] = {}
		CNVs_tmp[mutation_type][phase][2] = True
		CNVs_tmp[mutation_type][phase][4] = True
		CNVs.append(CNVs_tmp)
	
		# lineage index contained
		lineage_cnv = 2
		self.assertTrue(model.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

		# lineage index not contained
		lineage_cnv = 3
		self.assertFalse(model.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

		# wrong phase
		phase = cons.B
		self.assertFalse(model.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

		# wrong mutation type
		mutation_type = cons.LOSS
		self.assertFalse(model.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

	def test_cn_change_influences_ssms(self):

		# lineage_ssm < lineage_cnv, mutation = cons.LOSS
		# there is change!
		lin_num = 4
		seg_num = 6
		lineage_ssm = 2
		lineage_cnv = 3
		seg_index = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		phase = cons.A
		mutation = cons.LOSS

		model.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][model.other_phase(phase)][lineage_ssm] = True
		self.assertEqual(present_ssms, my_present_ssms)
		
		# lineage_ssm < lineage_cnv, mutation = cons.GAIN
		# there is change!
		lin_num = 4
		seg_num = 6
		lineage_ssm = 2
		lineage_cnv = 3
		seg_index = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		phase = cons.A
		mutation = cons.GAIN

		model.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][model.other_phase(phase)][lineage_ssm] = True
		self.assertEqual(present_ssms, my_present_ssms)

		# lineage_ssm > lineage_cnv, mutation = cons.LOSS
		# there is change!
		lin_num = 4
		seg_num = 6
		lineage_ssm = 3
		lineage_cnv = 2
		seg_index = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		phase = cons.A
		mutation = cons.LOSS

		model.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][model.other_phase(phase)][lineage_ssm] = True
		self.assertEqual(present_ssms, my_present_ssms)

		# lineage_ssm > lineage_cnv, mutation = cons.GAIN
		# there is change!
		lin_num = 4
		seg_num = 6
		lineage_ssm = 3
		lineage_cnv = 2
		seg_index = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		phase = cons.A
		mutation = cons.GAIN

		model.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
		self.assertEqual(present_ssms, my_present_ssms)

	def test_move_unphased_SSMs(self):

		lin_num = 4
		seg_num = 6
		current_lin = 2
		seg_index = 3
		present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		present_ssms[seg_index][cons.UNPHASED][current_lin] = True
		phase = cons.A

		model.move_unphased_SSMs(present_ssms, seg_index, current_lin, phase)
		my_present_ssms = [[[False] * lin_num for _ in xrange(3)] for _ in xrange(seg_num)]
		my_present_ssms[seg_index][phase][current_lin] = True
		self.assertEqual(present_ssms, my_present_ssms)

		with self.assertRaises(eo.MyException):
			present_ssms[seg_index][model.other_phase(phase)][current_lin] = True
			model.move_unphased_SSMs(present_ssms, seg_index, current_lin, phase)
		

	def test_get_CN_changes_SSM_apperance(self):
		# 4 lineages, 3 segments
		# lin1 only has unphased SSM in seg0
		# lin2 only one loss in seg1, lin3 also one in seg1
		# seg2: CN gains in all lineages
		lin0 = lineage.Lineage([1, 2, 3], 1.0, None, None, None, None, None, None, None, None)
		ssm1 = snp_ssm.SNP_SSM()
		ssm1.pos = 1
		ssm1.seg_index = 0
		ssm11 = snp_ssm.SNP_SSM()
		ssm11.pos = 11
		ssm11.seg_index = 1
		ssm11.infl_cnv_same_lin = True
		cnv12 = cnv.CNV(1, 2, 1, 21, 30)
		cnv122 = cnv.CNV(1, 2, 1, 21, 30)
		lin1 = lineage.Lineage([2, 3], 1.0, [cnv12], [cnv122], None, None, None, [ssm1], [ssm11], None)
		cnv2 = cnv.CNV(-1, 1, 1, 11, 20)
		ssm21 = snp_ssm.SNP_SSM()
		ssm21.pos = 12
		ssm21.seg_index = 1
		cnv22 = cnv.CNV(1, 2, 1, 21, 30)
		lin2 = lineage.Lineage([3], 1.0, None, [cnv2, cnv22], None, None, None, None, None, [ssm21])
		cnv3 = cnv.CNV(-1, 1, 1, 11, 20)
		cnv32 = cnv.CNV(1, 2, 1, 21, 30)
		cnv322 = cnv.CNV(1, 2, 1, 21, 30)
		lin3 = lineage.Lineage([], 1.0, [cnv3, cnv32], [cnv322], None, None, None, None, None, None)

		seg_num = 3
		gain_num = []
		loss_num = []
		CNVs = [] 
		present_ssms = [] 
		ssm_infl_cnv_same_lineage = []
		my_lineages = [lin0, lin1, lin2, lin3]
		lineage_num = len(my_lineages)

		model.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)

		self.assertEqual(gain_num, [0, 0, 5])
		self.assertEqual(loss_num, [0, 2, 0])
		self.assertEqual(present_ssms[0], [[False, False, False, False], [False, False, False, False], 
			[False, True, False, False]])
		self.assertEqual(present_ssms[1], [[False, True, False, False], [False, False, True, False], 
			[False, False, False, False]])
		self.assertEqual(present_ssms[2], [[False, False, False, False], [False, False, False, False],
			[False, False, False, False]])
		self.assertEqual(ssm_infl_cnv_same_lineage[0], [[False, False, False, False], [False, False, False, False]])
		self.assertEqual(ssm_infl_cnv_same_lineage[1], [[False, True, False, False], [False, False, False, False]])
		self.assertEqual(ssm_infl_cnv_same_lineage[2], [[False, False, False, False], [False, False, False, False]])
		self.assertEqual(len(CNVs[0].keys()), 0)
		self.assertEqual(len(CNVs[1][cons.LOSS].keys()), 2)
		self.assertEqual(len(CNVs[2][cons.GAIN].keys()), 2)
		self.assertEqual(sorted(CNVs[2][cons.GAIN][cons.A].keys()), [1, 3])

	def test_post_analysis_Z_matrix(self):
		# example where 1c is not applied because SSMs lie on different segment then CNVs
		# 	1f is used
		#	2f is used
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7, 8], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2, 3], 0.5, None, None, None, None, None, None, None, None)
		cnv2 = cnv.CNV(-1, 0, 1, 1, 10)
		lin2 = lineage.Lineage([], 0.25, None, [cnv2], None, None, None, None, None, None)
		ssm3 = snp_ssm.SNP_SSM()
		ssm3.pos = 11
		ssm3.seg_index = 1
		lin3 = lineage.Lineage([], 0.24, None, None, None, None, None, None, None, [ssm3])
		cnv4 = cnv.CNV(-1, 2, 1, 21, 30)
		lin4 = lineage.Lineage([5], 0.2, None, [cnv4], None, None, None, None, None, None)
		cnv5 = cnv.CNV(-1, 2, 1, 21, 30)
		lin5 = lineage.Lineage([], 0.19, [cnv5], None, None, None, None, None, None, None)
		ssm6 = snp_ssm.SNP_SSM()
		ssm6.pos = 21
		ssm6.seg_index = 2
		lin6 = lineage.Lineage([], 0.1, None, None, None, None, None, [ssm6], None, None)
		ssm7 = snp_ssm.SNP_SSM()
		ssm7.pos = 31
		ssm7.seg_index = 3
		lin7 = lineage.Lineage([], 0.05, None, None, None, None, None, [ssm7], None, None)
		cnv8_A = cnv.CNV(1, 3, 1, 31, 40)
		cnv8_B = cnv.CNV(1, 3, 1, 31, 40)
		lin8 = lineage.Lineage([], 0.04, [cnv8_A], [cnv8_B], None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7, lin8]
		seg_num = 4

		my_z_matrix = [
			[-1, 1, 1, 1, 1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 0, 0, 0, 0, 0],
			[-1, -1, -1, 0, 0, 0, 0, 0, 0],
			[-1, -1, -1, -1, 0, 0, 0, 0, 0],
			[-1, -1, -1, -1, -1, 1, 0, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, -1, -1, -1, -1]
			]
		
		z_matrix = model.get_Z_matrix(my_lins)[0]
		zero_count = model.get_0_number_in_z_matrix(z_matrix)
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		z_matrix_list, z_matrix_fst_rnd,  origin_present_ssms, present_ssms_list, = (
			model.post_analysis_Z_matrix(
			my_lins, seg_num, z_matrix, 
			zero_count, triplet_xys, triplet_ysx, triplet_xsy))
			
		self.assertEqual(z_matrix, my_z_matrix)
		
		# same example, only lineage frequencies are now too high
		# example where 1c is not applied because SSMs lie on different segment then CNVs
		# 	1f is used
		#	2f is used
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7, 8], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2, 3], 0.5, None, None, None, None, None, None, None, None)
		cnv2 = cnv.CNV(-1, 0, 1, 1, 10)
		lin2 = lineage.Lineage([], 0.25, None, [cnv2], None, None, None, None, None, None)
		ssm3 = snp_ssm.SNP_SSM()
		ssm3.pos = 11
		ssm3.seg_index = 1
		lin3 = lineage.Lineage([], 0.24, None, None, None, None, None, None, None, [ssm3])
		cnv4 = cnv.CNV(-1, 2, 1, 21, 30)
		lin4 = lineage.Lineage([5], 0.2, None, [cnv4], None, None, None, None, None, None)
		cnv5 = cnv.CNV(-1, 2, 1, 21, 30)
		lin5 = lineage.Lineage([], 0.19, [cnv5], None, None, None, None, None, None, None)
		ssm6 = snp_ssm.SNP_SSM()
		ssm6.pos = 21
		ssm6.seg_index = 2
		lin6 = lineage.Lineage([], 0.19, None, None, None, None, None, [ssm6], None, None)
		ssm7 = snp_ssm.SNP_SSM()
		ssm7.pos = 31
		ssm7.seg_index = 3
		lin7 = lineage.Lineage([], 0.19, None, None, None, None, None, [ssm7], None, None)
		cnv8_A = cnv.CNV(1, 3, 1, 31, 40)
		cnv8_B = cnv.CNV(1, 3, 1, 31, 40)
		lin8 = lineage.Lineage([], 0.19, [cnv8_A], [cnv8_B], None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7, lin8]
		seg_num = 4

		z_matrix = model.get_Z_matrix(my_lins)[0]
		zero_count = model.get_0_number_in_z_matrix(z_matrix)
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, = (
			model.post_analysis_Z_matrix(
			my_lins, seg_num, z_matrix, 
			zero_count, triplet_xys, triplet_ysx, triplet_xsy))


		# same example, only this time 4 is not a parent of 5, thus case 1f can't be decided
		# example where 1c is not applied because SSMs lie on different segment then CNVs
		# 	1f is used, but can't be decided in first round, second needed
		#	2f is used
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7, 8], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2, 3], 0.5, None, None, None, None, None, None, None, None)
		cnv2 = cnv.CNV(-1, 0, 1, 1, 10)
		lin2 = lineage.Lineage([], 0.25, None, [cnv2], None, None, None, None, None, None)
		ssm3 = snp_ssm.SNP_SSM()
		ssm3.pos = 11
		ssm3.seg_index = 1
		lin3 = lineage.Lineage([], 0.24, None, None, None, None, None, None, None, [ssm3])
		cnv4 = cnv.CNV(-1, 2, 1, 21, 30)
		lin4 = lineage.Lineage([], 0.2, None, [cnv4], None, None, None, None, None, None)
		cnv5 = cnv.CNV(-1, 2, 1, 21, 30)
		lin5 = lineage.Lineage([], 0.19, [cnv5], None, None, None, None, None, None, None)
		ssm6 = snp_ssm.SNP_SSM()
		ssm6.pos = 21
		ssm6.seg_index = 2
		lin6 = lineage.Lineage([], 0.1, None, None, None, None, None, [ssm6], None, None)
		ssm7 = snp_ssm.SNP_SSM()
		ssm7.pos = 31
		ssm7.seg_index = 3
		lin7 = lineage.Lineage([], 0.05, None, None, None, None, None, [ssm7], None, None)
		cnv8_A = cnv.CNV(1, 3, 1, 31, 40)
		cnv8_B = cnv.CNV(1, 3, 1, 31, 40)
		lin8 = lineage.Lineage([], 0.04, [cnv8_A], [cnv8_B], None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7, lin8]
		seg_num = 4

		z_matrix = model.get_Z_matrix(my_lins)[0]
		zero_count = model.get_0_number_in_z_matrix(z_matrix)
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, = (
			model.post_analysis_Z_matrix(
			my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy))

		self.assertEqual(len(z_matrix_list), 4)

		# other example that shows that drawing a ancestor-descendat relation because of one
		# segment can influence the phasing in other segments and thus the forking of
		# other segments
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, None, None, None, None, None, None, None, None)
		ssm1_1 = snp_ssm.SNP_SSM()
		ssm1_1.pos = 1
		ssm1_1.seg_index = 0
		ssm1_2 = snp_ssm.SNP_SSM()
		ssm1_2.pos = 10
		ssm1_2.seg_index = 1
		lin1 = lineage.Lineage([], 0.2, None, None, None, None, None, [ssm1_1, ssm1_2], None, None)
		cnv2 = cnv.CNV(-1, 1, 1, 10, 19)
		lin2 = lineage.Lineage([], 0.19, [cnv2], None, None, None, None, None, None, None)
		cnv3 = cnv.CNV(-1, 1, 1, 10, 19)
		lin3 = lineage.Lineage([], 0.18, None, [cnv3], None, None, None, None, None, None)
		cnv4 = cnv.CNV(-1, 0, 1, 1, 9)
		lin4 = lineage.Lineage([], 0.17, [cnv4], None, None, None, None, None, None, None)
		cnv5_1 = cnv.CNV(-1, 0, 1, 1, 9)
		cnv5_2 = cnv.CNV(-1, 1, 1, 10, 19)
		lin5 = lineage.Lineage([], 0.16, None, [cnv5_1, cnv5_2], None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4, lin5]
		seg_num = 2

		z_matrix = model.get_Z_matrix(my_lins)[0]
		zero_count = model.get_0_number_in_z_matrix(z_matrix)
		lineage_num = len(my_lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num)
		z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, = (
			model.post_analysis_Z_matrix(
			my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy))

		self.assertEqual(len(z_matrix_list), 24)

		# Test currently not used because all hard cases possible from matrix after the first round
		#	are getting checked in detail
		## example that shows that even though invalid hard cases might be possible because
		## entries of a hard case are set through other hard cases, the phasing constraints
		## prevent this from happening
		#lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, None, None, None, None, None, None, None, None)
		#ssm1_1 = snp_ssm.SNP_SSM()
		#ssm1_1.pos = 1
		#ssm1_1.seg_index = 0
		#lin1 = lineage.Lineage([], 0.2, None, None, None, None, None, [ssm1_1], None, None)
		#cnv2 = cnv.CNV(-1, 0, 1, 1, 9)
		#ssm2_1 = snp_ssm.SNP_SSM()
		#ssm2_1.pos = 21
		#ssm2_1.seg_index = 2
		#ssm2_2 = snp_ssm.SNP_SSM()
		#ssm2_2.pos = 31
		#ssm2_2.seg_index = 3
		#lin2 = lineage.Lineage([], 0.19, [cnv2], None, None, None, None, [ssm2_1, ssm2_2], None, None)
		#cnv3 = cnv.CNV(-1, 0, 1, 1, 9)
		#ssm3 = snp_ssm.SNP_SSM()
		#ssm3.pos = 11
		#ssm3.seg_index = 1
		#cnv3_2 = cnv.CNV(-1, 3, 1, 31, 39)
		#lin3 = lineage.Lineage([], 0.18, [cnv3_2], [cnv3], None, None, None, [ssm3], None, None)
		#cnv4 = cnv.CNV(-1, 1, 1, 10, 19)
		#cnv4_2 = cnv.CNV(-1, 2, 1, 20, 29)
		#cnv4_3 = cnv.CNV(-1, 3, 1, 30, 39)
		#lin4 = lineage.Lineage([], 0.17, [cnv4, cnv4_2], [cnv4_3], None, None, None, None, None, None)
		#cnv5_1 = cnv.CNV(-1, 1, 1, 11, 19)
		#cnv5_2 = cnv.CNV(-1, 2, 1, 20, 29)
		#lin5 = lineage.Lineage([], 0.16, None, [cnv5_1, cnv5_2], None, None, None, None, None, None)
		#my_lins = [lin0, lin1, lin2, lin3, lin4, lin5]
		#seg_num = 4

		#(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy) = model.check_and_update_complete_Z_matrix(
		#	my_lins)
		#z_matrix_list, present_ssms_list, = model.post_analysis_Z_matrix(
		#	my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)	

		## the forbidden hard case is not allowed in any of the Z-matrices
		#forbidden_hard_case = False
		#for i, matrix in enumerate(z_matrix_list):
		#	if matrix[2][3] == 1 and matrix[2][4] == 1 and matrix[3][4] == 1:
		#		forbidden_hard_case = True
		#		break
		#self.assertFalse(forbidden_hard_case)

	def test_post_opt_lineage_divergence_rule_feasibility_check(self):
		# matrix and frequencies are feasible
		lin0 = lineage.Lineage([], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([], 0.8, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([], 0.5, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([], 0.2, None, None, None, None, None, None, None, None)
		lin4 = lineage.Lineage([], 0.01, None, None, None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4]
		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 0, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]

		self.assertTrue(model.post_opt_lineage_divergence_rule_feasibility_check([z_matrix], my_lins)[0])

		# matrix and frequencies are not feasible
		lin0 = lineage.Lineage([], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([], 0.8, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([], 0.5, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([], 0.2, None, None, None, None, None, None, None, None)
		lin4 = lineage.Lineage([], 0.15, None, None, None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4]
		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 0, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]

		self.assertFalse(model.post_opt_lineage_divergence_rule_feasibility_check([z_matrix], my_lins)[0])

		# one matrix is feasible, the other not
		lin0 = lineage.Lineage([], 1.0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([], 0.8, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([], 0.5, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([], 0.2, None, None, None, None, None, None, None, None)
		lin4 = lineage.Lineage([], 0.15, None, None, None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4]
		z_matrix_1 = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 1, 1],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]
		z_matrix_2 = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 0, 1, 1],
			[-1, -1, -1, 0, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]
		self.assertEqual(model.post_opt_lineage_divergence_rule_feasibility_check([z_matrix_1, z_matrix_2], my_lins),
			[True, False])


	def test_check_and_update_complete_Z_matrix(self):
		# minimal filled Z matrix
		# will get filled completely with 1's
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7], 0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2], 0, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([3], 0, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([4], 0, None, None, None, None, None, None, None, None)
		lin4 = lineage.Lineage([5], 0, None, None, None, None, None, None, None, None)
		lin5 = lineage.Lineage([6], 0, None, None, None, None, None, None, None, None)
		lin6 = lineage.Lineage([7], 0, None, None, None, None, None, None, None, None)
		lin7 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
		lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7]

		z_matrix = model.get_Z_matrix(lins)[0]
		zero_count = model.get_0_number_in_z_matrix(z_matrix)
		lineage_num = len(lins)
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = model.check_and_update_complete_Z_matrix_from_matrix(
		 	z_matrix, zero_count, lineage_num)
		self.assertEqual(z_matrix, [[-1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, 1, 1, 1, 1],
			[-1, -1, -1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, 1, 1, 1],
			[-1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, 1], [-1, -1, -1, -1, -1, -1, -1, -1]])
		self.assertEqual(zero_count, 0)
		self.assertEqual(triplet_xys, {})
		self.assertEqual(triplet_ysx, {})
		self.assertEqual(triplet_xsy, {})
		

	def test_update_Z_matrix_iteratively(self):
		# no triplets in hashes, no influence by changed index pair
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		z_matrix = [[0] * 4 for _ in xrange(4)]
		my_z_matrix = [[0] * 4 for _ in xrange(4)]
		zero_count = 0
		index_pair = (2, 3)

		zero_count = model.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
			index_pair)
		self.assertEqual(z_matrix, my_z_matrix)
		self.assertEqual(zero_count, 0)

		# triplets with 0's exist, but non are influences by the changed index pair
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 1, 2, 3)
		model.update_triplet_hash(triplet_ysx, 2, 3, 1)
		model.update_triplet_hash(triplet_xsy, 1, 3, 2)
		z_matrix = [[0] * 12 for _ in xrange(12)]
		my_z_matrix = [[0] * 12 for _ in xrange(12)]
		zero_count = 1
		index_pair = (5,7)

		zero_count = model.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
			index_pair)
		self.assertEqual(z_matrix, my_z_matrix)
		self.assertEqual(zero_count, 1)

		# 4 triplets, at least one in each triplet category
		# 1 triplet in each category is changed
		# 1 triplet is not changed
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 5, 7, 8)
		model.update_triplet_hash(triplet_xsy, 5, 7, 6)
		model.update_triplet_hash(triplet_ysx, 5, 7, 1)
		model.update_triplet_hash(triplet_ysx, 5, 7, 2)
		z_matrix = [[0] * 12 for _ in xrange(12)]
		z_matrix[5][7] = 1
		z_matrix[5][8] = -1
		z_matrix[6][7] = 1
		z_matrix[1][7] = 1
		my_z_matrix = [[0] * 12 for _ in xrange(12)]
		my_z_matrix[5][7] = 1
		my_z_matrix[5][8] = -1
		my_z_matrix[6][7] = 1
		my_z_matrix[1][7] = 1
		my_z_matrix[7][8] = -1
		my_z_matrix[5][6] = 1
		my_z_matrix[1][5] = 1
		zero_count = 4
		index_pair = (5,7)

		zero_count = model.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
			index_pair)
		self.assertEqual(z_matrix, my_z_matrix)
		self.assertEqual(zero_count, 1)
		self.assertEqual(triplet_xys, {})
		self.assertEqual(triplet_xsy, {})
		self.assertEqual(triplet_ysx[5][7][2], True)
		self.assertEqual(len(triplet_ysx.keys()), 1)

		# iterative update
		# 3 triplets, influence each other iterativly
		# 4th triplet includes value that gets changes before triplet is processed
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		model.update_triplet_hash(triplet_xys, 5, 7, 9)
		model.update_triplet_hash(triplet_ysx, 5, 7, 4)
		model.update_triplet_hash(triplet_ysx, 5, 9, 1)
		model.update_triplet_hash(triplet_xsy, 1, 5, 4)
		z_matrix = [[0] * 12 for _ in xrange(12)]
		z_matrix[5][7] = 1
		z_matrix[7][9] = 1
		z_matrix[1][9] = 1
		z_matrix[1][4] = -1
		z_matrix[4][7] = -1
		my_z_matrix = [[0] * 12 for _ in xrange(12)]
		my_z_matrix[5][7] = 1
		my_z_matrix[7][9] = 1
		my_z_matrix[1][9] = 1
		my_z_matrix[1][4] = -1
		my_z_matrix[4][7] = -1
		my_z_matrix[5][9] = 1
		my_z_matrix[1][5] = 1
		my_z_matrix[4][5] = -1
		zero_count = 4
		index_pair = (5,7)

		zero_count = model.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
			index_pair)
		self.assertEqual(z_matrix, my_z_matrix)
		self.assertEqual(zero_count, 1)
		self.assertEqual(triplet_xys, {})
		self.assertEqual(triplet_xsy, {})
		self.assertEqual(triplet_ysx, {})

	def test_remove_triplet_from_all_hashes(self):
		# add one entry to all three hashes manually
		# then remove them all with the function to be tested
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		x = 1
		y = 2
		s = 3
		model.update_triplet_hash(triplet_xys, x, y, s)
		model.update_triplet_hash(triplet_ysx, y, s, x)
		model.update_triplet_hash(triplet_xsy, x, s, y)

		model.remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x, y, s)
		self.assertEqual(triplet_xys, {})
		self.assertEqual(triplet_ysx, {})
		self.assertEqual(triplet_xsy, {})

	def test_remove_triplet_from_hash(self):
		# hash with more entries at second index
		my_hash = {}
		model.update_triplet_hash(my_hash, 1, 2, 3)
		model.update_triplet_hash(my_hash, 1, 2, 4)

		model.remove_triplet_from_hash(my_hash, 1, 2, 4)
		self.assertEqual(my_hash[1][2].keys(), [3])

		# hash with only one entry, is empty afterwards
		model.remove_triplet_from_hash(my_hash, 1, 2, 3)
		self.assertEqual(my_hash, {})

	def test_update_triplet_hash(self):
		my_hash = {}
		model.update_triplet_hash(my_hash, 1, 2, 3)
		self.assertTrue(my_hash[1][2][3])

	def test_check_1f_2d_2g_2j_losses_gains(self):
		# two losses on two alleles in two lineages with relation
		# lower lineages can't be descendants if they have SSMs
		# lineage has SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][2] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][2] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][3] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages with relation
		# lower lineages can't be descendants if they have SSMs
		# lineage has no SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][2] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][2] = 1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages without known relation
		# nothing can be said about lower lineages
		# lineage has SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][3] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages without relation
		# nothing can be said about lower lineages
		# lineage has SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][2] = -1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][2] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][3] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages with relation
		# higher lineage has SSM
		# can't be ancestor of both
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[2][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[2][3] = 1
		my_z_matrix[1][2] = -1
		my_z_matrix[1][3] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][1] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages with relation
		# higher lineage has SSM
		# is only ancestor of one --> not possible
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[2][3] = 1
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[2][3] = 1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][1] = True
		zero_count = 2
		first_run = True

		with self.assertRaises(eo.MyException):
			model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, present_ssms, 
				triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)

		# two losses on two alleles in two lineages with relation
		# higher lineage has no SSMs
		# no result can be drawn
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[2][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[2][3] = 1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two gains on two alleles in two lineages with relation
		# higher lineage has SSM
		# can't be ancestor of both
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][2] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][3] = "something"
		gain_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[2][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[2][3] = 1
		my_z_matrix[1][2] = -1
		my_z_matrix[1][3] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][1] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
			first_run, mutations=cons.GAIN)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)
		
		# two losses on two alleles in two lineages with relation
		# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
		# lineage has unphased SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][3] = 1
		my_z_matrix[1][2] = -1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][2] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages with relation
		# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
		# lineage has no unphased SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][3] = 1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages without known relation
		# nothing can be said about middle lineages
		# lineage has unphased SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][2] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages without relation
		# nothing can be said about middle lineages
		# lineage has unphased SSMs
		# first run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][3] = -1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][3] = -1
		present_ssms = [[False] * lin_num for _ in xrange(3)]
		present_ssms[cons.UNPHASED][2] = True
		zero_count = 2
		first_run = True

		zero_count = model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
			present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# two losses on two alleles in two lineages without known relation
		# lineage 3 has SSMs
		# second run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		present_ssms = [[[False] * lin_num for _ in xrange(3)]]
		present_ssms[0][cons.UNPHASED][3] = True
		zero_count = 3
		first_run = False
		z_matrix_fst_rnd = copy.deepcopy(z_matrix)
		z_matrix_list = [z_matrix]
		triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
		present_ssms_list = [present_ssms]
		CNVs_all = [CNVs]

		# function to test
		model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, None, zero_count, None, 
			triplet_xys, triplet_ysx, triplet_xsy, first_run, mutations=cons.LOSS,
			z_matrix_fst_rnd=z_matrix_fst_rnd, z_matrix_list=z_matrix_list, triplets_list=triplets_list,
			present_ssms_list=present_ssms_list, seg_index=0, CNVs_all=CNVs_all)

		self.assertEqual(len(z_matrix_list), 4)
		self.assertEqual(len(triplets_list), 4)
		self.assertEqual(len(present_ssms_list), 4)

		# two losses on two alleles in two lineages without relation
		# nothing can be said about lower lineages
		# lineage has SSMs
		# second run
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		loss_num = 2
		lin_num = 4
		z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		z_matrix[1][2] = -1
		my_z_matrix = [[0] * (lin_num) for _ in xrange(lin_num)]
		my_z_matrix[1][2] = -1
		present_ssms = [[[False] * lin_num for _ in xrange(3)]]
		present_ssms[0][cons.UNPHASED][3] = True
		zero_count = 2
		first_run = False
		z_matrix_fst_rnd = copy.deepcopy(z_matrix)
		z_matrix_list = [z_matrix]
		triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
		present_ssms_list = [present_ssms]
		CNVs_all = [CNVs]
		# function to test
		model.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, None, zero_count,
			None, triplet_xys, triplet_ysx, triplet_xsy,
			first_run, mutations=cons.LOSS, z_matrix_fst_rnd=z_matrix_fst_rnd, z_matrix_list=z_matrix_list,
			triplets_list=triplets_list, present_ssms_list=present_ssms_list, seg_index=0, CNVs_all=CNVs_all)

		self.assertEqual(len(z_matrix_list), 1)
		self.assertEqual(len(triplets_list), 1)
		self.assertEqual(len(present_ssms_list), 1)

	def test_check_2i_phased_changes(self):
		# no influence of upstream lineages
		# current example is not possible in practice but should test everything
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][2] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][2] = 1 
		my_z_matrix[1][3] = -1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.A][1] = True
		present_ssms[cons.B][1] = True
		present_ssms[cons.A][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms, 
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)

		# some influence to upstream lineages
		# loss in A
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.A][1] = True
		present_ssms[cons.A][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# loss in B
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.B][1] = True
		present_ssms[cons.B][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# gain in A
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.A][1] = True
		present_ssms[cons.A][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# gain in B
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.B][1] = True
		present_ssms[cons.B][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# gain in B
		# adding a -1 to the matrix has an influence on further triplets and thus fields in the matrix
		triplet_xys = {}
		model.update_triplet_hash(triplet_xys, 2, 3, 4)
		triplet_ysx = {}
		triplet_xsy = {}
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][3] = "something"
		z_matrix = [[0] * (5) for _ in xrange(5)]
		z_matrix[1][3] = 1
		z_matrix[3][4] = 1
		my_z_matrix = [[0] * (5) for _ in xrange(5)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		my_z_matrix[3][4] = 1
		my_z_matrix[2][4] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.B][1] = True
		present_ssms[cons.B][2] = True
		zero_count = 2

		zero_count = model.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)

	def test_check_2h_LOH(self):
		# loss and gain in different lineages
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 1
		loss_num = 1
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		with self.assertRaises(eo.MyException):
			model.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)

		# LOH but no SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 1
		loss_num = 1
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# LOH with upstream SSMs, some are already in relation
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 1
		loss_num = 1
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.A][1] = True
		present_ssms[cons.B][2] = True
		zero_count = 2

		zero_count = model.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)


	def test_check_2f_CN_gains(self):
		# CN gains on both alleles and in different lineages
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 2
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)

		# CN on both alleles and in the same lineages, no upstream SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 2
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# CN on one alleles and in the different lineages, no upstream SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 2
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.A][1] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# CN on both alleles and in the same lineages, multiple upstream SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		gain_num = 2
		CNVs = {}
		CNVs[cons.GAIN] = {} 
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][3] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][3] = 1
		my_z_matrix[2][3] = -1
		
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.UNPHASED][1] = True
		present_ssms[cons.UNPHASED][2] = True
		zero_count = 2

		zero_count = model.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

	def test_check_1d_2c_CN_losses(self):

		# CN losses on both alleles and different lineages
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)

		# CN losses on both alleles and in same lineage, no lower or higher lineage with SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)] 
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)


		# CN losses on both alleles and in same lineage, lower lineages have SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)] 
		my_z_matrix[2][3] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.UNPHASED][3] = True
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# CN losses on both alleles and in same lineage, lower lineages have SSMs as well as higher
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)] 
		my_z_matrix[2][3] = -1
		my_z_matrix[1][2] = -1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.UNPHASED][3] = True
		present_ssms[cons.UNPHASED][1] = True
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 0)
		self.assertEqual(z_matrix, my_z_matrix)

		# CN losses on both alleles and in same lineage, higher lineage has SSMs but is phased
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][2] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		z_matrix[1][2] = 1
		my_z_matrix = [[0] * (4) for _ in xrange(4)] 
		my_z_matrix[1][2] = 1
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.UNPHASED][1] = True
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# CN losses on only one allele and same lineage
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {} 
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		CNVs[cons.LOSS][cons.A][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		present_ssms = [[False] * 4 for _ in xrange(3)]
		zero_count = 2

		zero_count = model.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
	
	def test_check_1c_CN_loss(self):

		# already lowest lineage
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 1
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		zero_count = 2
		phase = cons.A
		present_ssms = [[False] * 4 for _ in xrange(3)]

		zero_count = model.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# lower lineage has no SSMs phased to A
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 1
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		zero_count = 2
		phase = cons.A
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.B][2] = True
		present_ssms[cons.UNPHASED][2] = True

		zero_count = model.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 2)
		self.assertEqual(z_matrix, my_z_matrix)

		# lower lineage 2 has SSMs phased to B, lineage 3 doesn't have SSMs
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 1
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][1] = "something"
		z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix = [[0] * (4) for _ in xrange(4)]
		my_z_matrix[1][2] = -1
		zero_count = 2
		phase = cons.B
		present_ssms = [[False] * 4 for _ in xrange(3)]
		present_ssms[cons.B][2] = True
		present_ssms[cons.UNPHASED][2] = True

		zero_count = model.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)


	def test_check_1a_CN_LOSS(self):

		# only one loss
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 1
		CNVs = {}
		z_matrix = None
		zero_count = 1

		zero_count = model.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)

		# 2 losses, same lineage, different alleles
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][3] = "something" 
		z_matrix = None
		zero_count = 1

		zero_count = model.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)

		# 2 losses, different lineages, different allele
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][3] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][5] = "something" 
		z_matrix = None
		zero_count = 1

		zero_count = model.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)

		# 2 losses, different lineages, same allele
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 2
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.A][2] = "something" 
		z_matrix = [[0] * 3 for _ in xrange(3)]
		my_z_matrix = [[0] * 3 for _ in xrange(3)]
		my_z_matrix[1][2] = -1
		zero_count = 2

		zero_count = model.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

		# 3 losses, three pairs, three pairs lead to -1, whereas one already has a -1
		triplet_xys = {}
		triplet_ysx = {}
		triplet_xsy = {}
		loss_num = 3
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][1] = "something"
		CNVs[cons.LOSS][cons.B][2] = "something" 
		CNVs[cons.LOSS][cons.B][3] = "something" 
		z_matrix = [[0] * 4 for _ in xrange(4)]
		z_matrix[2][3] = -1
		my_z_matrix = [[0] * 4 for _ in xrange(4)]
		my_z_matrix[2][3] = -1
		my_z_matrix[1][2] = -1
		my_z_matrix[1][3] = -1
		zero_count = 3

		zero_count = model.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
			triplet_xys, triplet_ysx, triplet_xsy)
		self.assertEqual(zero_count, 1)
		self.assertEqual(z_matrix, my_z_matrix)

	def test_get_present_ssms(self):
		ssm_1 = snp_ssm.SNP_SSM()
		ssm_1.seg_index = 0
		ssm_2 = snp_ssm.SNP_SSM()
		ssm_2.seg_index = 0
		ssm_a_1 = snp_ssm.SNP_SSM()
		ssm_a_1.seg_index = 0
		ssm_b_1 = snp_ssm.SNP_SSM()
		ssm_b_1.seg_index = 3
		ssm_b_2 = snp_ssm.SNP_SSM()
		ssm_b_2.seg_index = 6

		lin1 = lineage.Lineage([], 0, None, None, None, None, None, [ssm_1, ssm_2], None, 
			[ssm_b_1, ssm_b_2])
		lin2 = lineage.Lineage([], 0, None, None, None, None, None, None, [ssm_a_1], None)
		lins = [lin1, lin2]

		# unphased SSMs are present in segment 0 and the first lineage
		onctos_present_ssms = [[False] * 2 for _ in xrange(3)]
		lin_index = 0
		seg_index = 0
		phase = cons.UNPHASED
		ssms_index_list = [0, 0]
		model.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
			ssms_index_list)
		my_present_ssms = [[False] * 2 for _ in xrange(3)]
		my_present_ssms[cons.UNPHASED][0] = True
		self.assertEqual(onctos_present_ssms, my_present_ssms)
		self.assertEqual(ssms_index_list, [2, 0])

		# no unphased SSMs are present for this position in ssm index list
		onctos_present_ssms = [[False] * 2 for _ in xrange(3)]
		lin_index = 0
		seg_index = 1
		phase = cons.UNPHASED
		ssms_index_list = [10, 0]
		model.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
			ssms_index_list)
		my_present_ssms = [[False] * 2 for _ in xrange(3)]
		self.assertEqual(onctos_present_ssms, my_present_ssms)
		self.assertEqual(ssms_index_list, [10, 0])

		# phased to A SSMs are present in second lineage
		onctos_present_ssms = [[False] * 2 for _ in xrange(3)]
		lin_index = 1
		seg_index = 0
		phase = cons.A
		ssms_index_list = [0, 0]
		model.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
			ssms_index_list)
		my_present_ssms = [[False] * 2 for _ in xrange(3)]
		my_present_ssms[cons.A][1] = True
		self.assertEqual(onctos_present_ssms, my_present_ssms)
		self.assertEqual(ssms_index_list, [0, 1])

		# phased to B in first lineage, other segment
		onctos_present_ssms = [[False] * 2 for _ in xrange(3)]
		lin_index = 0
		seg_index = 6
		phase = cons.B
		ssms_index_list = [1, 0]
		model.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
			ssms_index_list)
		my_present_ssms = [[False] * 2 for _ in xrange(3)]
		my_present_ssms[cons.B][0] = True
		self.assertEqual(onctos_present_ssms, my_present_ssms)
		self.assertEqual(ssms_index_list, [2, 0])


	def test_is_it_LOH(self):

		cnv1 = cnv.CNV(1, 0, 1, 1, 1)
		cnv2 = cnv.CNV(-1, 0, 1, 1, 1)

		# proper LOH
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][1] = cnv1
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][1] = cnv2
		gain_num = 1
		loss_num = 1
		self.assertTrue(model.is_it_LOH(gain_num, loss_num, CNVs))

		# only CN changes in one direction
		gain_num = 1
		loss_num = 0
		self.assertFalse(model.is_it_LOH(gain_num, loss_num, CNVs))

		# too many CN changes
		gain_num = 1
		loss_num = 2
		with self.assertRaises(eo.NotProperLOH):
			model.is_it_LOH(gain_num, loss_num, CNVs)

		# CN changes on different chromosome
		gain_num = 1
		loss_num = 1
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][1] = cnv1
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = cnv2
		with self.assertRaises(eo.NotProperLOH):
			model.is_it_LOH(gain_num, loss_num, CNVs)


	def test_add_CN_change_to_hash(self):
		# two CN changes, gain and loss, an two different segments
		cnv1 = cnv.CNV(1, 0, 1, 1, 1)	
		cnv2 = cnv.CNV(-1, 2, 1, 1, 1)	
		cnv_list = [cnv1, cnv2]
		# other CN change on other alleles
		cnv3 = cnv.CNV(-1, 0, 1, 1, 1)
		cnv_list_b = [cnv3]
		# lineages
		lin0 = lineage.Lineage([1, 2, 3], 0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([], 0, cnv_list, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([], 0, None, cnv_list_b, None, None, None, None, None, None)
		lin_list = [lin0, lin1, lin2]

		# test insertion of cnv, phase A, cnv is present
		CNVs = {}
		gain_num = 0
		loss_num = 0
		phase = cons.A
		lin_index = 1
		seg_index = 0
		cnv_index_list = [0, 0, 0]
		gain_num, loss_num = model.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
			CNVs, gain_num, loss_num, phase, cnv_index_list)
		self.assertEqual(gain_num, 1)
		self.assertEqual(loss_num, 0)
		self.assertEqual(cnv_index_list, [0, 1, 0])
		self.assertEqual(CNVs[cons.GAIN][cons.A][1], cnv1)

		# test insertion of cnv, phase B, cnv is present
		CNVs = {}
		gain_num = 0
		loss_num = 0
		phase = cons.B
		lin_index = 2
		seg_index = 0
		cnv_index_list = [0, 0, 0]
		gain_num, loss_num = model.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
			CNVs, gain_num, loss_num, phase, cnv_index_list)
		self.assertEqual(gain_num, 0)
		self.assertEqual(loss_num, 1)
		self.assertEqual(cnv_index_list, [0, 0, 1])
		self.assertEqual(CNVs[cons.LOSS][cons.B][2], cnv3)

		# test insertion of cnv, phase A, no cnv present
		CNVs = {}
		gain_num = 0
		loss_num = 0
		phase = cons.A
		lin_index = 1
		seg_index = 1
		cnv_index_list = [0, 1, 0]
		gain_num, loss_num = model.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
			CNVs, gain_num, loss_num, phase, cnv_index_list)
		self.assertEqual(gain_num, 0)
		self.assertEqual(loss_num, 0)
		self.assertEqual(cnv_index_list, [0, 1, 0])
		self.assertEqual(len(CNVs.keys()), 0)

		# test insertion of cnv, phase A, cnv present at position
		CNVs = {}
		gain_num = 0
		loss_num = 0
		phase = cons.A
		lin_index = 1
		seg_index = 2
		cnv_index_list = [0, 1, 0]
		gain_num, loss_num = model.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
			CNVs, gain_num, loss_num, phase, cnv_index_list)
		self.assertEqual(gain_num, 0)
		self.assertEqual(loss_num, 1)
		self.assertEqual(cnv_index_list, [0, 2, 0])
		self.assertEqual(CNVs[cons.LOSS][cons.A][1], cnv2)

		# test insertion of cnv, phase A, no cnv anymore
		CNVs = {}
		gain_num = 0
		loss_num = 0
		phase = cons.A
		lin_index = 1
		seg_index = 5
		cnv_index_list = [0, 2, 0]
		gain_num, loss_num = model.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
			CNVs, gain_num, loss_num, phase, cnv_index_list)
		self.assertEqual(gain_num, 0)
		self.assertEqual(loss_num, 0)
		self.assertEqual(cnv_index_list, [0, 2, 0])
		self.assertEqual(len(CNVs.keys()), 0)

	def test_create_fixed_values_new_for_normal_segments(self):

		# with already fixed segments
		normal_seg_indices = [0, 2, 5]
		fixed_cnv_list_new = [[1, [[1, 1, cons.A]]], [3, [[1, 1, cons.A]]], [4, [[1, 1, cons.A]]]]

		new_list = model.create_fixed_values_new_for_normal_segments(normal_seg_indices, fixed_cnv_list_new)

		self.assertEqual(new_list, [[0, [[0, 0, cons.A]]], [1, [[1, 1, cons.A]]], [2, [[0, 0, cons.A]]], 
			[3, [[1, 1, cons.A]]], [4, [[1, 1, cons.A]]], [5, [[0, 0, cons.A]]]])

		# without already fixed segments
		normal_seg_indices = [0, 2, 5]
		fixed_cnv_list_new = None

		new_list = model.create_fixed_values_new_for_normal_segments(normal_seg_indices, fixed_cnv_list_new)

		self.assertEqual(new_list, [[0, [[0, 0, cons.A]]], [2, [[0, 0, cons.A]]], 
			[5, [[0, 0, cons.A]]]])

	def test_update_linage_relations_based_on_z_matrix(self):

		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 0, 1],
			[-1, -1, -1, 1, 0],
			[-1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1]
			]
		lin0 = lineage.Lineage([1, 2, 3, 4], 0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2], 0, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([3], 0, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
		lin4 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
		my_lins = [lin0, lin1, lin2, lin3, lin4]

		model.update_linage_relations_based_on_z_matrix(my_lins, z_matrix)

		self.assertEqual(my_lins[1].sublins, [2,4])
		self.assertEqual(my_lins[2].sublins, [3])

	def test_get_Z_matrix(self):
		lin0 = lineage.Lineage([1, 2, 3], 0, None, None, None, None, None, None, None, None)
		lin1 = lineage.Lineage([2, 3], 0, None, None, None, None, None, None, None, None)
		lin2 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
		lin3 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
		my_lineages = [lin0, lin1, lin2, lin3]

		my_z = [
			[-1, 1, 1, 1],
			[-1, -1, 1, 1],
			[-1, -1, -1, 0],
			[-1, -1, -1, -1]
			]

		z_matrix, zero_count = model.get_Z_matrix(my_lineages)
		self.assertEqual(my_z, z_matrix)
		self.assertEqual(1, zero_count)

	def test_get_phis_from_lineages(self):
		lineages = [lineage.Lineage([], 1, [], [], [], [], [], [], [], []),
			lineage.Lineage([], 0.6, [], [], [], [], [], [], [], []),
			lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])]
		self.assertEqual(model.get_phis_from_lineages(lineages), [0.6, 0.2])

	def test_get_fixed_z_matrices(self):
		fixed_five = model.get_fixed_z_matrices(5)
		self.assertEqual(64, len(fixed_five))
		self.assertEqual([0, 0, 0, 0, 0, 0], fixed_five[0])
		self.assertEqual([0, 0, 0, 0, 0, 1], fixed_five[1])
		self.assertEqual([1, 1, 1, 1, 1, 1], fixed_five[63])

	def test_combine_lineages_lists_fixed_phi_z(self):
		# create two lineage lists with numbers in lists instead of SSMs and co
		lin_list_1 = [lineage.Lineage([0], 0.5, [], [1], [2], [3], [4], [5], [6], [7]),
			lineage.Lineage([0,2], 0.4, [], [1], [2], [3], [4], [5], [6], [7])]
		lin_list_2 = [lineage.Lineage([0], 0.5, [8], [9], [10], [11], [12], [13], [14], [15]),
			lineage.Lineage([0,2], 0.4, [], [1], [2], [3], [4], [5], [6], [7])]
		combination = [lineage.Lineage([0], 0.5, [8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]),
			lineage.Lineage([0,2], 0.4, [], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7])]
		self.assertEqual(model.combine_lineages_lists_fixed_phi_z([lin_list_1, lin_list_2]), combination)

	def test_get_seg_indices_w_cn_changes(self):
		seg_num = 10
		normal_seg_indices = [1,2,3,5,6,7]
		other_seg_indices = [0, 4, 8, 9]
		self.assertEqual(model.get_seg_indices_w_cn_changes(seg_num, normal_seg_indices),
			other_seg_indices)

	def test_get_lin_ass_for_ssms_w_ssms_from_super_ssms(self):
		ssm_31 = snp_ssm.SSM()
		ssm_31.chr = 3
		ssm_31.pos = 1
		ssm_32 = snp_ssm.SSM()
		ssm_32.chr = 3
		ssm_32.pos = 2
		ssm_33 = snp_ssm.SSM()
		ssm_33.chr = 3
		ssm_33.pos = 3
		ssm_34 = snp_ssm.SSM()
		ssm_34.chr = 3
		ssm_34.pos = 4
		ssm_310 = snp_ssm.SSM()
		ssm_310.chr = 3
		ssm_310.pos = 10
		ssm_41 = snp_ssm.SSM()
		ssm_41.chr = 4
		ssm_41.pos = 1
		ssm_42 = snp_ssm.SSM()
		ssm_42.chr = 4
		ssm_42.pos = 2
		ssm_43 = snp_ssm.SSM()
		ssm_43.chr = 4
		ssm_43.pos = 3

		ssm_list = [ssm_31, ssm_32, ssm_33, ssm_34, ssm_310, ssm_41, ssm_42, ssm_43]
		super_ssms = [ssm_310, ssm_31, ssm_32]
		lineage_assignment_for_ssms_w_ssms = [[ssm_31, 4], [ssm_32, 2], [ssm_310, 5]]
		ssms_of_segment = [0, 1, 2, 4, 5, 7]
		cluster_labels = [1, 2, 2, 0, 0, 1]

		lineage_assignment_for_normal_ssms_w_ssms = model.get_lin_ass_for_ssms_w_ssms_from_super_ssms(
			lineage_assignment_for_ssms_w_ssms, super_ssms, cluster_labels, ssms_of_segment, ssm_list)
		true_assignment = [[ssm_31, 4], [ssm_32, 2], [ssm_33, 2], [ssm_310, 5], [ssm_41, 5], [ssm_43, 4]]
		self.assertEqual(true_assignment, lineage_assignment_for_normal_ssms_w_ssms)



	def test_get_lineage_assignment_for_subset_of_ssms(self):
		# working scenario
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 2
		ssm3.pos = 1
		lineage_assignment_for_ssms_w_ssms = [[ssm1, 2], [ssm2, 1], [ssm3, 1]]
		ssm_subset = [ssm1, ssm3]

		lineage_assignments = model.get_lineage_assignment_for_subset_of_ssms(
			lineage_assignment_for_ssms_w_ssms, ssm_subset)
		self.assertEqual([2, 1], lineage_assignments)

		# subset has SSM in middel not included in list
		ssm_subset = [ssm1, ssm2, ssm3]
		lineage_assignment_for_ssms_w_ssms = [[ssm1, 2], [ssm3, 1]]
		with self.assertRaises(eo.SSMNotFoundException):
		 	model.get_lineage_assignment_for_subset_of_ssms(
				lineage_assignment_for_ssms_w_ssms, ssm_subset)

		# subset has SSM at end not included in list
		ssm_subset = [ssm1, ssm2, ssm3]
		lineage_assignment_for_ssms_w_ssms = [[ssm1, 2], [ssm2, 1]]
		with self.assertRaises(eo.SSMNotFoundException):
		 	model.get_lineage_assignment_for_subset_of_ssms(
				lineage_assignment_for_ssms_w_ssms, ssm_subset)


	def test_get_lineage_assignment_for_ssms_w_ssms(self):
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 2
		ssm3.pos = 1
		my_lineage0 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		my_lineage1 = lineage.Lineage([], 0, [], [], [], [], [], [ssm3], [], [])
		my_lineage2 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [ssm1])
		my_lineage3 = lineage.Lineage([], 0, [], [], [], [], [], [], [ssm2], [])
		lineage_list = [my_lineage0, my_lineage1, my_lineage2, my_lineage3]

		lineage_assignment_for_ssms_w_ssms = model.get_lineage_assignment_for_ssms_w_ssms(
			lineage_list)

		self.assertEqual([[ssm1, 2], [ssm2, 3], [ssm3, 1]], lineage_assignment_for_ssms_w_ssms)

	# 2 segments, 3 clusters
	# in 2nd segment only 1 SSM
	def test_create_superSSMs_and_backtransformation(self):
		# data creation
		# SSM1, seg: 0, VAF = 0.5
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 1
		ssm1.variant_count = 10
		ssm1.ref_count = 10
		ssm1.seg_index = 0
		# SSM2, seg: 0, VAF = 0.5
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 2
		ssm2.variant_count = 10
		ssm2.ref_count = 10
		ssm2.seg_index = 0
		# SSM3, seg: 0, VAF = 0.25
		ssm3 = snp_ssm.SSM()
		ssm3.chr = 1
		ssm3.pos = 3
		ssm3.variant_count = 5
		ssm3.ref_count = 15
		ssm3.seg_index = 0
		# SSM4, seg: 0, VAF = 0.75
		ssm4 = snp_ssm.SSM()
		ssm4.chr = 1
		ssm4.pos = 4
		ssm4.variant_count = 15
		ssm4.ref_count = 5
		ssm4.seg_index = 0
		# SSM5, seg: 1, VAF = 0.75
		ssm5 = snp_ssm.SSM()
		ssm5.chr = 1
		ssm5.pos = 5
		ssm5.variant_count = 15
		ssm5.ref_count = 5
		ssm5.seg_index = 1
		# further data
		ssm_list = [ssm1, ssm2, ssm3, ssm4, ssm5]
		seg_indices = [0, 1]
		cluster_num = [3, 3]

		# correct data
		superSSM1 = snp_ssm.SSM()
		superSSM1.chr = 1
		superSSM1.pos = 1
		superSSM1.variant_count = 20
		superSSM1.ref_count = 20
		superSSM1.seg_index = 0
		my_superSSMs = [superSSM1, ssm3, ssm4, ssm5]

		# use functions to create superSSMs
		(ssm_indices_per_cluster_per_seg, ssm_objects_of_segment_per_seg) = (
			clusmod.choose_SSMs_cluster_create_indices(seg_indices, ssm_list,
			[], cluster_num, []))
		superSSMs, superSSM_hash = model.create_superSSMs(ssm_indices_per_cluster_per_seg,
			ssm_objects_of_segment_per_seg)

		# test
		self.assertEqual(my_superSSMs, superSSMs)
		self.assertEqual(1, len(superSSM_hash.keys()))
		self.assertEqual(4, len(superSSM_hash[1].keys()))
		self.assertEqual(ssm3, superSSM_hash[1][3][0])
		self.assertEqual([ssm1, ssm2], superSSM_hash[1][1])

		# create lineage with superSSMs
		lin1 = lineage.Lineage([], 0, [], [], [], [], [], [superSSMs[0]], [superSSMs[1]],
			[superSSMs[2]])
		lin2 = lineage.Lineage([], 0, [], [], [], [], [], [superSSMs[0], superSSMs[3]], [superSSMs[1]],
			[superSSMs[2]])
		my_lins_superSSMs = [lin1, lin2]

		# correct data
		lin1_indi = lineage.Lineage([], 0, [], [], [], [], [], [ssm1, ssm2], [ssm3],
			[ssm4])
		lin2_indi = lineage.Lineage([], 0, [], [], [], [], [], [ssm1, ssm2, ssm5], [ssm3],
			[ssm4])
		my_lins_indi = [lin1_indi, lin2_indi]

		# transform lineage back
		model.replace_superSSMs_in_lineages(my_lins_superSSMs, superSSM_hash)

		# test if lineages are equal
		self.assertEqual(my_lins_superSSMs, my_lins_indi)


	def test_create_ssm_indices_per_cluster(self):

		cluster_labels_list = [np.array([2, 1, 2, 0, 1]), np.array([2])]
		#cluster_assingment_list = [[[0,2], [1, 1], [2, 2], [3, 0], [4,1]],
		#	[[5, 2]]]
		cluster_num = [3, 3]
		true_ssm_indices_per_cluster = [[[3], [1, 4],
			[0, 2]], [[], [], [0]]]

		ssm_indices_per_cluster = model.create_ssm_indices_per_cluster(
			cluster_labels_list, cluster_num)
	
		self.assertTrue(true_ssm_indices_per_cluster==ssm_indices_per_cluster)

	def test_cluster_VAFs(self):
		segment_VAFs = np.asarray([[0.1], [0.1], [0.5], [0.6]])
		clusters = model.cluster_VAFs(segment_VAFs, 2)
		clusters_test = np.asarray([[0, 0, 1, 1]])
		self.assertTrue((clusters==clusters_test).all())
		# use too many clusters
		clusters = model.cluster_VAFs(segment_VAFs, 20)
		clusters_test = np.asarray([[0, 0, 2, 1]])
		self.assertTrue((clusters==clusters_test).all())

	def test_choose_seg_subset(self):
		# segments
		seg1 = segment.Segment_allele_specific(1, 0, 10, 1, 0.25, 1, 0.25)
		seg1.index = 0
		seg2 = segment.Segment_allele_specific(1, 0, 10, 1, 0.25, 1, 0.25)
		seg2.index = 1
		seg3 = segment.Segment_allele_specific(1, 0, 10, 1, 0.25, 1, 0.25)
		seg3.index = 2

		normal_indices = [0, 2]

		chosen_segments = model.choose_seg_subset(normal_indices, [seg1, seg2, seg3])

		self.assertEqual(chosen_segments, [seg1, seg3])

	def test_choose_ssm_subset(self):
		# SSMs
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 3

		########
		# for normal segments
		########

		# 1 SSM belongs to 1 segment
		segid = [1]
		ssm_list = [ssm1]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=True)
		self.assertEqual(ssms_of_segment, [0])

		# 3 SSMs, first belongs to 1 segment
		segid = [1]
		ssm_list = [ssm1, ssm2, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=True)
		self.assertEqual(ssms_of_segment, [0])

		# 3 SSMs, third belongs to 1 segment
		segid = [3]
		ssm_list = [ssm1, ssm1, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=True)
		self.assertEqual(ssms_of_segment, [2])

		# 2 SSMs belong to 2 following segments
		segid = [1, 3]
		ssm_list = [ssm1, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=True)
		self.assertEqual(ssms_of_segment, [0, 1])

		# 2 SSMs belong to 3 segments, middle segment is unused
		segid = [1, 2, 3]
		ssm_list = [ssm1, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=True)
		self.assertEqual(ssms_of_segment, [0, 1])

		########
		# for segments with CN changes
		########

		# 2 SSMs, 2 segments, last SSM is in last segment
		segid = [1, 3]
		ssm_list = [ssm1, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=False)
		self.assertEqual(ssms_of_segment, [[0], [1]])

		# 1 SSM, 1 segment, last SSMs come after last segment
		segid = [1]
		ssm_list = [ssm1, ssm2, ssm2] 
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=False)
		self.assertEqual(ssms_of_segment, [[0]])

		# 2 SSM, 6 segments, 4 segments without SSM
		segid = [1, 2, 3, 4, 5, 6]
		ssm2.seg_index = 4
		ssm_list = [ssm1, ssm2]
		ssms_of_segment = model.choose_ssm_subset(segid, ssm_list, normal_segments=False)
		self.assertEqual(ssms_of_segment, [[0], [], [], [1], [], []])


	def test_get_num_k_for_dc_des(self):
		sublin_num = 4
		k = 1
		self.assertEqual(2, model.get_num_k_for_dc_des(sublin_num, k))

		sublin_num = 4
		k = 2
		self.assertEqual(1, model.get_num_k_for_dc_des(sublin_num, k))

		sublin_num = 5
		k = 2
		self.assertEqual(2, model.get_num_k_for_dc_des(sublin_num, k))

	def test_get_sum_num_k_for_dc_des(self):
		sublin_num = 4
		k = 1
		self.assertEqual(2, model.get_sum_num_k_for_dc_des(sublin_num, k))

		sublin_num = 4
		k = 2
		self.assertEqual(3, model.get_sum_num_k_for_dc_des(sublin_num, k))

		sublin_num = 5
		k = 2
		self.assertEqual(5, model.get_sum_num_k_for_dc_des(sublin_num, k))

	def test_get_num_k_for_dc_anc(self):
		k_prime = 1
		self.assertEqual(0, model.get_num_k_for_dc_anc(k_prime))

		k_prime = 2
		self.assertEqual(1, model.get_num_k_for_dc_anc(k_prime))

		k_prime = 3
		self.assertEqual(2, model.get_num_k_for_dc_anc(k_prime))

	def test_get_sum_num_k_for_dc_anc(self):
		k_prime = 2
		self.assertEqual(1, model.get_sum_num_k_for_dc_anc(k_prime))

		k_prime = 3
		self.assertEqual(3, model.get_sum_num_k_for_dc_anc(k_prime))

		k_prime = 4
		self.assertEqual(6, model.get_sum_num_k_for_dc_anc(k_prime))

	def test_create_z_as_maxtrix_w_values(self):
		sublin_num = 4
		values = [1, 1, 1]

		z_matrix = model.create_z_as_maxtrix_w_values(sublin_num, values)

		# test
		self.assertEqual(z_matrix[1][2], 1)
		self.assertEqual(z_matrix[1][3], 1)
		self.assertEqual(z_matrix[2][3], 1)
		self.assertEqual(z_matrix[0][2], 1)
		self.assertEqual(z_matrix[0][0], 0)
		self.assertEqual(z_matrix[3][2], 0)

	def test_create_fixed_frequencies_data(self):
		
		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_phis_five_lins"

		model.create_fixed_file(lineages, cons.FREQ, test=True, 
			result_file_name=input_file_name, output_file=output_file_name)
		(fixed_values, start_index, stop_index) = oio.read_fixed_value_file(output_file_name)

		self.assertListEqual(fixed_values, [0.5, 0.3, 0.2, 0.15])
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)

	def test_create_fixed_segments_all_but_one(self):

		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_prefix = "testdata/unittests/fixed_seg_but_one_five_lins"

		model.create_fixed_segments_all_but_one(lineages, output_file_prefix, test=True, 
			result_file_name=input_file_name)
		
		# test segments
		(fixed_segs_2, start_index, stop_index) = oio.read_fixed_value_file(
			"testdata/unittests/fixed_seg_but_one_five_lins_CNV_unfixed_segment_2")
		my_fixed_segs_2 = [
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 1,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 0, 0, 0
			]

		self.assertListEqual(fixed_segs_2, my_fixed_segs_2)
		self.assertEqual(start_index, 2)
		self.assertEqual(stop_index, 2)


		# test SNPs for segment where no SNPs occurs, at the end
		(fixed_SNPs_4, start_index, stop_index) = oio.read_fixed_value_file(
			"testdata/unittests/fixed_seg_but_one_five_lins_SNP_unfixed_segment_4")
		my_fixed_SNPs_4 = [
			1,
			0,
			1,
			0,

			0,
			1,
			0,
			0,

			0,
			0,
			0,
			1
			]
		self.assertListEqual(fixed_SNPs_4, my_fixed_SNPs_4)
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)

		# test SNPs for segment where no SNPs occur, in the middle
		(fixed_SNPs_1, start_index, stop_index) = oio.read_fixed_value_file(
			"testdata/unittests/fixed_seg_but_one_five_lins_SNP_unfixed_segment_1")
		# are same fixed values as for segment 4
		self.assertListEqual(fixed_SNPs_1, my_fixed_SNPs_4)
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)


		# test SNPS in segment where they occur
		(fixed_SNPs_0, start_index, stop_index) = oio.read_fixed_value_file(
			"testdata/unittests/fixed_seg_but_one_five_lins_SNP_unfixed_segment_0")
		my_fixed_SNPs_0 = [
			1,
			0,

			0,
			0,

			0,
			1
			]
		
		self.assertListEqual(fixed_SNPs_0, my_fixed_SNPs_0)
		self.assertEqual(start_index, 0)
		self.assertEqual(stop_index, 1)

		# test SSMs
		(fixed_SSMs_2, start_index, stop_index) = oio.read_fixed_value_file(
			"testdata/unittests/fixed_seg_but_one_five_lins_SSM_unfixed_segment_2")
		my_fixed_SSMs_2 = [
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 1
			]

		self.assertListEqual(fixed_SSMs_2, my_fixed_SSMs_2)
		self.assertEqual(start_index, 3)
		self.assertEqual(stop_index, 3)

	def test_create_fixed_Z_data(self):
	
		# test with 2 lineages
		input_file_name = "testdata/unittests/result_file_two_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_z_two_lins"

		model.create_fixed_file(lineages, cons.Z, test=True, 
			result_file_name=input_file_name, output_file=output_file_name)
		(fixed_values, start_index, stop_index) = oio.read_fixed_value_file(output_file_name)

		self.assertListEqual(fixed_values, [])
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)

		# test with 5 lineages 
		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_z_five_lins"
		
		model.create_fixed_file(lineages, cons.Z, test=True,
			result_file_name=input_file_name, output_file=output_file_name) 
		fixed_values = oio.read_fixed_value_file(output_file_name)[0]

		my_fixed_values = [
			1, 0, 1,
			0, 0,
			1
			]
		self.assertListEqual(fixed_values, my_fixed_values) 

		# test with 7 lineages
		input_file_name = "testdata/unittests/result_file_seven_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_z_seven_lins"
		
		model.create_fixed_file(lineages, cons.Z, test=True,
			result_file_name=input_file_name, output_file=output_file_name) 
		fixed_values = oio.read_fixed_value_file(output_file_name)[0]

		my_fixed_values = [
			0, 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0,
			0, 0,
			0
			]
		self.assertListEqual(fixed_values, my_fixed_values) 


	def test_create_fixed_SSMs_data(self):
		
		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_ssm_five_lins"
	
		model.create_fixed_file(lineages, cons.SSM, test=True,
			result_file_name=input_file_name, output_file=output_file_name)
		(fixed_values, start_index, stop_index) = oio.read_fixed_value_file(output_file_name)

		my_fixed_values = [
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 1,
			0, 1, 0, 0, 0
			]
		self.assertListEqual(fixed_values, my_fixed_values)
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)


	def test_create_fixed_SNPs_data(self):
		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_snp_five_lins"

		model.create_fixed_file(lineages, cons.SNP, test=True, 
			result_file_name=input_file_name, output_file=output_file_name)
		(fixed_values, start_index, stop_index) = oio.read_fixed_value_file(output_file_name)

		my_fixes_list = [
			1,
			0,
			1,
			0,

			0,
			1,
			0,
			0,

			0,
			0,
			0,
			1
			]
		self.assertListEqual(fixed_values, my_fixes_list)
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)

	def test_create_fixed_SSM_data_new(self):

		input_file_name = "testdata/unittests/result_file_two_lins_2"
		lineages = oio.read_result_file(input_file_name)

		fixed_ssm_list = model.create_fixed_SSM_data_new(lineages)

		my_fixed_ssm_list = [
			[0, 2, cons.B], [1, 1, cons.A], [2, 1, cons.UNPHASED], [3, 2, cons.UNPHASED], [4, 2, cons.B],
			[5, 2, cons.UNPHASED]
			]
		self.assertListEqual(fixed_ssm_list, my_fixed_ssm_list)


	def test_create_fixed_CNV_data_new(self):

		input_file_name = "testdata/unittests/result_file_two_lins_2"
		lineages = oio.read_result_file(input_file_name)

		fixed_cnv_list = model.create_fixed_CNV_data_new(lineages, 4)

		my_fixed_list = [
			[0, [[2, -1, cons.A], [2, -1, cons.B]]],
			[1, [[0, 0, cons.A]]],
			[2, [[1, 1, cons.A], [1, 1, cons.B], [2, 1, cons.A]]],
			[3, [[1, -1, cons.B]]]
			]
		self.assertListEqual(fixed_cnv_list, my_fixed_list)

	def test_create_fixed_CNV_data(self):

		# test with two lineages and only CN change of "+1"
		input_file_name = "testdata/unittests/result_file_two_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_cnv_two_lins"

		cnv_state_num = 1
		model.create_fixed_file(lineages, cons.CNV, cnv_state_num, test=True,
			result_file_name=input_file_name, output_file=output_file_name)
		(fixed_values, start_index, stop_index) = oio.read_fixed_value_file(output_file_name)

		my_fixed_list = [
			0, 0,
			1, 0,
			1, 0,
			0, 0,
			0, 1,
			0, 0,
			0, 0, 
			1, 0
			]
		self.assertListEqual(fixed_values, my_fixed_list)
		self.assertEqual(start_index, -1)
		self.assertEqual(stop_index, -1)

		# test with five lineages and CN changes of "+1" and "-1"
		input_file_name = "testdata/unittests/result_file_five_lins"
		lineages = oio.read_result_file(input_file_name)
		output_file_name = "testdata/unittests/fixed_cnv_five_lins"

		cnv_state_num = 2
		model.create_fixed_file(lineages, cons.CNV, cnv_state_num, test=True,
			result_file_name=input_file_name, output_file=output_file_name)
		fixed_values = oio.read_fixed_value_file(output_file_name)[0]

		my_fixed_list = [
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 1,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,

			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 0, 0, 0
			]
		self.assertListEqual(fixed_values, my_fixed_list)

	def test_get_mut_index_from_muts_indices_on_seg(self):
		muts_indices_on_seg = [[0, 1, 2], []]

		self.assertEqual(model.get_mut_index_from_muts_indices_on_seg(muts_indices_on_seg, 0, 0),
			0)
		self.assertEqual(model.get_mut_index_from_muts_indices_on_seg(muts_indices_on_seg, 0, -1),
			2)
		self.assertEqual(model.get_mut_index_from_muts_indices_on_seg(muts_indices_on_seg, 1, 0),
			-1)

	def test_get_mutation_index_list_for_segments(self):
		# SNPs are assigned in different order to segment
		# one segment is without SNPs
		snp1 = snp_ssm.SNP()
		snp1.seg_index = 2
		snp2 = snp_ssm.SNP()
		snp2.seg_index = 0
		snp3 = snp_ssm.SNP()
		snp3.seg_index = 2
		snp_list = [snp1, snp2, snp3]

		seg_num = 3

		self.assertListEqual(model.get_mutation_index_list_for_segments(snp_list, seg_num),
			[[1], [], [0, 2]])

	
	def test_compute_avg_cn_from_lineages_per_segment(self):
		
		lineages = oio.read_result_file("testdata/unittests/lineage_avg_cn")

		# no CN change, CN = 2
		self.assertEqual(model.compute_average_cn(lineages, 3)[0], 2)

		# CN change in last lineage, CN = 1.8
		self.assertEqual(model.compute_average_cn(lineages, 3)[1], 1.8)

		# several CN changes in several lineages, CN = 3.6
		self.assertAlmostEqual(model.compute_average_cn(lineages, 3)[2], 3.6)

	def test_compute_average_cn_difference(self):
		lineages = oio.read_result_file("testdata/unittests/lineage_avg_cn")
		lineages_2 = oio.read_result_file("testdata/unittests/lineage_avg_cn_2")

		self.assertEqual(0, model.compute_average_cn_difference(lineages, lineages, 3))
		self.assertAlmostEqual(2.0/3, model.compute_average_cn_difference(lineages, lineages_2, 3))

	def test_z_matrix_phis_feasible(self):
		# branching, phis too high
		direct_descendants = [[1, 2], []]
		fixed_phis = [0.7, 0.6]
		self.assertFalse(model.z_matrix_phis_feasible(direct_descendants, fixed_phis))

		# branching, phis okay
		direct_descendants = [[1, 2], []]
		fixed_phis = [0.7, 0.2]
		self.assertTrue(model.z_matrix_phis_feasible(direct_descendants, fixed_phis))

		# linear, no role of phis
		direct_descendants = [[1], [2]]
		fixed_phis = [0.7, 0.6]
		self.assertTrue(model.z_matrix_phis_feasible(direct_descendants, fixed_phis))

	def test_parse_direct_descendants_for_constraints(self):
		direct_descendants = [[1, 4], [2, 3], [5], [], [7, 8]]
		direct_descendants_for_constraints = model.parse_direct_descendants_for_constraints(
			direct_descendants)
		self.assertEqual([[0, 1, 4], [1, 2, 3], [4, 7, 8]], direct_descendants_for_constraints)

	def test_get_direct_descendants(self):
		z_matrix = [[1, 1, 1, 1], [1, 1, 1], [0, 1], [0]]
		sublin_num = 5

		descendant_list = model.get_direct_descendants(z_matrix, sublin_num)

		self.assertEqual([[1], [2, 3], [4], [], []], descendant_list)

	def test_get_quadratic_z_matrix(self):
		triangle_z_matrix = [[1, 1, 1], [1, 0], [0]]
		right_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		z_matrix = model.get_quadratic_z_matrix(triangle_z_matrix)
		self.assertEqual(z_matrix, right_matrix)

	def testparse_fixed_z_matrix_list_to_matrix(self):
		# test with 3 sublineages
		sublin_num = 3
		fixed_z_matrix_list = [1]

		z_matrix = model.parse_fixed_z_matrix_list_to_matrix(fixed_z_matrix_list, sublin_num)
		self.assertEqual([[1, 1], [1]], z_matrix)

		# test with 5 sublineages
		sublin_num = 5
		fixed_z_matrix_list = [1, 0, 0, 1, 1, 0]

		z_matrix = model.parse_fixed_z_matrix_list_to_matrix(fixed_z_matrix_list, sublin_num)
		self.assertEqual([[1, 1, 1, 1], [1, 0, 0], [1, 1], [0]], z_matrix)

	def test_change_labeling_of_lineage(self):

		# empty lists
		indices = []
		old_lin = []
		old_lin_true = []
		self.assertEqual(([], []), model.change_labeling_of_lineage(indices, old_lin, old_lin_true))

		empty_lineage = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])

		# true lineage list has more entries, old_lineage is in right order
		# true lineage has two entries, old lineage only one (can be seen via indicides)
		# new_lin: first entry like in old lineage, new entry: freq = 0, all other entries empty
		snp_0 = snp_ssm.SNP()
		snp_0.set_all_but_seg_index(1, 0, -1, -1)
		old_lin_0 = lineage.Lineage([], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin = [old_lin_0]
		old_lin_true_0 = lineage.Lineage([1], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin_true_1 = lineage.Lineage([], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin_true = [old_lin_true_0, old_lin_true_1]
		indices = [(0,0), (1,1)]
		(new_lin, new_lin_true) =  model.change_labeling_of_lineage(indices, old_lin, old_lin_true)
		self.assertEqual(2, len(new_lin))
		# add new lineage as a sublines in normal one
		self.assertNotEqual(old_lin_0, new_lin[0])
		old_lin_0.sublins.append(1)
		self.assertEqual(old_lin_0, new_lin[0])
		self.assertEqual(empty_lineage, new_lin[1])

		# true has more lineages, order is different
		old_lin_0 = lineage.Lineage([1, 2], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin_1 = lineage.Lineage([2], 0.8, [], [], [], [], [], [], [], [snp_0])
		old_lin_2 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [snp_0])
		old_lin = [old_lin_0, old_lin_1, old_lin_2]
		lin_true_0 = lineage.Lineage([1, 2, 3], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true_1 = lineage.Lineage([], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true = [lin_true_0, lin_true_1, lin_true_1, lin_true_1]
		indices = [(0, 0), (1, 2), (2, 3), (3, 1)]
		(new_lin, new_lin_true) = model.change_labeling_of_lineage(indices, old_lin, lin_true)
		self.assertEqual(4, len(new_lin))
		self.assertEqual(new_lin_true, lin_true)
		self.assertListEqual(new_lin[0].sublins, [3, 1, 2])
		self.assertListEqual(new_lin[1].sublins, [])
		self.assertListEqual(new_lin[2].sublins, [])
		self.assertListEqual(new_lin[3].sublins, [1])

		# same number of entries, old_lineage is in right order
		old_lin_1 = lineage.Lineage([], 0.5, [], [], [], [], [], [snp_0], [], [])
		old_lin = [old_lin_0, old_lin_1]
		indices = [(0,0), (1,1)]
		(new_lin, new_lin_true) = model.change_labeling_of_lineage(indices, old_lin, old_lin_true)
		self.assertEqual(2, len(new_lin))
		self.assertEqual(old_lin_0, new_lin[0])
		self.assertEqual(old_lin_1, new_lin[1])
	
		# same number of entries, old_lineage is not in righht order
		old_lin_3 = lineage.Lineage([1, 2, 3, 4], 1, [], [], [snp_0], [], [], [], [], [])
		old_lin_4 = lineage.Lineage([2, 3, 4], 0.8, [], [], [], [], [], [snp_0], [], [])
		old_lin_5 = lineage.Lineage([3, 4], 0.6, [], [], [], [], [], [snp_0], [snp_0], [])
		old_lin_6 = lineage.Lineage([4], 0.4, [], [], [], [], [], [], [snp_0], [])
		old_lin_7 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [snp_0])
		old_lin = [old_lin_3, old_lin_4, old_lin_5, old_lin_6, old_lin_7]
		old_lin_true_0.sublins = [1, 2, 3, 4]
		old_lin_true = [old_lin_true_0, old_lin_true_1, old_lin_true_1, old_lin_true_1, old_lin_true_1]
		indices = [(0,0), (1,4), (2,2), (3,1), (4,3)]
		(new_lin, new_lin_true) = model.change_labeling_of_lineage(indices, old_lin, old_lin_true)
		self.assertEqual(5, len(new_lin))
		self.assertTrue(old_lin_3.same_lineage_expect_sublins(new_lin[0]))
		self.assertEqual([3, 2, 4, 1], new_lin[0].sublins)
		self.assertTrue(old_lin_4.same_lineage_expect_sublins(new_lin[3]))
		self.assertEqual([2, 4, 1], new_lin[3].sublins)
		self.assertTrue(old_lin_5.same_lineage_expect_sublins(new_lin[2]))
		self.assertEqual([4, 1], new_lin[2].sublins)
		self.assertTrue(old_lin_6.same_lineage_expect_sublins(new_lin[4]))
		self.assertEqual([1], new_lin[4].sublins)
		self.assertTrue(old_lin_7.same_lineage_expect_sublins(new_lin[1]))
		self.assertEqual([], new_lin[1].sublins)

		# inferred has more lineages, right order
		old_lin_0 = lineage.Lineage([1, 2], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin_1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [snp_0])
		old_lin_2 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [snp_0])
		old_lin = [old_lin_0, old_lin_1, old_lin_2]
		lin_true_0 = lineage.Lineage([1], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true_1 = lineage.Lineage([], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true = [lin_true_0, lin_true_1]
		indices = [(0, 0), (1, 1), (2, 2)]
		(new_lin, new_lin_true) = model.change_labeling_of_lineage(indices, old_lin, lin_true)
		self.assertEqual(3, len(new_lin_true))
		self.assertEqual([1, 2], new_lin_true[0].sublins)
		self.assertListEqual(old_lin, new_lin)

		# inferred has more lineages, different order
		old_lin_0 = lineage.Lineage([1, 2, 3], 1, [], [], [], [], [], [], [], [snp_0])
		old_lin_1 = lineage.Lineage([2], 0.8, [], [], [], [], [], [], [], [snp_0])
		old_lin_2 = lineage.Lineage([3], 0.5, [], [], [], [], [], [], [], [snp_0])
		old_lin_3 = lineage.Lineage([], 0.3, [], [], [], [], [], [], [], [snp_0])
		old_lin = [old_lin_0, old_lin_1, old_lin_2, old_lin_3]
		lin_true_0 = lineage.Lineage([1], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true_1 = lineage.Lineage([], 1, [], [], [], [], [], [], [], [snp_0])
		lin_true = [lin_true_0, lin_true_1]
		indices = [(0, 0), (1, 3), (2, 1), (3, 2)]
		(new_lin, new_lin_true) = model.change_labeling_of_lineage(indices, old_lin, lin_true)
		self.assertEqual(4, len(new_lin_true))
		self.assertEqual([1, 2, 3], new_lin_true[0].sublins)
		self.assertListEqual(new_lin[0].sublins, [2, 3, 1])
		self.assertListEqual(new_lin[1].sublins, [])
		self.assertListEqual(new_lin[2].sublins, [3])
		self.assertListEqual(new_lin[3].sublins, [1])

	def test_compute_diff_between_lineages(self):
		test_lineages = oio.read_result_file("testdata/unittests/compute_diff_between_lineages_1")
		# test 1, both lineages are none
		# expected 0
		lineage1 = None
		lineage2 = None
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 0)

		# test 2, one existing empty lineage and None
		# expected 0
		lineage1 = test_lineages[0]
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 0)

		# test 2.5, None and one existing empty lineage
		# expected 0
		self.assertEqual(model.compute_diff_between_lineages(lineage2, lineage1), 0)

		# test 3, non-empty lineage and None
		# lineage1 = 8 mutations
		# expected 8
		lineage1 = test_lineages[1]
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 8)

		# test 3.5, None and non-empty lineage
		# lineage1 = 8 mutations
		# expected 8
		self.assertEqual(model.compute_diff_between_lineages(lineage2, lineage1), 8)

		# test 4, non-empty lineage and empty lineage
		# lineage1 = 8 mutations
		# expected 8
		lineage2 = test_lineages[0]
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 8)

		# test 4.5, empty lineage and non-empty lineage
		# lineage1 = 8 mutations
		# expected 8
		self.assertEqual(model.compute_diff_between_lineages(lineage2, lineage1), 8)

		# test 5, two non-emtpy lineages with no shared mutations
		# lineage1 = 8 mutations
		# lineage2 = 6 mutations(3 CNVS with different start, end or change value to lineage 1), 
		# 	SNPs are different
		# expected = 14
		lineage2 = test_lineages[2]
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 14)
		# test 5.5, two non-emtpy lineages with no shared mutations
		self.assertEqual(model.compute_diff_between_lineages(lineage2, lineage1), 14)

		# test 6 and 6.5, two non-empty lineages with shared mutations
		# lineage1 and lineage2 have 8 shared mutations(same mutations as in test 3) 
		# and different ssms(lineage1 4 and lineage2 3)
		# expected 7
		lineage1 = test_lineages[3]
		lineage2 = test_lineages[4]
		self.assertEqual(model.compute_diff_between_lineages(lineage1, lineage2), 7)
		self.assertEqual(model.compute_diff_between_lineages(lineage2, lineage1), 7)
 
	def test_compute_BAF(self):
		# no variance count
		variance_count = 0
		total = 1
		self.assertEqual(0, model.compute_BAF(variance_count, total))

		# no variance count, no total count
		variance_count = 0
		total = 0
		self.assertEqual(0, model.compute_BAF(variance_count, total))

		# no total count
		variance_count = 1
		total = 0
		with self.assertRaises(eo.BAFComputationException):
			model.compute_BAF(variance_count, total)

		# variance count > total
		variance_count = 2
		total = 1
		with self.assertRaises(eo.BAFComputationException):
			model.compute_BAF(variance_count, total)

		# variance count and total count
		variance_count = 1
		total = 2
		self.assertEqual(0.5, model.compute_BAF(variance_count, total))

	def test_compute_LogR(self):
		count = 1
		avg_count = 1
		self.assertEqual(0, model.compute_LogR(count, avg_count))

		count = 0
		avg_count = 1
		self.assertEqual(-np.log2(2), model.compute_LogR(count, avg_count))

	def test_check_result_file(self):
		import sys
		"""
		errors in resultfile 2:
		missing snp: 1,1,8

		missing ssms:1,1,6; 0,1,4

		missing cnv for segment 3,0,6
		duplicat of cnv +1,1,5,10
		"""
		# test file names
		test_segments = "testdata/unittests/checkresultfile_segment"
		test_snps = "testdata/unittests/checkresultfile_snp"
		test_ssms = "testdata/unittests/checkresultfile_ssm"
		correct_resultfile = "testdata/unittests/checkresultfile_resultfile1"
		incorrect_resultfile = "testdata/unittests/checkresultfile_resultfile2"

		# expected output
		# output 1 should be empty, because the result file matches the data
		output_1 = ""
		# all expected errors with error message
		output_2 = "Segment on chromosome 1 start: 5, end: 10 from file \"testdata/unittests/checkresultfile_segment\" has wrong CNVs in result-file \"testdata/unittests/checkresultfile_resultfile2\"\n\tCNV on chromosome 1 start: 5, end: 10, change: 1\n\tCNV on chromosome 1 start: 5, end: 10, change: 1\nSegment on chromosome 3 start: 0, end: 6 from file \"testdata/unittests/checkresultfile_segment\" has no CNV in result-file \"testdata/unittests/checkresultfile_resultfile2\"\nSNP on chromosome 1 at position 8 from file \"testdata/unittests/checkresultfile_snp\" not found in result-file \"testdata/unittests/checkresultfile_resultfile2\"\nSSM on chromosome 1 at position 4 from file \"testdata/unittests/checkresultfile_ssm\" not found in result-file \"testdata/unittests/checkresultfile_resultfile2\"\nSSM on chromosome 1 at position 6 from file \"testdata/unittests/checkresultfile_ssm\" not found in result-file \"testdata/unittests/checkresultfile_resultfile2\""

		try:
			# direct output in StringIO object
			out = StringIO()
			sys.stdout = out
			# run function
			model.check_result_file(correct_resultfile, test_segments, test_snps, test_ssms)
			# get output
			output = out.getvalue().strip()
			self.assertEqual(output, output_1)

			# reset output
			out = StringIO()
			sys.stdout = out
			model.check_result_file(incorrect_resultfile, test_segments, test_snps, test_ssms)
			output = out.getvalue().strip()
			self.assertEqual(output, output_2)

		# restore standard output
		finally:
			sys.stdout = sys.__stdout__


	def test_get_number_of_segments(self):
		
		list1 = [
			cnv.CNV(0, 0, 1, 1, 1),
			cnv.CNV(1, 1, 1, 2, 2)
			]
		list2 = [
			cnv.CNV(1, 1, 2, 2, 3),
			cnv.CNV(0, 0, 1, 1, 1),
			cnv.CNV(1, 1, 2, 3, 3)
			]

		self.assertEqual(model.get_number_of_segments(list1), 2)
		self.assertEqual(model.get_number_of_segments(list2), 2)

	def test_compare_results(self):

		# test function where found_ssm/cnv_percentage is calculated 
		# per lineage
		simulated_file = "testdata/unittests/out_result1"
		cplex_created_file = "testdata/unittests/out_result1_cplex"

		lineages_true = oio.read_result_file(simulated_file)
		lineages_cplex = oio.read_result_file(cplex_created_file)

		# compare the file with itself
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_true, lineages_true))
		self.assertEqual(freq_difference, 0.0)
		self.assertEqual(found_ssm_percentage, 1.0)
		self.assertEqual(found_cnv_percentage, 1.0)

		# compare with each other
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_true, lineages_cplex))

		# written files by myself
		test_file_1 = "testdata/unittests/lineage_for_test_compare_1"
		test_file_2 = "testdata/unittests/lineage_for_test_compare_2"
		lineages_test_1 = oio.read_result_file(test_file_1)
		lineages_test_2 = oio.read_result_file(test_file_2)

		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_1, lineages_test_2))
		self.assertAlmostEqual(freq_difference, 0.1)
		self.assertAlmostEqual(found_ssm_percentage, 0.75)
		self.assertAlmostEqual(found_cnv_percentage, 0.5)
	
		# test other way around
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_2, lineages_test_1))
		self.assertAlmostEqual(freq_difference, 0.1)
		self.assertAlmostEqual(found_ssm_percentage, 1.0)
		self.assertAlmostEqual(found_cnv_percentage, 1.0)

		# test function where found_ssm/cnv_percentage is calculated per overall 
		# number of mutation
		
		test_file_3 = "testdata/unittests/lineage_for_test_compare_3"
		lineages_test_3 = oio.read_result_file(test_file_3)

		# test with same number of lineages
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_3, lineages_test_3, percentage_per_mutation=True))
		self.assertEqual(freq_difference, 0)
		# true has more lineages than cplex created solution
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_3, lineages_test_2, percentage_per_mutation=True))
		self.assertAlmostEqual(freq_difference, 0.1)
		# true has less lineages than cplex created solution
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_2, lineages_test_3, percentage_per_mutation=True))
		self.assertAlmostEqual(freq_difference, 0.1)

		# test different scenarios for SSMs and CNVs
		# 2 SSMs in lin 1, 1 in lin 2
		# 2 CNVs in lin 1, 1 in lin 2
		test_file_4 = "testdata/unittests/lineage_for_test_compare_4"
		lineages_test_4 = oio.read_result_file(test_file_4)
		# all are found
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_4, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 1)
		self.assertEqual(found_cnv_percentage, 1)
		# 1 in lin 1 is found correctly, 1 in lin 2 (same for SSM/CNV)
		test_file_5 = "testdata/unittests/lineage_for_test_compare_5"
		lineages_test_5 = oio.read_result_file(test_file_5)
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_5, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 2.0/3)
		self.assertEqual(found_cnv_percentage, 2.0/3)
		# 1 lin 1 found correctly, others in new lin 3
		test_file_6 = "testdata/unittests/lineage_for_test_compare_6"
		lineages_test_6 = oio.read_result_file(test_file_6)
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_6, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 1.0/3)
		self.assertEqual(found_cnv_percentage, 1.0/3)
		# 2 are found correctly in 1, no 2 exists
		test_file_7 = "testdata/unittests/lineage_for_test_compare_7"
		lineages_test_7 = oio.read_result_file(test_file_7)
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_7, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 2.0/3)
		self.assertEqual(found_cnv_percentage, 2.0/3)
		# 2 are found correctly in 1, no 2 exists, but there's lineage with freq 0
		test_file_9 = "testdata/unittests/lineage_for_test_compare_9"
		lineages_test_9 = oio.read_result_file(test_file_9)
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_9, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 2.0/3)
		self.assertEqual(found_cnv_percentage, 2.0/3)
		# nothing found correctly
		test_file_8 = "testdata/unittests/lineage_for_test_compare_8"
		lineages_test_8 = oio.read_result_file(test_file_8)
		(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
			model.compare_results(lineages_test_4, lineages_test_8, percentage_per_mutation=True))
		self.assertEqual(found_ssm_percentage, 0)
		self.assertEqual(found_cnv_percentage, 0)


	# see not efrom 26.9.15, problem with z-matrix
	'''def test_refine_z_matrix(self):
		
		sublin_nums = [3]

		seg_overdispersion = 1000
		seg_points = 1000
		snp_overdispersion = 1000
		snp_points = 1000 
		ssm_overdispersion = 1000
		ssm_points = 1000
		
		# create mutations and segments
		seg_list = [segment.Segment(1, 0, 9, 260, 100), 
			segment.Segment(1, 10, 19, 230, 100)]
		snp1 = snp_ssm.SNP()
		snp1.set_all_but_seg_index(1, 0, 160, 100)
		snp1.seg_index = 0
		snp2 = snp_ssm.SNP()
		snp2.set_all_but_seg_index(1, 10, 130, 100)
		snp2.seg_index = 1
		snp_list = [snp1, snp2]
		ssm = snp_ssm.SSM()
		ssm.set_all_but_seg_index(1, 11, 30, 200)
		ssm.seg_index = 1
		ssm_list = [ssm]

		# create splines
		seg_splines = log_pdf.compute_piecewise_linear_for_seg_list(seg_list, seg_overdispersion,
			seg_points)
		snp_splines = log_pdf.compute_piecewise_linear_for_snp_ssm_list(snp_list, 
			snp_overdispersion, snp_points)
		ssm_splines = log_pdf.compute_piecewise_linear_for_snp_ssm_list(ssm_list,
			ssm_overdispersion, ssm_points)

		# optimize
		cplex_obj = optimization.Optimization_with_CPLEX(seg_splines, snp_splines, ssm_splines) 
		cplex_obj.opt_with_CPLEX(sublin_nums[0], snp_list, ssm_list, seg_list)

		# check if number of entries of variables were changed
		model.check_number_entries_z_matrix(cplex_obj)

		# set z_1_2 to right value of 0 and test
		cplex_obj.save_part_of_solution_in_class()
		self.assertEqual(cplex_obj.solution_z[2], 1.0)

		model.refine_z_matrix(cplex_obj, ssm_list)
		self.assertEqual(cplex_obj.solution_z[2], 0.0)'''



	def test_sort_segments(self):
		# create segments
		end = 0
		count = 0
		hm = 0
		s1 = segment.Segment(2, 2, end, count, hm)
		s2 = segment.Segment(2, 1, end, count, hm)
		s3 = segment.Segment(1, 1, end, count, hm)
		s4 = segment.Segment(1, 2, end, count, hm)
		list = [s1, s2, s3, s4]

		# test
		list = model.sort_segments(list)

		self.assertListEqual(list, [s3, s4, s2, s1])

	def test_sort_snps_ssms(self):
		# create mutations
		s1 = snp_ssm.SNP()
		s2 = snp_ssm.SNP()
		s3 = snp_ssm.SNP()
		s4 = snp_ssm.SNP()
		s1.chr = 2
		s1.pos = 2
		s2.chr = 2
		s2.pos = 1
		s3.chr = 1
		s3.pos = 1
		s4.chr = 1
		s4.pos = 2
		list = [s2, s3, s1, s4]

		# test
		list = model.sort_snps_ssms(list)

		self.assertListEqual(list, [s3, s4, s2, s1])

	def test_assign_muts_to_segments(self):
		# create segments and mutation
		count = 0
		hm = 0
		seg1 = segment.Segment(1, 1, 1, count, hm)
		seg2 = segment.Segment(2, 1, 2, count, hm)
		seg3 = segment.Segment(2, 6, 7, count, hm)
		seg_list = [seg1, seg2, seg3]
		mut1 = snp_ssm.SNP()
		mut1.chr = 2
		mut1.pos = 5
		mut_list = [mut1]

		# test, should not work
		with self.assertRaises(eo.SegmentAssignmentException):
			model.assign_muts_to_segments(seg_list, mut_list)

		# create other segmets and mutations
		seg4 = segment.Segment(2, 5, 6, count, hm)
		seg5 = segment.Segment(2, 7, 8, count, hm)
		seg6 = segment.Segment(2, 9, 11, count, hm)
		seg_list2 = [seg1, seg4, seg5, seg6]
		mut2 = snp_ssm.SNP()
		mut2.chr = 2
		mut2.pos = 5
		mut3 = snp_ssm.SNP()
		mut3.chr = 2
		mut3.pos = 8
		mut4 = snp_ssm.SNP()
		mut4.chr = 2
		mut4.pos = 10
		mut_list2 = [mut2, mut3, mut4]

		# test, should work for all 3 mutations
		model.assign_muts_to_segments(seg_list2, mut_list2)
		self.assertEqual(mut2.seg_index, 1)
		self.assertEqual(mut3.seg_index, 2)
		self.assertEqual(mut4.seg_index, 3)

		# allele-specific, no mutations in middle segment
		seg1 = segment.Segment_allele_specific(1, 1, 10, 1.0, 0.25, 1.0 ,0.25)
		seg1.index = 0
		seg2 = segment.Segment_allele_specific(1, 11, 20, 1.0, 0.25, 1.0 ,0.25)
		seg2.index = 1
		seg3 = segment.Segment_allele_specific(1, 21, 30, 1.0, 0.25, 1.0 ,0.25)
		seg3.index = 2
		seg_list = [seg1, seg2, seg3]

		ssm1 = snp_ssm.SSM()
		ssm1.chr = 1
		ssm1.pos = 5
		ssm1.variant_count = 20
		ssm1.variant_count = 30
		ssm2 = snp_ssm.SSM()
		ssm2.chr = 1
		ssm2.pos = 25
		ssm2.variant_count = 20
		ssm2.variant_count = 30
		ssm_list = [ssm1, ssm2]

		model.assign_muts_to_segments(seg_list, ssm_list)

		self.assertEqual(ssm_list[0].seg_index, 0)
		self.assertEqual(ssm_list[1].seg_index, 2)

	def test_snp_ssm_equality(self):
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 4
		snp2 = snp_ssm.SNP()
		snp2.chr = 1
		snp2.pos = 4
		# equality
		self.assertEqual(snp1, snp2)
		self.assertEqual(snp2, snp1)

		# difference in one attribute
		snp2.chr = 2
		self.assertNotEqual(snp1, snp2)
		self.assertNotEqual(snp2, snp1)
		snp2.chr = 1
		snp2.pos = 14
		self.assertNotEqual(snp1, snp2)
		self.assertNotEqual(snp2, snp1)

		# difference in both attributes
		snp2.chr = 12
		self.assertNotEqual(snp1, snp2)
		self.assertNotEqual(snp2, snp1)

	def test_snp_ssm_lt(self):
		snp1 = snp_ssm.SNP()
		snp1.chr = 1
		snp1.pos = 4
		snp2 = snp_ssm.SNP()
		snp2.chr = 1
		snp2.pos = 4
		# equality
		self.assertFalse(snp1 < snp2)
		self.assertFalse(snp2 < snp1)

		# difference in first attribute
		snp2.chr = 2
		self.assertTrue(snp1 < snp2)
		self.assertFalse(snp2 < snp1)

		# difference in second attribute
		snp2.chr = 1
		snp2.pos = 2
		self.assertFalse(snp1 < snp2)
		self.assertTrue(snp2 < snp1)

		# differences in both attributes
		# both attributes less than
		snp2.chr = 3
		snp2.pos = 823
		self.assertTrue(snp1 < snp2)
		self.assertFalse(snp2 < snp1)
		# one atribute less than, the other greater
		self.assertTrue(snp1 < snp2)
		self.assertFalse(snp2 < snp1)

		# chromosome is 0
		snp1.chr = 0
		snp1.pos = 10
		snp2.chr = 1
		snp2.pos = 0
		self.assertTrue(snp1 < snp2)
		self.assertFalse(snp2 < snp1)

def suite():
	 return unittest.TestLoader().loadTestsFromTestCase(ModelTest)
