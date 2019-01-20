import unittest
import snp_ssm
import segment
import log_pdf
import cnv
import lineage
import model
import constants as cons
import numpy as np
import exceptions_onctopus as eo

class LogPdfTest(unittest.TestCase):
	
	def test_find_piecewise_linear_new(self):

		# for CNS
		number_spline_points = 50
		xmin = 0
		xmax = 5.5
		npoints = 1000
		mutation_type = cons.CNV
		func = log_pdf.log_normal
		given_cn_A = 0.5
		standard_error_A = 0.001

		spline, max_point = log_pdf.find_piecewise_linear_new(number_spline_points, xmin, xmax, npoints, mutation_type,
			func, given_cn_A, standard_error_A)
		
		# spline contains no duplicated values
		duplicate_values = False
		for i in xrange(len(spline.get_knots())-1):
			if spline.get_knots()[i] == spline.get_knots()[i+1]:
				duplicate_values = True
				break
		self.assertFalse(duplicate_values)

		# spline contains maximum
		contains_max = False
		for i in xrange(len(spline.get_knots())):
			if spline.get_knots()[i] == max_point[0]:
				contains_max = True
				break
		self.assertTrue(contains_max)

		# starts with xmin and end with xmax
		self.assertEqual(spline.get_knots()[0], xmin)
		self.assertEqual(spline.get_knots()[-1], xmax)

		# spline is concave
		self.assertTrue(log_pdf.check_concavity(spline))


		# for SSMs
		number_spline_points = 50
		xmin = 0
		xmax = 1
		npoints = 1000
		mutation_type = cons.SSM
		func = log_pdf.beta_binomial
		V = 108
		D = 200
		s = 1000

		spline, max_point = log_pdf.find_piecewise_linear_new(number_spline_points, xmin, xmax, npoints, mutation_type,
			func, V, D, s)
		
		# spline contains no duplicated values
		duplicate_values = False
		for i in xrange(len(spline.get_knots())-1):
			if spline.get_knots()[i] == spline.get_knots()[i+1]:
				duplicate_values = True
				break
		self.assertFalse(duplicate_values)

		# spline contains maximum
		contains_max = False
		for i in xrange(len(spline.get_knots())):
			if spline.get_knots()[i] == max_point[0]:
				contains_max = True
				break
		self.assertTrue(contains_max)

		# starts with xmin and end with xmax
		self.assertAlmostEqual(spline.get_knots()[0], xmin, places=6)
		self.assertAlmostEqual(spline.get_knots()[-1], xmax, places=6)

		# spline is concave
		self.assertTrue(log_pdf.check_concavity(spline))

	def test_add_knots_to_right(self):

		# larger_interval: 1, end of list is multiple of 1
		xs = [4, 5]
		larger_interval = 1
		xmax = 8
		self.assertEqual(log_pdf.add_knots_to_right(xs, xmax, larger_interval), [4, 5, 6, 7, 8])

		# larger_interval: 1, end of list is not multiple of 1
		xs = [4, 5.6]
		larger_interval = 1
		xmax = 8
		self.assertEqual(log_pdf.add_knots_to_right(xs, xmax, larger_interval), [4, 5.6, 6, 7, 8])

		# larger_interval: 0.02, end of list is multiple of 0.02
		xs = [0.96]
		larger_interval = 0.02
		xmax = 1
		self.assertEqual(log_pdf.add_knots_to_right(xs, xmax, larger_interval), [0.96, 0.98, 1])

		# larger_interval: 0.02, end of list is not multiple of 0.02
		xs = [0.95]
		larger_interval = 0.02
		xmax = 1
		self.assertEqual(log_pdf.add_knots_to_right(xs, xmax, larger_interval), [0.95, 0.96, 0.98, 1])

		# larger_interval: 0.02, end of list is not multiple of 0.02 but veeeery close to it
		xs = [0.95, 0.95999999999]
		larger_interval = 0.02
		xmax = 1
		self.assertEqual(log_pdf.add_knots_to_right(xs, xmax, larger_interval), [0.95, 0.95999999999, 0.98, 1])

	def test_add_knots_to_left(self):

		# larger_interval: 1, start of list is multiple of 1
		xs = [2, 3]
		larger_interval = 1
		self.assertEqual(log_pdf.add_knots_to_left(xs, larger_interval), [0, 1, 2, 3])

		# start of list isn't multiple of larger_interval = 1
		xs = [2.1, 3]
		larger_interval = 1
		self.assertEqual(log_pdf.add_knots_to_left(xs, larger_interval), [0, 1, 2, 2.1, 3])

		# larger_interval = 0.02, start of list is multiple of 0.02
		xs = [0.1, 0.12]
		larger_interval = 0.02
		self.assertEqual(log_pdf.add_knots_to_left(xs, larger_interval), [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])

		# larger_interval = 0.02, start of list is not multiple of 0.02
		xs = [0.05]
		larger_interval = 0.02
		self.assertEqual(log_pdf.add_knots_to_left(xs, larger_interval), [0, 0.02, 0.04, 0.05])

		# larger_interval = 0.02, start of list is not multiple of 0.02 but veeeery close to it
		xs = [0.0400000001, 0.05]
		larger_interval = 0.02
		self.assertEqual(log_pdf.add_knots_to_left(xs, larger_interval), [0, 0.02, 0.0400000001, 0.05])

	def test_set_right_border_correct(self):
		# 1 is in list
		xs = [0.5, 0.8, 1, 1.2, 1.3]
		self.assertEqual(log_pdf.set_right_border_correct(xs), [0.5, 0.8, 1])

		# 1 is not in list
		xs = [0.5, 0.8, 1.1, 1.2, 1.3]
		self.assertEqual(log_pdf.set_right_border_correct(xs), [0.5, 0.8])
		
		# list ends with 1
		xs = [0.5, 0.8, 1]
		self.assertEqual(log_pdf.set_right_border_correct(xs), [0.5, 0.8, 1])

	def test_set_left_border_correct(self):
		# 0 is in list
		xs = [-2, -1, 0, 1, 2]
		self.assertEqual(log_pdf.set_left_border_correct(xs), [0, 1, 2])

		# 0 is not in list
		xs = [-2, -1, 0.3, 1, 2]
		self.assertEqual(log_pdf.set_left_border_correct(xs), [0.3, 1, 2])

		# list starts with 0
		xs = [0, 1, 2]
		self.assertEqual(log_pdf.set_left_border_correct(xs), [0, 1, 2])

	def test_compute_llh(self):
		# read segment and SSM file
		input_seg = "testdata/unittests/logpdf_compute_llh_seg"
		input_ssm = "testdata/unittests/logpdf_compute_llh_ssm"
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg, None, input_ssm, True)
		# set phase, CN influence of SSMs and lineage
		ssm_list[0].lineage = 1
		ssm_list[1].lineage = 1
		ssm_list[2].lineage = 1
		ssm_list[3].lineage = 1
		ssm_list[4].lineage = 1
		# create CNVs
		cnv_1 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		cnv_2 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		cnv_3 = cnv.CNV(change=-1, seg_index=1, chr=2, start=1, end=10)
		cnv_4 = cnv.CNV(change=-1, seg_index=1, chr=2, start=1, end=10)
		cnv_5 = cnv.CNV(change=1, seg_index=2, chr=3, start=1, end=10)
		cnv_6 = cnv.CNV(change=-1, seg_index=2, chr=3, start=1, end=10)
		# lineages
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([4, 5], 0.5, [cnv_1, cnv_5], [cnv_6], [], [], [], [],
			[ssm_list[0], ssm_list[1], ssm_list[2],	ssm_list[4]], [ssm_list[3]])
		lin2 = lineage.Lineage([], 0.3, [], [cnv_3], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.1, [cnv_2], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.05, [], [cnv_4], [], [], [], [], [], [])
		lins = [lin0, lin1, lin2, lin3, lin4, lin5]
		# phasing of SSMs is not known
		phasing_not_known = True
		overdispersion_parameter = 1000

		as_avg_cn_true_A = [1.6, 1.0, 1.5]
		as_avg_cn_true_B = [1.0, 0.65, 0.5]
		vaf_true = [110.0/260.0, 60.0/260.0, 50.0/165.0, 45.0/165.0, 50.0/200.0]
		true_llh_seg_a = sum([log_pdf.log_normal_complete_llh(as_avg_cn_true_A[i], seg_list[i].given_cn_A, 
			seg_list[i].standard_error_A) for i in xrange(len(seg_list))])
		true_llh_seg_b = sum([log_pdf.log_normal_complete_llh(as_avg_cn_true_B[i], seg_list[i].given_cn_B, 
			seg_list[i].standard_error_B) for i in xrange(len(seg_list))])
		true_llh_ssm = sum(log_pdf.compute_llh_ssms(vaf_true, ssm_list, overdispersion_parameter))

		llh = log_pdf.compute_llh(lins, ssm_list, seg_list, overdispersion_parameter=overdispersion_parameter, 
			phasing_not_known=phasing_not_known)

		self.assertAlmostEqual(llh, true_llh_seg_a+true_llh_seg_b+true_llh_ssm)

		# with known phasing
		ssm_list[0].infl_cnv_same_lin = True
		ssm_list[1].infl_cnv_same_lin = False
		ssm_list[2].infl_cnv_same_lin = False
		ssm_list[3].infl_cnv_same_lin = False
		ssm_list[4].infl_cnv_same_lin = False
		ssm_list[0].phase = cons.A
		ssm_list[1].phase = cons.A
		ssm_list[2].phase = cons.A
		ssm_list[3].phase = cons.B
		ssm_list[4].phase = cons.A
		llh = log_pdf.compute_llh(lins, ssm_list, seg_list, overdispersion_parameter=overdispersion_parameter, 
			phasing_not_known=False)

		self.assertAlmostEqual(llh, true_llh_seg_a+true_llh_seg_b+true_llh_ssm)



	def test_compute_average_SSM_cn(self):
		# CNV gains on segment 0
		cnv_1 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		cnv_2 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		cnv_3 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		# CNV gains on segment 1
		cnv_4 = cnv.CNV(change=1, seg_index=1, chr=2, start=1, end=10)
		cnv_5 = cnv.CNV(change=1, seg_index=1, chr=2, start=1, end=10)
		cnv_6 = cnv.CNV(change=1, seg_index=1, chr=2, start=1, end=10)
		# CN loss on segment 2
		cnv_7 = cnv.CNV(change=-1, seg_index=2, chr=3, start=1, end=10)
		# CN loss on segment 3
		cnv_8 = cnv.CNV(change=-1, seg_index=3, chr=4, start=1, end=10)
		# SSM on segment 0
		ssm_1 = snp_ssm.SSM()
		ssm_1.seg_index = 0
		ssm_1.phase = cons.A
		ssm_1.lineage = 1
		ssm_1.infl_cnv_same_lin = True
		# SSM on segment 1
		ssm_2 = snp_ssm.SSM()
		ssm_2.seg_index = 1
		ssm_2.phase = cons.B
		ssm_2.lineage = 1
		ssm_2.infl_cnv_same_lin = True
		# SSM on segment 2
		ssm_3 = snp_ssm.SSM()
		ssm_3.seg_index = 2
		ssm_3.phase = cons.A
		ssm_3.lineage = 1
		ssm_3.infl_cnv_same_lin = False
		# SSM on segment 3
		ssm_4 = snp_ssm.SSM()
		ssm_4.seg_index = 3
		ssm_4.phase = cons.B
		ssm_4.lineage = 1
		ssm_4.infl_cnv_same_lin = True
		# SSM on segment 0, unphased
		ssm_5 = snp_ssm.SSM()
		ssm_5.seg_index = 0
		ssm_5.phase = cons.UNPHASED
		ssm_5.lineage = 1
		ssm_5.infl_cnv_same_lin = False
		# SSM on segment 0, phased but not influenced by CN gains
		ssm_6 = snp_ssm.SSM()
		ssm_6.seg_index = 0
		ssm_6.phase = cons.A
		ssm_6.lineage = 1
		ssm_6.infl_cnv_same_lin = False
		# lineages
		lin_0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, [], [], [], [], [], [], [], [])
		lin_1 = lineage.Lineage([4, 5], 0.5, [cnv_1], [cnv_4, cnv_7], [], [], [], [ssm_5], [ssm_1, ssm_2, ssm_6], 
			[ssm_3, ssm_4])
		lin_2 = lineage.Lineage([], 0.3, [cnv_2], [cnv_5], [], [], [], [], [], [])
		lin_3 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		lin_4 = lineage.Lineage([], 0.1, [cnv_3], [cnv_8], [], [], [], [], [], [])
		lin_5 = lineage.Lineage([], 0.05, [], [cnv_6], [], [], [], [], [], [])
		lins = [lin_0, lin_1, lin_2, lin_3, lin_4, lin_5]
		# segment number
		seg_num = 4

		# build delta_C_A/B
		delta_C_A, delta_C_B = log_pdf.build_delta_C_from_reconstructions(lins, seg_num, phasing_not_known=False)
		# get Z-matrix
		z_matrix = model.get_Z_matrix(lins)[0]

		############## tests #######################################

		# average CN of unphased SSM 5
		avg_cn_5 = log_pdf.compute_average_SSM_cn(ssm_5, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_5, 0.5)

		# average CN of SSM 1, phased to A, on segment 0, influenced by change in same lineage 1
		# influenced by CN change in lineage 4
		avg_cn_1 = log_pdf.compute_average_SSM_cn(ssm_1, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_1, 1.1)

		# average CN of SSM 2, phased to B, on segment 1, influenced by change in same lineage 1
		# influenced by CN change in lineage 5
		avg_cn_2 = log_pdf.compute_average_SSM_cn(ssm_2, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_2, 1.05)

		# average CN of SSM 3, not influenced by CN loss in same lineae
		avg_cn_3 = log_pdf.compute_average_SSM_cn(ssm_3, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_3, 0.5)

		# average CN of SSM 4, influenced by CN loss in lineage 4
		avg_cn_4 = log_pdf.compute_average_SSM_cn(ssm_4, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_4, 0.4)

		# average CN of SSM 6, not influenced by CN gain in same lineage
		avg_cn_6 = log_pdf.compute_average_SSM_cn(ssm_6, lins, delta_C_A, delta_C_B, z_matrix)
		self.assertEqual(avg_cn_6, 0.6)



	def test_build_delta_C_from_reconstructions(self):
		cnv_1 = cnv.CNV(change=1, seg_index=0, chr=1, start=1, end=10)
		cnv_2 = cnv.CNV(change=1, seg_index=1, chr=1, start=11, end=20)
		cnv_3 = cnv.CNV(change=-1, seg_index=2, chr=2, start=1, end=10)
		cnv_4 = cnv.CNV(change=0, seg_index=3, chr=3, start=1, end=10)
		cnv_5 = cnv.CNV(change=0, seg_index=4, chr=4, start=1, end=10)
		lin1 = lineage.Lineage([], 0, [cnv_4, cnv_5], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0, [cnv_1], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0, [], [cnv_3], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0, [cnv_2], [], [], [], [], [], [], [])
		lineages = [lin1, lin2, lin3, lin4]
		seg_num = 5
		phasing_not_known = False

		true_delta_C_A = np.zeros(seg_num*len(lineages)).reshape(seg_num, len(lineages))
		true_delta_C_A[0][1] = 1
		true_delta_C_A[1][3] = 1
		true_delta_C_A[3][0] = 0
		true_delta_C_A[4][0] = 0
		true_delta_C_B = np.zeros(seg_num*len(lineages)).reshape(seg_num, len(lineages))
		true_delta_C_B[2][2] = -1
	
		delta_C_A, delta_C_B = log_pdf.build_delta_C_from_reconstructions(lineages, seg_num, phasing_not_known)
		self.assertTrue((delta_C_A == true_delta_C_A).all())
		self.assertTrue((delta_C_B == true_delta_C_B).all())

	def test_help_build_delta_C_from_reconstructions_allele(self):
		# phasing is known
		seg_num = 5
		lin_num = 4
		delta_C = np.zeros(seg_num*lin_num).reshape(seg_num, lin_num)
		lin_index = 3
		seg_index = 2
		# positive CN change
		my_cnv = cnv.CNV(change=1, seg_index=seg_index, chr=1, start=1, end=10)
		# negative CN change
		seg_index_2 = 3
		my_cnv_2 = cnv.CNV(change=-1, seg_index=seg_index_2, chr=1, start=1, end=10)
		phasing_not_known = False

		log_pdf.help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv, phasing_not_known)
		self.assertTrue(delta_C[seg_index][lin_index], 1)
		with self.assertRaises(eo.MyException):
			log_pdf.help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv, phasing_not_known)

		log_pdf.help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv_2, phasing_not_known)
		self.assertTrue(delta_C[seg_index_2][lin_index], -1)

		# phasing is not known
		seg_num = 5
		lin_num = 4
		delta_C = np.zeros(seg_num*lin_num).reshape(seg_num, lin_num)
		lin_index = 3
		seg_index = 2
		# positive CN change
		my_cnv = cnv.CNV(change=1, seg_index=seg_index, chr=1, start=1, end=10)
		# negative CN change
		my_cnv_2 = cnv.CNV(change=-1, seg_index=seg_index, chr=1, start=1, end=10)
		phasing_not_known = True

		log_pdf.help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv, phasing_not_known)
		self.assertTrue(delta_C[seg_index][lin_index], 1)
		with self.assertRaises(eo.MyException):
			log_pdf.help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv_2, phasing_not_known)

	def test_get_intervall_for_maxima(self):
		# for SNP/SSM, okay
		xmin = 0
		xmax = 1
		my_func = log_pdf.beta_binomial
		V = 50
		D = 100
		(new_min, new_max) = log_pdf.get_intervall_for_maxima(xmin, xmax, my_func, V, D)
		self.assertEqual(new_min, 0.45)
		self.assertEqual(new_max, 0.55)

		# for SNP/SSM, first interval too small
		xmin = 0.46
		xmax = 0.54
		my_func = log_pdf.beta_binomial
		V = 50
		D = 100
		(new_min, new_max) = log_pdf.get_intervall_for_maxima(xmin, xmax, my_func, V, D)
		self.assertEqual(new_min, 0.46)
		self.assertEqual(new_max, 0.54)

		# for segment, okay
		xmin = 0
		xmax = 200
		my_func = log_pdf.neg_binomial
		D = 100
		(new_min, new_max) = log_pdf.get_intervall_for_maxima(xmin, xmax, my_func, D)
		self.assertEqual(new_min, 95)
		self.assertEqual(new_max, 105)

	def test_check_and_change_boundary_elements(self):
		
		my_func = lambda x: x

		# boundary elements are ok
		xs = np.array([1.0, 2.0])
		ys = [1.0, 2.0]
		# xmin and xmax won't be changed
		xmin = xmax = 0
		(xmin, xmax) = log_pdf.check_and_change_boundary_elements(xmin, xmax, xs, ys, 0.5, my_func)
		self.assertTrue((xs == np.array([1, 2])).all())
		self.assertListEqual(ys, [1,2])
		self.assertEqual(xmin, 0)
		self.assertEqual(xmax, 0)

		# boundary elements are both -inf
		xmin = 0
		xmax = 1
		xs = np.array([0.0, 1.0])
		ys = [-float('Inf'), -float('Inf')]
		(xmin, xmax) = log_pdf.check_and_change_boundary_elements(xmin, xmax, xs, ys, 0.05, my_func)
		self.assertTrue((xs == np.array([0.05, 0.95])).all())
		self.assertListEqual(ys, [0.05, 0.95])
		self.assertEqual(xmin, 0.05)
		self.assertEqual(xmax, 0.95)

		# if small_value is too big at first, it gets increased until computation works
		xs = np.array([0.0, 7.0, 8.0])
		ys = [-float('Inf'), -float('Inf')]
		(xmin, xmax) = log_pdf.check_and_change_boundary_elements(xmin, xmax, xs, ys, 5, my_func)
		self.assertTrue((xs == np.array([0.625, 7.0, 7.375])).all())
		self.assertListEqual(ys, [0.625, 7.375])
		self.assertEqual(xmin, 0.625)
		self.assertEqual(xmax, 7.375)
		# when array consists only of two values, exception can be raised because boundary
		# 	computation doesn't work in this special case
		xs = np.array([0.0, 1.0])
		ys = [-float('Inf'), -float('Inf')]
		with self.assertRaises(eo.MyException):
			log_pdf.check_and_change_boundary_elements(xmin, xmax, xs, ys, 5, my_func)


	def test_transform_array(self):
		points = 50
		xmin = 10
		xmax = 20
		my_func = lambda x: x

		(xs, ys) = log_pdf.transform_array(points, xmin, xmax, my_func)
		self.assertEqual(xs[0], 10)
		self.assertEqual(xs[-1], 20)
		self.assertEqual(len(xs), points+1)

	def test_insert_max_point(self):
		# point inserted at beginning of array
		xs = np.array([2, 3])
		ys = np.array([2, 3])
		xs_max = 1
		ys_max = 4
		(xs, ys) = log_pdf.insert_max_point(xs, xs_max, ys, ys_max)
		self.assertTrue((xs == np.array([1, 2, 3])).all())
		self.assertTrue((ys == np.array([4, 2, 3])).all())

		# point inserted in middle of array
		xs = np.array([1, 2, 3])
		ys = np.array([1, 2, 3])
		xs_max = 2.5
		ys_max = 4
		(xs, ys) = log_pdf.insert_max_point(xs, xs_max, ys, ys_max)
		self.assertTrue((xs == np.array([1, 2, 2.5, 3])).all())
		self.assertTrue((ys == np.array([1, 2, 4, 3])).all())

		# point inserted at end of array
		xs = np.array([1, 2])
		ys = np.array([1, 2])
		xs_max = 3
		ys_max = 3
		(xs, ys) = log_pdf.insert_max_point(xs, xs_max, ys, ys_max)
		self.assertTrue((xs == np.array([1, 2, 3])).all()) 
		self.assertTrue((ys == np.array([1, 2, 3])).all())

		# point already in array
		xs = np.array([1, 2]) 
		ys = np.array([1, 2])
		xs_max = 1
		ys_max = 1
		(xs, ys) = log_pdf.insert_max_point(xs, xs_max, ys, ys_max)
		self.assertTrue((xs == np.array([1, 2])).all())
		self.assertTrue((ys == np.array([1, 2])).all())

	@unittest.skip("rewrite/not needed")
	def test_get_xmin_xmax_for_seg(self):
		# offset < 10, xmin < 0
		count = 0
		hm = 10
		self.assertTupleEqual(log_pdf.get_xmin_xmax_for_seg(count, hm), (0, 10))

		# offset > 100
		hm = 2000
		count = 200
		self.assertTupleEqual(log_pdf.get_xmin_xmax_for_seg(count, hm), (100, 300))

		# normal
		hm = 500
		count = 300
		self.assertTupleEqual(log_pdf.get_xmin_xmax_for_seg(count, hm), (250, 350))
	
	@unittest.skip("rewrite/not needed")
	def test_get_xmin_xmax_for_snp_ssm(self):
		# xmin < 0
		variant = 1
		total = 20
		(xmin, xmax) = log_pdf.get_xmin_xmax_for_snp_ssm(variant, total)
		self.assertEqual(xmin, 0)
		self.assertEqual(round(xmax,2), 0.15)

		# xmax > 1
		variant = 19
		total = 20
		(xmin, xmax) = log_pdf.get_xmin_xmax_for_snp_ssm(variant, total)
		self.assertEqual(round(xmin, 2), 0.85)
		self.assertEqual(xmax, 1)

		# normal
		variant = 10
		total = 20
		(xmin, xmax) = log_pdf.get_xmin_xmax_for_snp_ssm(variant, total)
		self.assertEqual(round(xmin, 2), 0.4)
		self.assertEqual(round(xmax, 2), 0.6)

	@unittest.skip("Think about how to test this properly")
	def test_compute_piecewise_linear_for_seg_list(self):
		# create variables
		count = 100
		hm = 100
		seg1 = segment.Segment(0, 0, 0, count, hm)
		seg_list = [seg1]
		seg_overdispersion = 1000
		seg_points = 1000

		seg_splines = log_pdf.compute_piecewise_linear_for_seg_list(seg_list, 
			seg_overdispersion, seg_points)

	@unittest.skip("Think about how to test this properly")
	def test_compute_piecewise_linear_for_snp_ssm_list(self):
		# create variables 
		s = snp_ssm.SNP()
		s.variant_count = 5
		s.ref_count = 5
		snp_list = [s]
		snp_overdispersion = 1000
		mut_points = 1000

		snp_splines = log_pdf.compute_piecewise_linear_for_snp_ssm_list(snp_list,
			snp_overdispersion, mut_points)

def suite(): 
	return unittest.TestLoader().loadTestsFromTestCase(LogPdfTest)
