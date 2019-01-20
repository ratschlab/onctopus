import unittest
import clustered_mutation_optimization as cmo
import onctopus_io as oio
import logging
import constants as cons
import main

class ClusterTest(unittest.TestCase):
	
	def setUp(self):
		numeric_logging_info = getattr(logging, "DEBUG".upper(), None)
		logging.basicConfig(filename="testdata/unittests/logger_clusterTest",
			filemode='w', level=numeric_logging_info)

	def test_compare_woClus_superSSMs(self):
		path = "testdata/unittests/"
		ssm_input_file = path + "test_clustering_superSSMs_ssms"
		seg_input_file = path + "test_clustering_superSSMs_seg"
		sublin_num = 3

		time = 1e+75
		threads = 1
		probing = 0
		emph_switch = 0
		coef_reduc = -1
		mipgap = 1e-04
		symmetry = 4
		strategy_file = 1
		workmem = 128
		workdir = "/scratch"
		treememory = 1e+75
		emphasis_memory = 0
		nodeselect = 1
		cplex_log_file = None
		number_spline_points = 50
	
		# run without clustering
		(lineages_no_clustering, cplex_object_no_clustering) = (
			main.go_onctopus(seg_input_file, None, ssm_input_file,
			sublin_num, None, time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
			strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect,
			cplex_log_file, number_spline_points,
			test_run=True, write_output_to_disk=False, allele_specific=True, use_super_SSMs=False,
			review_ambiguous_relations=False))

		# run with superSSMs
		(lineages_superSSMs, cplex_object_superSSMs) = (
			main.go_onctopus(seg_input_file, None, ssm_input_file,
			sublin_num, None, time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
			strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect,
			cplex_log_file, number_spline_points,
			test_run=True, write_output_to_disk=False, allele_specific=True, use_super_SSMs=True,
			cluster_SSM=True, review_ambiguous_relations=False))

		self.assertEqual(lineages_no_clustering, lineages_superSSMs)

	def test_start_cluster_heuristic_fixed_phi_pure_segmentwise_2(self):
		# tested with 3 segs, where seg 0 and 1 are normal
		# seg 2 has CN gain in lin 2
		# cluster num is too small therefore SSMs are assigned to same phase that should
		# be divided
		path = "testdata/unittests/"
		ssm_input_file = path + "test_start_cluster_heuristic_fixed_phi_pure_segmentwise_ssms_2"
		seg_input_file = path + "test_start_cluster_heuristic_fixed_phi_pure_segmentwise_segs_2"
		sublin_num = 3
		normal_seg_indices = [0, 1]
		cluster_num = 2
		result_file_name = (path + "test_start_cluster_heuristic_fixed_phi_"
			"pure_segmentwise_result_file_2")

		# method to test
		cmo.start_cluster_heuristic_fixed_phi_pure_segmentwise(ssm_input_file, seg_input_file,
			sublin_num, normal_seg_indices, cluster_num,
			test=True, write_result_file=True, 
			result_file_name=result_file_name)

		my_lineages = oio.read_result_file(result_file_name)
		# more SSMs are assigned to phase A because of clustering
		self.assertEqual(len(my_lineages[1].ssms_a), 3)

	def test_start_cluster_heuristic_fixed_phi_pure_segmentwise(self):
		# tested with 4 segments where seg 0 and 2 are normal
		# seg 1 has CN loss in lin 1
		# seg 3 has CN gain in lin 2
		path = "testdata/unittests/"
		ssm_input_file = path + "test_start_cluster_heuristic_fixed_phi_pure_segmentwise_ssms"
		seg_input_file = path + "test_start_cluster_heuristic_fixed_phi_pure_segmentwise_segs"
		sublin_num = 2
		normal_seg_indices = [0, 2]
		cluster_num = 4
		result_file_name = (path + "test_start_cluster_heuristic_fixed_phi_"
			"pure_segmentwise_result_file")

		# method to test
		cmo.start_cluster_heuristic_fixed_phi_pure_segmentwise(ssm_input_file, seg_input_file,
			sublin_num, normal_seg_indices, cluster_num,
			test=True, write_result_file=True, 
			result_file_name=result_file_name)

		my_lineages = oio.read_result_file(result_file_name)

		self.assertEqual(2, len(my_lineages))		

		###################################
		# with 3 lineages
		sublin_num = 3
		# method to test
		cmo.start_cluster_heuristic_fixed_phi_pure_segmentwise(ssm_input_file, seg_input_file,
			sublin_num, normal_seg_indices, cluster_num, 
			test=True, write_result_file=True, 
			result_file_name=result_file_name)

		my_lineages = oio.read_result_file(result_file_name)

		self.assertAlmostEqual(my_lineages[1].freq, 0.6)
		self.assertAlmostEqual(my_lineages[2].freq, 0.2)
		self.assertEqual(my_lineages[1].sublins, [2])
		self.assertEqual(len(my_lineages[0].cnvs_a), 2)
		self.assertEqual(len(my_lineages[1].cnvs_b), 1)
		self.assertEqual(len(my_lineages[2].cnvs_a), 1)
		self.assertEqual(len(my_lineages[1].ssms), 2)
		self.assertEqual(len(my_lineages[1].ssms_a), 5)
		self.assertEqual(len(my_lineages[1].ssms_b), 1)
		self.assertEqual(len(my_lineages[2].ssms), 5)
		self.assertEqual(len(my_lineages[2].ssms_a), 2)
		self.assertEqual(len(my_lineages[2].ssms_b), 0)

	def test_cluster_ssms_normal_segs_optimization(self):
		path = "testdata/unittests/"
		ssm_input_file = path + "test_cluster_ssms_normal_segs_optimization_ssms"
		seg_input_file = path + "test_cluster_ssms_normal_segs_optimization_segs"
		sublin_num = 3
		normal_seg_indices = [0, 2]
		cluster_num = 2
		allele_specific = True 
		number_spline_points = 50
		write_result_file= True
		result_file_name = path + "test_cluster_ssms_normal_segs_optimization_result_file"
		test = True

		# method to test
		(_, cplex_obj, bic1, _, _) = cmo.cluster_ssms_normal_segs_optimization(ssm_input_file, 
			seg_input_file, sublin_num,
			normal_seg_indices, 
			allele_specific=allele_specific, number_spline_points=number_spline_points,
			write_result_file=write_result_file, result_file_name=result_file_name,
			test=test)

		# read results
		my_lineages = oio.read_result_file(result_file_name)

		self.assertEqual(my_lineages[1].ssms[0].seg_index, 0)
		self.assertEqual(my_lineages[1].ssms[1].seg_index, 0)
		self.assertEqual(my_lineages[2].ssms[0].seg_index, 2)
		self.assertEqual(my_lineages[2].ssms[1].seg_index, 2)
		self.assertEqual(my_lineages[0].cnvs_a[0].start, 0)
		self.assertEqual(my_lineages[0].cnvs_a[1].start, 20)
		self.assertEqual(my_lineages[0].cnvs_a[1].seg_index, 2)

		# test BIC with a second scenario, where the sublin num is higher but doesn't bring a 
		# better likelihood
		sublin_num = 5
		write_result_file = False
		(_, cplex_obj, bic2, _, _,) = cmo.cluster_ssms_normal_segs_optimization(
			ssm_input_file, seg_input_file, sublin_num,
			normal_seg_indices, 
			allele_specific=allele_specific, number_spline_points=number_spline_points,
			write_result_file=write_result_file, result_file_name=result_file_name,
			test=test)

		self.assertTrue(bic1 > bic2)
		

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(ClusterTest)
