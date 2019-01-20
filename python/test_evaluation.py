import unittest
import evaluation
import onctopus_io as oio
import numpy as np
import clustered_mutation_optimization as cmo
from sklearn.metrics import average_precision_score
import logging
import exceptions_onctopus as eo
import lineage

class EvaluationTest(unittest.TestCase):

	def test_compute_adr_matrix(self):
		lin1 = lineage.Lineage([1,2,3,4,5], 0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([2], 0, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([5], 0, [], [], [], [], [], [], [], [])
		lin6 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		my_lins = [lin1, lin2, lin3, lin4, lin5, lin6]

		lineage_assignment = [[0,1], [0,2], [0,3], [0,4], [0,5]]

		z_matrix = [
			[-1, 1, 1, 1, 1, 1],
			[-1, -1, 1, 0, -1, -1],
			[-1, -1, -1, 0, -1, -1],
			[-1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, 1],
			[-1, -1, -1, -1, -1, -1]
			]

		matrix = [
			[0, 1, 0.5, 0, 0],
			[0, 0, 0.5, 0, 0],
			[0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0]
			]

		self.assertEqual(matrix, evaluation.compute_adr_matrix(lineage_assignment, my_lins, z_matrix).tolist())
	

	def test_create_sublin_matrix(self):
	
		lin1 = lineage.Lineage([1,2], 0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([2], 0, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0, [], [], [], [], [], [], [], [])
		my_lins = [lin1, lin2, lin3]

		sublins = [
			[0, 1, 1],
			[0, 0, 1],
			[0, 0, 0]
			]

		self.assertEqual(sublins, evaluation.create_sublin_matrix(my_lins).tolist())
		

	def test_compute_auprc_over_all_trees(self):
		# phyloWGS, most basic computation, only one phyloWGS result 
		summary_result_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.summ.json"
		mutation_attribute_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.muts.json"
		mutation_assignment_path = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/phyloWGS_mutation_assignment"
		input_ssm_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.ssms"
		input_seg_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.segs"
		result_file_true = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.results"
		start_id = 0
		end_id = 0

		auprc, co_clustering_matrix_true, co_clustering_phyloWGS = evaluation.compute_auprc_over_all_trees(
			phyloWGS=True, summary_result_file=summary_result_file, mutation_attribute_file=mutation_attribute_file, 
			mutation_assignment_path=mutation_assignment_path, input_ssm_file=input_ssm_file, 
			input_seg_file=input_seg_file, result_file_true=result_file_true,
			start_id=start_id, end_id=end_id)

		self.assertEqual(auprc, 1.0)
		self.assertTrue(co_clustering_matrix_true.all() == co_clustering_phyloWGS.all())

		# phyloWGS, basic computation, two phyloWGS results 
		summary_result_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.summ.json"
		mutation_attribute_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.muts.json"
		mutation_assignment_path = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/phyloWGS_mutation_assignment"
		input_ssm_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.ssms"
		input_seg_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.segs"
		result_file_true = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.results"
		start_id = 0
		end_id = 1

		auprc, co_clustering_matrix_true, co_clustering_phyloWGS = evaluation.compute_auprc_over_all_trees(
			phyloWGS=True, summary_result_file=summary_result_file, mutation_attribute_file=mutation_attribute_file, 
			mutation_assignment_path=mutation_assignment_path, input_ssm_file=input_ssm_file, 
			input_seg_file=input_seg_file, result_file_true=result_file_true,
			start_id=start_id, end_id=end_id)

		self.assertTrue(auprc <  1.0)

		# phyloWGS, with given lineage number, two phyloWGS results
		# no reconstruction with given lineage exists
		summary_result_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.summ.json"
		mutation_attribute_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.muts.json"
		mutation_assignment_path = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/phyloWGS_mutation_assignment"
		input_ssm_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.ssms"
		input_seg_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.segs"
		result_file_true = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.results"
		start_id = 0
		end_id = 1
		right_lin_num = 7

		with self.assertRaises(eo.NoReconstructionWithGivenLineageNumber):
			auprc, co_clustering_matrix_true, co_clustering_phyloWGS = evaluation.compute_auprc_over_all_trees(
				phyloWGS=True, summary_result_file=summary_result_file, 
				mutation_attribute_file=mutation_attribute_file, 
				mutation_assignment_path=mutation_assignment_path, input_ssm_file=input_ssm_file, 
				input_seg_file=input_seg_file, result_file_true=result_file_true,
				start_id=start_id, end_id=end_id, right_lin_num=right_lin_num)

		# phyloWGS, two phyloWGS results, co-clustering matrix is written to file
		summary_result_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.summ.json"
		mutation_attribute_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.muts.json"
		mutation_assignment_path = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/phyloWGS_mutation_assignment"
		input_ssm_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.ssms"
		input_seg_file = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.segs"
		result_file_true = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.results"
		start_id = 0
		end_id = 0
		co_clus_matrix_prefix = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.phylowgs.co_clustering_matrix"

		# call for first tree
		termination = evaluation.compute_auprc_over_all_trees(
			phyloWGS=True, summary_result_file=summary_result_file, mutation_attribute_file=mutation_attribute_file, 
			mutation_assignment_path=mutation_assignment_path, input_ssm_file=input_ssm_file, 
			input_seg_file=input_seg_file, result_file_true=result_file_true,
			start_id=start_id, end_id=end_id, co_clus_matrix_prefix=co_clus_matrix_prefix, test=True)
		# call for second tree
		start_id = 1
		end_id = 1
		termination = evaluation.compute_auprc_over_all_trees(
			phyloWGS=True, summary_result_file=summary_result_file, mutation_attribute_file=mutation_attribute_file, 
			mutation_assignment_path=mutation_assignment_path, input_ssm_file=input_ssm_file, 
			input_seg_file=input_seg_file, result_file_true=result_file_true,
			start_id=start_id, end_id=end_id, co_clus_matrix_prefix=co_clus_matrix_prefix, test=True)

		with open("{0}_{1}_{2}".format(co_clus_matrix_prefix, start_id, end_id), 'rb') as f:
			co_clustering_matrix_pre = np.load(f)

		self.assertTrue(termination)
		self.assertEqual(len(co_clustering_matrix_pre), 6)

		# phyloWGS, try approach where co-clustering matrices are read from file
		result_file_true = "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.results"
		used_samples = 2
		already_build_matrices = ["/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.phylowgs.co_clustering_matrix_0_0", "/cluster/home/sunderm/projects2015-clonalreconstruction/python/parsing/data_5.onctopus.phylowgs.co_clustering_matrix_1_1"]

		auprc2, co_clustering_matrix_true2, co_clustering_phyloWGS2 = evaluation.compute_auprc_over_all_trees(
			phyloWGS=True, result_file_true=result_file_true, used_samples=used_samples,
			already_build_matrices=already_build_matrices)

		self.assertTrue(co_clustering_phyloWGS.all() == co_clustering_phyloWGS2.all())

	def test_compute_auprc_of_inferred_and_true_result_files(self):

		# given are two result files
		# in the inferred file, SSM 1 and 2, and SSM 3, 4 and 5 cluster together
		# in the true file, SSM 1 and 3, and SSM 2, 4 and 5 cluster together

		result_file_inferred = "testdata/unittests/test_evaluation_auprc_result_file_inferred"
		result_file_true = "testdata/unittests/test_evaluation_auprc_result_file_true"

		co_clustering_inferred = np.asarray([
			1, 1, 0, 0, 0,
			1, 1, 0, 0, 0,
			0, 0, 1, 1, 1, 
			0, 0, 1, 1, 1, 
			0, 0, 1, 1, 1]).reshape(5, 5)	
		co_clustering_true = np.asarray([
			1, 0, 1, 0, 0,
			0, 1, 0, 1, 1,
			1, 0, 1, 0, 0,
			0, 1, 0, 1, 1,
			0, 1, 0, 1, 1]).reshape(5, 5)


		(auprc, co_clustering_matrix_inferred, co_clustering_matrix_true) = (
			evaluation.compute_auprc_of_inferred_and_true_result_files(
			result_file_inferred, result_file_true))

		self.assertTrue((co_clustering_inferred==co_clustering_matrix_inferred).all())
		self.assertTrue((co_clustering_true==co_clustering_matrix_true).all())
		self.assertAlmostEqual(0.4, auprc)

	def test_compute_auprc_of_inferred_lineages_and_true_lineage_assignment(self):

		# 4 SSMs, 3 segments
		# segment 0 and 2 are used
		# segment 0 has 2 SSMs, segment 2 1 SSM
		# SSM1 and SSM3 are clustered together, so there are 2 super SSMs
		# in inferred solution, SSM1 and SSM3 are put to the same lineage
		# in the true solution SSM2 is put togethre with SSM3
		# only SSM2 and 3 are used for the evaluation
		
		ssm_input_file = "testdata/unittests/test_evaluation_auprc_ssm"
		seg_input_file = "testdata/unittests/test_evaluation_auprc_seg"
		sublin_num = 3
		normal_seg_indices = [0, 2]
		cluster_num = 2
		write_result_file = True
		result_file_name = "testdata/unittests/test_evaluation_auprc_result_file"
		ssm_subset_file = "testdata/unittests/test_evaluation_auprc_ssm_subset"
		test = True
		lineage_assignment_file = "testdata/unittests/test_evaluation_auprc_true_lineage_assignment"
		auprc_file = "testdata/unittests/test_evaluation_auprc_auprc"

		numeric_logging_info = getattr(logging, "DEBUG".upper(), None)
		logging.basicConfig(filename=auprc_file+".log", filemode='w', level=numeric_logging_info)

		cmo.cluster_ssms_normal_segs_optimization(ssm_input_file, seg_input_file, sublin_num,
			normal_seg_indices, cluster_num, 
			write_result_file=write_result_file, result_file_name=result_file_name, 
			test=test)

		(auprc, inferred_co_clustering_matrix, true_co_clustering_matrix) = (evaluation.
			compute_auprc_of_inferred_lineages_and_true_lineage_assignment(result_file_name, 
			seg_input_file, ssm_input_file, 
			normal_seg_indices, ssm_subset_file, lineage_assignment_file, auprc_file, test,
			write_to_file=True))

		# read file
		auprc_in_file = -1
		with open(auprc_file) as f:
			for line in f:
				auprc_in_file = float(line.rstrip("\n"))	
		
		self.assertEqual(1, auprc)
		self.assertEqual(auprc, auprc_in_file)
		self.assertEqual(inferred_co_clustering_matrix.size, 4)
		self.assertFalse((true_co_clustering_matrix==inferred_co_clustering_matrix).all())

	def test_co_clustering_matrices_to_avg_precision_score(self):
		true_matrix = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1],
			[1, 1, 1, 0], [0, 0, 0, 1]])
		inferred_matrix = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1],
			[0, 0, 1, 0], [0, 0, 1, 1]])

		y_true = np.asarray([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
		y_score = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
		avg_pre_score = average_precision_score(y_true, y_score)

		self.assertEqual(avg_pre_score, evaluation.
			co_clustering_matrices_to_avg_precision_score(true_matrix, inferred_matrix))

	def test_compute_co_clustering_matrix(self):
		lineage_assignment = np.array([0, 0, 1])
		x = np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]])

		co_clustering = evaluation.compute_co_clustering_matrix(lineage_assignment)
		self.assertTrue((x == co_clustering).all())

		#lineage_assignment = [0, 0, 1]
		#x = np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
		#my_matrix = np.asmatrix(x)

		#co_clustering = evaluation.compute_co_clustering_matrix(lineage_assignment)
		#self.assertTrue((my_matrix == co_clustering).all())

	def test_compute_min_bipartite_match(self):
		# expected matches (0,0), (1,2), (2,4), (3,1), (4,3) (5,5)
		indexes = [(0,0),(1,2),(2,4),(3,1),(4,3),(5,5)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_1", 
			"testdata/unittests/evaluation_test_2", verbose=False)
		tested_indexes = sorted(tested_indexes)
		self.assertEqual(tested_indexes, indexes)
		# other direction
		indexes = [(0,0),(1,3),(2,1),(3,4),(4,2),(5,5)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_2", 
			"testdata/unittests/evaluation_test_1", verbose=False)
		self.assertEqual(tested_indexes, indexes)

		# equal number of lineages
		indexes = [(0,0),(1,2),(2,4),(3,1),(4,3)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_1", 
			"testdata/unittests/evaluation_test_3", verbose=False)
		tested_indexes = sorted(tested_indexes)
		self.assertEqual(tested_indexes, indexes)
		# other direction
		indexes = [(0,0),(1,3),(2,1),(3,4),(4,2)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_3", 
			"testdata/unittests/evaluation_test_1", verbose=False)
		self.assertEqual(tested_indexes, indexes)

		# smaller files
		# same size
		indices = [(0,0), (1,2), (2,1)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_4",
			"testdata/unittests/evaluation_test_5", verbose=False)
		self.assertEqual(tested_indexes, indices)

		# different size
		indices = [(0,0), (1,1), (2,2)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_4",
			"testdata/unittests/evaluation_test_6", verbose=False)
		self.assertEqual(tested_indexes, indices)
		# other direction
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_6",
			"testdata/unittests/evaluation_test_4", verbose=False)
		self.assertEqual(tested_indexes, indices)

		# different size, other files
		indices = [(0,0), (1,2), (2,1)]
		(tested_indexes, matrix) = evaluation.compute_min_bipartite_match(
			"testdata/unittests/evaluation_test_5",
			"testdata/unittests/evaluation_test_4", verbose=False)
		self.assertEqual(tested_indexes, indices)





	def test_construct_assignment_matrix(self):
		lineage1 = oio.read_result_file("testdata/unittests/evaluation_test_1")
		lineage2 = oio.read_result_file("testdata/unittests/evaluation_test_2")
		# test for exact match in lineages and combinations of differences, some entries are equal 
		# for the other test
		matrix = [[4,16,12,15,12,12],[16,14,6,11,10,10],[10,8,4,5,2,4],[14,0,8,11,8,8],[13,11,7,6,7,5],
			[8,6,2,5,2,2]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage1, lineage2), matrix)
		# switch lineage1 and lineage2
		matrix = [[4,16,10,14,13,8],[16,14,8,0,11,6],[12,6,4,8,7,2],[15,11,5,11,6,5],[12,10,2,8,7,2],
			[12,10,4,8,5,2]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage2, lineage1), matrix)
		# equal number of lineages
		lineage2 = oio.read_result_file("testdata/unittests/evaluation_test_3")
		matrix = [[4,16,12,15,12],[16,14,6,11,10],[10,8,4,5,2],[14,0,8,11,8],[13,11,7,6,7]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage1, lineage2), matrix)
		# switch lineages
		matrix = [[4,16,10,14,13],[16,14,8,0,11],[12,6,4,8,7],[15,11,5,11,6],[12,10,2,8,7]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage2, lineage1), matrix)

		# smaller files
		# same size
		lineage1 = oio.read_result_file("testdata/unittests/evaluation_test_4")
		lineage2 = oio.read_result_file("testdata/unittests/evaluation_test_5")
		matrix = [[0, 2, 2], [2, 2, 0], [2, 0, 2]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage1, lineage2), matrix)
		# different size
		lineage2 = oio.read_result_file("testdata/unittests/evaluation_test_6")
		matrix = [[0, 2, 1], [2, 0, 1], [2, 2, 1]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage1, lineage2), matrix)
		# other direction
		matrix = [[0, 2, 2], [2, 0, 2], [1, 1, 1]]
		self.assertEqual(evaluation.construct_assignment_matrix(lineage2, lineage1), matrix)

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(EvaluationTest)

