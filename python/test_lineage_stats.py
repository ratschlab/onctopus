import unittest
import lineage_stats
import cnv
import lineage
import numpy as np

class LineageStatsTest(unittest.TestCase):

	def test_get_CN_change_differences(self):

		seg_num = 4

		cnv1 = cnv.CNV(1, 0, 1, 1, 10)
		cnv2 = cnv.CNV(-1, 0, 1, 1, 10)
		cnv3 = cnv.CNV(1, 1, 1, 11, 100)
		cnv4 = cnv.CNV(-1, 1, 1, 11, 100)
		cnv5 = cnv.CNV(1, 2, 1, 101, 120)

		lin0 = lineage.Lineage([1,2], 1, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.5, [cnv1], [cnv4], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.5, [cnv1, cnv4], [], [], [], [], [], [], [])
		true_lin = [lin0, lin1, lin2]

		lin3 = lineage.Lineage([1,2], 1, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.5, [cnv1, cnv5, cnv5], [cnv2], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.5, [cnv3], [cnv3], [], [], [], [], [], [])
		inf_lin = [lin3, lin4, lin5]

		gain_overestimation, gain_underestimation, loss_overestimation, loss_underestimation = (
			lineage_stats.get_CN_change_differences(true_lin, inf_lin, seg_num))

		self.assertEqual(gain_overestimation, 1)
		self.assertEqual(gain_underestimation, -0.25)
		self.assertEqual(loss_overestimation, 0.25)
		self.assertEqual(loss_underestimation, -0.5)


	def test_build_lineage_relationship_confusion_matrix(self):

		true_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, -1, 1, -1],
			[-1, -1, -1, 0, -1],
			[-1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1]
			]
		inferred_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, -1, -1],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1]
			]
		true_confusion_matrix = [
			[0, 1, 0],
			[1, 2, 1],
			[0, 1, 0]
			]
		true_confusion_matrix = np.array(true_confusion_matrix)

		computed_confusion_matrix = lineage_stats.build_lineage_relationship_confusion_matrix(
			true_matrix, inferred_matrix)
		self.assertTrue((true_confusion_matrix == computed_confusion_matrix).all())

	def test_create_matrix_with_ambiguities(self):

		m1 = [[-1, 1, 1], [-1, -1, -1], [1, 1, 1]]
		m2 = [[-1, 1, 1], [-1, -1, 1], [1, 1, 1]]
		m3 = [[-1, 1, 1], [1, 1, 1], [1, 1, 1]]
		m4 = [[-1, 1, 1], [-1, 1, 1], [1, 1, 1]]
		true_combination = np.array([[-1, 1, 1], [-1, 0, 1], [1, 1, 1]])
		cutoff_absence = 0.33
		cutoff_ambiguity = 0.66

		z_matrix_list = [m1, m2, m3, m4]

		computed_combination = lineage_stats.create_matrix_with_ambiguities(z_matrix_list, cutoff_absence, 
			cutoff_ambiguity)
		self.assertTrue((true_combination == computed_combination).all())


def suite():
	 return unittest.TestLoader().loadTestsFromTestCase(LineageStatsTest)
