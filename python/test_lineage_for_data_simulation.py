import unittest
import lineage_for_data_simulation as lineage
import constants as cons
import snp_ssm
import exceptions_onctopus as eo

class LineageSimulationClassTest(unittest.TestCase):
	
	def setUp(self):
		sublin = [0,1]
		freq = 0.8
		segment_number = 2
		self.lin = lineage.Lineage_Simulation(sublin, freq, segment_number)

	def test_add_SNP(self):
		phase = cons.A
		segment = 0
		mutation = cons.SNP
		#nothing is inserted
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 0)
		# insert one SNP
		self.lin.add_mutation_to_segment(phase, segment, mutation)
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 1)
		self.assertIsInstance(self.lin.segments[phase][segment][mutation][0], snp_ssm.SNP)

	def test_add_SSM(self):
		phase = cons.A
		segment = 0
		mutation = cons.SSM
		# nothing is inserted
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 0)
		# insert one SSM
		self.lin.add_mutation_to_segment(phase, segment, mutation)
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 1)
		self.assertIsInstance(self.lin.segments[phase][segment][mutation][0], snp_ssm.SSM)  

	def test_add_CNV(self):
		phase = cons.A
		other_phase = cons.B
		segment = 0
		mutation = cons.CNV
		state = "+1"
		# nothing is inserted
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 0) 
		# insert one CNV
		self.lin.add_mutation_to_segment(phase, segment, mutation, state)
		self.assertEqual(len(self.lin.segments[phase][segment][mutation]), 1)
		self.assertEqual(self.lin.segments[phase][segment][mutation][0], "+1")
		# try to insert second CNV
		with self.assertRaises(eo.AddingException):
			self.lin.add_mutation_to_segment(phase, segment, mutation)
		self.lin.add_mutation_to_segment(other_phase, segment, mutation, state)
		self.assertEqual(self.lin.segments[other_phase][segment][mutation][0], "+1")

	def test_SNP_count(self):
		phase = cons.A
		segment = 0
		mutation = cons.SNP
		#nothing is inserted
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 0)
		# insert one SNP
		self.lin.add_mutation_to_segment(phase, segment, mutation) 
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 1)

	def test_SSM_count(self):
		phase = cons.A
		segment = 0
		mutation = cons.SSM
		# nothing is inserted
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 0)
		# insert one SSM
		self.lin.add_mutation_to_segment(phase, segment, mutation)
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 1)

	def test_CNV_count(self):
		phase = cons.A
		segment = 0
		mutation = cons.CNV
		# nothing is inserted
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 0)
		# insert one CNV
		self.lin.add_mutation_to_segment(phase, segment, mutation)
		self.assertEqual(self.lin.get_mut_count(phase, segment, mutation), 1)

	
def suite():
	return unittest.TestLoader().loadTestsFromTestCase(LineageSimulationClassTest)
