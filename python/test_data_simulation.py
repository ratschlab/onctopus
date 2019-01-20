import unittest
import data_simulation
import lineage_for_data_simulation as lin_class
import constants as cons
import snp_ssm
import exceptions_onctopus as eo

class DataSimulationTest(unittest.TestCase):

	def setUp(self):
		segment_number = 1
		noise = False
		self.freq0 = 1
		self.freq1 = 0.8
		sublin0 = [1]
		sublin1 = []
		lineages = [lin_class.Lineage_Simulation(sublin0, self.freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, self.freq1, segment_number)]
		self.sim = data_simulation.Data_Simulation()
		self.sim.noise = noise
		self.sim.segment_number = segment_number
		self.sim.lineages = lineages
		self.sim.new_version = True
		self.sim.allele_specific = False
		self.sim.clonal_cn_percentage = 0.75
		self.sim.p1_A_prop = 0.3125
		self.sim.p1_A_B_prop = 0.125
		self.sim.m1_B_prop = 0.3125
		self.sim.m1_A_B_prop = 0.21875
		self.sim.p1_m1_prop = 0.03125
		self.sim.SSM_before_CNV_LH = 0.5

	def test_choose_lineage_by_frequency(self):
		freq0 = 1
		freq1 = 0.8
		freq2 = 0.6
		freq3 = 0.3
		lins = [lin_class.Lineage_Simulation([], freq0, 1), lin_class.Lineage_Simulation([], freq1, 1),
			lin_class.Lineage_Simulation([], freq2, 1), lin_class.Lineage_Simulation([], freq3, 1)]
		self.sim.lineages = lins
		self.sim.frequency_table = [0.8, 1.4, 1.7]

		self.sim.choose_lineage_by_frequency()
		self.assertEqual(self.sim.choose_lineage_by_frequency(0.5), 1)
		self.assertEqual(self.sim.choose_lineage_by_frequency(0.8), 1)
		self.assertEqual(self.sim.choose_lineage_by_frequency(0.81), 2)
		self.assertEqual(self.sim.choose_lineage_by_frequency(1.4), 2)
		self.assertEqual(self.sim.choose_lineage_by_frequency(1.41), 3)
		self.assertEqual(self.sim.choose_lineage_by_frequency(1.7), 3)

	def test_get_frequency_table(self):
		freq0 = 1
		freq1 = 0.8
		freq2 = 0.6
		freq3 = 0.3
		lins = [lin_class.Lineage_Simulation([], freq0, 1), lin_class.Lineage_Simulation([], freq1, 1),
			lin_class.Lineage_Simulation([], freq2, 1), lin_class.Lineage_Simulation([], freq3, 1)]
		self.sim.lineages = lins

		frequency_table = self.sim.get_frequency_table()

		self.assertEqual(frequency_table, [0.8, 1.4, 1.7])

	def test_add_x_SSMs_per_unit(self):
		self.sim.lineages[1].add_mutation_to_segment(cons.A, 0, cons.CNV, state="-1")
		self.sim.add_x_SSMs_per_unit(2)

		self.assertEqual(0, self.sim.lineages[1].get_mut_count(cons.A, 0, cons.SSM))
		self.assertEqual(2, self.sim.lineages[1].get_mut_count(cons.B, 0, cons.SSM))
		self.assertEqual(2, self.sim.ssm_number)

	def test_check_SSM_assignment(self):
		seg_index = 0
		phase = cons.B

		# SSM assignment to chromatid that is lost
		self.sim.lineages[1].add_mutation_to_segment(phase, seg_index, cons.CNV, state="-1") 
		self.sim.lineages[1].add_mutation_to_segment(phase, seg_index, cons.SSM) 
		with self.assertRaises(eo.SSMAssignmentException):
			self.sim.check_SSM_assignment()

		# correct SSM assignment
		self.sim.lineages[1].segments[phase][seg_index][cons.CNV] = []
		self.assertTrue(self.sim.check_SSM_assignment())

	def test_is_chromatid_lost(self):
		freq = 0
		seg_num = 1
		seg_index = 0
		lin_index = 5
		phase = cons.B
		lin0 = lin_class.Lineage_Simulation([1, 2, 3, 4, 5], freq, seg_num)
		lin1 = lin_class.Lineage_Simulation([2, 3, 5], freq, seg_num)
		lin2 = lin_class.Lineage_Simulation([3, 5], freq, seg_num)
		lin3 = lin_class.Lineage_Simulation([5], freq, seg_num)
		lin4 = lin_class.Lineage_Simulation([], freq, seg_num)
		lin5 = lin_class.Lineage_Simulation([], freq, seg_num)
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5]

		self.sim.lineages = my_lineages
		self.sim.construct_ancestral_lineage_list()

		# no chromatid is lost
		self.assertFalse(self.sim.is_chromatid_lost(lin_index, seg_index, phase))

		# chromatid lost in ancestor 1 of lineage 5
		self.sim.lineages[1].add_mutation_to_segment(phase, seg_index, cons.CNV, state="-1")
		self.assertTrue(self.sim.is_chromatid_lost(lin_index, seg_index, phase))

		# chromatid lost in ancestor 3 of lineage 5
		self.sim.lineages[1].segments[phase][seg_index][cons.CNV] = []
		self.sim.lineages[3].add_mutation_to_segment(phase, seg_index, cons.CNV, state="-1")
		self.assertTrue(self.sim.is_chromatid_lost(lin_index, seg_index, phase))

		# chromatid lost in lineage 4, not an ancestor of lineage 5
		self.sim.lineages[3].segments[phase][seg_index][cons.CNV] = []
		self.sim.lineages[4].add_mutation_to_segment(phase, seg_index, cons.CNV, state="-1")
		self.assertFalse(self.sim.is_chromatid_lost(lin_index, seg_index, phase))

		# chromatid is lost in lineage 5 itself
		self.sim.lineages[4].segments[phase][seg_index][cons.CNV] = []
		self.sim.lineages[5].add_mutation_to_segment(phase, seg_index, cons.CNV, state="-1")
		self.assertTrue(self.sim.is_chromatid_lost(lin_index, seg_index, phase))

	def test_construct_ancestral_lineage_list(self):
		freq = 0
		seg_num = 1
		lin0 = lin_class.Lineage_Simulation([1, 2, 3, 4], freq, seg_num)
		lin1 = lin_class.Lineage_Simulation([2, 4], freq, seg_num)
		lin2 = lin_class.Lineage_Simulation([4], freq, seg_num)
		lin3 = lin_class.Lineage_Simulation([], freq, seg_num)
		lin4 = lin_class.Lineage_Simulation([], freq, seg_num)
		my_lineages = [lin0, lin1, lin2, lin3, lin4]

		self.sim.lineages = my_lineages

		# function to test
		self.sim.construct_ancestral_lineage_list()

		self.assertEqual([], self.sim.lineages[1].ancestral_lineages)
		self.assertEqual([1], self.sim.lineages[2].ancestral_lineages)
		self.assertEqual([], self.sim.lineages[3].ancestral_lineages)
		self.assertEqual([1, 2], self.sim.lineages[4].ancestral_lineages)

	def test_compute_frequency_with_sublins(self):
		# new lineages object because I need 3 lineages
		freq0 = 1
		freq1 = 0.8
		freq2 = 0.5
		sublin0 = [1, 2]
		sublin1 = [2]
		sublin2 = []
		segment_number = 1
		lineages = [lin_class.Lineage_Simulation(sublin0, freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, freq1, segment_number),
			lin_class.Lineage_Simulation(sublin2, freq2, segment_number)]
		sim = data_simulation.Data_Simulation()
		sim.noise = False
		sim.segment_number = segment_number
		sim.lineages = lineages
		sim.new_version = True

		phase = cons.A
		segment_index = 0
		lineage_index = 1

		# compute frequency, no CN changes occur
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, lineage_index,
			freq1), freq1)

		# compute frequency, CN duplication happens in same lineage (#TODO: rule of thumb)
		sim.lineages[lineage_index].segments[phase][segment_index][cons.CNV] = ["+1"]
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, lineage_index,
			freq1), freq1)

		sim.lineages[lineage_index].segments[phase][segment_index][cons.CNV] = []

		# compute frequency, CN duplication in lineage after current lineage
		sim.lineages[lineage_index + 1].segments[phase][segment_index][cons.CNV] = ["+1"]
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, lineage_index,
			freq1), freq1 + freq2)

		sim.lineages[lineage_index + 1].segments[phase][segment_index][cons.CNV] = [] 

		# compute frequency, two CN duplications in lineage after current lineage
		sim.lineages[1].segments[phase][segment_index][cons.CNV] = ["+1"]
		sim.lineages[2].segments[phase][segment_index][cons.CNV] = ["+1"]
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, 0,
			freq0), freq0 + freq1 + freq2)

		sim.lineages[1].segments[phase][segment_index][cons.CNV] = []
		sim.lineages[2].segments[phase][segment_index][cons.CNV] = []

		# compute frequency, CN loss in same lineage
		sim.lineages[lineage_index].segments[phase][segment_index][cons.CNV] = ["-1"]
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, lineage_index,
			freq1), 0)

		sim.lineages[lineage_index].segments[phase][segment_index][cons.CNV] = []

		# compute frequency, CN loss in lineage after current lineage
		sim.lineages[lineage_index + 1].segments[phase][segment_index][cons.CNV] = ["-1"]
		self.assertEqual(sim.compute_frequency_with_sublins(phase, segment_index, lineage_index,
			freq1), freq1 - freq2)


	def test_get_haploid_sequencing_mass(self):
		segment_number = 1
		self.freq0 = 1
		self.freq1 = 0.5
		sublin0 = [1]
		sublin1 = []
		lineages = [lin_class.Lineage_Simulation(sublin0, self.freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, self.freq1, segment_number)]
		my_simulation = data_simulation.Data_Simulation()
		my_simulation.segment_number = segment_number
		my_simulation.lineages = lineages

		my_simulation.lineages[1].segments[cons.A][0][cons.CNV] = ["+1"]
		reads_total = [25]
		haploid = my_simulation.get_haploid_sequencing_mass(reads_total)
		self.assertEqual(haploid[0], 10)


	def test_get_average_coverage(self):
		segment_number = 3
		freq0 = 1
		freq1 = 0.8 
		sublin0 = [1]
		sublin1 = []
		lineages = [lin_class.Lineage_Simulation(sublin0, self.freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, self.freq1, segment_number)] 
		my_simulation = data_simulation.Data_Simulation() 
		my_simulation.segment_number = segment_number
		my_simulation.lineages = lineages

		# SNPs
		snp_3 = snp_ssm.SNP()
		snp_3.variant_count = 3
		snp_3.ref_count = 3
		snp_5 = snp_ssm.SNP()
		snp_5.variant_count = 5
		snp_5.ref_count = 5
		# add SNPs to segment
		my_simulation.lineages[cons.NORMAL].segments[cons.A][0][cons.SNP] = [snp_3, snp_5]
		my_simulation.lineages[cons.NORMAL].segments[cons.B][0][cons.SNP] = [snp_3, snp_5]
		my_simulation.lineages[cons.NORMAL].segments[cons.A][1][cons.SNP] = [snp_3]
		
		average_coverage = my_simulation.get_average_coverage()
		self.assertListEqual(average_coverage, [8, 6, 0])


	def test_choose_segment(self):
		self.assertEqual(self.sim.choose_segment(), 0)

	def test_choose_phase(self):
		pha = self.sim.choose_phase()
		self.assertGreaterEqual(pha, 0)
		self.assertLessEqual(pha, 1)

	def test_choose_lineage_but_normal(self):
		self.assertEqual(self.sim.choose_lineage_but_normal(), 1)

	def test_add_SNP_to_normal_lineage(self):
		self.sim.add_SNP_to_normal_lineage()
		
		added_a = (len(self.sim.lineages[cons.NORMAL].segments[
			cons.A][0][cons.SNP]) == 1) 
		added_b = (len(self.sim.lineages[cons.NORMAL].segments[
			cons.B][0][cons.SNP]) == 1) 
		self.assertTrue(added_a or added_b)

	def test_add_SSM(self):
		self.sim.addSSMsAccoringToFreqs = False
		self.sim.clonal_ssm_percentage = -1
		self.sim.add_SSM()
		added_a = (len(self.sim.lineages[1].segments[
			cons.A][0][cons.SSM]) == 1) 
		added_b = (len(self.sim.lineages[1].segments[
			cons.B][0][cons.SSM]) == 1) 
		self.assertTrue(added_a or added_b)

	def test_add_CNV_according_assignment(self):
		assignment = [[1, cons.A, 0, "+1"]]
		self.sim.add_CNV_according_assignment(assignment)
		self.assertTrue(self.sim.lineages[1].segments[cons.A][0][cons.CNV], ["+1"])

		assignment = [[1, cons.B, 0, "-1"]]
		self.sim.add_CNV_according_assignment(assignment)
		self.assertTrue(self.sim.lineages[1].segments[cons.B][0][cons.CNV], ["-1"])

		assignment = [[0, cons.A, 0, "0"]]
		self.sim.add_CNV_according_assignment(assignment)
		self.assertTrue(self.sim.lineages[0].segments[cons.A][0][cons.CNV], ["0"])

	def test_add_CNV(self):
		# add CNV
		segment = 0
		self.sim.cnv_number = 1
		self.sim.add_CNV()

		p1_added_A = (self.sim.lineages[1].segments[cons.A][segment][cons.CNV] == ["+1"])
		p1_added_B = (self.sim.lineages[1].segments[cons.B][segment][cons.CNV] == ["+1"])
		m1_added_A = (self.sim.lineages[1].segments[cons.A][segment][cons.CNV] == ["-1"])
		m1_added_B = (self.sim.lineages[1].segments[cons.B][segment][cons.CNV] == ["-1"])

		p1 = (p1_added_A and not p1_added_B and not m1_added_A and not m1_added_B)
		m1 = (not p1_added_A and not p1_added_B and not m1_added_A and m1_added_B)
		p1_m1 = (p1_added_A and not p1_added_B and not m1_added_A and m1_added_B)
		p1_p1 = (p1_added_A and p1_added_B)
		m1_m1 = (m1_added_A and m1_added_B)

		self.assertTrue((p1 or m1 or p1_m1 or p1_p1 or m1_m1))

		## try to add second CNV
		#self.sim.add_CNV(segment)
		#A_added = (len(self.sim.lineages[1].segments[cons.A][segment][cons.CNV]) == 1)
		#B_added = (len(self.sim.lineages[1].segments[cons.B][segment][cons.CNV]) == 1)
		#self.assertTrue((A_added or B_added) and (not(A_added and B_added)))
		
	def test_get_mut_count_per_segment(self):
		# try empty
		segment = 0
		self.assertEqual(self.sim.get_mut_count_per_segment(segment, cons.SNP), 0)
		
		# insert one SNP and three SSMs to different phases
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.sim.lineages[1].add_mutation_to_segment(cons.A, segment, cons.SSM)
		self.sim.lineages[1].add_mutation_to_segment(cons.A, segment, cons.SSM)
		self.sim.lineages[1].add_mutation_to_segment(cons.B, segment, cons.SSM)
		self.assertEqual(self.sim.get_mut_count_per_segment(segment, cons.SNP), 1)
		self.assertEqual(self.sim.get_mut_count_per_segment(segment, cons.SSM), 3)

	def test_get_highest_single_mut_count_per_segment(self):
		segment = 0

		# SSM count is higher
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.sim.lineages[1].add_mutation_to_segment(cons.A, segment, cons.SSM)
		self.sim.lineages[1].add_mutation_to_segment(cons.A, segment, cons.SSM)
		self.assertEqual(self.sim.get_highest_single_mut_count_per_segment(segment), 2)

		# SNP count is higher
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.assertEqual(self.sim.get_highest_single_mut_count_per_segment(segment), 3)

	def test_compute_start_end_pos_of_one_segment(self):
		segment = 0

		# is first segment, no mutations in segment
		self.assertTupleEqual(self.sim.compute_start_end_pos_of_one_segment(segment, -1), (0, 999))

		# first segment, one mutation
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.assertTupleEqual(self.sim.compute_start_end_pos_of_one_segment(segment, -1), (0, 1999))

		# not first segment, two mutations
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, segment, cons.SNP)
		self.assertTupleEqual(self.sim.compute_start_end_pos_of_one_segment(segment, 0), (1, 3000))

	def test_compute_start_end_pos_of_all_segments(self):
		# test with two segments, each one position long
		segment_number = 2
		noise = False
		self.freq0 = 1
		self.freq1 = 0.8
		sublin0 = [1]
		sublin1 = []
		lineages = [lin_class.Lineage_Simulation(sublin0, self.freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, self.freq1, segment_number)]
		self.sim = data_simulation.Data_Simulation()
		self.sim.noise = noise
		self.sim.segment_number = 2
		self.sim.lineages = lineages
		
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, 0, cons.SNP)	
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(cons.A, 1, cons.SNP)	
		
		(seg_start, seg_end) = self.sim.compute_start_end_pos_of_all_segments()
		self.assertEqual(len(seg_start), 2)
		self.assertEqual(seg_start[0], 1)
		self.assertEqual(seg_end[0], 2000)
		self.assertEqual(seg_start[1], 2001)
		self.assertEqual(seg_end[1], 4000)

	def test_compute_phased_segment_read_number_float(self):
		segment = 0
		pha = cons.A
		lin = cons.NORMAL
		freq = self.sim.lineages[lin].freq
		self.sim.mass = 200
		
		# nothing inserted
		self.assertEqual(self.sim.compute_phased_segment_read_number_float(pha, segment, lin, freq), 
			1 * self.sim.mass)

		# +1 inserted
		self.sim.lineages[1].segments[pha][segment][cons.CNV] = ["+1"]
		self.assertEqual(self.sim.compute_phased_segment_read_number_float(pha, segment, lin, freq), 
			(1 + self.sim.lineages[1].freq) * self.sim.mass)

		# -1 inserted
		self.sim.lineages[1].segments[pha][segment][cons.CNV] = ["-1"]
		self.assertEqual(self.sim.compute_phased_segment_read_number_float(pha, segment, lin, freq),
			(1 - self.sim.lineages[1].freq) * self.sim.mass)

	@unittest.skip("not yet written")
	def test_add_noise(self):
		read_count = 1
		self.sim.add_noise(read_count)

	@unittest.skip("think how to handle noise case")
	def test_compute_ref_count(self):
		mut_count = 6
		total_count = 10
		
		# without noise
		self.assertEqual(self.sim.compute_ref_count(mut_count, total_count), total_count - mut_count)
		
		# with noise
		self.sim.noise = True
		with self.assertRaises(Exception):
			self.sim.compute_ref_count(mut_count, total_count)

	def test_compute_total_reads(self):

		# total number of reads when no CNV exists
		segment_index = 0
		self.sim.mass = 100
		
		self.assertEqual(self.sim.compute_total_reads()[segment_index], 200)

		# total number of reads when one segmented is duplicated
		segment_index = 0
		self.sim.lineages[1].segments[cons.A][segment_index][cons.CNV] = ["+1"]
		self.sim.mass = 100
		
		self.assertEqual(self.sim.compute_total_reads()[segment_index], 280)

		# total number of reads when one segmented is deleted
		segment_index = 0
		self.sim.lineages[1].segments[cons.A][segment_index][cons.CNV] = []
		self.sim.lineages[1].segments[cons.B][segment_index][cons.CNV] = ["-1"]
		self.sim.mass = 100
		
		self.assertEqual(self.sim.compute_total_reads()[segment_index], 120)

		# total number of reads when one segmented is deleted and the other duplicated
		segment_index = 0
		self.sim.lineages[1].segments[cons.A][segment_index][cons.CNV] = ["+1"]
		self.sim.lineages[1].segments[cons.B][segment_index][cons.CNV] = ["-1"]
		self.sim.mass = 100
		
		self.assertEqual(self.sim.compute_total_reads()[segment_index], 200)

		# set changed CN numbers back to no changes
		self.sim.lineages[1].segments[cons.A][segment_index][cons.CNV] = []
		self.sim.lineages[1].segments[cons.B][segment_index][cons.CNV] = []

	def test_compute_and_set_variant_reads_for_seg_pha_lin(self):

		# lineage object
		segment_number = 1
		freq0 = 1
		freq1 = 0.8
		sublin0 = [1]
		sublin1 = []
		lineages = [lin_class.Lineage_Simulation(sublin0, freq0, segment_number),
			lin_class.Lineage_Simulation(sublin1, freq1, segment_number)]
		sim = data_simulation.Data_Simulation()
		sim.segment_number = segment_number
		sim.lineages = lineages
		sim.mass = 100
		sim.new_version = True
		sim.overdispersion = False
		sim.allele_specific = False
		sim.SSM_before_CNV_LH = 0.0

		# input data
		reads_total = [280]
		seg_start = [0]
		seg = 0
		pha = cons.A
		lin = sim.lineages[1]
		offset = [0, 0]
		lin1 = 1
		# CN gain on A in lin 1
		# number of variant reads of SNP is influenced by this
		sim.lineages[lin1].add_mutation_to_segment(pha, seg, cons.CNV, state="+1")

		################### tests ##########################

		##############
		# 1) SNPs, no mutations, no noise
		offset = [0, 0]
		lin_index = cons.NORMAL
		sim.lineages[lin_index].segments[pha][seg][cons.SNP] = []
		sim.noise = False
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# no SNP in lineage
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP], [])
		# offset was not changed
		self.assertEqual(offset[cons.SNP], 0)


		##############
		# 2) SNPs, with mutations, no noise
		offset = [0, 0]
		lin_index = cons.NORMAL
		sim.lineages[lin_index].add_mutation_to_segment(pha, seg, cons.SNP)
		sim.noise = False
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# attributes of SNP
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].variant_count, 180)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].ref_count, 100)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].pos, 0)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].chr, 1)
		# offset
		self.assertEqual(offset[cons.SNP], 1)


		##############
		# 3) SNPs, with one mutation, no noise
		# variant read number higher than total number of reads
		offset = [0, 0]
		lin_index = cons.NORMAL
		# change total number of reads
		reads_total = [100]
		#delete old SNP from test before before adding a new one
		sim.lineages[lin_index].segments[pha][seg][cons.SNP] = []
		sim.lineages[lin_index].add_mutation_to_segment(pha, seg, cons.SNP)
		sim.noise = False
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# attributes of SNP
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].variant_count, 180)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].ref_count, 0)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].pos, 0)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].chr, 1)
		# offset
		self.assertEqual(offset[cons.SNP], 1)

		# set total number of reads back to normal for other tests
		reads_total = [280]


		##############
		# 4) SNPs, with two mutations, with noise
		# two mutations to check whether different values are simulated
		offset = [0, 0]
		lin_index = cons.NORMAL
		#delete old SNP from test before before adding a new one
		sim.lineages[lin_index].segments[pha][seg][cons.SNP] = []
		sim.lineages[lin_index].add_mutation_to_segment(pha, seg, cons.SNP)
		sim.lineages[lin_index].add_mutation_to_segment(pha, seg, cons.SNP)
		sim.noise = True
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# positions of SNPs
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].pos, 0)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][1].pos, 1000)
		# when SNPs are simulated with noise their variant counts shouldn't be equal
		self.assertNotEqual(sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].variant_count,
			sim.lineages[lin_index].segments[pha][seg][cons.SNP][1].variant_count)
		# also coverage of SNPs shoudn't be equal
		cov_SNP0 = (sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].variant_count +
			sim.lineages[lin_index].segments[pha][seg][cons.SNP][0].ref_count)
		cov_SNP1 = (sim.lineages[lin_index].segments[pha][seg][cons.SNP][1].variant_count +
			sim.lineages[lin_index].segments[pha][seg][cons.SNP][1].ref_count)
		self.assertNotEqual(cov_SNP0, cov_SNP1)
		# offset
		self.assertEqual(offset[cons.SNP], 2)


		##############
		# 5) SSMs, no mutations, no noise
		offset = [0, 0]
		lin_index = lin1
		sim.lineages[lin_index].segments[pha][seg][cons.SSM] = []
		sim.noise = False
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# no SSM in lineage
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SSM], [])
		# offset was not changed
		self.assertEqual(offset[cons.SSM], 0)


		##############
		# 6) SSMs, with one mutation, no noise
		offset = [0, 0]
		lin_index = lin1
		sim.lineages[lin1].add_mutation_to_segment(pha, seg, cons.SSM)
		sim.noise = False
		sim.compute_and_set_variant_reads_for_seg_pha_lin(reads_total, seg_start, seg, pha,
			lin_index, sim.lineages[lin_index], offset)

		# attributes of SSM
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SSM][0].variant_count, 80)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SSM][0].ref_count, 200)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SSM][0].pos, 0)
		self.assertEqual(sim.lineages[lin_index].segments[pha][seg][cons.SSM][0].chr, 1)
		# offset
		self.assertEqual(offset[cons.SSM], 1)


	def test_compute_and_set_variant_reads(self):
		
		# test with one SNP and one SSM in phase B on the first and only segment
		self.sim.mass = 200
		self.sim.new_version = False
		reads_total = [self.sim.mass * cons.PHASE_NUMBER]
		seg_start = [0]
		segment = 0
		pha = cons.B
		self.sim.lineages[cons.NORMAL].add_mutation_to_segment(pha, segment, cons.SNP)
		self.sim.lineages[1].add_mutation_to_segment(pha, segment, cons.SSM)

		self.sim.compute_and_set_variant_reads(reads_total, seg_start)

		self.assertEqual(self.sim.lineages[cons.NORMAL].segments[pha][segment][cons.SNP][0].pos, 
			seg_start[segment])
		self.assertEqual(self.sim.lineages[1].segments[pha][segment][cons.SSM][0].pos, 
			seg_start[segment] + 0)

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(DataSimulationTest)
























