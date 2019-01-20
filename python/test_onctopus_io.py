import unittest
import onctopus_io as oio
import constants as cons
import segment
import snp_ssm
import cnv
import lineage
import lineage_for_data_simulation
import exceptions_onctopus as eo
import os
import pdb
import sys
from StringIO import StringIO

class OnctopusIOTest(unittest.TestCase):
	
	def test_SNP_count_to_LogR_new(self):
		input_file_name = "testdata/unittests/SNP_count_small_normal.tsv"
		output_file_name_normal = "testdata/unittests/SNP_count_to_LogR_normal"
		output_file_name_tumor = "testdata/unittests/SNP_count_to_LogR_tumor"

		average_cov = 4

		# test for normal file
		oio.SNP_count_to_LogR_new(input_file_name, output_file_name_normal, avg_coverage=average_cov,
			test=True, normal=True)

		lines_ref = []
		with open(input_file_name) as f:
			lines_ref = f.readlines()

		lines = []
		with open(output_file_name_normal) as f:
			lines = f.readlines()

		self.assertEqual(len(lines_ref), len(lines))
		# header
		self.assertEqual("\tchrs\tpos\tsample\n", lines[0])
		# entry for chromosome and position
		self.assertEqual(lines_ref[1].split("\t")[:2], lines[1].split("\t")[1:3])
		self.assertEqual("4", lines[4].split("\t")[0])

		# test for tumor file
		oio.SNP_count_to_LogR_new(input_file_name, output_file_name_tumor, 
			test=True, normal=False, normal_SNP_file=input_file_name)
		lines_tumor = []
		with open(output_file_name_tumor) as f:
			lines_tumor = f.readlines()

		self.assertEqual(len(lines_tumor), len(lines_ref))
		self.assertNotEqual(lines_tumor, lines_ref)

	def test_remove_single_positions_with_low_coverage(self):
		input_file_name = "testdata/unittests/cleaned_tumor_2"
		ref_file_name = "testdata/unittests/cleaned_tumor_2_ref"
		output_file_name = "testdata/unittests/cleaned_tumor_2_after_test"
		removed_positions = [(1,0), (1,1), (1,7)]
		coverage_cutoff = 10
		length = 3

		my_list = oio.remove_single_positions_with_low_coverage(input_file_name, output_file_name,
			coverage_cutoff, length, test=True)

		ref_file = []
		output_file = []
		with open(ref_file_name) as f:
			ref_file = f.read()
		with open(output_file_name) as f:
			output_file = f.read()

		self.assertEqual(ref_file, output_file)
		self.assertListEqual(my_list, removed_positions)
		

	def test_remove_positions_from_list(self):
		input_file_name = "testdata/unittests/cleaned_tumor_1"
		ref_file_name = "testdata/unittests/cleaned_tumor_1_ref"
		output_file_name = "testdata/unittests/cleaned_tumor_1_after_test"
		removed_positions = [(1, 1), (1, 3)]

		oio.remove_positions_from_list(input_file_name, output_file_name,
			removed_positions, test=True)

		ref_file = []
		output_file = []
		with open(ref_file_name) as f:
			ref_file = f.read()
		with open(output_file_name) as f:
			output_file = f.read()

		self.assertEqual(ref_file, output_file)

	def test_remove_positions_in_normal_with_low_coverage(self):
		input_file_name = "testdata/unittests/cleaned_normal_1"
		output_file_name = "testdata/unittests/cleaned_normal_1_after_test"
		coverage_cutoff = 10

		removed_positions = oio.remove_positions_in_normal_with_low_coverage(
			input_file_name, output_file_name, coverage_cutoff,
			test=True)
		
		self.assertEqual([(1, 1)], removed_positions)
		
		output_content = []
		with open(output_file_name) as f:
			output_content = f.read()
		self.assertEqual("#CHR\tPOS\tCount_A\tCount_C\tCount_G\tCount_T\tGood depth\n1\t2\t10\t10\t0\t0\t20\n", output_content)


	def test_compute_av_coverage_from_SNP_count(self):
		
		input_file_name = "testdata/unittests/SNP_count_small.tsv"
		self.assertEqual(4, oio.compute_av_coverage_from_SNP_count(input_file_name))


	def test_read_vcf_file(self):
		input_file_name = "testdata/unittests/simulated.VCF"
		output_file_name = "testdata/unittests/read_simulated.VCF"

		oio.parse_vcf_file_for_onctopus(input_file_name, output_file_name, True)
		ssm_list = oio.read_snp_ssm_file(output_file_name, snp_ssm.SSM)

		self.assertEqual(len(ssm_list), 132)
		# test first entry
		self.assertEqual(ssm_list[0].chr, 1)
		self.assertEqual(ssm_list[0].pos, 30000001)
		self.assertEqual(ssm_list[0].variant_count, 20)
		self.assertEqual(ssm_list[0].ref_count, 22)
		# test third entry
		self.assertEqual(ssm_list[2].chr, 1)
		self.assertEqual(ssm_list[2].pos, 30020001)
		self.assertEqual(ssm_list[2].variant_count, 14)
		self.assertEqual(ssm_list[2].ref_count, 15)
		# test last entry
		self.assertEqual(ssm_list[-1].chr, 22)
		self.assertEqual(ssm_list[-1].pos, 30050001)
		self.assertEqual(ssm_list[-1].variant_count, 8)
		self.assertEqual(ssm_list[-1].ref_count, 46)


	def test_read_result_file(self):
		file_name = "testdata/unittests/out_result1"

		ll = oio.read_result_file(file_name, phasing_not_known=False)

		# test if list is correct

		# should have 4 lineages
		self.assertEqual(len(ll), 4)

		# frequencies
		self.assertEqual(ll[0].freq, 1.0)
		self.assertEqual(ll[1].freq, 0.9)

		# sublineages
		self.assertEqual(ll[0].sublins, [1, 2, 3])
		self.assertEqual(ll[1].sublins, [2, 3])
		self.assertEqual(ll[3].sublins, [])

		# snps
		self.assertEqual(len(ll[0].snps), 0)
		self.assertEqual(len(ll[0].snps_a), 5)
		self.assertEqual(len(ll[0].snps_b), 5)
		self.assertEqual(len(ll[1].snps_b), 0)
		self.assertEqual(ll[0].snps_a[0].seg_index, 0)
		self.assertEqual(ll[0].snps_a[0].chr, 1)
		self.assertEqual(ll[0].snps_a[0].pos, 0)

		# ssms
		self.assertEqual(len(ll[1].ssms), 0)
		self.assertEqual(len(ll[1].ssms_a), 4)
		self.assertEqual(ll[1].ssms_a[0].phase, cons.A)
		self.assertEqual(ll[1].ssms_a[0].lineage, 1)
		self.assertEqual(len(ll[3].ssms_b), 1)
		self.assertEqual(ll[3].ssms_b[0].phase, cons.B)
		self.assertEqual(ll[3].ssms_b[0].lineage, 3)

		# cnvs
		self.assertEqual(len(ll[1].cnvs_a), 1)
		self.assertEqual(len(ll[1].cnvs_b), 1)
		self.assertEqual(ll[1].cnvs_a[0].chr, 1)
		self.assertEqual(ll[1].cnvs_a[0].start, 0)
		self.assertEqual(ll[1].cnvs_a[0].change, 1)
		self.assertEqual(ll[1].cnvs_a[0].phase, cons.A)

		# read without phasing information
		ll = oio.read_result_file(file_name, phasing_not_known=True)
		self.assertEqual(ll[1].ssms_a[0].phase, None)

	def test_write_result_file(self):
		output_file = "testdata/unittests/out_result3"
		reference_file = "testdata/unittests/out_result3_reference"

		#create test_lineages, one empty(None), one only contains empty lists and the other with entries
		cnvs_a = [cnv.CNV(+1, 0, 1, 0, 7), cnv.CNV(-1, 2, 1, 12, 16)]
		cnvs_b = [cnv.CNV(-1, 1, 1, 8, 11)]
		snps = [snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM()]
		snps[0].chr = 1
		snps[0].pos = 2
		snps[0].seg_index = 0
		snps[1].chr = 1
		snps[1].pos = 4
		snps[1].seg_index = 0
		snps_a = [snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM()]
		snps_a[0].chr = 1
		snps_a[0].pos = 0
		snps_a[0].seg_index = 0
		snps_a[1].chr = 1
		snps_a[1].pos = 13
		snps_a[1].seg_index = 2
		snps_a[2].chr = 1
		snps_a[2].pos = 14
		snps_a[2].seg_index = 2
		snps_b = [snp_ssm.SNP_SSM()]
		snps_b[0].chr = 1
		snps_b[0].pos = 8
		snps_b[0].seg_index = 1
		ssms = snps_b[:]
		ssms_a = snps[:]
		ssms_b = snps_a[:]
		lineages = [lineage.Lineage(None, None, None, None, None, None, None, None, None, None), 
			lineage.Lineage([], 0.0, [], [], [], [], [], [], [], []), 
			lineage.Lineage([0,1], 0.2, cnvs_a, cnvs_b, snps, snps_a, snps_b, ssms, ssms_a, ssms_b)]

		oio.write_result_file(lineages, output_file, test=True)

		with open(output_file, 'r') as f:
			file_data = f.read()
		with open(reference_file, 'r') as f:
			reference_data = f.read()

		self.assertEqual(file_data, reference_data)

	def test_read_segment_file(self):
		file_name = "testdata/unittests/test1_segments.txt"
		segment_list = oio.read_segment_file(file_name)

		self.assertEqual(len(segment_list), 2)
		self.assertEqual(segment_list[0].chr, 1)
		self.assertEqual(segment_list[1].start, 11)
		self.assertEqual(segment_list[1].end, 20)
		self.assertEqual(segment_list[1].count, 250)
		self.assertEqual(segment_list[1].hm, 120)
		self.assertEqual(segment_list[1].cn, -1)

		# test allele-specific
		file_name = "testdata/unittests/test1_segments_allele_specific"
		segment_list = oio.read_segment_file(file_name, True)

		self.assertEqual(len(segment_list), 3)
		self.assertEqual(segment_list[0].chr, 1)
		self.assertEqual(segment_list[0].start, 1)
		self.assertEqual(segment_list[0].end, 10)
		self.assertEqual(segment_list[0].given_cn_A, 2.1)
		self.assertEqual(segment_list[0].standard_error_A, 0.25)
		self.assertEqual(segment_list[0].given_cn_B, 1.3)
		self.assertEqual(segment_list[0].standard_error_B, 0.12)
		self.assertEqual(segment_list[1].inferred_cn_A, -1)
		self.assertEqual(segment_list[2].given_cn_A, 1.4)
		self.assertEqual(segment_list[2].standard_error_A, 0.26)
		self.assertEqual(segment_list[2].given_cn_B, 0.4)
		self.assertEqual(segment_list[2].standard_error_B, 0.25)
		
	def test_read_snp_ssm_file(self):
		file_name = "testdata/unittests/test1_SNPs_SSMs.txt"

		# test with SNPs
		mut_list = oio.read_snp_ssm_file(file_name, cons.SNP)

		self.assertEqual(len(mut_list), 1)
		self.assertEqual(mut_list[0].chr, 1)
		self.assertEqual(mut_list[0].pos, 5)
		self.assertEqual(mut_list[0].variant_count, 120)
		self.assertEqual(mut_list[0].ref_count, 220)

	def test_raise_if_file_exists(self):
		file_name = "testdata/unittests/test1_SNPs_SSMs.txt" 

		# file does exist
		with self.assertRaises(eo.FileExistsException):
			oio.raise_if_file_exists(file_name)
		
		# file doesn't exist
		self.assertTrue(oio.raise_if_file_exists("kuhfkjh8kjhkjh"))

	def test_write_segment_file(self):
		file_name = "testdata/unittests/write_segment.txt"
		reads_total = [1, 1]
		seg_start = [0, 1]
		seg_end = [0, 1]
		segment_number = len(reads_total)
		mass = [1, 1]
		no_test = False

		# write results, no matter if file exists or not
		oio.write_segment_file(reads_total, seg_start, seg_end, segment_number, mass, file_name, no_test)

		segment_list = oio.read_segment_file(file_name)

		self.assertEqual(len(segment_list), segment_number)
		self.assertEqual(segment_list[0].start, seg_start[0])
		
		# now file exists already and should not be overwritten
		no_test = True
		with self.assertRaises(eo.FileExistsException):
			oio.write_segment_file(reads_total, seg_start, seg_end, segment_number, mass, file_name, no_test)

	def test_write_single_muts_file(self):
		file_name_snp = "testdata/unittests/write_snp.txt"
		file_name_ssm = "testdata/unittests/write_ssm.txt"
		reads_total = [1]
		seg_start = [0]
		segment_number = 1
		segment = 0
		no_test = False

		# create lineage with mutations
		lin0 = lineage_for_data_simulation.Lineage_Simulation([1], 1, segment_number)
		lin1 = lineage_for_data_simulation.Lineage_Simulation([], 0.8, segment_number)
		lineages = [lin0, lin1]
		lin0.add_mutation_to_segment(cons.A, segment, cons.SNP)
		lin0.add_mutation_to_segment(cons.A, segment, cons.SNP)
		lin1.add_mutation_to_segment(cons.B, segment, cons.SSM)
		lin0.segments[cons.A][segment][cons.SNP][0].pos = 0
		lin0.segments[cons.A][segment][cons.SNP][0].variant_count = 1
		lin0.segments[cons.A][segment][cons.SNP][0].ref_count = 1
		lin0.segments[cons.A][segment][cons.SNP][1].pos = 1
		lin0.segments[cons.A][segment][cons.SNP][1].variant_count = 2
		lin0.segments[cons.A][segment][cons.SNP][1].ref_count = 2
		lin1.segments[cons.B][segment][cons.SSM][0].pos = 0
		lin1.segments[cons.B][segment][cons.SSM][0].variant_count = 3
		lin1.segments[cons.B][segment][cons.SSM][0].ref_count = 3

		# overwrite file if it exists
		oio.write_single_muts_file(reads_total, lineages, seg_start, segment_number, file_name_snp,
			file_name_ssm, no_test)

		snp_list = oio.read_snp_ssm_file(file_name_snp, cons.SNP)
		ssm_list = oio.read_snp_ssm_file(file_name_ssm, cons.SSM)

		self.assertEqual(len(snp_list), len(lin0.segments[cons.A][segment][cons.SNP]))
		self.assertEqual(len(ssm_list), len(lin1.segments[cons.B][segment][cons.SSM]))
		self.assertEqual(snp_list[1].variant_count, lin0.segments[cons.A][segment][cons.SNP][1].variant_count)
		self.assertEqual(ssm_list[0].pos, lin1.segments[cons.B][segment][cons.SSM][0].pos)
		
		# don't overwrite file
		no_test = True
		with self.assertRaises(eo.FileExistsException):
			oio.write_single_muts_file(reads_total, lineages, seg_start, segment_number, file_name_snp,
				file_name_ssm, no_test)

	def test_get_sublineages_line(self):
		sublins = [2,4]
		self.assertEqual(oio.get_sublineages_line(sublins), "SUBLINEAGES: 2;4\n")

		sublins = [5]
		self.assertEqual(oio.get_sublineages_line(sublins), "SUBLINEAGES: 5\n")

		sublins = []
		self.assertEqual(oio.get_sublineages_line(sublins), "SUBLINEAGES: \n")

	def test_get_end_of_mutation_line(self):
		# create lineage with three segments and mutations
		# CNV: |-1|-|1|
		# SNP: |xx|-|x|
		segment_number = 3
		pha = cons.A
		segment_start = [0,2,3]
		segment_end = [1,2,3]
		lin = lineage_for_data_simulation.Lineage_Simulation([], 1, segment_number)
		lin.add_mutation_to_segment(pha, 0, cons.CNV, -1)
		lin.add_mutation_to_segment(pha, 2, cons.CNV, 1)
		lin.add_mutation_to_segment(pha, 0, cons.SNP)
		lin.segments[pha][0][cons.SNP][0].chr = 1
		lin.segments[pha][0][cons.SNP][0].pos = segment_start[0]
		lin.add_mutation_to_segment(pha, 0, cons.SNP)
		lin.segments[pha][0][cons.SNP][1].chr = 1
		lin.segments[pha][0][cons.SNP][1].pos = segment_start[0] + 1
		lin.add_mutation_to_segment(pha, 2, cons.SNP)
		lin.segments[pha][2][cons.SNP][0].chr = 1
		lin.segments[pha][2][cons.SNP][0].pos = segment_start[2]

		line = []

		# test case 1)
		mut = cons.CNV
		self.assertEqual(oio.get_end_of_mutation_line(line, mut, lin, pha, segment_number,
			segment_start, segment_end), 
			"{0},{1},1,{2},{3};{4},{5},1,{6},{7}\n".format(-1, 0, segment_start[0], segment_end[0],
			1, 2, segment_start[2], segment_end[2]))

		# test case 3), 6)
		mut = cons.SNP
		line = []
		self.assertEqual(oio.get_end_of_mutation_line(line, mut, lin, pha, segment_number,
			segment_start, segment_end), 
			"{0},{1},{2};{3},{4},{5};{6},{7},{8}\n".format(
			0, lin.segments[pha][0][cons.SNP][0].chr,
			lin.segments[pha][0][cons.SNP][0].pos, 0, lin.segments[pha][0][cons.SNP][1].chr,
			lin.segments[pha][0][cons.SNP][1].pos, 2, lin.segments[pha][2][cons.SNP][0].chr,
			lin.segments[pha][2][cons.SNP][0].pos))

		# empty segments
		line = []
		lin.segments[pha][0][cons.SNP] = []
		lin.segments[pha][2][cons.SNP] = []
		self.assertEqual(oio.get_end_of_mutation_line(line, mut, lin, pha, segment_number,
			segment_start, segment_end), "\n")

	def test_write_simulation_results(self):
		segment_number = 1
		segment_start = [0]
		segment_end = [0]
		lin = lineage_for_data_simulation.Lineage_Simulation([], 1, segment_number) 
		lineages = [lin]
		file_name = "testdata/unittests/write_results"
		no_test = False

		oio.write_simulation_results(segment_start, segment_end, lineages, segment_number, 
			file_name, no_test)

		# read file line for line and compare it 
		i = 0
		with open(file_name) as f:
			for line in f:
				if (i == 0):
					self.assertEqual(line, "@\n")
					i = 1
				elif (i == 1):
					self.assertEqual(line, "LINEAGE: 0\n")
					i = 2
				elif (i == 2):
					self.assertEqual(line, "FREQUENCY: 1\n")
					i = 3
				elif (i == 3):
					self.assertEqual(line, "SUBLINEAGES: \n")
					i = 4
				elif (i == 4):
					self.assertEqual(line, "SNPS_A: \n")
					i = 5
				elif (i == 5):
					self.assertEqual(line, "SNPS_B: \n")
					i = 6
				elif (i == 6):
					self.assertEqual(line, "SSMS_A: \n")
					i = 7
				elif (i == 7):
					self.assertEqual(line, "SSMS_B: \n")
					i = 8
				elif (i == 8):
					self.assertEqual(line, "CNVS_A: \n")
					i = 9
				elif (i == 9):
					self.assertEqual(line, "CNVS_B: \n")
					i = 10
		self.assertEqual(i, 10)

	def test_read_CNV_assignments(self):
		file_name = "testdata/unittests/assignment"

		assignment = oio.read_mutation_assignments(file_name)
		
		self.assertListEqual([1, cons.A, 1, "+1"], assignment[0])
		self.assertListEqual([2, cons.B, 2, "-1"], assignment[1])
		self.assertListEqual([3, cons.A, 1], assignment[2])


	def test_read_lineages_tree(self):
		file_name = "testdata/unittests/lin1.txt"
		segment_number = 2

		lin = oio.read_lineages_tree(file_name, segment_number)
		self.assertEqual(len(lin), 3)
		self.assertEqual(lin[0].sublin, [1,2])
		self.assertEqual(lin[1].sublin, [2])
		self.assertEqual(lin[2].sublin, [])
		self.assertEqual(lin[0].freq, 1)
		self.assertEqual(lin[1].freq, 0.5)

	def test_str_to_bool(self):
		s = 'True'
		self.assertTrue(oio.str_to_bool(s))

		s = 'False'
		self.assertFalse(oio.str_to_bool(s))

		s = 'something'
		with self.assertRaises(ValueError):
			oio.str_to_bool(s)

	def test_read_SNP_index(self):
		input_file = "testdata/unittests/SNP_index_full.txt"
		test_index = oio.read_SNP_index(input_file)
		reference_index = {(10, 60523):("T", "A"), (10, 60969):("T","C"), (10, 60975):("G","A"), (10, 60985):("A", "T"), (11, 45132):("C", "T"), (11, 45350):("A", "G"),(11, 72423):("G", "C"), (11, 72482):("C", "G")}

		self.assertEqual(test_index, reference_index)

	def test_SNP_count_to_BAF(self):
		"""
		tested cases:
		1: variance = 0, total = reference = 7, result = 0
		2: variance = reference = total = 0, result = 0
		3: variance = 0, reference = 8, total = 10, additional entry = 2 is new variance, 
			result = 0.2 and printed message
		4: variance = 3, reference = total = 0, result = 0 and catched exception with printed message
		5: variance = 5, reference = 2, total = 8, additional entry = 1, 
			result = 0.625 with printed warning for the addidtional entry
		6: variance = 5, reference = 1, total = 4, result = 0 and catched exception with printed message
		7: variance = 2, reference = 3, total = 5, no entry in index file, error and skipped
		8: variance = 2, reference = 4, total = 6, result = 0.333333333333
		9: variance = 2, reference = 0, total = 2, result = 1.0
		"""
		input_index = "testdata/unittests/SNP_index_full.txt"
		input_count = "testdata/unittests/SNP_count_full.tsv"

		output_file = "testdata/unittests/SNP_count_to_BAF"
		output_reference = "testdata/unittests/SNP_count_to_BAF_reference"
		print_reference = "SNP(chr: 10, pos: 60975) has variance_count 0. Set new variance(C) with count = 2.\nBAF computation failed for SNP(chr: 10, pos: 60985) with message:\"Total count is 0 but variant count isn't.\"\nWARNING: SNP(chr: 11, pos: 45132) has an entry unequal 0, which isn't the variance or reference.\nBAF computation failed for SNP(chr: 11, pos: 45350) with message:\'Variant count is bigger than total count.\'\nERROR: Skipping SNP(chr: 11, pos: 72314) with no entry in indexfile"

		try:
			# intercept printed messages
			out = StringIO()
			sys.stdout = out
			# run function
			index = oio.read_SNP_index(input_index)
			oio.SNP_count_to_BAF(input_count, index, output_file, test=True)
			# get output from standard output
			print_output = out.getvalue().strip()

		finally:
			# restore standard output
			sys.stdout = sys.__stdout__

		# read output reference and outputfile
		with open(output_file, 'r')as f:
			file_data = f.read().strip()
		with open(output_reference, 'r')as f:
			reference_data = f.read().strip()

		# check results
		self.assertEqual(file_data, reference_data)
		self.assertEqual(print_output, print_reference)

	def test_SNP_count_to_LogR(self):
		# files
		input_file = "testdata/unittests/SNP_count_full.tsv"
		output_file = "testdata/unittests/SNP_count_to_LogR"
		reference = "testdata/unittests/SNP_count_to_LogR_reference"
		# average coverage used in test
		avg_coverage = 15
		# run function
		oio.SNP_count_to_LogR(input_file, avg_coverage, output_file, test=True)

		# read reference file and output file
		with open(output_file, 'r') as f:
			file_data = f.read().strip()
		with open(reference, 'r') as f:
			reference_data = f.read().strip()
		# check result
		self.assertEqual(file_data, reference_data)

	def test_clean_SNP_count_naive(self):

		''' 12 cases are tested:
		1.  ref: ok, var: ok
		2.  ref: ok, var: ok, another entry != 0
		3.  ref: ok, var: ok, two other entries != 0
		4.  ref: ok, var: 0
		5.  ref: ok, var: 0, another entry or more ! = 0
		6.  ref: ok, var: 0, two other entries != 0
		7.  ref: 0, var: ok
		8.  ref: 0, var: ok, another entry or more! = 0
		9.  ref: 0, var: ok, two other entries != 0
		10. ref: 0, var: 0
		11. ref: 0, var: 0, another entry or more! = 0
		12. ref: 0, var: 0, two other entries != 0
		13. SNP is not in index
		'''

		# data for function
		input_SNP_count = "testdata/unittests/invalid_SNP_count"
		SNP_index = "testdata/unittests/SNP_index_for_cleanup"
		output_SNP_count = "testdata/unittests/cleaned_SNP_count"

		# reference data for result
		reference_SNP_count = "testdata/unittests/valid_SNP_count"
		print_reference = "ERROR: SNP(chr: 13, pos: 3543) has no entry in indexfile and is removed.\n#SNPs: 12\n#cleaned SNPs: 8\n#cleaned SNPs, where cleaned entry was greater 5: 3"

		index = oio.read_SNP_index(SNP_index)

		try:
			# intercept printed messages
			out = StringIO()
			sys.stdout = out
			# run function
			oio.clean_SNP_count_naive(input_SNP_count, index, output_SNP_count, test=True)
			# get output from standard output
			print_output = out.getvalue().strip()

		finally:
			# restore standard output
			sys.stdout = sys.__stdout__

		with open(output_SNP_count, 'r')as f:
			file_data = f.read().strip()
		with open(reference_SNP_count, 'r')as f:
			reference_data = f.read().strip()

		# check results
		self.assertEqual(file_data, reference_data)
		self.assertEqual(print_output, print_reference)	

	def test_resultfile_to_2A_file(self):
		# tested result file has 11 ssms, one is unphased
		inputname = "testdata/unittests/out_result2"
		outputname = "testdata/unittests/out_2A"
		oio.resultfile_to_2A_file(inputname, outputname, test=True)
		# read output 2A file
		with open(outputname, 'r') as test_file:
			test_file_content = test_file.read()
		# expected entries
		expected_content = "1\n1\n2\n2\n3\n1\n1\n2\n2\n2\n1\n"
		self.assertEqual(test_file_content, expected_content)

	def test_resultfile_to_pseudo_VCF_for_2A(self):
		# tested result file has 11 ssms, one is unphased
		inputname = "testdata/unittests/out_result2"
		outputname = "testdata/unittests/out_pseudo_VCF"
		oio.resultfile_to_pseudo_VCF_for_2A(inputname, outputname, test=True)
		# read generated pseudo VCFfile
		with open(outputname, 'r') as test_file:
			test_file_content = test_file.read()
		# 11 entries expected
		expected_content = "True\nTrue\nTrue\nTrue\nTrue\nTrue\nTrue\nTrue\nTrue\nTrue\nTrue\n"
		self.assertEqual(test_file_content, expected_content)

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(OnctopusIOTest)
