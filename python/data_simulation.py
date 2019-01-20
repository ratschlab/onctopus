#!/usr/bin/env python

import random
import constants as cons
import exceptions_onctopus as eo
import sys
import onctopus_io as oio
import sys
import numpy as np
from scipy import stats
import argparse
import exceptions_onctopus as eo

# currenty two versions in this file
# standard: new_version
#	For all segments, the exact coverage is computed. From this value, SNPs and SSMs are simulated
#	that can contain noise. At the end, the coverage of the segment is calculated as the
#	average of the coverages of the SNPs.
# old version:
# 	When noise should be introduced in the data, coverage counts of the segments are calculated
#	with noise. Then, when SNPs and SSMs are calculated, noise is introduced again.

class Data_Simulation(object):
	
	def __init__(self):
		self.segment_number = 0
		self.snp_number = 0
		self.ssm_number = 0
		self.cnv_number = 0
		self.noise = False
		self.lineages = []
		self.mass = 0

	# mass is haploid sequencing mass
	def do_simulation(self, lineages_file, segment_number, snp_number, ssm_number, cnv_number,
		mass, file_end_tag, file_start_tag="", noise=True, CN_standard_deviation=0.1,
		no_test=True, 
		CNV_assignment=None, SNP_assignment=None, SSM_assignment=None, new_version=True,
		coverage_overdispersion=1000, frequency_overdispersion=1000, overdispersion=True,
		allele_specific=False, SSM_num_per_unit=0, clonal_cn_percentage=0.75, p1_A_prop=0.2,
		p1_A_B_prop=0.2, m1_B_prop=0.2, m1_A_B_prop=0.2, p1_m1_prop=0.2,
		SSM_before_CNV_LH=0.5, addSSMsAccoringToFreqs=False, clonal_ssm_percentage=None,
		CN_noise=True, seg_min_length=1000000, CNAs_mult_lin_prop=0.0):

		# set variables
		self.allele_specific=allele_specific
		self.segment_number = segment_number
		if not self.allele_specific:
			self.snp_number = snp_number
		self.ssm_number = ssm_number
		self.cnv_number = cnv_number
		self.noise = noise
		self.CN_noise = CN_noise
		#self.CN_standard_deviation = CN_standard_deviation
		self.mass = mass
		self.new_version = new_version
		self.coverage_overdispersion = coverage_overdispersion
		self.frequency_overdispersion = frequency_overdispersion
		self.overdispersion = overdispersion
		self.clonal_cn_percentage = clonal_cn_percentage
		if clonal_ssm_percentage is None:
			self.clonal_ssm_percentage = -1
		else:
			self.clonal_ssm_percentage = clonal_ssm_percentage
		self.p1_A_prop = p1_A_prop
		self.p1_A_B_prop = p1_A_B_prop
		self.m1_B_prop = m1_B_prop
		self.m1_A_B_prop = m1_A_B_prop
		self.p1_m1_prop = p1_m1_prop
		self.CNAs_mult_lin_prop = CNAs_mult_lin_prop
		self.SSM_before_CNV_LH = SSM_before_CNV_LH
		self.addSSMsAccoringToFreqs = addSSMsAccoringToFreqs
		self.seg_min_length = seg_min_length
		# standrad CNs
		self.cn_A = [1] * self.segment_number
		self.cn_B = [1] * self.segment_number
		self.cn_total = [2] * self.segment_number
		output_segments = "{0}segments_{1}".format(file_start_tag, file_end_tag)
		output_snps = None
		if not self.allele_specific: 
			output_snps = "{0}snps_{1}".format(file_start_tag, file_end_tag) 
		output_ssms = "{0}ssms_{1}".format(file_start_tag, file_end_tag)
		output_results = "{0}results_{1}".format(file_start_tag, file_end_tag)  
		output_info = "{0}info_{1}".format(file_start_tag, file_end_tag)
		
		reads_total = []
		seg_start = []
		seg_end = []

		# check if cnv number isn't too high
		if(self.segment_number < self.cnv_number and CNV_assignment == None):
			sys.stderr.write('Segment number cannot be smaller than CNV number. CNV number was set ' +
				'to segment number ({0}).'.format(segment_number))
			self.cnv_number = self.segment_number

		# check if files that should be written don't exist
		if no_test:
			oio.raise_if_file_exists(output_segments)
			if not self.allele_specific:
				oio.raise_if_file_exists(output_snps)
			oio.raise_if_file_exists(output_ssms)
			oio.raise_if_file_exists(output_results)
			oio.raise_if_file_exists(output_info)

		# check if proportions of mutations make sense
		if (self.p1_A_prop + self.p1_A_B_prop + self.m1_B_prop + self.m1_A_B_prop
			+ self.p1_m1_prop + self.CNAs_mult_lin_prop != 1):
			raise eo.myException("Proportion of CN changes don't sum up to 1.")

		# check if only one way to sample SSMs are chosen
		if self.addSSMsAccoringToFreqs == True and self.clonal_ssm_percentage != -1:
			raise eo.myException("A single way how to sample the SSM lineages must be chosen.")

		# read lineages
		self.lineages = oio.read_lineages_tree(lineages_file, self.segment_number)
		self.construct_ancestral_lineage_list()

		# create mutations and add them to lineages, segments and phases without simulation read counts
		# if assignment files exists, mutations are put to lineages like assigned
		# add CNVs
		if CNV_assignment == None:
			self.add_CNV()
		else:
			self.add_CNV_according_assignment(oio.read_mutation_assignments(CNV_assignment))
		# add SNPs and SSMs
		if not self.allele_specific:
			if SNP_assignment: 
				self.add_SNP_SSM_according_assignment(oio.read_mutation_assignments(
					SNP_assignment), cons.SNP)
			else:
				for i in range(self.snp_number):
					self.add_SNP_to_normal_lineage()
		if SSM_assignment:
			self.add_SNP_SSM_according_assignment(oio.read_mutation_assignments(SSM_assignment),
				cons.SSM)
		elif SSM_num_per_unit > 0:
			self.add_x_SSMs_per_unit(SSM_num_per_unit)
		else:
			# if SSMs should be added to lineages according to the lineage frequencies
			if addSSMsAccoringToFreqs:
				self.frequency_table = self.get_frequency_table()
			for i in range(self.ssm_number):
				self.add_SSM()
		
		# derive/simulate read numbers/mutation frequencies
		if self.new_version:
			# set noise to false to get exact theoretical coverage per segment
			tmp_noise = noise
			self.noise = False
			reads_total = self.compute_total_reads()
			self.noise = tmp_noise
		else:
			reads_total = self.compute_total_reads()
		
		# compute boundaries of segments and VAF of SNPs and SSMs
		(seg_start, seg_end) = self.compute_start_end_pos_of_all_segments()
		self.compute_and_set_variant_reads(reads_total, seg_start)

		# compute either coverage of segment or allele-specific copy number
		haploid_sequencing_mass = []
		# allele-specific simulation
		if self.allele_specific:
			# standard error is computed
			self.standard_error_A = []
			self.standard_error_B = [] 
			calculated_cn_A, calculated_cn_B = self.compute_cn_standard_error(seg_start, seg_end)

			# if noise for CNs should be used, use above calculated values
			if self.CN_noise is True:
				self.cn_A = self.add_noise_to_cn(self.cn_A, self.standard_error_A)
				self.cn_B = self.add_noise_to_cn(self.cn_B, self.standard_error_B)

		# not allele-specific simulation
		else:
			if self.new_version:
				reads_total = self.get_average_coverage()
				haploid_sequencing_mass = self.get_haploid_sequencing_mass(reads_total)
			else: 
				haploid_sequencing_mass = [self.mass] * self.segment_number

		# write simulation for Onctopus to file
		if self.allele_specific:
			oio.write_segment_file_allele_specific(seg_start, seg_end, self.segment_number,
				self.cn_A, self.cn_B, self.standard_error_A, self.standard_error_B, output_segments,
				no_test)
		else:
			oio.write_segment_file(reads_total, seg_start, seg_end, self.segment_number,
				haploid_sequencing_mass, output_segments, no_test)
		oio.write_single_muts_file(reads_total, self.lineages, seg_start, self.segment_number,
			output_snps, output_ssms, no_test, self.allele_specific)
		# write simulation results
		oio.write_simulation_results(seg_start, seg_end, self.lineages, self.segment_number, 
			output_results, no_test)
		# write simulation info
                oio.write_simulation_info(lineages_file, self.segment_number,
                        self.snp_number, self.ssm_number, self.cnv_number,
                        self.noise, self.CN_noise,
			self.mass, output_segments, output_snps,
                        output_ssms, output_results,	output_info, no_test,
                        self.new_version, self.overdispersion,
                        self.coverage_overdispersion,
                        self.frequency_overdispersion, CNV_assignment,
                        SNP_assignment, SSM_assignment, allele_specific,
                        SSM_num_per_unit, self.clonal_cn_percentage, self.p1_A_prop,
			self.p1_A_B_prop, self.m1_B_prop, self.m1_A_B_prop, self.p1_m1_prop,
			self.SSM_before_CNV_LH, addSSMsAccoringToFreqs=self.addSSMsAccoringToFreqs,
			clonal_ssm_percentage=self.clonal_ssm_percentage, CNAs_mult_lin_prop=self.CNAs_mult_lin_prop)

	# constructs for each lineage a list with its ancestral lineages based on the list of
	# sublineages of each lineage
	def construct_ancestral_lineage_list(self):
		ancestral_lineages_list = [[] for __ in range(len(self.lineages) - 1)]
		for lin in range(1, len(self.lineages)):
			for sublin in self.lineages[lin].sublin:
				ancestral_lineages_list[sublin - 1].append(lin)
		for lin in range(1, len(self.lineages)):
			self.lineages[lin].ancestral_lineages = ancestral_lineages_list[lin - 1]

	def rand(self, stop):
		return random.randrange(stop)

	def choose_segment(self):
		return self.rand(self.segment_number)

	def choose_phase(self):
		return self.rand(cons.PHASE_NUMBER)

	def choose_lineage_but_normal(self):
		return self.rand(len(self.lineages) - 1) + 1

	# draws a number that is uniformly distributed between 0 and the frequency sum of all
	# non-normal lineages
	# according to this number the lineage number is returned to which the number belongs
	def choose_lineage_by_frequency(self, x=None):
		non_normal_lineage_num = len(self.frequency_table)

		# for testing purposes x can be given
		if x is None:
			freq_sum = self.frequency_table[-1]
			x = random.uniform(0, freq_sum)

		for i in xrange(non_normal_lineage_num):
			if x <= self.frequency_table[i]:
				return i+1

	def choose_lineage_but_normal_with_clonal_percentage(self, cn_percentage):
		x = random.random()
		# CN should be put in the first lineage (index 1)
		if x <= cn_percentage:
			return 1
		else:
			lin_num = len(self.lineages)
			# when only 2 lineages do exist, all mutations must be assigned to lineage with index 1
			if lin_num == 2:
				return 1
			# if there are 3 lineages and the mutation is not clonal, it can only appear in the lineage
			# with the index 2
			if lin_num == 3:
				return 2
			# otherwise the mutations is assigned to one of the subclonal lineages uniformly
			return int(round(x * 100) % (lin_num - 2)) + 2
	
	# chooses state for CNV, can be "+1", "-1" or "+1" of one chromatid and "-1" on the other
	def choose_state(self):
		number = random.random()
		if number < self.p1_A_prop:
			return "+1"
		elif number < self.p1_A_prop + self.p1_A_B_prop:
			return "+1/+1"
		elif number < self.p1_A_prop + self.p1_A_B_prop + self.m1_B_prop:
			return "-1"
		elif number < self.p1_A_prop + self.p1_A_B_prop + self.m1_B_prop + self.m1_A_B_prop:
			return "-1/-1"
		elif number < self.p1_A_prop + self.p1_A_B_prop + self.m1_B_prop + self.m1_A_B_prop + self.p1_m1_prop:
			return "+1/-1"
		else:
			return "CNAs_mult_lin"

	# chooses randomly a segment and a phase and adds SNP to normal lineage
	def add_SNP_to_normal_lineage(self):
		seg = self.choose_segment()
		pha = self.choose_phase()
		self.lineages[cons.NORMAL].add_mutation_to_segment(pha, seg, cons.SNP)

	# computes a frequency table that adds up the frequency of all non-normal lineages step-wise
	def get_frequency_table(self):
		non_normal_lin_num = len(self.lineages) -1
		frequency_table = [0] * non_normal_lin_num

		# first needed frequency is the one of the first non-normal lineage
		frequency_table[0] = self.lineages[1].freq
		for i in xrange(1, non_normal_lin_num):
			frequency_table[i] = frequency_table[i-1] + self.lineages[i+1].freq

		return frequency_table


	# chooses randomly a lineage that is not the normal lineage, a segment,
	# and a phase and adds SSM to it if the chromatid is not lost
	def add_SSM(self):
		lost_chromatid = True
		while_counter = 0
		while lost_chromatid:
			# lineage is chosen to which the mutation is assigned
			# either chosen according to lineage frequencie
			# or uniformly
			if self.addSSMsAccoringToFreqs:
				lin = self.choose_lineage_by_frequency()
			elif self.clonal_ssm_percentage != -1:
				lin = self.choose_lineage_but_normal_with_clonal_percentage(self.clonal_ssm_percentage)
			else:
				lin = self.choose_lineage_but_normal()

			seg = self.choose_segment()
			pha = self.choose_phase()
			lost_chromatid = self.is_chromatid_lost(lin, seg, pha)
			while_counter += 1
			if while_counter == 1000:
				raise eo.myException("While-loop in add_SSM doesn't terminate.")
		self.lineages[lin].add_mutation_to_segment(pha, seg, cons.SSM)

	# adds x SSMs per lineage per segment per phase if the chromatid is not lost
	def add_x_SSMs_per_unit(self, x):
		self.ssm_number = 0
		for lin in range(1, len(self.lineages)):
			for seg in range(self.segment_number):
				for pha in range(cons.PHASE_NUMBER):
					if not self.is_chromatid_lost(lin, seg, pha):
						for i in xrange(x):
							self.lineages[lin].add_mutation_to_segment(
								pha, seg, cons.SSM)
							self.ssm_number += 1

	# checks whether the chromatid part in the given segment and phase got lost in the current
	# lineage or in one of its ancestors
	def is_chromatid_lost(self, lin, seg, pha):
		# if chromatid got lost in ancestral lineage of lin
		for ancestor in self.lineages[lin].ancestral_lineages:
			if self.lineages[ancestor].segments[pha][seg][cons.CNV] == ["-1"]:
				return True
		# if chromatid got lost in current lineage lin
		if self.lineages[lin].segments[pha][seg][cons.CNV] == ["-1"]:
			return True
		# the chromatid did not got lost
		return False

	# chooses randomly a lineage that is not the normal one and a phase
	# and adds CNV if segment doesn't have another CNV already
	def add_CNV(self):
		# forbid that all chromatids are deleted in lineage 1
		all_chromatids_deleted = True
		while_counter = 0
		while all_chromatids_deleted:
			# add CNVs in first segments
			for seg in xrange(self.cnv_number):
				lin = self.choose_lineage_but_normal_with_clonal_percentage(self.clonal_cn_percentage)
				state = self.choose_state()
				if state == "+1":
					self.lineages[lin].add_mutation_to_segment(
						cons.A, seg, cons.CNV, state="+1")
					self.cn_A[seg] = self.cn_A[seg] + self.lineages[lin].freq
				elif state == "+1/+1":
					self.lineages[lin].add_mutation_to_segment(
						cons.A, seg, cons.CNV, state="+1")
					self.lineages[lin].add_mutation_to_segment(
						cons.B, seg, cons.CNV, state="+1")
					self.cn_A[seg] = self.cn_A[seg] + self.lineages[lin].freq
					self.cn_B[seg] = self.cn_B[seg] + self.lineages[lin].freq
				elif state == "-1":
					self.lineages[lin].add_mutation_to_segment(
						cons.B, seg, cons.CNV, state="-1")
					self.cn_B[seg] = self.cn_B[seg] - self.lineages[lin].freq
				elif state == "-1/-1":
					self.lineages[lin].add_mutation_to_segment(
						cons.A, seg, cons.CNV, state="-1")
					self.lineages[lin].add_mutation_to_segment(
						cons.B, seg, cons.CNV, state="-1")
					self.cn_A[seg] = self.cn_A[seg] - self.lineages[lin].freq
					self.cn_B[seg] = self.cn_B[seg] - self.lineages[lin].freq
				elif state == "+1/-1":
					self.lineages[lin].add_mutation_to_segment(
						cons.A, seg, cons.CNV, state="+1")
					self.lineages[lin].add_mutation_to_segment(
						cons.B, seg, cons.CNV, state="-1")
					self.cn_A[seg] = self.cn_A[seg] + self.lineages[lin].freq
					self.cn_B[seg] = self.cn_B[seg] - self.lineages[lin].freq
				elif state == "CNAs_mult_lin":
					self.add_CNAs_to_mult_lineages(lin, seg)
				self.cn_total[seg] = self.cn_A[seg] + self.cn_B[seg]
			# remaining segments don't get a CNV
			for seg in xrange(self.cnv_number, self.segment_number):
				self.lineages[cons.NORMAL].add_mutation_to_segment(
					cons.A, seg, cons.CNV, state="0")
			# check if all chromatids are deleted in lineage 1
			for seg in xrange(self.segment_number):
				# is chromatid A deleted?
				all_chromatids_deleted = self.is_chromatid_lost(1, seg, cons.A)
				if all_chromatids_deleted:
					# is chromatid B deleted?
					all_chromatids_deleted = self.is_chromatid_lost(1, seg, cons.B)
					if not all_chromatids_deleted:
						break
				else:
					break
			# if all chromatids are deleted in the first lineage, the CNVs are removed
			# and new ones are sampled in the next round
			# also the CNs are set back to default
			if all_chromatids_deleted:
				for seg in xrange(self.segment_number):
					self.lineages[1].segments[cons.A][seg][cons.CNV] = []
					self.lineages[1].segments[cons.B][seg][cons.CNV] = []
					self.cn_A = [1] * self.segment_number
					self.cn_B = [1] * self.segment_number
					self.cn_total = [2] * self.segment_number

			# check that while-loop doesn't run forever
			while_counter += 1
			if while_counter == 1000:
				raise eo.myException("While-loop in add_CNV doesn't terminate.")

	def add_CNAs_to_mult_lineages(self, first_lin, seg):
		# choose loss or gain
		number = random.random()
		if number < 0.5:
			mutation = cons.LOSS
			state = "-1"
		else:
			mutation = cons.GAIN
			state = "+1"
		# choose allele
		number = random.random()
		if number < 0.5:
			allele = cons.A
		else:
			allele = cons.B
		
		second_lin_valid = False
		loop_counter = 0
		while second_lin_valid == False:
			# check loop iteration
			if loop_counter == 1000:
				raise eo.MyException("Too many iterations when trying 'add_CNAs_to_mult_lineages'.")
			
			# choose second lineage
			second_lin = self.choose_lineage_but_normal_with_clonal_percentage(self.clonal_cn_percentage)
			# if lineage is the same than first, do again
			if second_lin == first_lin:
				loop_counter += 1
				continue

			# choose allele for second lineage
			number = random.random()
			if number < 0.5:
				second_allele = cons.A
			else:
				second_allele = cons.B
			# if mutation is loss and second lineage has same allele, check whether lineages are in
			# ancestor-descendant relationship
			if mutation == cons.LOSS and allele == second_allele:
				# if lins are in ADR, do again
				if self.lins_in_adr(first_lin, second_lin):
					loop_counter += 1
					continue
				else:
					second_lin_valid = True
			else:
				second_lin_valid = True

		# add mutations to lineages
		self.lineages[first_lin].add_mutation_to_segment(allele, seg, cons.CNV, state=state)
		self.lineages[second_lin].add_mutation_to_segment(second_allele, seg, cons.CNV, state=state)

		# change CN of allele A and B
		# for first lineage
		if allele == cons.A:
			if mutation == cons.GAIN:
				self.cn_A[seg] = self.cn_A[seg] + self.lineages[first_lin].freq
			else:
				self.cn_A[seg] = self.cn_A[seg] - self.lineages[first_lin].freq
		else:
			if mutation == cons.GAIN:
				self.cn_B[seg] = self.cn_B[seg] + self.lineages[first_lin].freq
			else:
				self.cn_B[seg] = self.cn_B[seg] - self.lineages[first_lin].freq
		# for second lineage
		if second_allele == cons.A:
			if mutation == cons.GAIN:
				self.cn_A[seg] = self.cn_A[seg] + self.lineages[second_lin].freq
			else:
				self.cn_A[seg] = self.cn_A[seg] - self.lineages[second_lin].freq
		else:
			if mutation == cons.GAIN:
				self.cn_B[seg] = self.cn_B[seg] + self.lineages[second_lin].freq
			else:
				self.cn_B[seg] = self.cn_B[seg] - self.lineages[second_lin].freq

	# checks whether to lineages are in an ancestor-descendant relationship
	def lins_in_adr(self, first_lin, second_lin):
		# determines which lineage could be the ancestor and which the descendant
		if first_lin < second_lin:
			ancestor = first_lin
			descendant = second_lin
		elif first_lin > second_lin:
			ancestor = second_lin
			descendant = first_lin
		else:
			raise eo.MyException("This should not happen.")

		# checks relationship
		if descendant in self.lineages[ancestor].sublin:
			return True
		else:
			return False
	
	# adds CNVs to lineages accoriding to an earlier assignment 
	def add_CNV_according_assignment(self, assignment):
		for a in assignment:
			lin_index = a[0]
			phase = a[1]
			seg = a[2]
			state = a[3]
			# store CN change in lineage
			self.lineages[lin_index].add_mutation_to_segment(phase=phase,
				segment=seg, mutation=cons.CNV, state=state)
			# set CN of segment
			if phase == cons.A:
				if state == "+1":
					self.cn_A[seg] = self.cn_A[seg] + self.lineages[lin_index].freq
				elif state == "-1":
					self.cn_A[seg] = self.cn_A[seg] - self.lineages[lin_index].freq
			else:
				if state == "+1":
					self.cn_B[seg] = self.cn_B[seg] + self.lineages[lin_index].freq
				elif state == "-1":
					self.cn_B[seg] = self.cn_B[seg] - self.lineages[lin_index].freq
			self.cn_total[seg] = self.cn_A[seg] + self.cn_B[seg]

	# adds CNVs to lineages accoriding to an earlier assignment 
	def add_SNP_SSM_according_assignment(self, assignment, mut):
		for a in assignment:
			self.lineages[a[0]].add_mutation_to_segment(a[1], a[2], mut)
		# if mutations are SSMs check whether assignment is correct, so if SSMs don't 
		# happen to be on a chromatid that is deleted in the given lineage or in an
		# ancestral lineage
		if mut == cons.SSM:
			self.check_SSM_assignment()
	
	# checks whether SSM assignment is correct, so if SSMs don't 
	# happen to be on a chromatid that is deleted in the given lineage or in an
	# ancestral lineage
	def check_SSM_assignment(self):
		for lin_index in range(1, len(self.lineages)):
			for seg_index in range(self.segment_number):
				for phase in range(cons.PHASE_NUMBER):
					# ancestral lineage and lineage lin_index itself have chromatid lost
					# and SSMs are assigned on that chromatid
					if (self.is_chromatid_lost(lin_index, seg_index, phase) 
						and self.lineages[lin_index].get_mut_count(phase, 
						seg_index, cons.SSM) > 0):
						error_message = "SSMs are assigned to chromatid on segment {0} in phase {1} in lineage {2} but this chromatid is lost.".format(seg_index, phase, lin_index)
						raise(eo.SSMAssignmentException(error_message))
		return True
		

	# returns the complete number of either SNP or SSM per segment
	# all lineages are considered
	def get_mut_count_per_segment(self, seg, mut):
		mut_count = 0
		for lin in self.lineages:
			for pha in range(cons.PHASE_NUMBER):
				mut_count = mut_count + lin.get_mut_count(pha, seg, 
					mut)
		return mut_count

	# checks whether more SNPs or SSMs are assigned to a segment
	# and returns highest count
	def get_highest_single_mut_count_per_segment(self, seg):
		snp_count = self.get_mut_count_per_segment(seg, cons.SNP)
		ssm_count = self.get_mut_count_per_segment(seg, cons.SSM)
		if (snp_count >= ssm_count):
			return snp_count
		else:
			return ssm_count
	
	# computes the start and end position of one segment
	# given the end position of the segment before, the new segments
	#	starts just after this position
	# it is as long as the highest number of either SNPs or SSMS in this
	#	segment times 1000 plus 1000
	#	so that no segment has 0 length even though no SSM appears in it
	def compute_start_end_pos_of_one_segment(self, seg, old_stop):
		highest_mut_count = self.get_highest_single_mut_count_per_segment(
			seg)
		start = old_stop + 1
		end = old_stop + (highest_mut_count * 1000) + 1000
		# check for minimal length of segment
		if end - start + 1 < self.seg_min_length:
			end = start + self.seg_min_length - 1
		return (start, end)

	# returns start and end positions of all segments
	def compute_start_end_pos_of_all_segments(self):
		seg_start = []
		seg_end = []
		old_end = 0
		for seg in range(self.segment_number):
			(start, end) = self.compute_start_end_pos_of_one_segment(seg,
				old_end)
			seg_start.append(start)
			seg_end.append(end)
			old_end = end
		return (seg_start, seg_end)

	# takes frequency of given lineage and when descendant lineage
	#	has a CN change, changes in frequency are considered
	#	also checks, whether a copy number loss happens in the 
	#	current lineage on the same chromatid
	#	TODO: can frequency also be influenced by a CN gain in the same lineage?
	#	--> for allele-specific scenario, the current setting works!
	def compute_frequency_with_sublins(self, pha, seg, lin, freq):
		# if CN loss happens in the same lineage on the same chromatid,
		# the frequency is 0 because this chromatid is lost
		if self.lineages[lin].segments[pha][seg][cons.CNV] == ["-1"]:
			return 0
		for sublin in self.lineages[lin].sublin:
			if(self.lineages[sublin].get_mut_count(pha, seg, 
				cons.CNV) > 0):
				if self.lineages[sublin].segments[pha][seg][cons.CNV] == ["+1"]:
					freq = freq + self.lineages[sublin].freq
				if self.lineages[sublin].segments[pha][seg][cons.CNV] == ["-1"]:
					freq = freq - self.lineages[sublin].freq
		return freq

	# computes the coverage per segment
	# coverage results in complete frequency times sequencing mass
	def compute_phased_segment_read_number_float(self, pha, seg, lin, freq):
		return self.compute_frequency_with_sublins(pha, seg, lin, freq) * self.mass

	# computes a value sampled from a beta binomial distribution
	# given the parameters n and p, which correspond to a binomal distribution,
	# and the parameter overdispersion
	def beta_binomial(self, n, p, overdispersion):
		a = p * overdispersion
		b = (1 - p) * overdispersion
		random_var_beta = np.random.beta(a, b)
		return np.random.binomial(n, random_var_beta)

	# computes a value sampled from a negative binomial distribution
	# given the parameters coverage (mu), which corresponds to a poisson distribution,
	# and overdispersion
	def negative_binomial(self, coverage, overdispersion):
		p = float(overdispersion) / (overdispersion + coverage)
		n = 0
		try:
			n = float(coverage * p) / (1 - p)
		except ZeroDivisionError:
			raise ZeroDivisionError
		return np.random.negative_binomial(n, p)

	# noise is added the the CNs
	# this means, we draw c' = max(0, N(\mu, \sigma^2))
	# note that np.random.normal takes the standard deviation, not the variance as input
	def add_noise_to_cn(self, cns, standard_errors):
		return [max(0, np.random.normal(cns[i], standard_errors[i])) for i in xrange(len(cns))]

	# standrad error of the segment is computed, depends on their length
	# computes also the CN of the segment, this can be used when not the true one but a noisy one should be used
	# seg_start, seg_end: start and end position of each segment
	def compute_cn_standard_error(self, seg_start, seg_end):
		# create empty lists for calculated CN
		calculated_cn_A = []
		calculated_cn_B = []

		# for each segment
		for i in xrange(self.segment_number):
			# length of segment
			seg_length = seg_end[i] - seg_start[i] + 1
			# number of heterozygous SNPs in segment
			het_snp_number = max(1, int(round(7.0/10000.0 * seg_length)))
			
			# compute standard error for allele A & B
			self.standard_error_A.append(self.compute_cn_standrad_error_as(self.cn_A[i], het_snp_number,
				calculated_cn_A))
			self.standard_error_B.append(self.compute_cn_standrad_error_as(self.cn_B[i], het_snp_number,
				calculated_cn_B))

		return calculated_cn_A, calculated_cn_B
			
	# allele-specific calculation of standard error and CN
	def compute_cn_standrad_error_as(self, real_cn, het_snp_number, calculated_cn_list):
		# simulate tumor and normal read count per heterozygous SNP position
		as_tumor_coverage = self.mass * real_cn
		tumor_count = [self.add_noise(as_tumor_coverage) for _ in xrange(het_snp_number)]
		normal_count = [max(1, self.add_noise(self.mass)) for _ in xrange(het_snp_number)]
		
		# calculate CN from this
		calculated_cn = (float(sum([float(tumor_count[i])/float(normal_count[i]) for i in xrange(het_snp_number)])) /
			het_snp_number)
		calculated_cn_list.append(calculated_cn)

		# compute standard error
		denominator = het_snp_number * (het_snp_number - 1)
		if denominator == 0:
			denominator = 1
		st_err = (np.sqrt((sum([(float(tumor_count[i])/float(normal_count[i]))**2 for i in xrange(het_snp_number)])
			- float(het_snp_number * (calculated_cn**2))) / (denominator)))
		# following only happens for very small segments, e.g. in simulation
		if st_err == 0 and het_snp_number == 1:
			return 0.00001

		return st_err

	# adds noise to the coverage
	# negative binomial distribution is used for noise with overdispersion
	# otherwise the poisson distribution is used
	def add_noise(self, read_count):
		if self.overdispersion:
			return self.negative_binomial(read_count, self.coverage_overdispersion)
		else:
			return np.random.poisson(lam=read_count, size=1)[0]

	def compute_ref_count(self, mut_count, total_count):
		ref_count = total_count - mut_count
		if ref_count < 0:
			return 0
		if (self.noise):
			ref_count = self.add_noise(ref_count)
		return ref_count

	# creates list of coverage per segment
	# noise can be added to coverage
	def compute_total_reads(self):

		reads_total = [int(round(self.cn_total[seg] * self.mass)) for seg in xrange(self.segment_number)]

		if self.noise:
			reads_total = map(self.add_noise, reads_total)

		return reads_total
	
	def compute_and_set_variant_reads_for_seg_pha_lin(self, reads_total, 
		seg_start, seg, pha, lin_index, lin, offset):

		# if allele-specific copy numbers are simulated, no
		# SNPs are needed
		if self.allele_specific and lin_index == cons.NORMAL:
			return

		# according to the index of the lineage, the mutation types
		#	get chosen
		# if the index belongs to the normal lineage, SNPs are considered
		# otherwise SSMs
		mut = cons.SNP
		if (lin_index != cons.NORMAL):
			mut = cons.SSM

		mut_count = lin.get_mut_count(pha, seg, mut)

		# mutations occur
		if (mut_count > 0):
			if self.new_version:
				# CN of mutation is computed
				# CN equals the sum of the frequency of all lineages the mutations is present in
				mut_cn = self.compute_frequency_with_sublins(pha, seg, lin_index, lin.freq)

				# for each mutation
				for mut_index in range(mut_count):
					mut_cn_per_ssm = mut_cn
					# if lineage has a CN gain in this phase, the SSM can be 
					# influenced by it, thus it's CN is higher
					cn_infl = False
					if lin.segments[pha][seg][cons.CNV] == ["+1"]:
						if random.random() <= self.SSM_before_CNV_LH:
							mut_cn_per_ssm += lin.freq
							cn_infl = True

					# segment_coverage need to be set to default value at the beginning of each round
					segment_coverage = reads_total[seg]

					# noise for the coverage is added
					if self.noise:
						if segment_coverage == 0:
							raise eo.MyException("Segment coverage is 0.")
						segment_coverage = self.add_noise(segment_coverage)

					# frequency of mutation is computed
					mutation_frequency = float(mut_cn_per_ssm)/float(self.cn_total[seg])

					# variation count is computed
					var_count = 0
					# noise for the mutation
					if self.noise:
						# if overdispersed data should be generated
						if self.overdispersion:
							var_count = self.beta_binomial(segment_coverage,
								mutation_frequency,
								self.frequency_overdispersion)
						# if data shouldn't be overdispersed
						else:
							var_count = np.random.binomial(
								segment_coverage, mutation_frequency)
					# if data should be created without noise
					else:
						var_count = int(round(segment_coverage * mutation_frequency))
					# variant has to be observed
					if var_count == 0:
						var_count = 1
					# reference count is computed from segment coverage and variation
					# count, no matter if noise is used or not
					ref_count = int(round(segment_coverage - var_count))
					if ref_count < 0:
						ref_count = 0

					# particular mutation in lineage gets values
					lin.segments[pha][seg][mut][mut_index].variant_count = var_count
					lin.segments[pha][seg][mut][mut_index].ref_count = ref_count
					lin.segments[pha][seg][mut][mut_index].pos = (
						seg_start[seg] + (mut_index + offset[mut]) * 1000)
					lin.segments[pha][seg][mut][mut_index].chr = 1
					if cn_infl == True:
						lin.segments[pha][seg][mut][mut_index].infl_cnv_same_lin = True

			# old version
			else:
				# number of reads that should cover the mutations on this
				#	segment, in this phase and lineage
				phased_read_number = int(round(self.
					compute_phased_segment_read_number_float(pha, seg, 
					lin_index, lin.freq)))
				
				for mut_index in range(mut_count):
					tmp_read_number = phased_read_number
					if self.noise:
						tmp_read_number = self.add_noise(
							tmp_read_number)
					# given a potentially individual number of reads
					#	covering the mutation, if noise is
					#	taken into account in the simulation,
					#	the number of reads covering the
					# 	refenence loci are calculated
					ref_count = self.compute_ref_count(tmp_read_number, 
						reads_total[seg])

					# particular mutation in lineage gets values
					lin.segments[pha][seg][mut][mut_index].variant_count = tmp_read_number
					lin.segments[pha][seg][mut][mut_index].ref_count = ref_count
					lin.segments[pha][seg][mut][mut_index].pos = (
						seg_start[seg] + (mut_index + offset[mut]) * 1000)
					lin.segments[pha][seg][mut][mut_index].chr = 1

			# offset for type of mutation is stored so that 
			#	positions of the next mutations of the same type
			#	in the same segment can be numbered in one row
			offset[mut] = offset[mut] + mut_count

	# computes the number of reads covering each SNP and SSM in all segments
	def compute_and_set_variant_reads(self, reads_total, seg_start):
		for seg in range(self.segment_number):
			# offset, first entry for SNP, second for SSM
			offset = [0, 0]
			for pha in range(cons.PHASE_NUMBER):
				for lin_index, lin in enumerate(self.lineages):
					self.compute_and_set_variant_reads_for_seg_pha_lin(
						reads_total, seg_start, seg, pha, 
						lin_index, lin, offset)

	# computes the average coverage of each segment given the counts of the SNPs
	# in this segment
	# TODO: if no SNPs are simulated for a segment, it's coverage is currently
	# set to 0
	def get_average_coverage(self):
		average_coverage = []
		for seg_index in xrange(self.segment_number):
			snp_num = 0
			snp_coverage = 0.0
			for pha in range(cons.PHASE_NUMBER):
				snps = self.lineages[cons.NORMAL].segments[pha][seg_index][cons.SNP]
				for i in xrange(len(snps)):
					snp_coverage = (snp_coverage + snps[i].variant_count +
						snps[i].ref_count)
				snp_num += len(snps)
			if snp_num > 0:
				snp_coverage = float(snp_coverage) / float(snp_num)
			else:
				snp_coverage = 0
			average_coverage.append(round(snp_coverage))
		return average_coverage

	# computes the haploid sequencing mass per segment, given the average coverage per segment
	# and the CN per segment
	def get_haploid_sequencing_mass(self, reads_total):
		haploid_sequencing_mass = []
		for seg_index in xrange(self.segment_number):
			copy_number = (self.compute_frequency_with_sublins(cons.A, seg_index, cons.NORMAL, 1)
				+ self.compute_frequency_with_sublins(cons.B, seg_index, cons.NORMAL, 1))
			haploid_sequencing_mass.append(reads_total[seg_index] / float(copy_number))
		return haploid_sequencing_mass


def main(lineages_file, segment_number, snp_number, ssm_number, cnv_number,
        mass, file_end_tag, file_start_tag="", noise=True, CN_standard_deviation=0.1,
	no_test=True, CNV_assignment=None, SNP_assignment=None, SSM_assignment=None, 
        new_version=True, coverage_overdispersion=1000, allele_specific=True,
        SSM_num_per_unit=0, frequency_overdispersion=1000, overdispersion=True,
	clonal_cn_percentage=0.75, p1_A_prop=0.3125, p1_A_B_prop=0.125,
	m1_B_prop=0.3125, m1_A_B_prop=0.21875, p1_m1_prop=0.03125,
	SSM_before_CNV_LH=0.5, addSSMsAccoringToFreqs=False, clonal_ssm_percentage=None,
	CN_noise=True, seg_min_length=1000000, CNAs_mult_lin_prop=0):



		sim = Data_Simulation()

                sim.do_simulation(lineages_file, segment_number, snp_number,
                        ssm_number, cnv_number, mass, file_end_tag,
                        file_start_tag=file_start_tag, noise=noise,
			CN_standard_deviation=CN_standard_deviation,
                        no_test=no_test, CNV_assignment=CNV_assignment,
                        SNP_assignment=SNP_assignment,
                        SSM_assignment=SSM_assignment,
                        SSM_num_per_unit=SSM_num_per_unit,
                        new_version=new_version,
                        coverage_overdispersion=coverage_overdispersion,
                        frequency_overdispersion=frequency_overdispersion,
                        overdispersion=overdispersion,
                        allele_specific=allele_specific,
			clonal_cn_percentage=clonal_cn_percentage,
			p1_A_prop=p1_A_prop, p1_A_B_prop=p1_A_B_prop, m1_B_prop=m1_B_prop,
			m1_A_B_prop=m1_A_B_prop, p1_m1_prop=p1_m1_prop,
			SSM_before_CNV_LH=SSM_before_CNV_LH, addSSMsAccoringToFreqs=addSSMsAccoringToFreqs,
			clonal_ssm_percentage=clonal_ssm_percentage, CN_noise=CN_noise,
			seg_min_length=seg_min_length, CNAs_mult_lin_prop=CNAs_mult_lin_prop)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--lineages_file", type = str, required=True)
	parser.add_argument("--segment_number", type=int, required=True)
	parser.add_argument("--snp_number", type=int, default=0)
	parser.add_argument("--ssm_number", type=int, required=True)
	parser.add_argument("--cnv_number", type=int, required=True)
	parser.add_argument("--mass", type=int, required=True)
	parser.add_argument("--file_end_tag", type=str, required=True)
	parser.add_argument("--file_start_tag", type=str, default="")
	parser.add_argument("--noise", default=True)
	parser.add_argument("--CN_standard_deviation", default=0.1)
	parser.add_argument("--no_test", default=True)
	parser.add_argument("--CNV_assignment", type=str, default=None)
	parser.add_argument("--SNP_assignment", type=str, default=None)
	parser.add_argument("--SSM_assignment", type=str, default=None)
	parser.add_argument("--new_version", default=True)
	parser.add_argument("--coverage_overdispersion", type=float, default=1000.0)
	parser.add_argument("--frequency_overdispersion", type=float, default=1000.0)
	parser.add_argument("--overdispersion", default=True)
	parser.add_argument("--allele_specific", default=True)
	parser.add_argument("--SSM_num_per_unit", default=0)
	parser.add_argument("--clonal_cn_percentage", default=0.75)
	parser.add_argument("--clonal_ssm_percentage", default=None)
	parser.add_argument("--p1_A_prop", default=0.3125)
	parser.add_argument("--p1_A_B_prop", default=0.125)
	parser.add_argument("--m1_B_prop", default=0.3125)
	parser.add_argument("--m1_A_B_prop", default=0.21875)
	parser.add_argument("--p1_m1_prop", default=0.03125)
	parser.add_argument("--CNAs_mult_lin_prop", default=0.0)
	parser.add_argument("--SSM_before_CNV_LH", default=0.5)
	parser.add_argument("--CN_noise", default=True)
	parser.add_argument("--addSSMsAccoringToFreqs", default=False)
	parser.add_argument("--seg_min_length", default=1000000)

	args = parser.parse_args()

	# convert to boolean
	noise = True
	if args.noise and isinstance(args.noise, basestring):
		noise = oio.str_to_bool(args.noise)
	CN_noise = True
	if args.CN_noise and isinstance(args.CN_noise, basestring):
		CN_noise = oio.str_to_bool(args.CN_noise)
	no_test = True
	if args.no_test and isinstance(args.no_test, basestring):
		no_test = oio.str_to_bool(args.no_test)
	overdispersion = True
	if args.overdispersion and isinstance(args.overdispersion, basestring):
		overdispersion = oio.str_to_bool(args.overdispersion)
	allele_specific = True
	if args.allele_specific  and isinstance(args.allele_specific, basestring):
		allele_specific = oio.str_to_bool(args.allele_specific)
	addSSMsAccoringToFreqs = False
	if args.addSSMsAccoringToFreqs and isinstance(args.addSSMsAccoringToFreqs, basestring):
		addSSMsAccoringToFreqs = oio.str_to_bool(args.addSSMsAccoringToFreqs)

	main(args.lineages_file, args.segment_number, args.snp_number, args.ssm_number, 
		args.cnv_number, args.mass, args.file_end_tag, file_start_tag=args.file_start_tag, 
		noise=noise, CN_standard_deviation=float(args.CN_standard_deviation), 
		no_test=no_test, CNV_assignment=args.CNV_assignment, 
		SNP_assignment=args.SNP_assignment, SSM_assignment=args.SSM_assignment, 
		new_version=args.new_version, coverage_overdispersion=args.coverage_overdispersion, 
		frequency_overdispersion=args.frequency_overdispersion, overdispersion=overdispersion,
		allele_specific=allele_specific, SSM_num_per_unit=int(args.SSM_num_per_unit),
		clonal_cn_percentage=float(args.clonal_cn_percentage), p1_A_prop=float(args.p1_A_prop),
		p1_A_B_prop=float(args.p1_A_B_prop), m1_B_prop=float(args.m1_B_prop),
		m1_A_B_prop=float(args.m1_A_B_prop), p1_m1_prop=float(args.p1_m1_prop), 
		SSM_before_CNV_LH=float(args.SSM_before_CNV_LH), addSSMsAccoringToFreqs=addSSMsAccoringToFreqs,
		clonal_ssm_percentage=float(args.clonal_ssm_percentage), CN_noise=CN_noise,
		seg_min_length=args.seg_min_length, CNAs_mult_lin_prop=float(args.CNAs_mult_lin_prop))


	#(prog, lineages_file, segment_number, snp_number, ssm_number, cnv_number, noise, mass, output_segments, 
	#	output_snps, output_ssms, output_results, output_info, no_test) = sys.argv
	#sim = Data_Simulation()
	#sim.do_simulation(lineages_file, int(segment_number), int(snp_number), int(ssm_number), int(cnv_number),  
	#	oio.str_to_bool(noise), int(mass), output_segments, output_snps, output_ssms, output_results, 
	#	output_info, oio.str_to_bool(no_test))


