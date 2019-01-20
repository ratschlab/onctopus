import snp_ssm
import segment
import constants as cons
import exceptions_onctopus as eo

class Lineage_Simulation(object):

	def __init__(self, sublin, freq, segment_number):
		self.sublin = sublin
		self.freq = freq
		phase_number = 2
		self.ancestral_lineages = []

		# segments list looks like: 
		# [phaseA [segment [SNP],[SSM],[CNV],[...]],[[[...]]]
		# 3 because of number of entries for SNP, SSM and CNV
		self.segments = [[[[] for __ in range(3)]
			for i in range(segment_number)]
			for j in range(phase_number)]

	def add_mutation_to_segment(self, phase, segment, mutation, state=None):
		if (mutation == cons.CNV):
			if (len(self.segments[phase][segment][mutation]) != 0):
				raise(eo.AddingException("Segment {0} already has one CNV"
					" in phase {1}.".format(segment, phase)))
			else: 
				self.segments[phase][segment][mutation].append(state)
		elif (mutation == cons.SNP):
			self.segments[phase][segment][mutation].append(snp_ssm.SNP())
		elif (mutation == cons.SSM):
			self.segments[phase][segment][mutation].append(snp_ssm.SSM())

	def get_mut_count(self, phase, segment, mutation):
		return len(self.segments[phase][segment][mutation])
		
