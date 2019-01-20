import cplex
from cplex.exceptions import CplexError
import sys
import constants as cons
import model
import logging
import time
import exceptions_onctopus as oe
import numpy as np
import onctopus_io as oio

class Optimization_with_CPLEX(object):

	def __init__(self, seg_splines, snp_splines, 
		ssm_splines, allele_specific=False, seg_splines_A=[], seg_splines_B=[],
		simple_CN_changes=True, max_x_CN_changes=-1, only_one_loss=True, 
		only_gains_losses_LOH=True, cn_weight=0, z_trans_weight=0.00001, single_segment=False,
		z_matrix_is_fixed=False, use_lineage_divergence_rule=True):

		self.allele_specific = allele_specific
		self.simple_CN_changes = simple_CN_changes
		self.max_x_CN_changes = max_x_CN_changes
		self.only_one_loss = only_one_loss
		self.only_gains_losses_LOH = only_gains_losses_LOH
		# weight must be >= 0
		if cn_weight >= 0:
			self.cn_weight = cn_weight
		else:
			self.cn_weight = - cn_weight

		self.z_trans_weight = z_trans_weight
		if z_trans_weight < 0:
			self.z_trans_weight = -z_trans_weight

		# used when only one segment is used in the segment-wise heuristic
		# then not the segment index of the SNP or SSM is used but the index 0
		# TODO do for SNPs
		self.single_segment = single_segment

		if self.allele_specific:
			self.seg_splines_A = seg_splines_A
			self.seg_splines_B = seg_splines_B
		else :
			self.seg_splines = seg_splines
		self.snp_splines = snp_splines
		self.ssm_splines = ssm_splines
		
		# seg and cnvs
		if self.allele_specific:
			self.seg_num = len(self.seg_splines_A)
		else:
			self.seg_num = len(self.seg_splines)
		# only one gain and one loss allowed so far
		self.cnv_state_num = 2
		# number of auxiliary matrices for delta_c_a/b, at the moment
		# binary/float for plus one copy, so 2
		self.aux_matrices_cnv_linear_types_num = 2
		
		# if lineage divergence rule should be used
		self.use_lineage_divergence_rule = use_lineage_divergence_rule

		# if Z-matrix is fixed
		# if Z-matrix is fixed, its values get fixed in a function
		# if this function is called, it sets self.z_matrix_is_fixed to True
		self.z_matrix_is_fixed = z_matrix_is_fixed

		# CN 
		self.normal_copy_number = 2.0
		self.starting_allele_specific_copy_number = 1.0

		# snps
		self.snp_num = len(self.snp_splines)
		# number of entries in one snp matrix is equal to snp number as we
		# already know that SNPs only appear in first lineage
		self.delta_snp_entry_num = self.snp_num
	
		# ssms
		self.ssm_num = len(self.ssm_splines)

		# whether all variables should be fixed in first run
		self.fix_all = False

		# arrays for variables and colnames
		self.my_obj = []
		self.my_ub = []
		self.my_lb = []
		self.my_ctype = []
		self.my_colnames = []
		self.my_rhs = []
		self.my_rownames = []
		self.my_sense = [] 
		self.my_rows = []
		
		self.my_colnames_seg = []
		self.my_colnames_seg_index = []
		self.my_colnames_seg_A = []
		self.my_colnames_seg_A_index = []
		self.my_colnames_seg_B = []
		self.my_colnames_seg_B_index = []
		self.my_colnames_snp = []
		self.my_colnames_ssm = []
		self.my_colnames_ssm_index = []
		self.my_phis = []
		self.my_phis_index = []
		self.my_colnames_dc_a_p1_binary = []
		self.my_colnames_dc_b_p1_binary = []
		self.my_colnames_dc_a_p1_binary_index = []
		self.my_colnames_dc_b_p1_binary_index = []
		self.my_colnames_dc_a_p1_float = []
		self.my_colnames_dc_b_p1_float = []
		self.my_colnames_dc_a_p1_float_index = []
		self.my_colnames_dc_b_p1_float_index = []
		self.my_colnames_dc_a_p1_binary = []
		self.my_colnames_dc_b_p1_binary = []
		self.my_colnames_dc_a_p1_binary_index = []
		self.my_colnames_dc_b_p1_binary_index = []
		self.my_colnames_dc_a_p1_float = []
		self.my_colnames_dc_b_p1_float = []
		self.my_colnames_dc_a_p1_float_index = []
		self.my_colnames_dc_b_p1_float_index = []
		self.my_colnames_dc_ancestral_a_m1 = []
		self.my_colnames_dc_ancestral_b_m1 = []
		self.my_colnames_dc_ancestral_a_m1_index = []
		self.my_colnames_dc_ancestral_b_m1_index = []
		self.my_colnames_z = []
		self.my_colnames_z_index = []
		self.my_colnames_dsnp = []
		self.my_colnames_dsnp_a = []
		self.my_colnames_dsnp_b = []
		self.my_colnames_snp_w_cnv_a_p1 = []
		self.my_colnames_snp_w_cnv_b_p1 = []
		self.my_colnames_snp_cn = []

		# default CPLEX parameters
		self.time = 1e+75
		# (thread is not default)
		self.threads = 1
		self.probing = 0
		self.emph_switch = 0
		self.coef_reduc = -1
		self.mipgap = 1e-04
		# 4 is not the default value but showed good perfomances
		# in simulation
		self.symmetry = 4
		self.strategy_file = 1
		self.nodeselect = 1
		self.workmem = 128.0
		self.workdir = "/scratch"
		self.treememory = 1e+75
		self.emphasis_memory = 0
		# minimal value of frequency sum over all samples that need to be achieved so that mutations are allowed
		# to be assigned to a lineage

	def set_CPLEX_parameters(self, time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
		strategy_file, workmem, workdir, treememory, emphasis_memory,
		nodeselect, cplex_log_file):
		self.time = time
		self.threads = threads
		self.probing = probing
		self.emph_switch = emph_switch
		self.coef_reduc = coef_reduc
		self.mipgap = mipgap
		self.symmetry = symmetry
		self.strategy_file = strategy_file
		self.nodeselect = nodeselect
		self.workmem = workmem
		self.workdir = workdir
		self.treememory = treememory
		self.emphasis_memory = emphasis_memory
		self.cplex_log_file = cplex_log_file
		message = ("Parameters used by CPLEX: time: {0}, threads: {1}, ".format(time, threads) 
			+ "probing: {0}, emph_switch: {1}, ".format(probing, emph_switch)
			+ "coef_reduc: {0}, mipgap: {1}, ".format(coef_reduc, mipgap)
			+ "symmetry: {0}, ".format(symmetry)
			+ "strategy_file: {0}, workmem: {1}, workdir: {2}, ".format(strategy_file, workmem, 
			self.workdir)
			+ "treememory: {0}, emphasis_memory: {1}, ".format(treememory, emphasis_memory)
			+ "cplex_log_file: {0}, nodeselect: {1}".format(cplex_log_file, nodeselect))
		
		logging.debug(message)

	# sets variables needed for optimization to empty arrays
	def empty_CPLEX_optimization_lists(self):
		# variables needed for optimization in CPLEX
		self.my_obj = []
		self.my_ub = []
		self.my_lb = []
		self.my_ctype = []
		self.my_colnames = []
		self.my_rhs = []
		self.my_rownames = []
		self.my_sense = [] 
		self.my_rows = []

	# removes specifc constraints from all relevant lists
	def remove_specific_constraints(self, start_index_constraints, end_index_constraints):
		if end_index_constraints > len(self.my_rhs):
			raise oe.MyException("Removing of constraints doesn't work here")
		del self.my_rhs[start_index_constraints : end_index_constraints]
		del self.my_rownames[start_index_constraints : end_index_constraints]
		del self.my_sense[start_index_constraints : end_index_constraints]
		del self.my_rows[start_index_constraints : end_index_constraints]

		
	# TODO remove list (snp, ssm, seg) entries, not used in this function...
	def set_other_parameter(self, sublin_num, snp_list, ssm_list, seg_list, fixed_avg_cn=None,
		fixed_avg_cn_start=-1, fixed_avg_cn_stop=-1):

		self.sublin_num = sublin_num

		# set CN for segments
		self.cn = []
		if self.allele_specific:
			for seg in seg_list:
				self.cn.append(seg.given_cn_A + seg.given_cn_B)
		else:
			for seg in seg_list:
				self.cn.append(float(seg.count) / float(seg.hm))
		# if fixed values for CN exist, set them for segments accordingly
		if fixed_avg_cn is not None:
			# if the CN for all segments are fixed
			if fixed_avg_cn_start == -1:
				self.cn = fixed_avg_cn
			# if the CN for some segments is not fixed, the previously calcualted
			# values are taken in that area
			else:
				self.cn = (fixed_avg_cn[:fixed_avg_cn_start] 
					+ self.cn[fixed_avg_cn_start:fixed_avg_cn_stop + 1] 
					+ fixed_avg_cn[fixed_avg_cn_start:])
	
		# number of entries in one delta_c matrix
		self.delta_c_entry_num = self.sublin_num * self.seg_num
		# number of auxilliary variables for entries in delta_c matrices 
		# to determine value in delta_c_a and delta_c_b
		self.cnv_aux_all_states_one_type_entry_num = (self.delta_c_entry_num * 
			cons.PHASE_NUMBER * self.cnv_state_num)

		if not self.allele_specific:
			# number of entries in all delta_S - SNP matrices
			if self.simple_CN_changes:
				self.all_delta_s_snp_entries_num = self.snp_num * cons.SNP_SSM_PHASING_MATRIX_NUM
			else:
				self.all_delta_s_snp_entries_num = self.snp_num * cons.PHASE_NUMBER

		# number of entries in one delta_S - ssm matrix
		self.delta_s_ssm_num = self.sublin_num * self.ssm_num
		# number of entries in all delta_s - ssm matrices
		if self.simple_CN_changes:
			self.all_delta_s_ssm_entries_num = self.delta_s_ssm_num * cons.SNP_SSM_PHASING_MATRIX_NUM
		else:
			self.all_delta_s_ssm_entries_num = self.delta_s_ssm_num * cons.PHASE_NUMBER

		if not self.allele_specific:
			# number of auxiliary variables for computing cn of snp in
			# linear way
			self.snp_aux_linear_variables_num = (self.snp_num * 
				self.cnv_state_num *
				cons.PHASE_NUMBER * (self.sublin_num - 1))

		# number of predefinded 0 entries in Z matrix
		self.pre_zero_z_num = ((self.sublin_num * self.sublin_num 
			+ self.sublin_num) / 2)
		# number of untrivial entries in Z-matrix
		self.untrivial_z_entries_num = model.get_number_of_untrivial_z_entries(self.sublin_num)

	#def save_part_of_solution_in_class(self):
	#	# saves the solution for the ancestral information (matrix Z) in a 
	#	# separate variable as some values might have to be changed later
	#	# skips the first row of matrix as this shows relation of normal
	#	# lineage to others, but this is known and does not have to be changed
	#	# TODO change later when this first row is skipped
	#	model.check_number_entries_z_matrix(self)
	#	self.solution_z = self.my_prob.solution.get_values()[self.z_index_start + self.sublin_num : 
	#		self.z_index_start + self.sublin_num * self.sublin_num]
	#	#print self.solution_z

	# optimizing function using CPLEX
	# creates all variables and constraints #1 - #29
	# also has constraints that can fix variables
	def opt_with_CPLEX(self, sublin_num, snp_list, ssm_list, 
		seg_list, fixed_cnv=None, unfixed_cnv_start=-1, unfixed_cnv_stop=-1, fixed_avg_cn=None,
		unfixed_avg_cn_start=-1, unfixed_avg_cn_stop=-1, fixed_snp=None, unfixed_snp_start=-1,
		unfixed_snp_stop=-1, fixed_ssm=None, unfixed_ssm_start=-1, unfixed_ssm_stop=-1, 
		fixed_z_matrix=None, 
		unfixed_z_matrix_start=-1, unfixed_z_matrix_stop=-1, fixed_phi=None, unfixed_phi_start=-1, 
		unfixed_phi_stop=-1, direct_descendants_for_constraints=[],
		warm_start_dc_binary=None, warm_start_dsnp=None, warm_start_dssm=None, warm_start_freqs=None,
		ssm_indices_per_cluster=None, fixed_cnv_indices=[], fixed_cnv_list_new=None,
		fixed_ssm_list_new=None, heuristic1=False, dont_break_z_symmetry=False,
		warm_start_z_matrix_list=None, start_freq=None, start_Z=None, start_SSMs=None, start_CNVs=None):

		# if SSMs should be clustered, instructions are contained in list
		message = ("Allowed copy number changes: simple_CN_changes: {0}, ".format(str(self.simple_CN_changes)) +
			"max_x_CN_changes: {0}, only_one_loss: {1}, ".format(self.max_x_CN_changes, str(self.only_one_loss)) +
			"only_gains_losses_LOH: {0}, cn weight: {1}, z_trans_weight: {2}, lin_div_rule: {3}, ".format(
			str(self.only_gains_losses_LOH), self.cn_weight, self.z_trans_weight, self.use_lineage_divergence_rule) +
			"dont_break_z_symmetry: {0}.".format(dont_break_z_symmetry))
		logging.debug(message)
		
		start_time = time.time()
		logging.info("Starting with the preparation for the optimization...")
		
		# set other variables
		self.set_other_parameter(sublin_num, snp_list, ssm_list, seg_list, fixed_avg_cn,
			unfixed_avg_cn_start, unfixed_avg_cn_stop)
	
		# empties lists needed for optimization
		self.empty_CPLEX_optimization_lists()
	
		# creation of variables that are optimized
		self.create_variables(fixed_z_matrix=fixed_z_matrix, dont_break_z_symmetry=dont_break_z_symmetry)

		#print "Going to exit..."
		#return None 

		# creation of constraints
		self.create_standard_constraints(snp_list, ssm_list, seg_list, fixed_z_matrix=fixed_z_matrix,
			fixed_phi=fixed_phi, dont_break_z_symmetry=dont_break_z_symmetry)
		warm_start_dc_binary_fixed, warm_start_dssm_fixed, warm_start_freqs_fixed, fixed_z_matrix_indices = (
			self.create_fixed_variables_constraints(fixed_cnv, unfixed_cnv_start, unfixed_cnv_stop, 
			fixed_avg_cn, unfixed_avg_cn_start, unfixed_avg_cn_stop, fixed_snp,
			unfixed_snp_start, unfixed_snp_stop, fixed_ssm, unfixed_ssm_start, unfixed_ssm_stop,
			fixed_z_matrix, unfixed_z_matrix_start, unfixed_z_matrix_stop, fixed_phi,
			unfixed_phi_start, unfixed_phi_stop, direct_descendants_for_constraints,
			ssm_indices_per_cluster, fixed_cnv_indices=fixed_cnv_indices,
			fixed_cnv_list_new=fixed_cnv_list_new, fixed_ssm_list_new=fixed_ssm_list_new))

		# check warm start values
		if warm_start_dc_binary is None and warm_start_dc_binary_fixed is not None:
			warm_start_dc_binary = warm_start_dc_binary_fixed
		if warm_start_dssm is None and warm_start_dssm_fixed is not None:
			warm_start_dssm = warm_start_dssm_fixed
		if warm_start_freqs is None and warm_start_freqs_fixed is not None:
			warm_start_freqs = warm_start_freqs_fixed
		z_matrix_for_warm_start = fixed_z_matrix
		if warm_start_z_matrix_list is not None:
			fixed_z_matrix_indices = self.get_z_matrix_indices()
			z_matrix_for_warm_start = warm_start_z_matrix_list

		self.last_regular_constraint = len(self.my_rows)
		if (start_freq is not None and start_Z is not None and start_SSMs is not None and start_CNVs is not None):
			self.fix_all = True
			# create fixing constraints
			self.create_fixed_variables_constraints(start_CNVs, -1, -1, None, None, None, None,
				None, None, start_SSMs, -1, -1, start_Z, -1, -1, start_freq, -1, -1, [], None)
			self.last_fixed_all_constraint = len(self.my_rows)

		end_time = time.time()
		used_time = end_time - start_time
		logging.debug("Time used to create variables and constraints: {0}".format(used_time))

		# Print variables and constraints
		#self.print_variables_constraints()

		if heuristic1 == True:
			return
	
		#return 0
		self.start_CPLEX(warm_start_dc_binary=warm_start_dc_binary, warm_start_dsnp=warm_start_dsnp, 
			warm_start_dssm=warm_start_dssm, warm_start_freqs=warm_start_freqs, z_matrix=z_matrix_for_warm_start,
			fixed_z_matrix_indices=fixed_z_matrix_indices)

	# creates variables that are going to be optimized
	def create_variables(self, fixed_z_matrix=None, dont_break_z_symmetry=False):
		logging.debug("Variables are created...")
		self.vars_aux_vars_mut_splines()
		self.vars_phi()
		self.vars_aux_dc()
		self.vars_z() 
		self.vars_z_trans()
		if not self.allele_specific:
			self.vars_three_snp_matrices()
			self.vars_aux_snp_linear()
		self.vars_three_ssm_matrices()
		self.vars_dssm_infl_cnv_same_lineage()
		self.vars_ssm_aux_1_cn()
		self.vars_ssm_aux_15_cn()
		self.vars_ssm_aux_2_cn()
		# Z-matrix is not fixed
		if fixed_z_matrix is None:
			# variables are only used to reduce symmetry in Z-matrix, not needed when Z is fixed
			# not needed when symmetry shouldn't be reduced
			if dont_break_z_symmetry == False:
				self.vars_dssm_infl_cnv()
			# further variables that are needed for the lineage divergence rule when the Z-matrix is not fixed
			if self.use_lineage_divergence_rule == True:
				self.vars_child()
				self.vars_child_freq()
				if dont_break_z_symmetry == False:
					self.vars_parent_freq()
					self.vars_sibling_freq()
					self.vars_ldr_active()
					self.vars_child_freq_minus_par_freq()
					self.vars_chf_m_pf_LDRa()
					self.vars_chf_m_pf_LDRi()
					self.vars_ldr_inactive()
		self.vars_dc_descendant()
		self.vars_dc_ancestral()
		
	# creates constraints #1 - #29
	# these are the standard constraints
	# 	including constraints that remove the symmetry
	def create_standard_constraints(self, snp_list, ssm_list, seg_list, fixed_z_matrix=None, fixed_phi=None,
		dont_break_z_symmetry=False):

		logging.debug("Constraints are created...")
		self.constraint_aux_vars_spline_sum_one()
		if self.allele_specific:
			self.constraint_cnv_value_spline_cn_allele_specific(seg_list)
		else:
			self.constraint_cnv_value_spline_mean(seg_list)
		self.constraint_phi_0()
		self.constraint_phi_k()
		if fixed_phi is None:
			self.constraints_phi_mutation_number()
		self.constraints_phi_0_no_mutations()
		self.constraints_aux_cnv_dc_matrices()
		if self.allele_specific:
			self.constraint_major_minor_cn()
			self.constraint_CN_direction(seg_list)
		else:
			self.constraint_remove_cn_symmetry()

		# copy number changes, constraints #9.x
		# simple CN changes
		if self.simple_CN_changes:
			self.constraint_delta_c_allowed_entries_per_row()
		# more complex CN changes in case of allele-specific modeling
		elif self.allele_specific:
			# max x CN changes per segment
			if self.max_x_CN_changes >= 0:
				self.constraint_max_cn_changes()
			# only one loss per segment per chromatid
			# also no simultaneous loss and gain in same lineage on same chromatid
			# and no gains of losses
			if self.only_one_loss:
				self.constraint_loss_per_chromatid()
				self.constraint_no_simultaneous_loss_and_gain()
				self.constraint_no_gains_of_lost_chromatids()
			# multiple losses of chromatid possible if in different lineages that still 
			#	had the chromatid
			# also no simultaneous loss and gain in same lineage on same chromatid
			# and no gains of losses in same constraint
			else:
				self.constraint_no_gains_or_losses_of_lost_chromatids()
				self.constraint_no_simultaneous_loss_and_gain()
			# only gains or losses or LOH in one segment
			if self.only_gains_losses_LOH:
				self.constraint_only_gains_losses_LOH()	
		# more complex CN changes in case of non-allele-specific modeling
		else:
			# is not tested
			raise(oe.MyException("Complex CN changes are not tested for non-allele-specific modeling."))
		# no CN changes in normal linage
		self.constraint_no_cnv_in_lineage_0()

		#   next constraint not tested
		self.constraint_z_matrix_half()
		#   next constraint not tested
		self.constraint_z_matrix_first_row_1()
		# Z-matrix is not fixed
		if fixed_z_matrix is None:
			# constraints are only used to reduce symmetry in Z-matrix, not needed when Z is fixed
			if dont_break_z_symmetry == False:
				self.constraint_remove_symmetry_z_matrix()
				self.constraint_dssm_infl_cnv(ssm_list)
			# constraints for lineage divergence rule
			if self.use_lineage_divergence_rule == True:
				self.constraint_define_children()
				self.constraint_child_frequency()
				# if Z-matrix symmetry should be broken
				if dont_break_z_symmetry == False:
					self.constraint_parent_frequency()
					self.constraint_sibling_frequency()
					self.constraint_chf_m_pf_LDRa()
					self.constraint_ldr_inactive()
					self.constraint_chf_m_pf_LDRi()
					self.constraint_ldr_active()
				self.constraint_ldr()
		else:
			pass
			## for testing purposes (put in commands when not testing)
			#self.vars_child()
			#self.vars_child_freq()
			#self.vars_parent_freq()
			#self.vars_sibling_freq()
			#self.vars_ldr_active()
			#self.vars_child_freq_minus_par_freq()
			#self.vars_chf_m_pf_LDRa()
			#self.vars_chf_m_pf_LDRi()
			#self.vars_ldr_inactive()
			#self.constraint_define_children()
			#self.constraint_child_frequency()
			#self.constraint_parent_frequency()
			#self.constraint_sibling_frequency()
			#self.constraint_chf_m_pf_LDRa()
			#self.constraint_ldr_inactive()
			#self.constraint_chf_m_pf_LDRi()
			#self.constraint_ldr_active()
			#self.constraint_ldr()
		# constraints for SNPs, when not in the allele_specific case
		if not self.allele_specific:
			self.constraint_snp_row_one_entry()
			self.constraint_remove_snp_symmetry(snp_list)
			self.constraint_aux_snp_w_cnv(snp_list)
			self.constraint_snp_value_spline_frequency(snp_list)
		self.constraint_no_ssm_normal_lineage()
		self.constraint_ssm_isa()
		self.constraint_remove_ssm_symmetry(ssm_list)
		self.constraint_dssm_infl_cnv_same_lineage(ssm_list)
		self.constraint_z_transitivity()
		self.constraint_ssm_aux_1_cn()
		self.constraint_ssm_aux_15_cn()
		self.constraint_ssm_aux_2_cn(ssm_list)
		self.constraint_ssm_value_spline_frequency(ssm_list)
		self.constraint_dc_descendant()
		self.constraint_dc_ancestral()
		
	# creates constraints #30 - #35 to fix certain variables
	def create_fixed_variables_constraints(self, fixed_cnv=None, unfixed_cnv_start=-1, 
		unfixed_cnv_stop=-1, 
		fixed_avg_cn=None, unfixed_avg_cn_start=-1, unfixed_avg_cn_stop=-1,
		fixed_snp=None, unfixed_snp_start=-1, unfixed_snp_stop=-1, fixed_ssm=None, 
		unfixed_ssm_start=-1, unfixed_ssm_stop=-1, fixed_z_matrix=None, 
		unfixed_z_matrix_start=-1, 
		unfixed_z_matrix_stop=-1, fixed_phi=None, unfixed_phi_start=-1, unfixed_phi_stop=-1, 
		direct_descendants_for_constraints=[], ssm_indices_per_cluster=None,
		fixed_cnv_indices=[], fixed_cnv_list_new=None, fixed_ssm_list_new=None):
		# parameters can be of type 'None'
		# then the corresponding variable isn't fixed

		warm_start_dc_binary_fixed = None
		warm_start_dssm_fixed = None
		warm_start_freqs_fixed = None
		fixed_z_matrix_indices = None
		
		if fixed_cnv:
			logging.info("Fixing CNVs.")
			self.constraint_fix_dc_binary(fixed_cnv, unfixed_cnv_start, unfixed_cnv_stop, fixed_cnv_indices)
			# if all CNVs are fixed, use values for warm start
			if unfixed_cnv_start == -1 and unfixed_cnv_stop == -1:
				warm_start_dc_binary_fixed = fixed_cnv
		if fixed_cnv_list_new:
			logging.info("Fixing CNVs.")
			self.constraint_fix_dc_binary_or_SSMs(fixed_cnv_list_new, cons.CNV)
		if fixed_avg_cn:
			logging.info("Fixing average CNs.")
			self.constraint_fix_avg_cn(fixed_avg_cn, unfixed_avg_cn_start, unfixed_avg_cn_stop)
		if fixed_snp and not self.allele_specific:
			logging.info("Fixing SNPs.")
			self.constraint_fix_dsnp(fixed_snp, unfixed_snp_start, unfixed_snp_stop)
		if fixed_ssm:
			logging.info("Fixing SSMs.")
			self.constraint_fix_dssm(fixed_ssm, unfixed_ssm_start, unfixed_ssm_stop)
			# if all SSMs are fixed, use values for warm start
			if unfixed_ssm_start == -1 and unfixed_ssm_stop == -1:
				warm_start_dssm_fixed = fixed_ssm
		if fixed_ssm_list_new:
			logging.info("Fixing SSMs.")
			self.constraint_fix_dc_binary_or_SSMs(fixed_ssm_list_new, cons.SSM)
		if fixed_z_matrix:
			logging.info("Fixing Z-matrix.")
			fixed_z_matrix_indices = self.constraint_fix_z_matrix(fixed_z_matrix, unfixed_z_matrix_start, 
				unfixed_z_matrix_stop)
			if direct_descendants_for_constraints != []:
				self.constraint_lineage_divergence_z_fixed(direct_descendants_for_constraints)
		if fixed_phi:
			logging.info("Fixing phis.")
			self.constraint_fix_phi(fixed_phi, unfixed_phi_start, unfixed_phi_stop)
			# if all frequencies are fixed, use values for warm start
			if unfixed_phi_start == -1 and unfixed_phi_stop == -1:
				warm_start_freqs_fixed = fixed_phi
		if ssm_indices_per_cluster:
			logging.info("Fixing SSM cluster.")
			self.constraint_clustered_ssms(ssm_indices_per_cluster)
			self.constraint_clustered_ssms_infl_cnv_same_lineage(ssm_indices_per_cluster)

		return warm_start_dc_binary_fixed, warm_start_dssm_fixed, warm_start_freqs_fixed, fixed_z_matrix_indices
	
	# prints variables and constraints
	def print_variables_constraints(self):
		for i in xrange(len(self.my_obj)):
			print "obj {0}: {1}".format(self.my_colnames[i], self.my_obj[i])
			print "upper bound {0}: {1}".format(self.my_colnames[i], self.my_ub[i])
			print "lower bound {0}: {1}".format(self.my_colnames[i], self.my_lb[i])
			print "type {0}: {1}".format(self.my_colnames[i], self.my_ctype[i])
			print "index {0}: {1}".format(self.my_colnames[i], i)
		#print "obj: ", self.my_obj
		#print "obj length: ", len(self.my_obj)
		#print "upper bound: ", self.my_ub
		#print "upper bound length: ", len(self.my_ub)
		#print "lower bound: ", self.my_lb
		#print "lb length: ", len(self.my_lb)
		#print "types: ", self.my_ctype
		#print "types length: ", len(self.my_ctype)
		#print "colnames: ", self.my_colnames
		#print "colnames length: ", len(self.my_colnames)

		for i in xrange(len(self.my_rhs)):
			print "index {0}: {1}".format(self.my_rownames[i], i)
			print "rhs {0}: {1}".format(self.my_rownames[i], self.my_rhs[i])
			print "sense {0}: {1}".format(self.my_rownames[i], self.my_sense[i])
			print "rows {0}: {1}".format(self.my_rownames[i], self.my_rows[i])
			var_string = []
			for my_var in self.my_rows[i][0]:
				var_string.append(self.my_colnames[my_var])
			print ", ".join(var_string)
			print ""
		#print "righthand side: ", self.my_rhs
		#print "rhs length: ", len(self.my_rhs)
		#print "rownames: ", self.my_rownames
		#print "rownames length: ", len(self.my_rownames)
		#print "senses: ", self.my_sense
		#print "senses length: ", len(self.my_sense)
		#print "rows: "
		#for i in range(len(self.my_rows)):
		#	print "%d: " % i, self.my_rows[i]
		#print "rows length: ", len(self.my_rows)
		#print ""

	# starts CPLEX
	def start_CPLEX(self, warm_start_dc_binary=None, warm_start_dsnp=None,
		warm_start_dssm=None, warm_start_freqs=None, z_matrix=None, 
		warm_start_values=None, warm_start_indices=None, fixed_z_matrix_indices=None):

		try:
			self.my_prob = cplex.Cplex()
			self.my_prob.objective.set_sense(self.my_prob.objective.sense.maximize)
			start_adding_variables = time.time()
			logging.info("Adding Variables...")
			# in allele-specific case, constraints work with variable indices, their
			# names are not needed
			if self.allele_specific:
				self.my_prob.variables.add(obj = self.my_obj, lb = self.my_lb, 
					ub = self.my_ub, 
					types = self.my_ctype)
			else:
				self.my_prob.variables.add(obj = self.my_obj, lb = self.my_lb, 
					ub = self.my_ub, 
					types = self.my_ctype, names = self.my_colnames)
			end_adding_variables = time.time()
			logging.info("Time to add variables: {0}".format(end_adding_variables 
				- start_adding_variables))
			logging.info("Adding Constraints...")
			start_adding_constraints = time.time()
			self.my_prob.linear_constraints.add(lin_expr = self.my_rows, 
				senses = self.my_sense,
				rhs = self.my_rhs)
			# rownames are not needed
			# self.my_prob.linear_constraints.add(lin_expr = self.my_rows, 
			#	senses = self.my_sense,
			#	rhs = self.my_rhs, names = self.my_rownames)
			end_adding_constraints = time.time()
			logging.info("Time to add constraints: {0}".format(end_adding_constraints 
				- start_adding_constraints))
			# warm starts
			if self.fix_all == False:
				start_adding_warm_start = time.time()
				self.create_warm_starts(warm_start_dc_binary, warm_start_dsnp, warm_start_dssm,
					warm_start_freqs, z_matrix, fixed_z_matrix_indices)
				self.add_values_for_complete_warm_start(warm_start_values, warm_start_indices)
				end_adding_warm_start = time.time()
				logging.info("Time to add warm start values: {0}".format(end_adding_warm_start 
					- start_adding_warm_start))

			self.my_prob.set_log_stream(self.cplex_log_file)
			self.my_prob.set_error_stream(self.cplex_log_file)
			self.my_prob.set_warning_stream(self.cplex_log_file)
			self.my_prob.set_results_stream(self.cplex_log_file)

			# sets parameters for CPLEX optimiation
			logging.info("Set other CPLEX parameters...")
			self.my_prob.parameters.timelimit.set(self.time)
			self.my_prob.parameters.threads.set(self.threads)
			self.my_prob.parameters.mip.strategy.probe.set(self.probing)
			self.my_prob.parameters.emphasis.mip.set(self.emph_switch)
			self.my_prob.parameters.preprocessing.coeffreduce.set(self.coef_reduc)
			self.my_prob.parameters.mip.tolerances.mipgap.set(self.mipgap)
			self.my_prob.parameters.preprocessing.symmetry.set(self.symmetry)
			self.my_prob.parameters.mip.strategy.file.set(self.strategy_file)
			self.my_prob.parameters.mip.strategy.nodeselect.set(self.nodeselect)
			self.my_prob.parameters.workmem.set(self.workmem)
			self.my_prob.parameters.workdir.set(self.workdir)
			self.my_prob.parameters.mip.limits.treememory.set(self.treememory)
			self.my_prob.parameters.emphasis.memory.set(self.emphasis_memory)

			# fix all at beginning to use this result as a warm start
			if self.fix_all == True:
				# lower run time
				self.my_prob.parameters.timelimit.set(600)
				logging.info("Starting CPLEX with fixed values...")
				self.my_prob.solve()

				# remove fixed constraints
				logging.debug("Removing fixing constraints")
				self.my_prob.linear_constraints.delete(self.last_regular_constraint, 
					self.last_fixed_all_constraint-1)
				# setting run time limit to original value again
				self.my_prob.parameters.timelimit.set(self.time)
				# lower feasibility tolerance
				self.my_prob.parameters.simplex.tolerances.feasibility = 1e-03

			logging.info("Starting CPLEX...")
			self.my_prob.solve()
	
			status = self.my_prob.solution.status[self.my_prob.solution.get_status()]
			value = self.my_prob.solution.get_objective_value()
			oio.print_status_value(status, value)
			numcols = self.my_prob.variables.get_num()
			#x = self.my_prob.solution.get_values()
			#for j in range(numcols):
			#	logging.debug("column {0}: value = {1}".format(self.my_colnames[j], x[j]))
	
			#logging.info("Finished optimization with CPLEX")

		except CplexError, exc:
			raise

	#####################################
	##### vars_aux_vars_mut_splines #####
	
	# variables for auxiliary variables in the splines are created 
	def vars_aux_vars_mut_splines(self):
		# create objective, upper and lower bounds, types
		if self.allele_specific:
			(obj_seg_A, u_seg_A, l_seg_A, t_seg_A) = (
				self.create_obj_ub_lb_types_aux_vars_mut_splines(
				self.seg_splines_A))
			(obj_seg_B, u_seg_B, l_seg_B, t_seg_B) = (
				self.create_obj_ub_lb_types_aux_vars_mut_splines(
				self.seg_splines_B))
		else:
			(obj_seg, u_seg, l_seg, t_seg) = self.create_obj_ub_lb_types_aux_vars_mut_splines(
				self.seg_splines)
			(obj_snp, u_snp, l_snp, t_snp) = self.create_obj_ub_lb_types_aux_vars_mut_splines(
				self.snp_splines)
		(obj_ssm, u_ssm, l_ssm, t_ssm) = self.create_obj_ub_lb_types_aux_vars_mut_splines(
			self.ssm_splines)

		# create column names
		if self.allele_specific:
			(col_seg_A_2d, col_seg_A_flat) = self.create_colnames_aux_vars_mut_splines(
				self.seg_splines_A, "seg_A")
			(col_seg_B_2d, col_seg_B_flat) = self.create_colnames_aux_vars_mut_splines(
				self.seg_splines_B, "seg_B")
		else:
			(col_seg_2d, col_seg_flat) = self.create_colnames_aux_vars_mut_splines(
				self.seg_splines, "seg")
			(col_snp_2d, col_snp_flat) = self.create_colnames_aux_vars_mut_splines(
				self.snp_splines, "snp")
		(col_ssm_2d, col_ssm_flat) = self.create_colnames_aux_vars_mut_splines(self.ssm_splines, "ssm")
	
		# concatenate results
		if self.allele_specific:
			self.my_obj.extend(obj_seg_A + obj_seg_B + obj_ssm)
			self.my_ub.extend(u_seg_A + u_seg_B + u_ssm)
			self.my_lb.extend(l_seg_A + l_seg_B + l_ssm)
			self.my_ctype.extend(t_seg_A + t_seg_B + t_ssm)
			self.my_colnames_seg_A = col_seg_A_2d
			self.my_colnames_seg_B = col_seg_B_2d
			self.start_index_colnames_seg = len(self.my_colnames)
			self.my_colnames_seg_A_index = self.create_2d_spline_indices_list(self.seg_splines_A,
				len(self.my_colnames))
			self.my_colnames.extend(col_seg_A_flat)
			self.my_colnames_seg_B_index = self.create_2d_spline_indices_list(self.seg_splines_B,
				len(self.my_colnames))
			self.my_colnames.extend(col_seg_B_flat)
		else:
			self.my_obj.extend(obj_seg + obj_snp + obj_ssm)
			self.my_ub.extend(u_seg + u_snp + u_ssm)
			self.my_lb.extend(l_seg + l_snp + l_ssm)
			self.my_ctype.extend(t_seg + t_snp + t_ssm)
			self.my_colnames_seg = col_seg_2d
			self.my_colnames_snp = col_snp_2d
			self.my_colnames_seg_index = self.create_2d_spline_indices_list(self.seg_splines,
				len(self.my_colnames))
			self.my_colnames.extend(col_seg_flat)
			self.my_colnames.extend(col_snp_flat)

		self.start_index_colnames_ssm = len(self.my_colnames)
		self.my_colnames_ssm = col_ssm_2d
		self.my_colnames_ssm_index = self.create_2d_spline_indices_list(self.ssm_splines,
			len(self.my_colnames))
		self.my_colnames.extend(col_ssm_flat)


	def create_obj_ub_lb_types_aux_vars_mut_splines(self, spl):
		''' creates objective, upper bound, lower bound and types
		 for auxiliary variables in the splines '''
		list = [self.part_obj_ub_lb_types_aux_vars_mut_splines(s) for s in spl]

		obj = [l[0] for l in list]
		obj = self.flatten_list(obj)
		ub = [l[1] for l in list]
		ub = self.flatten_list(ub)
		lb = [l[2] for l in list]
		lb = self.flatten_list(lb)
		t = [l[3] for l in list]
		t = self.flatten_list(t)

		return (obj, ub, lb, t)

	# part of create_obj_ub_lb_types_aux_vars_mut_splines
	def part_obj_ub_lb_types_aux_vars_mut_splines(self, spl):
		num = len(spl.get_coeffs())
		return ([spl.get_coeffs().tolist(), [1.0] * num, [0.0] * num, ["C"] * num])
	
	# creates the colnames for the auxiliary variables in the splines 
	def create_colnames_aux_vars_mut_splines(self, splines, mut):
		list_2d = [self.create_colnames_aux_vars_single_mut_spline(s, i, mut) for i, s in enumerate(splines)] 
		list_flat = self.flatten_list(list_2d)
		return (list_2d, list_flat)

	def create_2d_spline_indices_list(self, spl, start_index):
		my_list = []
		for spline in spl:
			spl_len = len(spline.get_coeffs())
			my_list.append(range(start_index, start_index + spl_len))
			start_index += spl_len
		return my_list

	# creates the colnames for the auxiliary variables in only one spline
	# so that a list of list can be created in the end in the 
	# function "create_colnames_aux_vars_mut_splines"
	def create_colnames_aux_vars_single_mut_spline(self, spline, i, mut):
		return ["{0}_a_{1}_{2}".format(mut, i, j) for j in  range(len(spline.get_coeffs()))]  

	##### vars_aux_vars_mut_splines #####
	#####################################

	# TODO: only variables for phis that need to be defined, as normal will 1, change!
	# variables for phis are created (subclonal lineage frequency)
	def vars_phi(self):
		# no influence in objective function
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", self.sublin_num)
		self.my_phis = []
		for i in range(self.sublin_num):
			name = "phi_%d" % i
			self.my_phis.append(name)
		# get index where in list my_colnames the phis start
		self.phi_start_index = len(self.my_colnames)
		# put colnames in list
		self.my_phis_index = np.array([i for i in xrange(len(self.my_colnames), 
			len(self.my_colnames) + self.sublin_num)])
		self.my_colnames.extend(self.my_phis)
	
	############################
	##### vars_aux_dc #####
	# TODO: make more flexible concerning multiple copy number states

	# variables for auxiliary variables for finding values in delta_C_A and delta_C_B
	# are created
	# dc_a_STATE_binary and dc_a_STATE_float, same for b
	def  vars_aux_dc(self):
		# can have influcence in objective function when self.cn_weight is set
		var_num = self.cnv_aux_all_states_one_type_entry_num * self.aux_matrices_cnv_linear_types_num
		var_num_half = var_num / 2
		self.my_obj.extend([-self.cn_weight] * var_num_half)
		self.my_obj.extend([0.0] * var_num_half)

		self.my_ub.extend([1.0] * var_num)
		self.my_lb.extend([0.0] * var_num)
		# types for binary auxilliary variables for delta_C_A and delta_C_B
		# types for float auxilliary variables for delta_C_A and delta_C_B 
		self.my_ctype.extend(["B"] * self.cnv_aux_all_states_one_type_entry_num +
			["C"] * self.cnv_aux_all_states_one_type_entry_num)
		
		(self.my_colnames_dc_a_p1_binary, col_dc_a_p1_b_flat) = self.create_colnames_vars_aux_dc("a", "binary", "p1")
		(self.my_colnames_dc_b_p1_binary, col_dc_b_p1_b_flat) = self.create_colnames_vars_aux_dc("b", "binary", "p1")
		(self.my_colnames_dc_a_m1_binary, col_dc_a_m1_b_flat) = self.create_colnames_vars_aux_dc("a", "binary", "m1")
		(self.my_colnames_dc_b_m1_binary, col_dc_b_m1_b_flat) = self.create_colnames_vars_aux_dc("b", "binary", "m1")
		(self.my_colnames_dc_a_p1_float, col_dc_a_p1_f_flat) = self.create_colnames_vars_aux_dc("a", "float", "p1")
		(self.my_colnames_dc_b_p1_float, col_dc_b_p1_f_flat) = self.create_colnames_vars_aux_dc("b", "float", "p1")
		(self.my_colnames_dc_a_m1_float, col_dc_a_m1_f_flat) = self.create_colnames_vars_aux_dc("a", "float", "m1")
		(self.my_colnames_dc_b_m1_float, col_dc_b_m1_f_flat) = self.create_colnames_vars_aux_dc("b", "float", "m1")
		
		# get index for list my_colnames where dc_{a,b}_STATES_binary variables start
		# first there are p1 variables on A, for one i all sublineages, then in same order on B
		self.dc_binary_index_start_p1 = len(self.my_colnames)
		self.dc_binary_index_start_m1 = (self.dc_binary_index_start_p1 + (self.seg_num * self.sublin_num *
			cons.PHASE_NUMBER))

		self.my_colnames_dc_a_p1_binary_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_a_p1_b_flat)
		self.my_colnames_dc_b_p1_binary_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_b_p1_b_flat)
		self.my_colnames_dc_a_m1_binary_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_a_m1_b_flat)
		self.my_colnames_dc_b_m1_binary_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_b_m1_b_flat) 

		self.my_colnames_dc_a_p1_float_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_a_p1_f_flat)
		self.my_colnames_dc_b_p1_float_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_b_p1_f_flat)
		self.my_colnames_dc_a_m1_float_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_a_m1_f_flat)
		self.my_colnames_dc_b_m1_float_index = self.create_2d_index_array(len(self.my_colnames), self.seg_num,
			self.sublin_num)
		self.my_colnames.extend(col_dc_b_m1_f_flat)

	# creates colnames for auxiliary variables for finding values in delta_C_A and delta_C_B
	def create_colnames_vars_aux_dc(self, phase, type, state):
		local_function = self.create_colnames_vars_aux_dc_per_sublin
		list_2d = [local_function(phase, type, i, state) for i in xrange(self.seg_num)]
		list_flat = self.flatten_list(list_2d)
		return (list_2d, list_flat)

	# helps to creates colnames for auxiliary variables for finding values in delta_C_A and delta_C_B
	# does so for inner loop
	def create_colnames_vars_aux_dc_per_sublin(self, phase, type, i, state):
		sublin_num = self.sublin_num
		return ["dc_{0}_{4}_{1}_{2}_{3}".format(phase, type, i, j, state) for j in range(sublin_num)]
	
	##### vars_aux_dc #####
	############################

	
	# variables for Z matrix are created
	# TODO: rewrite: only upper half of matrix
	def vars_z(self):
		# no influence in objective function
		entry_num = self.sublin_num * self.sublin_num
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entry_num)
		my_list = ["z_{0}_{1}".format(i, j) for i in range(self.sublin_num)
			for j in range(self.sublin_num)]
		self.my_colnames_z = [my_list[i:i+self.sublin_num] for i in xrange(0, len(my_list), self.sublin_num)]

		# get index for list my_colnames where z variables start
		self.z_index_start = len(self.my_colnames)

		# create index list
		self.my_colnames_z_index = np.array([i for i in xrange(self.z_index_start, 
			self.z_index_start+entry_num)]).reshape(self.sublin_num, self.sublin_num)

		# put colnames in list
		self.my_colnames.extend(self.flatten_list(self.my_colnames_z))
		
	# variables for Z transitivity rules
	def vars_z_trans(self):
		# names
		self.my_colnames_z_trans_i = self.create_z_trans_names_1("z_trans_i")
		self.my_colnames_z_trans_c = self.create_z_trans_names_1("z_trans_c")
		# flat names list
		self.my_colnames_z_trans_i_flat = self.flatten_3d_list(self.my_colnames_z_trans_i)
		self.my_colnames_z_trans_c_flat = self.flatten_3d_list(self.my_colnames_z_trans_c)

		# element number
		entry_num = len(self.my_colnames_z_trans_i_flat)

		# objective, upper and lower bound and type
		self.my_obj.extend([-self.z_trans_weight] * entry_num*2)
		self.my_ub.extend([1.0] * entry_num*2)
		self.my_lb.extend([0.0] * entry_num*2)
		self.my_ctype.extend(["B"] * entry_num*2)

		# get index for names
		self.z_trans_i_index_start = len(self.my_colnames)
		self.z_trans_c_index_start = len(self.my_colnames) + entry_num

		# create index list
		self.z_trans_index_start_tmp = self.z_trans_i_index_start
		self.z_trans_i_index = self.create_z_trans_indices_s_1()
		self.z_trans_index_start_tmp = self.z_trans_c_index_start
		self.z_trans_c_index = self.create_z_trans_indices_s_1()

		# put colnames in list
		self.my_colnames.extend(self.my_colnames_z_trans_i_flat + self.my_colnames_z_trans_c_flat)


	#############
	# creating list with indices for Z transitivity variables
	def create_z_trans_indices_s_1(self):
		return [self.create_z_trans_indices_s_2(k) for k in xrange(1, self.sublin_num-2)]

	def create_z_trans_indices_s_2(self, k):
		return [self.create_z_trans_indices_s_3(k_prime) for k_prime in xrange(k+1, self.sublin_num-1)]

	def create_z_trans_indices_s_3(self, k_prime):
		my_list = [self.z_trans_index_start_tmp+i for i in xrange(self.sublin_num - (k_prime + 1))]
		self.z_trans_index_start_tmp += self.sublin_num - (k_prime + 1)
		return my_list
	# creating list with indices for Z transitivity variables
	#############

	############# 
	## creating names for the Z transitivity variables
	def create_z_trans_names_1(self, name):
		return [self.create_z_trans_names_2(name, k) for k in xrange(1, self.sublin_num-2)]

	def create_z_trans_names_2(self, name, k):
		return [self.create_z_trans_names_3(name, k, k_prime) for k_prime in xrange(k+1, self.sublin_num-1)]

	def create_z_trans_names_3(self, name, k, k_prime):
		return ["{0}_{1}_{2}_{3}".format(name, k, k_prime, k_prime_prime) 
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
	## creating names for the Z transitivity variables
	############# 

	###############################
	# creating vars needed for lineage divergence rule when Z is not fixed

	# creating variable child_k_k' that tells whether lineage k' is a child of lineage k
	def vars_child(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
			
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entry_num)

		# my colnames
		self.my_colnames_children = [["child_{0}_{1}".format(k, k_prime) for k_prime in xrange(k+1, self.sublin_num)]
			for k in xrange(self.sublin_num - 1)]

		# get start index
		self.children_start_index = len(self.my_colnames)

		# put newly created colnames in list
		self.my_colnames_children_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_children_index_friendly_form = [[None for _ in xrange(self.sublin_num)] 
			for a in xrange(self.sublin_num)]
		current_index = self.children_start_index
		for k in xrange(self.sublin_num - 1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_children_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.flatten_list(self.my_colnames_children))

	def vars_child_freq(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
			
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entry_num)

		# my colnames
		self.my_colnames_children_freq = [
			["child_{0}_{1}_freq".format(k, k_prime) for k_prime in xrange(k+1, self.sublin_num)]
			for k in xrange(self.sublin_num - 1)]

		# get start index
		self.children_freq_start_index = len(self.my_colnames)

		# put newly created colnames in list
		self.my_colnames_children_freq_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_children_freq_index_friendly_form = [[None for _ in xrange(self.sublin_num)] 
			for a in xrange(self.sublin_num)]
		current_index = self.children_freq_start_index
		for k in xrange(self.sublin_num - 1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_children_freq_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.flatten_list(self.my_colnames_children_freq))
	
	def vars_parent_freq(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1

		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entry_num)

		# my colnames
		self.my_colnames_parent_freq = [
			["parent_{0}_{1}_freq".format(k, k_prime) for k_prime in xrange(k+1, self.sublin_num)]
			for k in xrange(self.sublin_num - 1)]

		# get start index
		self.parent_freq_start_index = len(self.my_colnames)

		# put newly created colnames in list
		self.my_colnames_parent_freq_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_parent_freq_index_friendly_form = [[None for _ in xrange(self.sublin_num)]
			for a in xrange(self.sublin_num)]
		current_index = self.parent_freq_start_index
		for k in xrange(self.sublin_num - 1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_parent_freq_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.flatten_list(self.my_colnames_parent_freq))

	def vars_sibling_freq(self):
		# my colnames
		self.my_colnames_sibling_freq = [
			"sibling_{0}_{1}_{2}_freq".format(k, k_circ, k_bullet) 
				for k in xrange(self.sublin_num-2)
				for k_circ in xrange(k+1, self.sublin_num)
				for k_bullet in xrange(k+1, self.sublin_num)
				if k_bullet != k_circ]

		entry_num = len(self.my_colnames_sibling_freq)

		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entry_num)


		# get start index
		self.sibling_freq_start_index = len(self.my_colnames)

		# put newly created colnames in list
		self.my_colnames_sibling_freq_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_sibling_freq_index_friendly_form = [[[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num)] for y in xrange(self.sublin_num-2)]
		current_index = self.sibling_freq_start_index
		for k in xrange(self.sublin_num - 2):
			for k_circ in xrange(k+1, self.sublin_num):
				for k_bullet in xrange(k+1, self.sublin_num):
					if k_bullet != k_circ:
						self.my_colnames_sibling_freq_index_friendly_form[k][k_circ][k_bullet] = (
							current_index)
						current_index += 1
		self.my_colnames.extend(self.my_colnames_sibling_freq)

	# variable that stores whether the lineage divergence rule is active for lineage k and
	# its child k'
	def vars_ldr_active(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entry_num)
		# get start index
		self.ldr_active_start_index = len(self.my_colnames)
		# put newly created colnames in list
		self.my_colnames_ldr_active = ["LDR_active_{0}_{1}".format(k, k_prime)
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)]
		self.my_colnames_ldr_active_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_ldr_active_index_friendly_form = [[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num-1)]
		current_index = self.ldr_active_start_index
		for k in xrange(1, self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_ldr_active_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.my_colnames_ldr_active)

	# variable that gives the difference between the added up frequencies of lineage k, all
	# its siblings and its child k', and the frequency of k's parent
	def vars_child_freq_minus_par_freq(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.create_vars_obj_ub_lb_t(1.0, -1.0, "C", entry_num)
		# get start index
		self.child_freq_minus_par_freq_start_index = len(self.my_colnames)
		# put newly created colnames in list
		self.my_colnames_child_freq_minus_par_freq = ["child_freq_minus_par_freq_{0}_{1}".format(k, k_prime)
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)]
		self.my_colnames_child_freq_minus_par_freq_index_friendly_form = [[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num-1)]
		current_index = self.child_freq_minus_par_freq_start_index
		for k in xrange(1, self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_child_freq_minus_par_freq_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.my_colnames_child_freq_minus_par_freq)

	# continuous variable that stores when lineage divergence rule between lineages k and
	# k' is active
	def vars_chf_m_pf_LDRa(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entry_num)
		# get start index
		self.chf_m_pf_LDRa_start_index = len(self.my_colnames)
		# put newly created colnames in list
		self.my_colnames_chf_m_pf_LDRa = ["chf_m_pf_LDRa_{0}_{1}".format(k, k_prime)
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)]
		self.my_colnames_chf_m_pf_LDRa_index_friendly_form = [[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num-1)]
		current_index = self.chf_m_pf_LDRa_start_index
		for k in xrange(1, self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_chf_m_pf_LDRa_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.my_colnames_chf_m_pf_LDRa)

	# continuous variable that stores when lineage divergence rule between lineages k and
	# k' is inactive
	def vars_chf_m_pf_LDRi(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.create_vars_obj_ub_lb_t(0.0, -1.0, "C", entry_num)
		self.chf_m_pf_LDRi_start_index = len(self.my_colnames)
		self.my_colnames_chf_m_pf_LDRi = ["chf_m_pf_LDRi_{0}_{1}".format(k, k_prime)
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)]
		self.my_colnames_chf_m_pf_LDRi_index_friendly_form = [[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num-1)]
		current_index = self.chf_m_pf_LDRi_start_index
		for k in xrange(1, self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_chf_m_pf_LDRi_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.my_colnames_chf_m_pf_LDRi)

	# variable that stores whether the lineage divergence rule is inactive for lineage k and
	# its child k'
	def vars_ldr_inactive(self):
		entry_num = model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entry_num)
		# get start index
		self.ldr_inactive_start_index = len(self.my_colnames)
		# put newly created colnames in list
		self.my_colnames_ldr_inactive = ["LDR_inactive_{0}_{1}".format(k, k_prime)
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)]
		self.my_colnames_ldr_inactive_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + entry_num)])
		self.my_colnames_ldr_inactive_index_friendly_form = [[None for i in xrange(self.sublin_num)]
			for j in xrange(self.sublin_num-1)]
		current_index = self.ldr_inactive_start_index
		for k in xrange(1, self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				self.my_colnames_ldr_inactive_index_friendly_form[k][k_prime] = current_index
				current_index += 1
		self.my_colnames.extend(self.my_colnames_ldr_inactive)

	# end of creating vars needed for lineage divergence rule when Z is not fixed
	###############################

	###############################
	##### _three_snp_matrices #####

	# variables for the three SNP matrices
	def vars_three_snp_matrices(self):
		# no influence in objective function
		entry_num = self.delta_snp_entry_num * cons.SNP_SSM_PHASING_MATRIX_NUM
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entry_num)

		self.my_colnames_dsnp = self.create_colnames_vars_three_snp_matrices("")
		self.my_colnames_dsnp_a = self.create_colnames_vars_three_snp_matrices("_a")
		self.my_colnames_dsnp_b = self.create_colnames_vars_three_snp_matrices("_b")

		# get start index for variables in list my_colames
		# first all unphased variables, then phased to A and then to B
		self.dsnp_start_index = len(self.my_colnames)

		# put colnames in list
		self.my_colnames_dsnp_index = np.array([i for i in xrange(len(self.my_colnames), 
			len(self.my_colnames) + self.snp_num)])
		self.my_colnames.extend(self.my_colnames_dsnp) 
		self.my_colnames_dsnp_a_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + self.snp_num)])    
		self.my_colnames.extend(self.my_colnames_dsnp_a)
		self.my_colnames_dsnp_b_index = np.array([i for i in xrange(len(self.my_colnames),
			len(self.my_colnames) + self.snp_num)])
		self.my_colnames.extend(self.my_colnames_dsnp_b)
	
	def create_colnames_vars_three_snp_matrices(self, phase):
		return ["dsnp{0}_{1}".format(phase, i) for i in xrange(self.snp_num)]
	
	##### _three_snp_matrices #####
	###############################
	
	###############################
	##### vars_aux_snp_linear #####
	
	# variables for auxiliary variable for computing the long snp equation in
	# a linear way
	def vars_aux_snp_linear(self):
		# no influence in objective function
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", self.snp_aux_linear_variables_num)
		
		(self.my_colnames_snp_w_cnv_a_p1, snp_w_cnv_a_p1_flat) = self.create_colnames_vars_aux_snp_linear("a", "p1")
		(self.my_colnames_snp_w_cnv_b_p1, snp_w_cnv_b_p1_flat) = self.create_colnames_vars_aux_snp_linear("b", "p1") 
		(self.my_colnames_snp_w_cnv_a_m1, snp_w_cnv_a_m1_flat) = self.create_colnames_vars_aux_snp_linear("a", "m1")
		(self.my_colnames_snp_w_cnv_b_m1, snp_w_cnv_b_m1_flat) = self.create_colnames_vars_aux_snp_linear("b", "m1") 
		self.my_colnames.extend(snp_w_cnv_a_p1_flat + snp_w_cnv_b_p1_flat + snp_w_cnv_a_m1_flat + snp_w_cnv_b_m1_flat)
		
	def create_colnames_vars_aux_snp_linear(self, phase, state):
		list_2d = [self.create_colnames_vars_aux_snp_linear_per_sublin(phase, i, state) for 
			i in xrange(self.snp_num)]
		list_flat = self.flatten_list(list_2d)
		return (list_2d, list_flat)

	def create_colnames_vars_aux_snp_linear_per_sublin(self, phase, i, state):
		return ["snp_w_cnv_{0}_{3}_{1}_{2}".format(phase, i, j + 1, state) for j in range(self.sublin_num - 1)]
	
	##### vars_aux_snp_linear #####
	###############################
		
	###################################
	##### vars_three_ssm_matrices #####
	# TODO cannot be in normal lineage, change variables + constraints!

	# if only simple copy number changes are modeled (+1, -1, +1/-1 once per segment),
	#	unphased state is used,
	# if more complicated cases are modeled, the unphased state is not used, but calculated
	#	 later in a post optimization analysis
	def vars_three_ssm_matrices(self):
		entries = 0
		if self.simple_CN_changes:
			entries = self.ssm_num * self.sublin_num * cons.SNP_SSM_PHASING_MATRIX_NUM
		else:
			entries = self.ssm_num * self.sublin_num * cons.PHASE_NUMBER
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entries)

		self.my_colnames_dssm = dssm = []
		if self.simple_CN_changes:
			(self.my_colnames_dssm, dssm) = self.create_colnames_vars_three_ssm_matrices("")
		(self.my_colnames_dssm_a, dssm_a) = self.create_colnames_vars_three_ssm_matrices("_a")
		(self.my_colnames_dssm_b, dssm_b) = self.create_colnames_vars_three_ssm_matrices("_b")
		
		# get start index for variables in list my_colames
		# first all unphased variables(if they exist), then phased to A and then to B
		self.dssm_start_index = len(self.my_colnames)
		
		self.my_colnames_dssm_index = self.create_2d_index_array(len(self.my_colnames), self.ssm_num,
			self.sublin_num)
		self.my_colnames.extend(dssm)
		self.my_colnames_dssm_a_index = self.create_2d_index_array(len(self.my_colnames), self.ssm_num,
			self.sublin_num)
		self.my_colnames.extend(dssm_a)
		self.my_colnames_dssm_b_index = self.create_2d_index_array(len(self.my_colnames), self.ssm_num,
			self.sublin_num)
		self.my_colnames.extend(dssm_b)
	
	def create_colnames_vars_three_ssm_matrices(self, phase):
		list_2d = [self.create_colnames_vars_three_ssm_matrices_per_sublin(phase, i) for
			i in xrange(self.ssm_num)]
		list_flat = self.flatten_list(list_2d)
		return (list_2d, list_flat)

	def create_colnames_vars_three_ssm_matrices_per_sublin(self, phase, i):
		return ["dssm{0}_{1}_{2}".format(phase, i, j) for j in range(self.sublin_num)]
	
	##### vars_three_ssm_matrices #####
	###################################
	
	def vars_dssm_infl_cnv_same_lineage(self):
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entries)

		self.my_colnames_dssm_infl_cnv_same_lineage_a = [["dssm_infl_cnv_same_lineage_a_p1_{0}_{1}".format(
			j, k) for k in xrange(1, self.sublin_num)] for j in xrange(self.ssm_num)]
		self.my_colnames_dssm_infl_cnv_same_lineage_a_index = (
			self.create_2d_index_array(len(self.my_colnames), self.ssm_num, self.sublin_num-1))
		self.my_colnames.extend(self.flatten_list(self.my_colnames_dssm_infl_cnv_same_lineage_a))
		self.my_colnames_dssm_infl_cnv_same_lineage_b = [["dssm_infl_cnv_same_lineage_b_p1_{0}_{1}".format(
			j, k) for k in xrange(1, self.sublin_num)] for j in xrange(self.ssm_num)]
		self.my_colnames_dssm_infl_cnv_same_lineage_b_index = (
			self.create_2d_index_array(len(self.my_colnames), self.ssm_num, self.sublin_num-1))
		self.my_colnames.extend(self.flatten_list(self.my_colnames_dssm_infl_cnv_same_lineage_b))


	#########################
	##### _ssm_aux_1_cn #####

	# auxiliary variables part 1 for calculating the copy number of ssms
	def vars_ssm_aux_1_cn(self):
		entries = self.ssm_num * self.sublin_num
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entries)

		(self.my_colnames_dssm_aux_1_cn, dssm_aux_1_cn_flat) = self.create_colnames_vars_ssm_aux_1_cn()
		self.my_colnames_dssm_aux_1_cn_index = self.create_2d_index_array(len(self.my_colnames), self.ssm_num,
			self.sublin_num)
		self.my_colnames.extend(dssm_aux_1_cn_flat)

	def create_colnames_vars_ssm_aux_1_cn(self):
		list_2d = [self.create_colnames_vars_ssm_aux_1_cn_per_sublin(i) for i in xrange(self.ssm_num)]
		list_flat = self.flatten_list(list_2d)
		return (list_2d, list_flat)

	def create_colnames_vars_ssm_aux_1_cn_per_sublin(self, i):
		return ["dssm_aux_1_cn_{0}_{1}".format(i, j) for j in range(self.sublin_num)]
	
	##### _ssm_aux_1_cn #####
	#########################
	
	def vars_ssm_aux_15_cn(self):
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entries)

		self.my_colnames_dssm_aux_15_cn_a_p1 = [["dssm_aux_15_cn_a_p1_{0}_{1}".format(j, k)
			for k in range(1, self.sublin_num)] for j in xrange(self.ssm_num)]
		self.my_colnames_dssm_aux_15_cn_a_p1_index = self.create_2d_index_array(len(self.my_colnames),
			self.ssm_num, self.sublin_num-1)
		self.my_colnames.extend(self.flatten_list(self.my_colnames_dssm_aux_15_cn_a_p1))
		self.my_colnames_dssm_aux_15_cn_b_p1 = [["dssm_aux_15_cn_b_p1_{0}_{1}".format(j, k)
			for k in range(1, self.sublin_num)] for j in xrange(self.ssm_num)]
		self.my_colnames_dssm_aux_15_cn_b_p1_index = self.create_2d_index_array(len(self.my_colnames),
			self.ssm_num, self.sublin_num-1)
		self.my_colnames.extend(self.flatten_list(self.my_colnames_dssm_aux_15_cn_b_p1))

	##################################
	##### vars_ssm_aux_2_cn ##########
	# TODO: refine --> less variables that are actually not needed

	# auxiliary variables part 2 for calculating the copy number of ssms 
	def vars_ssm_aux_2_cn(self):
		entries = (self.ssm_num * self.sublin_num * self.sublin_num 
			* cons.PHASE_NUMBER * self.cnv_state_num)
		self.create_vars_obj_ub_lb_t(1.0, 0.0, "C", entries)
	
		(self.my_colnames_dssm_aux_2_cn_a_p1, dssm_aux_2_cn_a_p1_flat) = (
			self.create_colnames_vars_ssm_aux_2_cn("a", "p1"))
		(self.my_colnames_dssm_aux_2_cn_b_p1, dssm_aux_2_cn_b_p1_flat) = (
			self.create_colnames_vars_ssm_aux_2_cn("b", "p1"))
		(self.my_colnames_dssm_aux_2_cn_a_m1, dssm_aux_2_cn_a_m1_flat) = (
			self.create_colnames_vars_ssm_aux_2_cn("a", "m1"))
		(self.my_colnames_dssm_aux_2_cn_b_m1, dssm_aux_2_cn_b_m1_flat) = (
			self.create_colnames_vars_ssm_aux_2_cn("b", "m1"))
		self.my_colnames_dssm_aux_2_cn_a_p1_index = self.create_3d_index_array(
			len(self.my_colnames), self.ssm_num, self.sublin_num, self.sublin_num)
		self.my_colnames.extend(dssm_aux_2_cn_a_p1_flat)
		self.my_colnames_dssm_aux_2_cn_b_p1_index = self.create_3d_index_array(
			len(self.my_colnames), self.ssm_num, self.sublin_num, self.sublin_num)
		self.my_colnames.extend(dssm_aux_2_cn_b_p1_flat)
		self.my_colnames_dssm_aux_2_cn_a_m1_index = self.create_3d_index_array(
			len(self.my_colnames), self.ssm_num, self.sublin_num, self.sublin_num)
		self.my_colnames.extend(dssm_aux_2_cn_a_m1_flat)
		self.my_colnames_dssm_aux_2_cn_b_m1_index = self.create_3d_index_array(
			len(self.my_colnames), self.ssm_num, self.sublin_num, self.sublin_num)
		self.my_colnames.extend(dssm_aux_2_cn_b_m1_flat)

	def create_colnames_vars_ssm_aux_2_cn(self, phase, state):
		list_3d = [self.create_colnames_vars_ssm_aux_2_cn_first_lin(phase, k, state) for k in xrange(self.ssm_num)]
		flat_list = self.flatten_3d_list(list_3d)
		return (list_3d, flat_list)

	def create_colnames_vars_ssm_aux_2_cn_first_lin(self, phase, ssm_index, state):
		return [self.create_colnames_vars_ssm_aux_2_cn_second_lin(phase, ssm_index, i, state) for i in
			range(self.sublin_num)]

	def create_colnames_vars_ssm_aux_2_cn_second_lin(self, phase, ssm_index, first_lin_index, state):
		return ["dssm_aux_2_cn_{0}_{4}_{1}_{2}_{3}".format(phase, ssm_index, first_lin_index, j, state) for
			j in range(self.sublin_num)]

		#self.my_colnames_dssm_aux_2_cn_a_p1 = self.create_vars_three_indices(1.0, 0.0, "C", "dssm_aux_2_cn_a_p1",
		#	self.ssm_num, self.sublin_num, self.sublin_num)
		#self.my_colnames_dssm_aux_2_cn_b_p1 = self.create_vars_three_indices(1.0, 0.0, "C", "dssm_aux_2_cn_b_p1",
		#	self.ssm_num, self.sublin_num, self.sublin_num)
	
	##### vars_ssm_aux_2_cn ##########
	##################################
	
	##################################
	##### vars_dssm_infl_cnv #########

	# auxiliary variables to see whether an SSM of lineage k is phased to the same chromatid as a CNV 
	# of lineage k'
	def vars_dssm_infl_cnv(self):

		if self.sublin_num > 2:
			entries = (self.untrivial_z_entries_num * self.ssm_num
				* cons.PHASE_NUMBER * self.cnv_state_num)
			self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entries)

			(self.my_colnames_dssm_infl_cnv_a_p1, self.my_colnames_dssm_infl_cnv_a_p1_flat) = (
				self.create_colnames_vars_using_upper_half_z("a", "p1", self.ssm_num, 
				"dssm_infl_cnv"))
			(self.my_colnames_dssm_infl_cnv_b_p1, self.my_colnames_dssm_infl_cnv_b_p1_flat) = (
				self.create_colnames_vars_using_upper_half_z("b", "p1", self.ssm_num,
				"dssm_infl_cnv"))
			(self.my_colnames_dssm_infl_cnv_a_m1, self.my_colnames_dssm_infl_cnv_a_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("a", "m1", self.ssm_num,
				"dssm_infl_cnv"))
			(self.my_colnames_dssm_infl_cnv_b_m1, self.my_colnames_dssm_infl_cnv_b_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("b", "m1", self.ssm_num,
				"dssm_infl_cnv"))

			element_number = len(self.my_colnames_dssm_infl_cnv_a_p1_flat)
			self.my_colnames_dssm_infl_cnv_a_p1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.ssm_num, element_number)
			self.my_colnames.extend(self.my_colnames_dssm_infl_cnv_a_p1_flat)
			self.my_colnames_dssm_infl_cnv_b_p1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.ssm_num, element_number)
			self.my_colnames.extend(self.my_colnames_dssm_infl_cnv_b_p1_flat)
			self.my_colnames_dssm_infl_cnv_a_m1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.ssm_num, element_number)
			self.my_colnames.extend(self.my_colnames_dssm_infl_cnv_a_m1_flat)
			self.my_colnames_dssm_infl_cnv_b_m1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.ssm_num, element_number)
			self.my_colnames.extend(self.my_colnames_dssm_infl_cnv_b_m1_flat)
	
	def create_colnames_vars_using_upper_half_z(self, phase, state, num, name, k_before_k_prime=True):
		list_3d = [self.create_colnames_vars_using_upper_half_z_first_lin(phase, state, j, name,
			k_before_k_prime) for j in xrange(num)]
		flat_list = self.flatten_3d_list(list_3d)
		return (list_3d, flat_list)

	def create_colnames_vars_using_upper_half_z_first_lin(self, phase, state, j, name, k_before_k_prime):
		if k_before_k_prime:
			return [self.create_colnames_vars_using_upper_half_z_second_lin(phase, state, j, k, 
				name, k_before_k_prime) for k in range(1, self.sublin_num - 1)]
		else:
			return [self.create_colnames_vars_using_upper_half_z_second_lin(phase, state, j, k_prime,
				name, k_before_k_prime) for k_prime in range(2, self.sublin_num)]

	def create_colnames_vars_using_upper_half_z_second_lin(self, phase, state, j, first_index_k, 
		name, k_before_k_prime):
		if k_before_k_prime:
			return ["{5}_{0}_{1}_{2}_{3}_{4}".format(phase, state, j, first_index_k, k_prime, name) 
				for k_prime in range(first_index_k + 1, self.sublin_num)]
		else:
			return ["{5}_{0}_{1}_{2}_{3}_{4}".format(phase, state, j, first_index_k, k, name) 
				for k in range(1, first_index_k)]
	
	def create_colnames_indices_decreasing_list_size(self, start_index, first_dimension, element_number):
		entries_per_seg = self.sublin_num - 2
		list_to_acc = [i for i in range(self.sublin_num-2, 0, -1)] * first_dimension
		return self.create_2d_uneven_indices_list(start_index, entries_per_seg, list_to_acc, element_number)

	##### vars_dssm_infl_cnv #########
	##################################

	##################################
	##### vars_dc_descendant #########

	# auxiliary variables to see whether a change in copy number happens in a lineage k' 
	# that is descendant to lineage k
	def vars_dc_descendant(self):
		if self.sublin_num > 2:
			entries = (self.untrivial_z_entries_num * self.seg_num
				* cons.PHASE_NUMBER * self.cnv_state_num)
			self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entries)

			(self.my_colnames_dc_descendant_a_p1, self.my_colnames_dc_descendant_a_p1_flat) = (
				self.create_colnames_vars_using_upper_half_z("a", "p1", self.seg_num,
				"dc_descendant"))
			(self.my_colnames_dc_descendant_b_p1, self.my_colnames_dc_descendant_b_p1_flat) = (
				self.create_colnames_vars_using_upper_half_z("b", "p1", self.seg_num,
				"dc_descendant"))
			(self.my_colnames_dc_descendant_a_m1, self.my_colnames_dc_descendant_a_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("a", "m1", self.seg_num,
				"dc_descendant"))
			(self.my_colnames_dc_descendant_b_m1, self.my_colnames_dc_descendant_b_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("b", "m1", self.seg_num,
				"dc_descendant"))

			self.dc_descdendant_start_index = len(self.my_colnames)
			element_number = len(self.my_colnames_dc_descendant_a_p1_flat)
			self.my_colnames_dc_descendant_a_p1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.seg_num, element_number)
			self.my_colnames.extend(self.my_colnames_dc_descendant_a_p1_flat)
			self.my_colnames_dc_descendant_b_p1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.seg_num, element_number)
			self.my_colnames.extend(self.my_colnames_dc_descendant_b_p1_flat)
			self.my_colnames_dc_descendant_a_m1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.seg_num, element_number)
			self.my_colnames.extend(self.my_colnames_dc_descendant_a_m1_flat) 
			self.my_colnames_dc_descendant_b_m1_index = self.create_colnames_indices_decreasing_list_size(
				len(self.my_colnames), self.seg_num, element_number)
			self.my_colnames.extend(self.my_colnames_dc_descendant_b_m1_flat)

	##### vars_dc_descendant #########
	##################################

	##################################
	##### vars_dc_ancestral #########

	# auxiliary variables to see whether a chromatid was lost in an ancestral lineage k of lineage k'
	def vars_dc_ancestral(self):
		if self.sublin_num > 2:
			entries = (self.untrivial_z_entries_num * self.seg_num
				* cons.PHASE_NUMBER)
			self.create_vars_obj_ub_lb_t(1.0, 0.0, "B", entries)

			(self.my_colnames_dc_ancestral_a_m1, self.my_colnames_dc_ancestral_a_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("a", "m1", self.seg_num,
				"dc_ancestral", k_before_k_prime=False))
			(self.my_colnames_dc_ancestral_b_m1, self.my_colnames_dc_ancestral_b_m1_flat) = (
				self.create_colnames_vars_using_upper_half_z("b", "m1", self.seg_num,
				"dc_ancestral", k_before_k_prime=False))

			self.dc_ancestral_start_index = len(self.my_colnames)
			self.my_colnames_dc_ancestral_a_m1_index = self.create_dc_ancestral_indices(
				len(self.my_colnames))
			self.my_colnames.extend(self.my_colnames_dc_ancestral_a_m1_flat) 
			self.my_colnames_dc_ancestral_b_m1_index = self.create_dc_ancestral_indices(
				len(self.my_colnames))
			self.my_colnames.extend(self.my_colnames_dc_ancestral_b_m1_flat)

	def create_dc_ancestral_indices(self, start_index):
		entries_per_seg = self.sublin_num - 2
		list_to_acc = [i+1 for i in xrange(entries_per_seg)] * self.seg_num
		element_number = len(self.my_colnames_dc_ancestral_a_m1_flat)
		return self.create_2d_uneven_indices_list(start_index, entries_per_seg, list_to_acc, element_number)

	def create_2d_uneven_indices_list(self, start_index, entries_per_seg, list_to_acc, element_number):
		# array with accumulating indices, to cut the next array
		acc_array = np.add.accumulate(list_to_acc)
		acc_array = np.insert(acc_array, 0, 0)
		# array with indices of variables
		indices = [i + start_index for i in xrange(element_number)]
		# list with indices is cut properly
		my_list = [indices[acc_array[i]:acc_array[i+1]] for i in xrange(len(acc_array)-1)]
		return [my_list[i:i+entries_per_seg] for i in xrange(0, len(my_list), entries_per_seg)]

	##### vars_dc_ancestral #########
	##################################

	# method for creation of parts of variables
	def create_vars_obj_ub_lb_t(self, item_ub, item_lb, item_type, vars_num):
		# no influence in objective function
		self.my_obj.extend([0.0] * vars_num)
		self.my_ub.extend([item_ub] * vars_num)
		self.my_lb.extend([item_lb] * vars_num)
		self.my_ctype.extend([item_type] * vars_num)


	################################################################################################
	####################   CONSTRAINTS   ###########################################################
	################################################################################################
	
	def create_rownames_one_index(self, name, num):
		return ["{0}_{1}".format(name, i) for i in xrange(num)]

	def create_row_vars_one_colname_val_ones(self, my_list):
		return [[my_list[i], [1.0] * len(my_list[i])] for i in xrange(len(my_list))]

	# constraint #1: auxilliary variables of spline sum up to one
	# seg: sum_i seg_a_i_i' = 1 (for both chromatids in allele-specific CN case)
	# snp: sum_l snp_a_l_l' = 1 
	# ssm: sum_l ssm_a_j_j' = 1
	def constraint_aux_vars_spline_sum_one(self):

		entries = 0
		if self.allele_specific:
			entries = self.seg_num * 2 + self.snp_num + self.ssm_num
		else:
			entries = self.seg_num + self.snp_num + self.ssm_num

		# rhs
		self.my_rhs.extend([1.0] * entries)
		
		# sense
		self.my_sense.extend(["E"] * entries)
		
		# rownames
		if self.allele_specific:
			self.my_rownames.extend(self.create_rownames_one_index("seg_sum_A_a", self.seg_num))
			self.my_rownames.extend(self.create_rownames_one_index("seg_sum_B_a", self.seg_num))
		else:
			self.my_rownames.extend(self.create_rownames_one_index("seg_sum_a", self.seg_num))
			self.my_rownames.extend(self.create_rownames_one_index("snp_sum_a", self.snp_num))
		self.my_rownames.extend(self.create_rownames_one_index("ssm_sum_a", self.ssm_num))

		# rows
		if self.allele_specific:
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_seg_A_index))
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_seg_B_index))
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_ssm_index))
		else:
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_seg))
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_snp))
			self.my_rows.extend(self.create_row_vars_one_colname_val_ones(self.my_colnames_ssm))
		
	# constraint #2: seg and cnv: value of spline equals mean cnv
	# a_i_l * x_i_l ... - mi * dc_a_p1_float_i_k - ... - mi * dc_b_p1_float_i_k - ... 
	#	+ mi * dc_a_m1_float_i_k + ... + mi * dc_b_m1_float_i_k + ... = mi * 2
	def constraint_cnv_value_spline_mean(self, seg_list):
		self.my_rhs.extend([seg_list[i].hm * self.normal_copy_number for i in xrange(self.seg_num)])
		self.my_sense.extend(["E"] * self.seg_num)
		self.my_rownames.extend(self.create_rownames_one_index("cnv_mean", self.seg_num))

		self.my_rows.extend([[self.my_colnames_seg_index[i] + 
			self.my_colnames_dc_a_p1_float_index[i].tolist() + 
			self.my_colnames_dc_b_p1_float_index[i].tolist() + 
			self.my_colnames_dc_a_m1_float_index[i].tolist() +
			self.my_colnames_dc_b_m1_float_index[i].tolist(), self.seg_splines[i].get_knots().tolist() + 
			[-seg_list[i].hm] * self.sublin_num * self.aux_matrices_cnv_linear_types_num +
			[seg_list[i].hm] * self.sublin_num * self.aux_matrices_cnv_linear_types_num]
			for i in xrange(self.seg_num)])

		#self.my_rows.extend([[self.my_colnames_seg[i] + self.my_colnames_dc_a_p1_float[i] + 
		#	self.my_colnames_dc_b_p1_float[i] + self.my_colnames_dc_a_m1_float[i] +
		#	self.my_colnames_dc_b_m1_float[i], self.seg_splines[i].get_knots().tolist() + 
		#	[-seg_list[i].hm] * self.sublin_num * self.aux_matrices_cnv_linear_types_num +
		#	[seg_list[i].hm] * self.sublin_num * self.aux_matrices_cnv_linear_types_num]
		#	for i in xrange(self.seg_num)])
			
	# constraint #2.1: x value of seg-spline (A/B) equals allele-specific CN (A/B)
	# a_i_l * x_i_l ... - dc_a_p1_float_i_k - ... + dc_a_m1_float_i_k + ... = 1
	def constraint_cnv_value_spline_cn_allele_specific(self, seg_list):
		self.my_rhs.extend([self.starting_allele_specific_copy_number] * self.seg_num * 2)
		self.my_sense.extend(["E"] * self.seg_num * 2)
		self.my_rownames.extend(self.create_rownames_one_index("seg_A", self.seg_num))
		self.my_rownames.extend(self.create_rownames_one_index("seg_B", self.seg_num))
		self.constraint_cnv_value_spline_cn_allele_specific_helping(self.my_colnames_seg_A_index,
			self.my_colnames_dc_a_p1_float_index, self.my_colnames_dc_a_m1_float_index,
			self.seg_splines_A)
		self.constraint_cnv_value_spline_cn_allele_specific_helping(self.my_colnames_seg_B_index,
			self.my_colnames_dc_b_p1_float_index, self.my_colnames_dc_b_m1_float_index,
			self.seg_splines_B)

	def constraint_cnv_value_spline_cn_allele_specific_helping(self, colnames_seg_index, colnames_dc_p1_float_index,
		colnames_dc_m1_float_index, seg_splines):
		self.my_rows.extend([[colnames_seg_index[i] + colnames_dc_p1_float_index[i].tolist() 
			+ colnames_dc_m1_float_index[i].tolist(), 
			seg_splines[i].get_knots().tolist() + [-1] * self.sublin_num 
			+ [1] * self.sublin_num] for i in xrange(self.seg_num)])

	# contraint #3: phi_0 = 1 as it is the "normal" clone
	# Lineages 0 is predecessor to all
	def constraint_phi_0(self):
		self.my_rhs.append(1.0)
		self.my_rownames.append("phi_0")
		self.my_sense.append("E")
		self.my_rows.append([[self.my_phis_index[0]], [1]])
		
	# contraint #4: phi_k >= phi_k+1
	# Lineages are ordered
	def constraint_phi_k(self):
		entries = self.sublin_num - 1
		self.my_rhs.extend([0.0] * entries)
		# formulate constraint
		c_vars_phi_k_greater = []
		c_values_phi_k_greater = []
		for i in range(entries):
			c_vars_phi_k_greater.append([self.my_phis_index[i], self.my_phis_index[i + 1]])
			c_values_phi_k_greater.append([1.0, -1.0])
			tmp_name = "phi_%d_greater_equal_phi_%d" % (i, i+1)
			self.my_rownames.append(tmp_name)
			self.my_sense.append("G")
		# add constraint
		for i in range(entries):
			self.my_rows.append([c_vars_phi_k_greater[i], c_values_phi_k_greater[i]])

	# constraints #4.5
	# If no mutations are assigned to a lineage, its frequency has to be 0
	# Constraint is not active if lineage frequencies are fixed
	# phi_k - sum_i dc_a_p1_binary_i_k - sum_i dc_b_p1_binary_i_k - sum_i dc_a_m1_binary_i_k
	#	- sum_i dc_b_m1_binary_i_k - sum_j dssm_j_k - sum_j dssm_a_j_k - sum_j dssm_b_j_k <= 0
	def constraints_phi_mutation_number(self):
		entries = self.sublin_num - 1
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["phi_mutation_number_{0}".format(k) for k in xrange(1, self.sublin_num)])
		if self.simple_CN_changes:
			self.my_rows.extend([[[self.my_phis_index[k]] + self.my_colnames_dc_a_p1_binary_index[:,k].tolist()
				+ self.my_colnames_dc_b_p1_binary_index[:,k].tolist() 
				+ self.my_colnames_dc_a_m1_binary_index[:,k].tolist()
				+ self.my_colnames_dc_b_m1_binary_index[:,k].tolist()
				+ self.my_colnames_dssm_index[:,k].tolist()
				+ self.my_colnames_dssm_a_index[:,k].tolist()
				+ self.my_colnames_dssm_b_index[:,k].tolist(), 
				[1.0] + [-1.0] * ((self.seg_num * 4) + (self.ssm_num * 3))]
				for k in xrange(1, self.sublin_num)])
		else:
			self.my_rows.extend([[[self.my_phis_index[k]] + self.my_colnames_dc_a_p1_binary_index[:,k].tolist()
				+ self.my_colnames_dc_b_p1_binary_index[:,k].tolist() 
				+ self.my_colnames_dc_a_m1_binary_index[:,k].tolist()
				+ self.my_colnames_dc_b_m1_binary_index[:,k].tolist()
				+ self.my_colnames_dssm_a_index[:,k].tolist()
				+ self.my_colnames_dssm_b_index[:,k].tolist(), 
				[1.0] + [-1.0] * ((self.seg_num * 4) + (self.ssm_num * 2))]
				for k in xrange(1, self.sublin_num)])
	
	# constraint #4.7
	# if a lineage has a frequency of ~ 0 in all samples, no mutations are allowed to be assigned to it
	# phi_k - mut_k >= \epsilon_freq - 1
	# where mut_k is all dc_a/m_p1/m1_binary_i_k and dssm_(a/b)_j_k in separate constraints
	def constraints_phi_0_no_mutations(self):
		if self.simple_CN_changes:
			number_ssm_matrices = 3
		else:
			number_ssm_matrices = 2
		entries = ((self.sublin_num - 1) * (self.ssm_num * number_ssm_matrices) 
			+ (self.sublin_num - 1) * (self.seg_num * cons.PHASE_NUMBER * self.cnv_state_num))
		self.my_rhs.extend([cons.EPSILON_FREQUENCY - 1] * entries)
		self.my_sense.extend(["G"] * entries)
		# constraints for dc_a/m_p1/m1_binary_i_k
		self.constraints_phi_0_no_mutations_help("dc_a_p1", self.my_colnames_dc_a_p1_binary_index, self.seg_num)
		self.constraints_phi_0_no_mutations_help("dc_b_p1", self.my_colnames_dc_b_p1_binary_index, self.seg_num)
		self.constraints_phi_0_no_mutations_help("dc_a_m1", self.my_colnames_dc_a_m1_binary_index, self.seg_num)
		self.constraints_phi_0_no_mutations_help("dc_b_m1", self.my_colnames_dc_b_m1_binary_index, self.seg_num)
		if self.simple_CN_changes:
			self.constraints_phi_0_no_mutations_help("ssm_unphased", self.my_colnames_dssm_index, self.ssm_num)
		self.constraints_phi_0_no_mutations_help("ssm_a", self.my_colnames_dssm_a_index, self.ssm_num)
		self.constraints_phi_0_no_mutations_help("ssm_b", self.my_colnames_dssm_b_index, self.ssm_num)


	def constraints_phi_0_no_mutations_help(self, mutation_name, mutation_index, mutation_num):
		self.my_rownames.extend(["phi_{0}_0_no_mutations_{1}_{2}".format(k, mutation_name, i)
			for k in xrange(1, self.sublin_num) for i in xrange(mutation_num)])
		self.my_rows.extend([[[self.my_phis_index[k], mutation_index[i][k]], [1.0, -1.0]] 
			for k in xrange(1, self.sublin_num) for i in xrange(mutation_num)])

	# constraints #5, #6, #7: auxiliary variables for delta_C_A and delta_C_B matrices
	# so that they can be computed with linear constraints
	# constraint #5: dc_a_p1_float_i_k <= dc_a_p1_binary_i_k, same for b, similar for m1
	# constraint #6: dc_a_p1_float_i_k - phi_k - dc_a_p1_binary_i_k >= -1, same for b, similar for m1
	# constraint #7: dc_a_p1_float_i_k <= phi_k, same for b, same for m1
	def constraints_aux_cnv_dc_matrices(self):
		# Constraint #5
		self.cons_aux_cnv_dc_matrices_float_le_binary()
		# Constraint #6
		self.cons_aux_cnv_dc_matrices_float_ge_phi_c()
		# Constraint $7
		self.cons_aux_cnv_dc_matrices_float_le_phi()
	
	# c #5
	# constraint #5: dc_a_p1_float_i_k <= dc_a_p1_binary_i_k, same for b
	# 		dc_a_m1_float_i_k <= dc_a_m1_binary_i_k, same for b
	def cons_aux_cnv_dc_matrices_float_le_binary(self):
		self.my_rhs.extend([0.0] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_sense.extend(["L"] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_p1_float_less_c"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_p1_float_less_c"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_m1_float_less_c"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_m1_float_less_c"))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_le_binary_inner_loop(i, 
			self.my_colnames_dc_a_p1_float_index, self.my_colnames_dc_a_p1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_le_binary_inner_loop(i, 
			self.my_colnames_dc_b_p1_float_index, self.my_colnames_dc_b_p1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_le_binary_inner_loop(i, 
			self.my_colnames_dc_a_m1_float_index, self.my_colnames_dc_a_m1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_le_binary_inner_loop(i, 
			self.my_colnames_dc_b_m1_float_index, self.my_colnames_dc_b_m1_binary_index) 
			for i in xrange(self.seg_num)]))

	def create_rows_cons_aux_cnv_dc_matrices_float_le_binary_inner_loop(self, index_1, float_phase, 
		binary_phase):
		return [[[float_phase[index_1][j], binary_phase[index_1][j]],
			[1.0, -1.0]] for j in range(self.sublin_num)]

	# c #6
	# constraint #6: dc_a_p1_float_i_k - phi_k - dc_a_p1_binary_i_k >= -1, same for b
	# 		dc_a_m1_float_i_k - phi_k - dc_a_m1_binary_i_k >= -1, same for b
	def cons_aux_cnv_dc_matrices_float_ge_phi_c(self):
		self.my_rhs.extend([-1.0] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_sense.extend(["G"] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_p1_float_greater_than"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_p1_float_greater_than"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_m1_float_greater_than"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_m1_float_greater_than"))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_ge_phi_c_inner_loop(
			i, self.my_colnames_dc_a_p1_float_index, self.my_colnames_dc_a_p1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_ge_phi_c_inner_loop(
			i, self.my_colnames_dc_b_p1_float_index, self.my_colnames_dc_b_p1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_ge_phi_c_inner_loop(
			i, self.my_colnames_dc_a_m1_float_index, self.my_colnames_dc_a_m1_binary_index) 
			for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list(
			[self.create_rows_cons_aux_cnv_dc_matrices_float_ge_phi_c_inner_loop(
			i, self.my_colnames_dc_b_m1_float_index, self.my_colnames_dc_b_m1_binary_index) 
			for i in xrange(self.seg_num)]))

	def create_rows_cons_aux_cnv_dc_matrices_float_ge_phi_c_inner_loop(self, index_1, float_phase, binary_phase):
		return [[[float_phase[index_1][j], self.my_phis_index[j], binary_phase[index_1][j]], [1.0, -1.0, -1.0]] 
			for j in range(self.sublin_num)]

	# c #7
	# constraint #7: dc_a_p1_float_i_k <= phi_k, same for b
	# 		dc_a_m1_float_i_k <= phi_k, same for b
	def cons_aux_cnv_dc_matrices_float_le_phi(self):
		self.my_rhs.extend([0.0] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_sense.extend(["L"] * self.cnv_aux_all_states_one_type_entry_num)
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_p1_float_smaller_phi"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_p1_float_smaller_phi"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_a_m1_float_smaller_phi"))
		self.my_rownames.extend(self.create_rownames_two_indices(self.seg_num, self.sublin_num, 
			"dc_b_m1_float_smaller_phi"))
		self.my_rows.extend(self.create_rows_cons_aux_cnv_dc_matrices_float_le_phi(
			self.seg_num, self.sublin_num, self.my_colnames_dc_a_p1_float_index))
		self.my_rows.extend(self.create_rows_cons_aux_cnv_dc_matrices_float_le_phi(
			self.seg_num, self.sublin_num, self.my_colnames_dc_b_p1_float_index))
		self.my_rows.extend(self.create_rows_cons_aux_cnv_dc_matrices_float_le_phi(
			self.seg_num, self.sublin_num, self.my_colnames_dc_a_m1_float_index))
		self.my_rows.extend(self.create_rows_cons_aux_cnv_dc_matrices_float_le_phi(
			self.seg_num, self.sublin_num, self.my_colnames_dc_b_m1_float_index))
		
	def create_rows_cons_aux_cnv_dc_matrices_float_le_phi(self, index_1, index_2, float_phase):
		return [[[float_phase[i][j], self.my_phis_index[j]], [1.0, -1.0]] for i in xrange(index_1) 
			for j in xrange(index_2)]

	def create_rownames_two_indices(self, index_1, index_2, name):
		return ["{0}_{1}_{2}".format(name, i, j) for i in xrange(index_1) for j in xrange(index_2)]

	# constraint #8.1: symmetry breaking, no CN change of plus one on B chromatid
	# dc_b_p1_binary_i_k + ... = 0
	# constraint #8.2: symmetry breaking, no CN change of minus one on A chromatid
	# dc_a_m1_binary_i_k + ... = 0
	def constraint_remove_cn_symmetry(self):
		# constraint #8.1
		self.constraint_remove_cn_symmetry_b_p1_zero()
		# constraint #8.2
		self.constraint_remove_cn_symmetry_a_m1_zero()

	# constraint #8.1
	def constraint_remove_cn_symmetry_b_p1_zero(self):
		self.my_rhs.extend([0.0] * self.seg_num)
		self.my_sense.extend(["E"] * self.seg_num)
		self.my_rownames.extend(self.create_rownames_one_index("remove_cn_symmetry_b_p1_zero", self.seg_num))
		self.my_rows.extend([[self.my_colnames_dc_b_p1_binary_index[i].tolist(), [1.0] * self.sublin_num] 
			for i in xrange(self.seg_num)])

	# constraint #8.2
	def constraint_remove_cn_symmetry_a_m1_zero(self):
		self.my_rhs.extend([0.0] * self.seg_num)
		self.my_sense.extend(["E"] * self.seg_num)
		self.my_rownames.extend(self.create_rownames_one_index("remove_cn_symmetry_a_m1_zero", self.seg_num))
		self.my_rows.extend([[self.my_colnames_dc_a_m1_binary_index[i].tolist(), [1.0] * self.sublin_num] 
			for i in xrange(self.seg_num)])

	# constraint #8.3: takes care of major and minor CN
	# CN of A is always greater equal CN of B
	# dc_a_p1_float_i_k + ... - dc_a_m1_float_i_k - ... - dc_b_p1_float_i_k - ...
	# 	+ dc_b_m1_float_i_k + ... >= 0
	def constraint_major_minor_cn(self):
		self.my_rhs.extend([0.0] * self.seg_num)
		self.my_sense.extend(["G"] * self.seg_num)
		self.my_rownames.extend(self.create_rownames_one_index("major_minor_cn", self.seg_num))
		self.my_rows.extend([[self.my_colnames_dc_a_p1_float_index[i].tolist() 
			+ self.my_colnames_dc_a_m1_float_index[i].tolist() 
			+ self.my_colnames_dc_b_p1_float_index[i].tolist()
			+ self.my_colnames_dc_b_m1_float_index[i].tolist(), [1.0] * self.sublin_num 
			+ [-1.0] * self.sublin_num * 2 + [1.0] * self.sublin_num]
			 for i in xrange(self.seg_num)])

	# constraints #9.1, 9.2, 9.3: only one entry per row in delta_C_i is non-zero
	# dc_a_p1_binary_i_k + ... + dc_b_p1_binary_i_k + ... <= 1
	def constraint_delta_c_allowed_entries_per_row(self):
		self.constraint_dc_row_same_state()
		self.constraint_dc_row_same_chromatid()
		self.constraint_dc_row_different()

	# constraint 9.1: only one entry per row with same state
	# dc_a_p1_binary_i_k + ... + dc_b_p1_binary_i_k + ... <= 1, same for m1
	def constraint_dc_row_same_state(self):
		self.my_rhs.extend([1.0] * self.seg_num * self.cnv_state_num)
		self.my_sense.extend(["L"] * self.seg_num * self.cnv_state_num)
		self.my_rownames.extend(self.create_rownames_one_index("dc_row_same_state_p1", self.seg_num))
		self.my_rownames.extend(self.create_rownames_one_index("dc_row_same_state_m1", self.seg_num))
		entries = self.sublin_num * cons.PHASE_NUMBER
		self.my_rows.extend(self.flatten_list([[[self.my_colnames_dc_a_p1_binary_index[i].tolist() 
			+ self.my_colnames_dc_b_p1_binary_index[i].tolist(),
			[1.0] * entries]] for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list([[[self.my_colnames_dc_a_m1_binary_index[i].tolist() 
			+ self.my_colnames_dc_b_m1_binary_index[i].tolist(),
			[1.0] * entries]] for i in xrange(self.seg_num)]))

	# constraint 9.2: only one entry per row with same chromatid
	# dc_a_p1_binary_i_k + ... + dc_a_m1_binary_i_k + ... <= 1, same for b
	def constraint_dc_row_same_chromatid(self):
		self.my_rhs.extend([1.0] * self.seg_num * self.cnv_state_num)
		self.my_sense.extend(["L"] * self.seg_num * self.cnv_state_num)
		self.my_rownames.extend(self.create_rownames_one_index("dc_row_same_chromatid_a", self.seg_num))
		self.my_rownames.extend(self.create_rownames_one_index("dc_row_same_chromatid_b", self.seg_num))
		entries = self.sublin_num * self.cnv_state_num
		self.my_rows.extend(self.flatten_list([[[self.my_colnames_dc_a_p1_binary_index[i].tolist() 
			+ self.my_colnames_dc_a_m1_binary_index[i].tolist(),
			[1.0] * entries]] for i in xrange(self.seg_num)]))
		self.my_rows.extend(self.flatten_list([[[self.my_colnames_dc_b_p1_binary_index[i].tolist() 
			+ self.my_colnames_dc_b_m1_binary_index[i].tolist(),
			[1.0] * entries]] for i in xrange(self.seg_num)]))

	# constraint 9.3: only one entry for combination of variable on one chromatid with one state and other variables
	#	on different chromatid with different state and different sublineage
	# dc_b_p1_binary_i_k' + \sum_{k \ne k'} dc_a_m1_binary_i_k for k', k > 0, equal for other combination of a and b
	def constraint_dc_row_different(self):
		constraint_number = self.seg_num * cons.PHASE_NUMBER * (self.sublin_num - 1)
		self.my_rhs.extend([1.0] * constraint_number)
		self.my_sense.extend(["L"] * constraint_number)
		self.my_rownames.extend(self.create_rownames_dc_row_different("a"))
		self.my_rownames.extend(self.create_rownames_dc_row_different("b"))
		self.my_rows.extend(self.create_constraint_rows_dc_row_different(self.my_colnames_dc_a_p1_binary_index,
			self.my_colnames_dc_b_m1_binary_index))
		self.my_rows.extend(self.create_constraint_rows_dc_row_different(self.my_colnames_dc_b_p1_binary_index,
			self.my_colnames_dc_a_m1_binary_index))

	def create_constraint_rows_dc_row_different(self, dc_p1_binary_index, dc_m1_binary_index):
		values = [1.0] * (self.sublin_num - 1)
		return [[[dc_p1_binary_index[i][k]] + dc_m1_binary_index[i][1:k].tolist() 
			+ dc_m1_binary_index[i][k+1:].tolist(), values] 
			for i in xrange(self.seg_num) for k in range(1, self.sublin_num)]
		
	def create_rownames_dc_row_different(self, phase):
		return ["dc_row_different_{0}_p1_{1}_{2}".format(phase, i, k) for i in xrange(self.seg_num)
			for k in range(1, self.sublin_num)]

	# constraint #9.4: no CNV in lineage 0 as this is the "normal" lineages
	def constraint_no_cnv_in_lineage_0(self):
		self.my_rhs.extend([0.0] * self.seg_num)
		# formulate constraint
		c_vars_no_CNV_in_first_lineage = []
		c_values_no_CNV_in_first_lineage = ([1.0] * self.cnv_state_num * 
			self.aux_matrices_cnv_linear_types_num)
		self.my_sense.extend(["E"] * self.seg_num)
		self.my_rownames.extend(["no_CNV_in_first_lineage_{0}".format(i)
			for i in xrange(self.seg_num)])
		self.my_rows.extend([[[self.my_colnames_dc_a_p1_binary_index[i][0], 
			self.my_colnames_dc_b_p1_binary_index[i][0],
			self.my_colnames_dc_a_m1_binary_index[i][0], 
			self.my_colnames_dc_b_m1_binary_index[i][0]],
			c_values_no_CNV_in_first_lineage] for i in xrange(self.seg_num)])
	
	# constraint #9.5: allow x copy number changes per segment
	# dc_a_p1_binary_i_k + ... + dc_b_p1_binary_i_k + ... + dc_a_m1_binary_i_k
	#	+ ... + dc_b_m1_binary_i_k + ... <= x for 0 <= k < K
	def constraint_max_cn_changes(self):
		self.my_rhs.extend([self.max_x_CN_changes] * self.seg_num)
		self.my_sense.extend(["L"] * self.seg_num)
		self.my_rownames.extend(["max_cn_changes_{0}".format(i) for i in xrange(self.seg_num)])
		self.my_rows.extend([[self.my_colnames_dc_a_p1_binary_index[i].tolist() 
			+ self.my_colnames_dc_b_p1_binary_index[i].tolist()
			+ self.my_colnames_dc_a_m1_binary_index[i].tolist() 
			+ self.my_colnames_dc_b_m1_binary_index[i].tolist(), 
			[1.0] * cons.PHASE_NUMBER * self.cnv_state_num * self.sublin_num]
			for i in xrange(self.seg_num)])
	
	# constraint #9.55: CN gain and loss in same lineage and on same chromatid are not allowed
	# dc_a_p1_binary_i_k + dc_a_m1_binary_i_k <= 1, for 1 <= k <= K
	def constraint_no_simultaneous_loss_and_gain(self):
		entries = self.seg_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_rhs.extend([1.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["no_simultaneous_loss_and_gain_a_{0}_{1}".format(i, k)
			for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)])
		self.my_rownames.extend(["no_simultaneous_loss_and_gain_b_{0}_{1}".format(i, k)
			for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_dc_a_p1_binary_index[i][k], 
			self.my_colnames_dc_a_m1_binary_index[i][k]],
			[1.0, 1.0]] for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_dc_b_p1_binary_index[i][k], 
			self.my_colnames_dc_b_m1_binary_index[i][k]],
			[1.0, 1.0]] for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)])


	# constraint #9.6: allow loss of each chromatid only once per segment
	# dc_a_m1_binary_i_k + ... <= 1 for 0 <= k < K, same for b
	def constraint_loss_per_chromatid(self):
		entries = self.seg_num * cons.PHASE_NUMBER
		self.my_rhs.extend([1.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["loss_per_chromatid_a_{0}".format(i) for i in xrange(self.seg_num)])
		self.my_rownames.extend(["loss_per_chromatid_b_{0}".format(i) for i in xrange(self.seg_num)])
		self.my_rows.extend([[self.my_colnames_dc_a_m1_binary_index[i], [1.0] * self.sublin_num]
			for i in xrange(self.seg_num)])
		self.my_rows.extend([[self.my_colnames_dc_b_m1_binary_index[i], [1.0] * self.sublin_num]
			for i in xrange(self.seg_num)])

	# constraint #9.7: not possible to get a copy of a chromatid that was already lost in ancestral lineage
	# dc_ancestral_a_m1_i_k'_k + ... + dc_a_p1_binary_i_k' <= 1, for k > k'
	# for 2 <= k' < K
	# same for b
	def constraint_no_gains_of_lost_chromatids(self):
		if self.sublin_num > 2:
			entries = cons.PHASE_NUMBER * self.seg_num * (self.sublin_num - 2)
			self.my_rhs.extend([1.0] * entries)
			self.my_sense.extend(["L"] * entries)
			self.my_rownames.extend(["no_gains_of_lost_chromatids_a_{0}_{1}".format(i, k_prime)
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rownames.extend(["no_gains_of_lost_chromatids_b_{0}_{1}".format(i, k_prime)
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rows.extend([[self.my_colnames_dc_ancestral_a_m1_index[i][k_prime-2] + 
				[self.my_colnames_dc_a_p1_binary_index[i][k_prime]], [1.0] * (k_prime - 1 + 1)]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rows.extend([[self.my_colnames_dc_ancestral_b_m1_index[i][k_prime-2] + 
				[self.my_colnames_dc_b_p1_binary_index[i][k_prime]], [1.0] * (k_prime - 1 + 1)]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])


	# constraint #9.75: not possible to get a copy of or to lose a chromatid that was already lost in ancestral lineage
	# dc_ancestral_a_m1_i_k'_k + ... + dc_a_p1_binary_i_k' + dc_a_m1_binary_i_k' <= 1, for k > k'
	# for 2 <= k' < K
	# same for b
	def constraint_no_gains_or_losses_of_lost_chromatids(self):
		if self.sublin_num > 2:
			entries = cons.PHASE_NUMBER * self.seg_num * (self.sublin_num - 2)
			self.my_rhs.extend([1.0] * entries)
			self.my_sense.extend(["L"] * entries)
			self.my_rownames.extend(["no_gains_or_losses_of_lost_chromatids_a_{0}_{1}".format(i, k_prime)
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rownames.extend(["no_gains_or_losses_of_lost_chromatids_b_{0}_{1}".format(i, k_prime)
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rows.extend([[self.my_colnames_dc_ancestral_a_m1_index[i][k_prime-2] + 
				[self.my_colnames_dc_a_p1_binary_index[i][k_prime]] 
				+ [self.my_colnames_dc_a_m1_binary_index[i][k_prime]], 
				[1.0] * (k_prime - 1 + 2)]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])
			self.my_rows.extend([[self.my_colnames_dc_ancestral_b_m1_index[i][k_prime-2] + 
				[self.my_colnames_dc_b_p1_binary_index[i][k_prime]] 
				+ [self.my_colnames_dc_b_m1_binary_index[i][k_prime]], 
				[1.0] * (k_prime - 1 + 2)]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)])

	####################################################################
	################## constraint_only_gains_losses_LOH ###############

	# constraint #9.8: only gains or losses or LOH once per segment are allowed
	# dc_a_p1_binary_i_k + dc_a_m1_binary_i_k' <= 1 for 1 <= k < K, 1 <= k' < K, k != k'
	# dc_a_p1_binary_i_k + dc_b_p1_binary_i_k' <= 1, same for b as first summand
	def constraint_only_gains_losses_LOH(self):
		if self.sublin_num > 2:
			entries = (cons.PHASE_NUMBER * cons.PHASE_NUMBER * self.seg_num *
				(self.sublin_num - 1) * (self.sublin_num - 2))
			self.my_rhs.extend([1.0] * entries)
			self.my_sense.extend(["L"] * entries)
			self.my_rownames.extend(self.constraint_only_gains_losses_LOH_rownames("a", "a"))
			self.my_rownames.extend(self.constraint_only_gains_losses_LOH_rownames("a", "b"))
			self.my_rownames.extend(self.constraint_only_gains_losses_LOH_rownames("b", "a"))
			self.my_rownames.extend(self.constraint_only_gains_losses_LOH_rownames("b", "b"))
			self.my_rows.extend(self.constraint_only_gains_losses_LOH_rows(
				self.my_colnames_dc_a_p1_binary_index, 
				self.my_colnames_dc_a_m1_binary_index))
			self.my_rows.extend(self.constraint_only_gains_losses_LOH_rows(
				self.my_colnames_dc_a_p1_binary_index, 
				self.my_colnames_dc_b_m1_binary_index))
			self.my_rows.extend(self.constraint_only_gains_losses_LOH_rows(
				self.my_colnames_dc_b_p1_binary_index, 
				self.my_colnames_dc_a_m1_binary_index))
			self.my_rows.extend(self.constraint_only_gains_losses_LOH_rows(
				self.my_colnames_dc_b_p1_binary_index, 
				self.my_colnames_dc_b_m1_binary_index))

	def constraint_only_gains_losses_LOH_rownames(self, first_summand, second_summand):
		return ["only_gains_losses_LOH_{0}_{1}_{2}_{3}_{4}".format(first_summand, second_summand,
			i, k, k_prime) for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)
			for k_prime in xrange(1, self.sublin_num) if k != k_prime]

	def constraint_only_gains_losses_LOH_rows(self, first_summand, second_summand):
		return [[[first_summand[i][k], second_summand[i][k_prime]], [1.0, 1.0]]
			for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num)
			for k_prime in xrange(1, self.sublin_num) if k != k_prime]

	################## constraint_only_gains_losses_LOH ###############
	####################################################################

	####################################################################
	################## constraint_CN_direction           ###############

	# constraint #9.91, #9.92: direction of CN change ci defines direction of inferred CN change
	# if c_i_A < 1: dc_a_p1_binary_i_k = 0, for 1 <= k < K, same for B
	# if c_i_A > 1: dc_a_m1_binary_i_k = 0, for 1 <= k < K, same for B
	def constraint_CN_direction(self, seg_list):
		# constraints created for allele A
		[self.constraint_CN_direction_per_allele(seg_list[i].given_cn_A, i, self.my_colnames_dc_a_p1_binary_index,
			self.my_colnames_dc_a_m1_binary_index, "A") for i in xrange(len(seg_list))]
		# constraints created for allele B
		[self.constraint_CN_direction_per_allele(seg_list[i].given_cn_B, i, self.my_colnames_dc_b_p1_binary_index,
			self.my_colnames_dc_b_m1_binary_index, "B") for i in xrange(len(seg_list))]

	def constraint_CN_direction_per_allele(self, given_cn, seg_index, local_colnames_dc_gain_index,
		local_colnames_dc_loss_index, phase):
		# if the CN isn't changed, no constraint needs to be created
		if given_cn == 1:
			return
		# if the CN is decreased, no gain is allowed
		elif given_cn < 1:
			forbidden_direction = "gain"
			local_colnames_binary_change = local_colnames_dc_gain_index
		# if the CN is increased, no loss is allowed
		elif given_cn > 1:
			forbidden_direction = "loss"
			local_colnames_binary_change = local_colnames_dc_loss_index

		# constraint is created
		self.my_rhs.extend([0.0] * (self.sublin_num - 1))
		self.my_sense.extend(["E"] * (self.sublin_num - 1))
		self.my_rownames.extend(["CN_direction_no_{0}_{1}_{2}_{3}".format(forbidden_direction, phase, seg_index, k)
			for k in xrange(1, self.sublin_num)])
		self.my_rows.extend([[[local_colnames_binary_change[seg_index][k]], [1.0]] for k in xrange(1, self.sublin_num)])

	################## constraint_CN_direction           ###############
	####################################################################


	### ATTENTION: constraint is not tested
	# constraint #10: diagonal and lower triangle of Z matrix are 0
	# since lineage cannot be its own ancestor and they are orderd in decreasing
	# order of frequency
	# z_k_k' = 0 for 0 <= k' <= k < K
	def constraint_z_matrix_half(self):
		self.my_rhs.extend([0.0] * self.pre_zero_z_num)
		self.my_sense.extend(["E"] * self.pre_zero_z_num)
		self.my_rownames.extend(["zero_z_{0}_{1}".format(i, j) for i in range(self.sublin_num)
			for j in range(i+1)])
		self.my_rows.extend([[[self.my_colnames_z_index[i][j]], [1.0]] for i in range(self.sublin_num)
			for j in range(i+1)])
		
	### ATTENTION: constraint is not tested
	# constraint #11: first row without first field in Z matrix are 1
	# because first lineage is ancestor of all other lineages
	# z_0_k' = 1 for 0 < k' < K
	def constraint_z_matrix_first_row_1(self):
		entry_num = self.sublin_num - 1
		self.my_rhs.extend([1.0] * entry_num)
		self.my_sense.extend(["E"] * entry_num)
		self.my_rownames.extend(["one_z_0_{0}".format(i+1) for i in range(entry_num)])
		self.my_rows.extend([[[self.my_colnames_z_index[0][i+1]], [1.0]] for i in range(entry_num)])
	
	# constraint #11.1, #11.2 and #11.3 to create auxiliary variables dssm_infl_cnv_a/b_STATE_j_k_k'
	#	to model multiplication of two variables for symmetry breaking of Z-matrix
	# #11.1: dssm_infl_cnv_a_p1_j_k_k' - dssm_a_j_k <= 0, same for b, same for m1
	# #11.2: dssm_infl_cnv_a_p1_j_k_k' - dc_a_p1_binary_i_k' <= 0, same for b, same for m1
	# #11.3: dssm_infl_cnv_a_p1_j_k_k' - dssm_a_j_k - dc_a_p1_binary_i_k' >= -1, same for b, same for m1
	def constraint_dssm_infl_cnv(self, ssm_list):
		if self.sublin_num > 2:
			constraint_num = (self.untrivial_z_entries_num * self.ssm_num 
				* cons.PHASE_NUMBER * self.cnv_state_num)
			# #11.1
			self.constraint_dssm_infl_cnv_le_dssm(constraint_num)
			# #11.2
			self.constraint_dssm_infl_cnv_le_dc_binary(constraint_num, ssm_list)
			# #11.3
			self.constraint_dssm_infl_cnv_ge_dssm_dc_binary(constraint_num, ssm_list)

	def create_rownames_constraint_mut_untrivial_z_entries(self, name, mut_num, k_before_k_prime=True):
		if k_before_k_prime:
			return ["{0}_{1}_{2}_{3}".format(name, j, k, k_prime) for j in xrange(mut_num) 
				for k in range(1, self.sublin_num - 1) 
				for k_prime in range(k + 1, self.sublin_num)]
		else:
			return ["{0}_{1}_{2}_{3}".format(name, j, k_prime, k) for j in xrange(mut_num) 
				for k_prime in range(2, self.sublin_num) 
				for k in range(1, k_prime)]


	# constraint #11.1
	def constraint_dssm_infl_cnv_le_dssm(self, constraint_num):
		self.my_rhs.extend([0.0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		# rownames
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dssm_a_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dssm_b_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dssm_a_m1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dssm_b_m1", self.ssm_num))
		# rows
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dssm(
			self.my_colnames_dssm_infl_cnv_a_p1_index, self.my_colnames_dssm_a_index))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dssm(
			self.my_colnames_dssm_infl_cnv_b_p1_index, self.my_colnames_dssm_b_index))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dssm(
			self.my_colnames_dssm_infl_cnv_a_m1_index, self.my_colnames_dssm_a_index))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dssm(
			self.my_colnames_dssm_infl_cnv_b_m1_index, self.my_colnames_dssm_b_index))
	
	def create_rows_constraint_dssm_infl_cnv_le_dssm(self, colnames_dssm_index_infl_cnv_index, colnames_dssm_index):
		return [[[colnames_dssm_index_infl_cnv_index[j][k1 - 1][k2 - k1 - 1], 
			colnames_dssm_index[j][k1]],[1.0, -1.0]]
			for j in xrange(self.ssm_num) for k1 in range(1, self.sublin_num - 1)
			for k2 in range(k1 + 1, self.sublin_num)]

	# constraint #11.2
	def constraint_dssm_infl_cnv_le_dc_binary(self, constraint_num, ssm_list):
		self.my_rhs.extend([0.0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		# rownames
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dc_binary_a_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dc_binary_b_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dc_binary_a_m1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_le_dc_binary_b_m1", self.ssm_num))
		# rows
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dc_binary(
			self.my_colnames_dssm_infl_cnv_a_p1_index, self.my_colnames_dc_a_p1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dc_binary(
			self.my_colnames_dssm_infl_cnv_b_p1_index, self.my_colnames_dc_b_p1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dc_binary(
			self.my_colnames_dssm_infl_cnv_a_m1_index, self.my_colnames_dc_a_m1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_le_dc_binary(
			self.my_colnames_dssm_infl_cnv_b_m1_index, self.my_colnames_dc_b_m1_binary_index, ssm_list))

	def create_rows_constraint_dssm_infl_cnv_le_dc_binary(self, colnames_dssm_infl_cnv_index, 
		colnames_dc_binary_index, ssm_list):
		my_list = [self.create_rows_constraint_dssm_infl_cnv_le_dc_binary_per_ssm(colnames_dssm_infl_cnv_index,
			colnames_dc_binary_index, ssm_list, j) for j in xrange(self.ssm_num)]
		return self.flatten_list(my_list)

	def create_rows_constraint_dssm_infl_cnv_le_dc_binary_per_ssm(self, colnames_dssm_infl_cnv_index,
		colnames_dc_binary_index, ssm_list, j):
		cnv_index = self.get_seg_index(ssm_list, j)
		return [[[colnames_dssm_infl_cnv_index[j][k1 - 1][k2 - k1 - 1], colnames_dc_binary_index[cnv_index][k2]],
			[1.0, -1.0]] for k1 in range(1, self.sublin_num - 1) for k2 in range(k1 + 1, 
			self.sublin_num)]

	def get_seg_index(self, ssm_list, j):
		seg_index = 0
		if not self.single_segment:
			return ssm_list[j].seg_index
		return seg_index

	# constraint #11.3
	def constraint_dssm_infl_cnv_ge_dssm_dc_binary(self, constraint_num, ssm_list):
		self.my_rhs.extend([-1] * constraint_num)
		self.my_sense.extend(["G"] * constraint_num)
		# rownames
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_ge_dssm_dc_binary_a_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_ge_dssm_dc_binary_b_p1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_ge_dssm_dc_binary_a_m1", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dssm_infl_cnv_ge_dssm_dc_binary_b_m1", self.ssm_num))
		# rows
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary(
			self.my_colnames_dssm_infl_cnv_a_p1_index, self.my_colnames_dssm_a_index,
			self.my_colnames_dc_a_p1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary(
			self.my_colnames_dssm_infl_cnv_b_p1_index, self.my_colnames_dssm_b_index,
			self.my_colnames_dc_b_p1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary(
			self.my_colnames_dssm_infl_cnv_a_m1_index, self.my_colnames_dssm_a_index,
			self.my_colnames_dc_a_m1_binary_index, ssm_list))
		self.my_rows.extend(self.create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary(
			self.my_colnames_dssm_infl_cnv_b_m1_index, self.my_colnames_dssm_b_index,
			self.my_colnames_dc_b_m1_binary_index, ssm_list))

	def create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary(self, colnames_dssm_index_infl_cnv_index, 
		colnames_dssm_index, colnames_dc_binary_index, ssm_list):
		my_list = [self.create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary_per_ssm(
			colnames_dssm_index_infl_cnv_index, colnames_dssm_index, colnames_dc_binary_index, ssm_list, j)
			for j in xrange(self.ssm_num)]
		return self.flatten_list(my_list)

	def create_rows_constraint_dssm_infl_cnv_ge_dssm_dc_binary_per_ssm(self, colnames_dssm_index_infl_cnv_index,
		colnames_dssm_index, colnames_dc_binary_index, ssm_list, j):
		cnv_index = self.get_seg_index(ssm_list, j)
		return [[[colnames_dssm_index_infl_cnv_index[j][k1 - 1][k2 - k1 -1], colnames_dssm_index[j][k1],
			colnames_dc_binary_index[cnv_index][k2]], [1.0, -1.0, -1.0]]
			for k1 in range(1, self.sublin_num - 1)
			for k2 in range(k1 + 1, self.sublin_num)]

	# constraints #11.31, #11.32 and #11.33, and #11.34, #11.35 and #11.36 to create auxiliarly variables
	#	z_trans_i_k_k'_k'' and z_trans_c_k_k'_k'' to apply transitivity on the matrix
	# #11.31: z_trans_i_k_k'_k'' - z_k_k' - z_k'_k'' >= -1
	# #11.32: z_trans_i_k_k'_k'' - z_k_k' <= 0
	# #11.33: z_trans_i_k_k'_k'' - z_k'_k'' <= 0
	# #11.34: z_trans_c_k_k'_k'' - z_k'_k'' - z_k_k'' >= -1
	# #11.35: z_trans_c_k_k'_k'' - z_k'_k'' <= 0
	# #11.36: z_trans_c_k_k'_k'' - z_k_k'' <= 0
	# constraints #11.37 and #11.38 which apply the auxiliary variables
	# #11.37: z_k_k'' - z_trans_i_k_k'_k'' >= 0 for all k': k < k' < k''
	# #11.38: z_k_k' - z_trans_c_k_k'_k'' >= 0 for all k'': k' < k'' < K
	def constraint_z_transitivity(self):
		if self.sublin_num > 3:
			self.constraint_z_trans_i_k_kp_kpp_both_z()
			self.constraint_z_trans_i_k_kp_kpp_z_k_kp()
			self.constraint_z_trans_i_k_kp_kpp_z_kp_kpp()
			self.constraint_z_trans_c_k_kp_kpp_both_z()
			self.constraint_z_trans_c_k_kp_kpp_z_kp_kpp()
			self.constraint_z_trans_c_k_kp_kpp_z_k_kpp()
			self.constraint_z_trans_i_z_k_kpp()
			self.constraint_z_trans_i_z_k_kp()

	# #11.31
	def constraint_z_trans_i_k_kp_kpp_both_z(self):
		# rownames
		new_constraints = (["constraint_z_trans_i_{0}_{1}_{2}_both_z".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([-1] * constraint_num)
		self.my_sense.extend(["G"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_i_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k][k_prime], self.my_colnames_z_index[k_prime][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.32
	def constraint_z_trans_i_k_kp_kpp_z_k_kp(self):
		# rownames
		new_constraints = (["constraint_z_trans_i_{0}_{1}_{2}_z_{0}_{1}".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_i_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k][k_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.33
	def constraint_z_trans_i_k_kp_kpp_z_kp_kpp(self):
		# rownames
		new_constraints = (["constraint_z_trans_i_{0}_{1}_{2}_z_{1}_{2}".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_i_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k_prime][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.34
	def constraint_z_trans_c_k_kp_kpp_both_z(self):
		# rownames
		new_constraints = (["constraint_z_trans_c_{0}_{1}_{2}_both_z".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([-1] * constraint_num)
		self.my_sense.extend(["G"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_c_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k_prime][k_prime_prime], self.my_colnames_z_index[k][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.35
	def constraint_z_trans_c_k_kp_kpp_z_kp_kpp(self):
		# rownames
		new_constraints = (["constraint_z_trans_c_{0}_{1}_{2}_z_{1}_{2}".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_c_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k_prime][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.36
	def constraint_z_trans_c_k_kp_kpp_z_k_kpp(self):
		# rownames
		new_constraints = (["constraint_z_trans_c_{0}_{1}_{2}_z_{0}_{2}".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["L"] * constraint_num)
		
		# my_rows
		rows_trans = [[self.z_trans_c_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		rows_z = [[self.my_colnames_z_index[k][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		self.my_rows.extend([[rows_trans[i] + rows_z[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.37
	def constraint_z_trans_i_z_k_kpp(self):
		# rownames
		new_constraints = (["constraint_z_trans_i_z_{0}_{1}_{2}".format(k, k_prime_prime, k_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["G"] * constraint_num)

		# my_rows
		rows_z = [[self.my_colnames_z_index[k][k_prime_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		rows_trans = [[self.z_trans_i_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		self.my_rows.extend([[rows_z[i] + rows_trans[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# #11.38
	def constraint_z_trans_i_z_k_kp(self):
		# rownames
		new_constraints = (["constraint_z_trans_c_z_{0}_{1}_{2}".format(k, k_prime, k_prime_prime)
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)])
		self.my_rownames.extend(new_constraints)
		constraint_num = len(new_constraints)

		self.my_rhs.extend([0] * constraint_num)
		self.my_sense.extend(["G"] * constraint_num)

		# my_rows
		rows_z = [[self.my_colnames_z_index[k][k_prime]]
			for k in xrange(1, self.sublin_num - 2) for k_prime in xrange(k+1, self.sublin_num - 1)
			for k_prime_prime in xrange(k_prime+1, self.sublin_num)]
		rows_trans = [[self.z_trans_c_index[k][k_prime][k_prime_prime]]
			for k in xrange(len(self.z_trans_i_index)) for k_prime in xrange(len(self.z_trans_i_index[k]))
			for k_prime_prime in xrange(len(self.z_trans_i_index[k][k_prime]))]
		self.my_rows.extend([[rows_z[i] + rows_trans[i], [1.0, -1.0]] for i in xrange(len(rows_trans))])

	# constraint #11.4: break symmetry in Z matrix construction
	# When after SSMs in lineage k no CNV appears in lineage k' > k and the SSMs are not phased 
	# because of this CNV, or if the transitivity rules are not applied, or if the lineage divergence rule is not
	# active, 
	# no conclusion can be drawn whether lineage k' is a descendant of k or not, so entry Zk,k' should be 0
	# z_k_k' - dssm_infl_cnv_a_p1_j_k_k' - ... - dssm_infl_cnv_b_p1_j_k_k' - ...
	#	- dssm_infl_cnv_a_m1_j_k_k' - ... - dssm_infl_cnv_b_m1_j_k_k' - ... - LDR_active_k_k' <= 0 
	#	for all j and 0 < k < k' < K
	def constraint_remove_symmetry_z_matrix(self):
		# if there are only 2 or less lineage, z values are all already determined
		if self.sublin_num > 2:
			entries = self.untrivial_z_entries_num
			self.my_rhs.extend([0.0] * entries)
			self.my_sense.extend(["L"] * entries)
			self.my_rownames.extend(["remove_symmetry_z_matrix_{0}_{1}".format(k, k_prime)
					for k in range(1, self.sublin_num) 
					for k_prime in range(k + 1, self.sublin_num)])
			rows_dssm = ([[[self.my_colnames_z_index[k][k_prime]]
				+ [my_list[k - 1][k_prime - k - 1] 
				for my_list in self.my_colnames_dssm_infl_cnv_a_p1_index]
				+ [my_list[k - 1][k_prime - k - 1] 
				for my_list in self.my_colnames_dssm_infl_cnv_b_p1_index]
				+ [my_list[k - 1][k_prime - k - 1] 
				for my_list in self.my_colnames_dssm_infl_cnv_a_m1_index]
				+ [my_list[k - 1][k_prime - k - 1] 
				for my_list in self.my_colnames_dssm_infl_cnv_b_m1_index],
				[1.0] + [-1] * self.ssm_num * cons.PHASE_NUMBER * self.cnv_state_num]
				for k in range(1, self.sublin_num - 1) 
				for k_prime in range(k + 1, self.sublin_num)])
			rows_c_vars = [self.z_trans_c_index[k-1][k_prime-k-1] 
				if (k < self.sublin_num - 2 and k_prime < self.sublin_num - 1)
				else [] 
				for k in xrange(1, self.sublin_num - 1) for k_prime in xrange(k+1, self.sublin_num)]
			rows_c_values= [[-1] * len(self.z_trans_c_index[k-1][k_prime-k-1]) 
				if (k < self.sublin_num - 2 and k_prime < self.sublin_num - 1)
				else [] 
				for k in xrange(1, self.sublin_num - 1) for k_prime in xrange(k+1, self.sublin_num)]
			rows_i = [self.help_with_rows_i(k, k_prime)
				if (k < self.sublin_num - 2 and k_prime < self.sublin_num)
				else [[], []]
				for k in xrange(1, self.sublin_num - 1) for k_prime in xrange(k+1, self.sublin_num)]
			if self.use_lineage_divergence_rule:
				rows_ldr_vars = [self.my_colnames_ldr_active_index_friendly_form[k][k_prime]
					for k in xrange(1, self.sublin_num - 1) for k_prime in xrange(k+1, self.sublin_num)]
				self.my_rows.extend([[rows_dssm[i][0] + rows_c_vars[i] + rows_i[i][0] + [rows_ldr_vars[i]],
					rows_dssm[i][1] + rows_c_values[i] + rows_i[i][1] + [-1.0]]
					for i in xrange(len(rows_dssm))])
			else:	
				self.my_rows.extend([[rows_dssm[i][0] + rows_c_vars[i] + rows_i[i][0],
					rows_dssm[i][1] + rows_c_values[i] + rows_i[i][1]]
					for i in xrange(len(rows_dssm))])

	def help_with_rows_i(self, k, k_prime):
		my_vars = [self.z_trans_i_index[k-1][k_star-k-1][k_prime-k_star-1] for k_star in xrange(k+1, k_prime)]
		values = [-1.0] * len(my_vars)
		return [my_vars, values]

	# constraints #11.5, 11.6, 11.7 to create auxiliary variables dc_descendant_{a,b}_STATES_i_k_k' 
	# to see whether a change in copy number happens in a lineage k' that is descendant
	# to lineage k
	# c #11.5: dc_descendant_a_p1_i_k_k' - dc_a_p1_binary_i_k' <= 0
	# c #11.6: dc_descendant_a_p1_i_k_k' - z_k_k' <= 0
	# c #11.7: dc_descendant_a_p1_i_k_k' - dc_a_p1_binary_i_k' - z_k_k- >= -1
	def constraint_dc_descendant(self):
		if self.sublin_num > 2:

			dc_descendant = [self.my_colnames_dc_descendant_a_p1_index, 
				self.my_colnames_dc_descendant_b_p1_index, self.my_colnames_dc_descendant_a_m1_index,
				self.my_colnames_dc_descendant_b_m1_index]
			entries = (self.untrivial_z_entries_num * self.seg_num * self.cnv_state_num
				* cons.PHASE_NUMBER)

			# c #11.5
			self.constraint_dc_relation_le_dc_binary(dc_descendant, cons.DESCENDANT, entries)
			# c #11.6
			self.constraint_dc_relation_le_z(dc_descendant, cons.DESCENDANT, entries)
			# c #11.7
			self.constraint_dc_relation_ge_dc_binary_z(dc_descendant, cons.DESCENDANT, entries)

	# constraints #11.8, 11.91, 11.92 to create auxiliary variables dc_ancestral_{a,b}_m1_i_k_k' 
	# to see whether a chromatid was lost in an ancestral lineage k of lineage k'
	# c #11.8: dc_ancestral_a_m1_i_k_k' - dc_a_m1_binary_i_k <= 0
	# c #11.91: dc_ancestral_a_m1_i_k_k' - z_k_k' <= 0
	# c #11.92: dc_ancestral_a_m1_i_k_k' - dc_a_m1_binary_i_k - z_k_k- >= -1
	def constraint_dc_ancestral(self):
		if self.sublin_num > 2:

			dc_ancestral = [[],[], self.my_colnames_dc_ancestral_a_m1_index,
				self.my_colnames_dc_ancestral_b_m1_index]
			entries = (self.untrivial_z_entries_num * self.seg_num * cons.PHASE_NUMBER)
			k_before_k_prime = False

			# c #11.8
			self.constraint_dc_relation_le_dc_binary(dc_ancestral, cons.ANCESTRAL, entries,
				k_before_k_prime)
			# c #11.91
			self.constraint_dc_relation_le_z(dc_ancestral, cons.ANCESTRAL, entries,
				k_before_k_prime)
			# c #11.92
			self.constraint_dc_relation_ge_dc_binary_z(dc_ancestral, cons.ANCESTRAL, entries,
				k_before_k_prime)

	# constraint 11.5, 11.8
	# function can be used for the creation of two constraints, constraints are nearly similar
	def constraint_dc_relation_le_dc_binary(self, dc_relation, relation, entries, k_before_k_prime=True):
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		
		if relation == cons.DESCENDANT:
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_le_dc_binary_a_p1", self.seg_num))
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_le_dc_binary_b_p1", self.seg_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_le_dc_binary_a_m1", self.seg_num, k_before_k_prime))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_le_dc_binary_b_m1", self.seg_num, k_before_k_prime))

		if relation == cons.DESCENDANT: 
			self.my_rows.extend(self.create_rows_constraint_dc_relation_le_dc_binary(
				dc_relation[cons.A_P1], self.my_colnames_dc_a_p1_binary_index, relation))
			self.my_rows.extend(self.create_rows_constraint_dc_relation_le_dc_binary(
				dc_relation[cons.B_P1], self.my_colnames_dc_b_p1_binary_index, relation))
		self.my_rows.extend(self.create_rows_constraint_dc_relation_le_dc_binary(
			dc_relation[cons.A_M1], self.my_colnames_dc_a_m1_binary_index, relation))
		self.my_rows.extend(self.create_rows_constraint_dc_relation_le_dc_binary(
			dc_relation[cons.B_M1], self.my_colnames_dc_b_m1_binary_index, relation))

	def create_rows_constraint_dc_relation_le_dc_binary(self, dc_relation_var, dc_binary_index, relation):
		if relation == cons.DESCENDANT:
			return [[[dc_relation_var[i][k - 1][k_prime - k - 1], dc_binary_index[i][k_prime]], 
				[1.0, -1.0]]
				for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num - 1)
				for k_prime in xrange(k + 1, self.sublin_num)]
		elif relation == cons.ANCESTRAL:
			return [[[dc_relation_var[i][k_prime - 2][k - 1], dc_binary_index[i][k]], [1.0, -1.0]]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)
				for k in xrange(1, k_prime)]

	# constraint 11.6, 11.91
	# function can be used for the creation of two constraints, constraints are nearly similar
	def constraint_dc_relation_le_z(self, dc_relation, relation, entries, k_before_k_prime=True):
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)

		if relation == cons.DESCENDANT:
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_le_z_a_p1", self.seg_num))
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_le_z_b_p1", self.seg_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_le_z_a_m1", self.seg_num, k_before_k_prime))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_le_z_b_m1", self.seg_num, k_before_k_prime))

		if relation == cons.DESCENDANT:  
			self.my_rows.extend(self.create_rows_constraints_dc_relation_le_z(
				dc_relation[cons.A_P1], relation))
			self.my_rows.extend(self.create_rows_constraints_dc_relation_le_z(
				dc_relation[cons.B_P1], relation))
		self.my_rows.extend(self.create_rows_constraints_dc_relation_le_z(
			dc_relation[cons.A_M1], relation))
		self.my_rows.extend(self.create_rows_constraints_dc_relation_le_z(
			dc_relation[cons.B_M1], relation))

	def create_rows_constraints_dc_relation_le_z(self, dc_relation_var, relation):
		z_matrix = self.my_colnames_z_index
		if relation == cons.DESCENDANT:
			return [[[dc_relation_var[i][k - 1][k_prime - k - 1], z_matrix[k][k_prime]], [1.0, -1.0]]
				for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num - 1)
				for k_prime in xrange(k + 1, self.sublin_num)]
		elif relation == cons.ANCESTRAL: 
			return [[[dc_relation_var[i][k_prime - 2][k - 1], z_matrix[k][k_prime]], [1.0, -1.0]]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)
				for k in xrange(1, k_prime)]

	# constraint 11.7, 11.92
	# function can be used for the creation of two constraints, constraints are nearly similar
	def constraint_dc_relation_ge_dc_binary_z(self, dc_relation, relation, entries, k_before_k_prime=True):
		self.my_rhs.extend([-1.0] * entries)
		self.my_sense.extend(["G"] * entries)

		if relation == cons.DESCENDANT: 
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_ge_dc_binary_z_a_p1", self.seg_num))
			self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
				"dc_descendant_ge_dc_binary_z_b_p1", self.seg_num))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_ge_dc_binary_z_a_m1", self.seg_num, k_before_k_prime))
		self.my_rownames.extend(self.create_rownames_constraint_mut_untrivial_z_entries(
			"dc_" + relation + "_ge_dc_binary_z_b_m1", self.seg_num, k_before_k_prime))

		if relation == cons.DESCENDANT:
			self.my_rows.extend(self.create_rows_constraint_dc_relation_ge_dc_binary_z(
				dc_relation[cons.A_P1], self.my_colnames_dc_a_p1_binary_index, relation))
			self.my_rows.extend(self.create_rows_constraint_dc_relation_ge_dc_binary_z(
				dc_relation[cons.B_P1], self.my_colnames_dc_b_p1_binary_index, relation))
		self.my_rows.extend(self.create_rows_constraint_dc_relation_ge_dc_binary_z(
			dc_relation[cons.A_M1], self.my_colnames_dc_a_m1_binary_index, relation))
		self.my_rows.extend(self.create_rows_constraint_dc_relation_ge_dc_binary_z(
			dc_relation[cons.B_M1], self.my_colnames_dc_b_m1_binary_index, relation))

	def create_rows_constraint_dc_relation_ge_dc_binary_z(self, dc_relation_var, dc_binary_index, relation):
		z_matrix = self.my_colnames_z_index
		if relation == cons.DESCENDANT: 
			return [[[dc_relation_var[i][k - 1][k_prime - k -1], 
				dc_binary_index[i][k_prime], z_matrix[k][k_prime]], [1.0, -1.0, -1.0]]
				for i in xrange(self.seg_num) for k in xrange(1, self.sublin_num - 1)
				for k_prime in xrange(k + 1, self.sublin_num)]
		elif relation == cons.ANCESTRAL:
			return [[[dc_relation_var[i][k_prime - 2][k - 1], 
				dc_binary_index[i][k], z_matrix[k][k_prime]], [1.0, -1.0, -1.0]]
				for i in xrange(self.seg_num) for k_prime in xrange(2, self.sublin_num)
				for k in xrange(1, k_prime)]
			

	# constraint #12: only one entry for SNP row in three matrices is 1
	# because snp appears accoring to infinite sites assumption only ones
	def constraint_snp_row_one_entry(self):
		self.my_rhs.extend([1.0] * self.snp_num)
		self.my_sense.extend(["E"] * self.snp_num)
		self.my_rownames.extend(self.create_rownames_one_index("snp_row", self.snp_num))
		self.my_rows.extend(
			[[[self.my_colnames_dsnp_index[i], self.my_colnames_dsnp_a_index[i], 
			self.my_colnames_dsnp_b_index[i]], 
			[1.0, 1.0, 1.0]] for i in xrange(self.snp_num)])
	
	
	# constraint #12.1: break symmetry of SNP assignment
	# when no CN change in segment appears, SNP should be unphased
	# dsnp_a_l - dc_a_p1_binary_i_k - ... - dc_b_p1_binary_i_k - ... 
	# 	- dc_a_m1_binary_i_k  - ... - dc_b_m1_binary_i_k - ... <= 0, same for dsnp_b_l
	# constraint #12.2: break symmetry of SNP assignment 
	# when change appears on one chromatid, uneffected SNPs should be phased to 
	#	other chromatid
	# dsnp_l + dc_a_p1_binary_i_k + ... + dc_b_p1_binary_i_k + ... <= 1, same for state m1
	def constraint_remove_snp_symmetry(self, snp_list):
		# constraint #12.1
		self.constraint_remove_snp_symmetry_no_CN_change_unphased(snp_list)
		# constraint #12.2
		self.constraint_remove_snp_symmetry_CN_uneffected_not_unphased(snp_list)

	def constraint_remove_snp_symmetry_no_CN_change_unphased(self, snp_list):
		entries = self.snp_num * cons.PHASE_NUMBER
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(self.create_rownames_one_index(
			"constraint_remove_snp_symmetry_no_CN_change_unphased_a", self.snp_num))
		self.my_rownames.extend(self.create_rownames_one_index(
			"constraint_remove_snp_symmetry_no_CN_change_unphased_b", self.snp_num))
		self.my_rows.extend([[[self.my_colnames_dsnp_a_index[i]] 
			+ self.my_colnames_dc_a_p1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_p1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_a_m1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_m1_binary_index[snp_list[i].seg_index].tolist(),
			[1.0] + [-1] * cons.PHASE_NUMBER * self.cnv_state_num * self.sublin_num]
			for i in xrange(self.snp_num)])
		self.my_rows.extend([[[self.my_colnames_dsnp_b_index[i]] 
			+ self.my_colnames_dc_a_p1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_p1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_a_m1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_m1_binary_index[snp_list[i].seg_index].tolist(),
			[1.0] + [-1] * cons.PHASE_NUMBER * self.cnv_state_num * self.sublin_num]
			for i in xrange(self.snp_num)])

	def constraint_remove_snp_symmetry_CN_uneffected_not_unphased(self, snp_list):
		entries = self.snp_num * self.cnv_state_num
		self.my_rhs.extend([1.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(self.create_rownames_one_index(
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_p1", self.snp_num))
		self.my_rownames.extend(self.create_rownames_one_index(
			"constraint_remove_snp_symmetry_CN_uneffected_not_unphased_m1", self.snp_num))
		self.my_rows.extend([[[self.my_colnames_dsnp_index[i]]
			+ self.my_colnames_dc_a_p1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_p1_binary_index[snp_list[i].seg_index].tolist(),
			[1.0] + [1.0] * self.sublin_num * cons.PHASE_NUMBER]
			for i in xrange(self.snp_num)])
		self.my_rows.extend([[[self.my_colnames_dsnp_index[i]]
			+ self.my_colnames_dc_a_m1_binary_index[snp_list[i].seg_index].tolist()
			+ self.my_colnames_dc_b_m1_binary_index[snp_list[i].seg_index].tolist(),
			[1.0] + [1.0] * self.sublin_num * cons.PHASE_NUMBER]
			for i in xrange(self.snp_num)])

	# constraint #13, #14, #15, 16: create auxiliary variable snp_w_cnv_a/b_p1 to model
	# snp with cnv in linear way
	# #13: snp_w_cnv_a_p1_l_k' - phi_k' <= 0, same for b, same for m1
	# #14: snp_w_cnv_a_p1_l_k' - dsnp_a_l <= 0, same for b, same for m1
	# #15: snp_w_cnv_a_p1_l_k - dc_a_p1_binary_i_k <= 0, same for b, similar for m1
	# #16: snp_w_cnv_a_p1_l_k' - phi_k' - dsnp_a_l - dc_a_p1_binary_i_k' >= -2, same for b, similar for m1
	def constraint_aux_snp_w_cnv(self, snp_list):
		# constraint #13: Constraint to create auxiliary variable snp_w_cnv_a/b_p1
		self.constraint_aux_snp_w_cnv_le_phi()
		# constraint #14: Constraint to create auxiliary variable snp_w_cnv_a/b_p1
		self.constraint_aux_snp_w_cnv_le_dsnp()
		# constraint #15: Constraint to create auxiliary variable snp_w_cnv_a/b_p1
		self.constraint_aux_snp_w_cnv_le_dc_binary(snp_list)
		# constraint #16: Constraint to create auxiliary variable snp_w_cnv_a/b_p1
		self.constraint_aux_snp_w_cnv_combined(snp_list)

	def create_rowname_constraint_aux_snp_w_cnv(self, name):
		return ["{0}_{1}_{2}".format(name, i, j + 1) for i in xrange(self.snp_num)
			for j in range(self.sublin_num - 1)]
	
	# #13: snp_w_cnv_a_p1_l_k' - phi_k' <= 0, same for b
	# 	snp_w_cnv_a_m1_l_k' - phi_k' <= 0, same for b
	def constraint_aux_snp_w_cnv_le_phi(self):
		self.my_rhs.extend([0.0] * self.snp_aux_linear_variables_num)
		self.my_sense.extend(["L"] * self.snp_aux_linear_variables_num)
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_p1_phi"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_p1_phi"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_m1_phi"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_m1_phi"))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_phi(self.my_colnames_snp_w_cnv_a_p1))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_phi(self.my_colnames_snp_w_cnv_b_p1))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_phi(self.my_colnames_snp_w_cnv_a_m1))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_phi(self.my_colnames_snp_w_cnv_b_m1))

	def create_rows_constraint_aux_snp_w_cnv_le_phi(self, colnames_snp_w_cnv_phased):
		return [[[colnames_snp_w_cnv_phased[i][j], self.my_phis[j + 1]], [1.0, -1.0]] for i in xrange(self.snp_num)
			for j in range(self.sublin_num - 1)]	
	
	# #14: snp_w_cnv_a_p1_l_k' - dsnp_a_l <= 0, same for b
	# 	snp_w_cnv_a_m1_l_k' - dsnp_a_l <= 0, same for b 
	def constraint_aux_snp_w_cnv_le_dsnp(self):
		self.my_rhs.extend([0.0] * self.snp_aux_linear_variables_num)
		self.my_sense.extend(["L"] * self.snp_aux_linear_variables_num)
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_p1_dsnp_a"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_p1_dsnp_b"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_m1_dsnp_a"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_m1_dsnp_b"))
		self.my_rows.extend(
			self.create_rows_constraint_aux_snp_w_cnv_le_dsnp(self.my_colnames_snp_w_cnv_a_p1, self.my_colnames_dsnp_a))
		self.my_rows.extend(
			self.create_rows_constraint_aux_snp_w_cnv_le_dsnp(self.my_colnames_snp_w_cnv_b_p1, self.my_colnames_dsnp_b))
		self.my_rows.extend(
			self.create_rows_constraint_aux_snp_w_cnv_le_dsnp(self.my_colnames_snp_w_cnv_a_m1, self.my_colnames_dsnp_a))
		self.my_rows.extend(
			self.create_rows_constraint_aux_snp_w_cnv_le_dsnp(self.my_colnames_snp_w_cnv_b_m1, self.my_colnames_dsnp_b))

	def create_rows_constraint_aux_snp_w_cnv_le_dsnp(self, colnames_snp_w_cnv_phased, dsnp_phased):
		return [[[colnames_snp_w_cnv_phased[i][j], dsnp_phased[i]], [1.0, -1.0]] for i in xrange(self.snp_num)
			for j in range(self.sublin_num - 1)]

	# #15: snp_w_cnv_a_p1_l_k - dc_a_p1_binary_i_k <= 0, same for b
	#	snp_w_cnv_a_m1_l_k - dc_a_m1_binary_i_k <= 0, same for b
	def constraint_aux_snp_w_cnv_le_dc_binary(self, snp_list):
		self.my_rhs.extend([0.0] * self.snp_aux_linear_variables_num)
		self.my_sense.extend(["L"] * self.snp_aux_linear_variables_num)
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_p1_dc_a"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_p1_dc_b"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_a_m1_dc_a"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_w_cnv_b_m1_dc_b"))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_dc_binary(self.my_colnames_snp_w_cnv_a_p1,
			self.my_colnames_dc_a_p1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_dc_binary(self.my_colnames_snp_w_cnv_b_p1,
			self.my_colnames_dc_b_p1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_dc_binary(self.my_colnames_snp_w_cnv_a_m1,
			self.my_colnames_dc_a_m1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_le_dc_binary(self.my_colnames_snp_w_cnv_b_m1,
			self.my_colnames_dc_b_m1_binary, snp_list))

	def create_rows_constraint_aux_snp_w_cnv_le_dc_binary(self, colnames_snp_w_cnv_phased, dc_binary_phased, snp_list):
		return [[[colnames_snp_w_cnv_phased[i][j], dc_binary_phased[snp_list[i].seg_index][j + 1]], [1.0, -1.0]]
			for i in xrange(self.snp_num) for j in range(self.sublin_num - 1)]
	
	# #16: snp_w_cnv_a_p1_l_k' - phi_k' - dsnp_a_l - dc_a_p1_binary_i_k' >= -2, same for b
	#	snp_w_cnv_a_m1_l_k' - phi_k' - dsnp_a_l - dc_a_m1_binary_i_k' >= -2, same for b 
	def constraint_aux_snp_w_cnv_combined(self, snp_list):
		self.my_rhs.extend([-2.0] * self.snp_aux_linear_variables_num)
		self.my_sense.extend(["G"] * self.snp_aux_linear_variables_num)
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_cnv_a_p1_together"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_cnv_b_p1_together"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_cnv_a_m1_together"))
		self.my_rownames.extend(self.create_rowname_constraint_aux_snp_w_cnv("snv_cnv_b_m1_together"))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_combined(self.my_colnames_snp_w_cnv_a_p1,
			self.my_colnames_dsnp_a, self.my_colnames_dc_a_p1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_combined(self.my_colnames_snp_w_cnv_b_p1,
			self.my_colnames_dsnp_b, self.my_colnames_dc_b_p1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_combined(self.my_colnames_snp_w_cnv_a_m1,
			self.my_colnames_dsnp_a, self.my_colnames_dc_a_m1_binary, snp_list))
		self.my_rows.extend(self.create_rows_constraint_aux_snp_w_cnv_combined(self.my_colnames_snp_w_cnv_b_m1,
			self.my_colnames_dsnp_b, self.my_colnames_dc_b_m1_binary, snp_list))

	def create_rows_constraint_aux_snp_w_cnv_combined(self, colnames_snp_w_cnv_phased, dsnp_phased, dc_binary_phased, snp_list):
		return [[[colnames_snp_w_cnv_phased[i][j], self.my_phis[j+1], dsnp_phased[i], 
			dc_binary_phased[snp_list[i].seg_index][j+1]],
			[1.0, -1.0, -1.0, -1.0]] for i in xrange(self.snp_num) for j in range(self.sublin_num - 1)]
	
	# constraint #18: Set CN of SNP and CN of segment equal to observed frequency of SNP
	def constraint_snp_value_spline_frequency(self, snp_list):
		self.my_rhs.extend([1.0] * self.snp_num)
		self.my_sense.extend(["E"] * self.snp_num)
		self.my_rownames.extend(self.create_rownames_one_index("snp_frequency", self.snp_num))
		snp_w_cnv_same_direction = (self.sublin_num - 1) * cons.PHASE_NUMBER * (self.cnv_state_num / 2)
		self.my_rows.extend([[self.my_colnames_snp[i] 
			+ self.my_colnames_snp_w_cnv_a_p1[i]
			+ self.my_colnames_snp_w_cnv_b_p1[i]
			+ self.my_colnames_snp_w_cnv_a_m1[i] 
			+ self.my_colnames_snp_w_cnv_b_m1[i], 
			[self.snp_splines[i].get_knots()[j] * self.cn[snp_list[i].seg_index] 
			for j in xrange(len(self.snp_splines[i].get_knots()))] 
			+ [-1.0] * snp_w_cnv_same_direction
			+ [1.0] * snp_w_cnv_same_direction] 
			for i in xrange(self.snp_num)])

	# constraint #19: ssm cannot appear in normal population (lineage 0)
	# dssm_j_0 = 0
	# no unphased if CN changes are not simple
	def constraint_no_ssm_normal_lineage(self):
		entries = 0
		if self.simple_CN_changes:
			entries = cons.SNP_SSM_PHASING_MATRIX_NUM * self.ssm_num
		else:
			entries = cons.PHASE_NUMBER * self.ssm_num
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["E"] * entries)
		if self.simple_CN_changes:
			self.my_rownames.extend(self.create_rownames_one_index("no_ssm_normal_lin", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_one_index("no_ssm_a_normal_lin", self.ssm_num))
		self.my_rownames.extend(self.create_rownames_one_index("no_ssm_b_normal_lin", self.ssm_num))
		if self.simple_CN_changes:
			self.my_rows.extend(self.create_rows_constraint_no_ssm_normal_lineage(
				self.my_colnames_dssm_index))
		self.my_rows.extend(self.create_rows_constraint_no_ssm_normal_lineage(self.my_colnames_dssm_a_index))
		self.my_rows.extend(self.create_rows_constraint_no_ssm_normal_lineage(self.my_colnames_dssm_b_index))

	def create_rows_constraint_no_ssm_normal_lineage(self, dssm_phase):
		return [[[dssm_phase[i][0]], [1.0]] for i in xrange(self.ssm_num)]
	
	# constraint #20: infinite site assumption for SSMs, only one entry per row 
	# for all three lines in matrices
	# dssm_j_0 + dssm_j_1 ... + d_ssm_a_j_0 ... + d_ssm_b_j_0 ... = 1
	def constraint_ssm_isa(self):
		self.my_rhs.extend([1.0] * self.ssm_num)
		self.my_sense.extend(["E"] * self.ssm_num)
		self.my_rownames.extend(self.create_rownames_one_index("ssm_isa", self.ssm_num))
		if self.simple_CN_changes:
			self.my_rows.extend([[self.my_colnames_dssm_index[i].tolist() 
				+ self.my_colnames_dssm_a_index[i].tolist() 
				+ self.my_colnames_dssm_b_index[i].tolist(),
				[1.0] * self.sublin_num * cons.SNP_SSM_PHASING_MATRIX_NUM] for i in xrange(self.ssm_num)])
		else:
			self.my_rows.extend([[self.my_colnames_dssm_a_index[i].tolist() 
				+ self.my_colnames_dssm_b_index[i].tolist(),
				[1.0] * self.sublin_num * cons.PHASE_NUMBER] for i in xrange(self.ssm_num)])
	
	# constraint #20.1: break symmetry of SSM assignment
	# when no CN change in segment after lineage of SSM appears, or no CN loss before lineage appears,
	# or no CN loss in actual lineage appears, the SSM should be unphased
	# when more compicated CN changes are allowed to be modeled, the unphased state does not exist
	#	so every SSM that can't be phased is assigned to A
	#	then only dssm_b_j_k is set to be 0, no dssm_a_j_k in following constraints
	# dssm_a_j_k + dssm_b_j_k - dc_descendant_a_p1_i_k_k' - ... - dc_descendant_b_p1_i_k_k' - ... 
	# 	- dc_descendant_a_m1_i_k_k'  - ... - dc_descendant_b_m1_i_k_k' - ... 
	#	- dc_ancestral_a_m1_i_k_k^ - ... - dc_ancestral_b_m1_i_k_k^ - ...
	#	- dc_a_m1_binary_i_k - dc_b_m1_binary_i_k <= 0, for 1 <= k <= K - 1, k'> k, k^ < k
	#
	# constraint #20.2: break symmetry of SSM assignment 
	# When change appears on one chromatid in a lineage k' descendent to lineage k of SSM, 
	# all SSMs in lineage k have to be phased
	# When more complicated CN changes are allowed to be modeled, all SSMs are phased bei default
	# because there is no unphased state, so the following constraint is not needed
	# dssm_j_k + dc_descendant_a_p1_i_k' + ... + dc_descendant_b_p1_i_k' + ... <= 1, k' > k, 1 <= k < K - 1, 
	#	same for state m1
	#
	# constraint #20.3: break symmetry of SSM assignment
	# When a lineage k' is descendant to an ancestral lineage k where one chromatid is lost, 
	# or if one chromatid is lost in k' itself, all SSMs in k' have to be phased to the other chromatid
	# When more complicated CN changes are allowed to be modeled, the unphased state does not exist,
	#	so it doesn't appear in constraints
	# dssm_j_k' + dssm_a_j_k' + dc_ancestral_a_m1_i_k'_k + ... + dc_a_m1_binary_i_k' <= 1, 
	#	1 <= k <= k' <= K - 1, same for chromatid B
	def constraint_remove_ssm_symmetry(self, ssm_list):
		# constraint #20.1
		self.constraint_remove_ssm_symmetry_no_CN_change_unphased(ssm_list)
		if self.sublin_num > 2 and self.simple_CN_changes:
			# constraint #20.2
			self.constraint_remove_ssm_symmetry_uneffected_not_unphased(ssm_list)
		# constraint #20.3
		self.constraint_remove_ssm_symmetry_chromatid_loss(ssm_list)

	# constraint #20.1
	def constraint_remove_ssm_symmetry_no_CN_change_unphased(self, ssm_list):
		entries = self.ssm_num * (self.sublin_num - 1)
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["constraint_remove_ssm_symmetry_no_CN_change_unphased_{0}_{1}".format(
			j, k) for k in range(1, self.sublin_num) for j in xrange(self.ssm_num)])

		second_lin = 1
		last_lin = self.sublin_num - 1

		get_seg_index = self.get_seg_index
		# for k = 1, second lineage, the first in which mutations can occur
		# only, when second lineage is not the last one (that's the case when there are only
		# 2 lineages)
		if second_lin != last_lin:
			if self.simple_CN_changes:
				self.my_rows.extend([[[self.my_colnames_dssm_a_index[j][second_lin]] 
					+ [self.my_colnames_dssm_b_index[j][second_lin]]
					+ self.my_colnames_dc_descendant_a_p1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_b_p1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_a_m1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_b_m1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][second_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][second_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][second_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][second_lin-1]],
					[1.0, 1.0] + [-1.0] * ((self.sublin_num - second_lin - 1) * self.cnv_state_num 
					* cons.PHASE_NUMBER + 4)]
					for j in xrange(self.ssm_num)])
			else:
				self.my_rows.extend([[ 
					[self.my_colnames_dssm_b_index[j][second_lin]]
					+ self.my_colnames_dc_descendant_a_p1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_b_p1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_a_m1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ self.my_colnames_dc_descendant_b_m1_index[
					get_seg_index(ssm_list,j)][second_lin - 1]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][second_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][second_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][second_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][second_lin-1]],
					[1.0] + [-1.0] * ((self.sublin_num - second_lin - 1) * self.cnv_state_num 
					* cons.PHASE_NUMBER + 4)]
					for j in xrange(self.ssm_num)])
		# for 1 < k < K - 1, lineages between second and last lineage
		if self.simple_CN_changes:
			self.my_rows.extend([[[self.my_colnames_dssm_a_index[j][k]] 
				+ [self.my_colnames_dssm_b_index[j][k]]
				+ self.my_colnames_dc_descendant_a_p1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_b_p1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_a_m1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_b_m1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_ancestral_a_m1_index[get_seg_index(ssm_list,j)][k - 2]
				+ self.my_colnames_dc_ancestral_b_m1_index[get_seg_index(ssm_list,j)][k - 2]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][k]]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][k]]
				+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][k-1]]
				+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][k-1]],
				[1.0, 1.0] + [-1.0] * ((self.sublin_num - k - 1) * self.cnv_state_num 
				* cons.PHASE_NUMBER + (k - 1) * cons.PHASE_NUMBER + 4)]
				for k in xrange(2, self.sublin_num - 1) for j in xrange(self.ssm_num)])
		else:
			self.my_rows.extend([[[self.my_colnames_dssm_b_index[j][k]]
				+ self.my_colnames_dc_descendant_a_p1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_b_p1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_a_m1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_descendant_b_m1_index[get_seg_index(ssm_list,j)][k - 1]
				+ self.my_colnames_dc_ancestral_a_m1_index[get_seg_index(ssm_list,j)][k - 2]
				+ self.my_colnames_dc_ancestral_b_m1_index[get_seg_index(ssm_list,j)][k - 2]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][k]]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][k]]
				+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][k-1]]
				+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][k-1]],
				[1.0] + [-1.0] * ((self.sublin_num - k - 1) * self.cnv_state_num 
				* cons.PHASE_NUMBER + (k - 1) * cons.PHASE_NUMBER + 4)]
				for k in xrange(2, self.sublin_num - 1) for j in xrange(self.ssm_num)])
		# for k = K - 1, last lineage
		# only when there are more than 2 lineages
		if self.sublin_num > 2:
			if self.simple_CN_changes:
				self.my_rows.extend([[[self.my_colnames_dssm_a_index[j][last_lin]] 
					+ [self.my_colnames_dssm_b_index[j][last_lin]]
					+ self.my_colnames_dc_ancestral_a_m1_index[
					get_seg_index(ssm_list,j)][last_lin - 2]
					+ self.my_colnames_dc_ancestral_b_m1_index[
					get_seg_index(ssm_list,j)][last_lin - 2]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][last_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][last_lin-1]],
					[1.0, 1.0] + [-1.0] * ((last_lin - 1) * cons.PHASE_NUMBER + 4)]
					for j in xrange(self.ssm_num)])
			else:
				self.my_rows.extend([[ 
					[self.my_colnames_dssm_b_index[j][last_lin]]
					+ self.my_colnames_dc_ancestral_a_m1_index[
					get_seg_index(ssm_list,j)][last_lin - 2]
					+ self.my_colnames_dc_ancestral_b_m1_index[
					get_seg_index(ssm_list,j)][last_lin - 2]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][last_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][last_lin-1]],
					[1.0] + [-1.0] * ((last_lin - 1) * cons.PHASE_NUMBER + 4)]
					for j in xrange(self.ssm_num)])
		# only 2 lineages
		elif self.sublin_num == 2:
			if self.simple_CN_changes:
				self.my_rows.extend([[[self.my_colnames_dssm_a_index[j][last_lin]] 
					+ [self.my_colnames_dssm_b_index[j][last_lin]]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][last_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][last_lin-1]],
					[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]]
					for j in xrange(self.ssm_num)])
			else:
				self.my_rows.extend([[ 
					[self.my_colnames_dssm_b_index[j][last_lin]]
					+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][last_lin]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_a_index[j][last_lin-1]]
					+ [self.my_colnames_dssm_infl_cnv_same_lineage_b_index[j][last_lin-1]],
					[1.0, -1.0, -1.0, -1.0, -1.0]]
					for j in xrange(self.ssm_num)])

	# constraint #20.2
	def constraint_remove_ssm_symmetry_uneffected_not_unphased(self, ssm_list):
		entries = self.ssm_num * (self.sublin_num - 2) * self.cnv_state_num
		self.my_rhs.extend([1.0] * entries)
		self.my_sense.extend(["L"] * entries)
		get_seg_index = self.get_seg_index
		self.my_rownames.extend(
			["constraint_remove_ssm_symmetry_uneffected_not_unphased_p1_{0}_{1}".format(j,k)
			for j in xrange(self.ssm_num) for k in range(1, self.sublin_num - 1)])
		self.my_rownames.extend([
			"constraint_remove_ssm_symmetry_uneffected_not_unphased_m1_{0}_{1}".format(j,k)
			for j in xrange(self.ssm_num) for k in range(1, self.sublin_num - 1)])
		self.my_rows.extend([[[self.my_colnames_dssm_index[j][k]] 
			+ self.my_colnames_dc_descendant_a_p1_index[get_seg_index(ssm_list,j)][k - 1]
			+ self.my_colnames_dc_descendant_b_p1_index[get_seg_index(ssm_list,j)][k - 1],
			[1.0] + [1.0] * (self.sublin_num - k - 1) * cons.PHASE_NUMBER]
			for j in xrange(self.ssm_num) for k in range(1, self.sublin_num - 1)])
		self.my_rows.extend([[[self.my_colnames_dssm_index[j][k]] 
			+ self.my_colnames_dc_descendant_a_m1_index[get_seg_index(ssm_list,j)][k - 1]
			+ self.my_colnames_dc_descendant_b_m1_index[get_seg_index(ssm_list,j)][k - 1],
			[1.0] + [1.0] * (self.sublin_num - k - 1) * cons.PHASE_NUMBER]
			for j in xrange(self.ssm_num) for k in range(1, self.sublin_num - 1)])

	#constraint #20.3
	def constraint_remove_ssm_symmetry_chromatid_loss(self, ssm_list):      
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_rhs.extend([1.0] * entries)
		self.my_sense.extend(["L"] * entries)

		self.my_rownames.extend(
			["constraint_remove_ssm_symmetry_chromatid_loss_a_{0}_{1}".format(j, k_prime)
			for k_prime in range(1, self.sublin_num) for j in xrange(self.ssm_num)])
		self.my_rownames.extend(
			["constraint_remove_ssm_symmetry_chromatid_loss_b_{0}_{1}".format(j, k_prime)
			for k_prime in range(1, self.sublin_num) for j in xrange(self.ssm_num)])

		get_seg_index = self.get_seg_index

		if self.simple_CN_changes:
			self.my_rows.extend([[[self.my_colnames_dssm_index[j][1]] 
				+ [self.my_colnames_dssm_a_index[j][1]]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][1]],
				[1.0, 1.0, 1.0]] for j in xrange(self.ssm_num)])
			self.my_rows.extend([[[self.my_colnames_dssm_index[j][k_prime]] 
				+ [self.my_colnames_dssm_a_index[j][k_prime]] 
				+ self.my_colnames_dc_ancestral_a_m1_index[get_seg_index(ssm_list,j)][k_prime - 2]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][k_prime]],
				[1.0] * (3 + (k_prime - 1))]
				for k_prime in xrange(2, self.sublin_num) for j in xrange(self.ssm_num)])
			self.my_rows.extend([[[self.my_colnames_dssm_index[j][1]] 
				+ [self.my_colnames_dssm_b_index[j][1]]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][1]],
				[1.0, 1.0, 1.0]] for j in xrange(self.ssm_num)])
			self.my_rows.extend([[[self.my_colnames_dssm_index[j][k_prime]] 
				+ [self.my_colnames_dssm_b_index[j][k_prime]] 
				+ self.my_colnames_dc_ancestral_b_m1_index[get_seg_index(ssm_list,j)][k_prime - 2]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][k_prime]],
				[1.0] * (3 + (k_prime - 1))]
				for k_prime in xrange(2, self.sublin_num) for j in xrange(self.ssm_num)])
		else:
			self.my_rows.extend([[[self.my_colnames_dssm_a_index[j][1]]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][1]],
				[1.0, 1.0]] for j in xrange(self.ssm_num)])
			self.my_rows.extend([[
				[self.my_colnames_dssm_a_index[j][k_prime]] 
				+ self.my_colnames_dc_ancestral_a_m1_index[get_seg_index(ssm_list,j)][k_prime - 2]
				+ [self.my_colnames_dc_a_m1_binary_index[get_seg_index(ssm_list,j)][k_prime]],
				[1.0] * (2 + (k_prime - 1))]
				for k_prime in xrange(2, self.sublin_num) for j in xrange(self.ssm_num)])
			self.my_rows.extend([[[self.my_colnames_dssm_b_index[j][1]]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][1]],
				[1.0, 1.0]] for j in xrange(self.ssm_num)])
			self.my_rows.extend([[
				[self.my_colnames_dssm_b_index[j][k_prime]] 
				+ self.my_colnames_dc_ancestral_b_m1_index[get_seg_index(ssm_list,j)][k_prime - 2]
				+ [self.my_colnames_dc_b_m1_binary_index[get_seg_index(ssm_list,j)][k_prime]],
				[1.0] * (2 + (k_prime - 1))]
				for k_prime in xrange(2, self.sublin_num) for j in xrange(self.ssm_num)])

	# constraints #20.4 and #20.5
	# for dssm_infl_cnv_same_lineage
	# Variable is 1 when SSM is influenced by CN change in same lineage; variable is 0 when SSM in not 
	# influenced, or no CN change exists, or SSM is not assigned to lineage
	# c #20.4: dssm_infl_cnv_same_lineage_a_p1_j_k - dc_a_p1_binary_i_k <= 0, for k > 1, also for b
	# c #20.5: dssm_infl_cnv_same_lineage_a_p1_j_k - dssm_a_j_k <= 0, for k > 1, also for b
	def constraint_dssm_infl_cnv_same_lineage(self, ssm_list):
		#20.4
		self.constraint_dssm_infl_cnv_same_lineage_cn_change(ssm_list)
		#20.5
		self.constraint_dssm_infl_cnv_same_lineage_ssm()

	# c #20.4
	def constraint_dssm_infl_cnv_same_lineage_cn_change(self, ssm_list):
		entries = (self.sublin_num - 1) * self.ssm_num * cons.PHASE_NUMBER
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["constraint_dssm_infl_cnv_same_lineage_cn_change_{0}_p1_{1}_{2}".format(
			phase, j, k) for phase in ["a", "b"] for j in xrange(self.ssm_num)
			for k in range(1, self.sublin_num)])
		dssm_infl_cnv_same_lineage_phase_index = [self.my_colnames_dssm_infl_cnv_same_lineage_a_index,
			self.my_colnames_dssm_infl_cnv_same_lineage_b_index]
		dc_a_p1_binary_index = [self.my_colnames_dc_a_p1_binary_index,
			self.my_colnames_dc_b_p1_binary_index]
		self.my_rows.extend([[[dssm_infl_cnv_same_lineage_phase_index[p][j][k],
			dc_a_p1_binary_index[p][self.get_seg_index(ssm_list, j)][k+1]], [1.0, -1.0]] 
			for p in range(cons.PHASE_NUMBER)
			for j in xrange(self.ssm_num)
			for k in range(self.sublin_num-1)])

	# c #20.5
	def constraint_dssm_infl_cnv_same_lineage_ssm(self):
		entries = (self.sublin_num - 1) * self.ssm_num * cons.PHASE_NUMBER
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["constraint_dssm_infl_cnv_same_lineage_ssm_{0}_p1_{1}_{2}".format(
			phase, j, k) for phase in ["a", "b"] for j in xrange(self.ssm_num)
			for k in range(1, self.sublin_num)])
		dssm_infl_cnv_same_lineage_phase_index = [self.my_colnames_dssm_infl_cnv_same_lineage_a_index,
			self.my_colnames_dssm_infl_cnv_same_lineage_b_index]
		dssm_index = [self.my_colnames_dssm_a_index, self.my_colnames_dssm_b_index]
		self.my_rows.extend([[[dssm_infl_cnv_same_lineage_phase_index[p][j][k],
			dssm_index[p][j][k+1]], [1.0, -1.0]]
			for p in range(cons.PHASE_NUMBER)
			for j in xrange(self.ssm_num)
			for k in range(self.sublin_num-1)])
		

	# constraints #21, #22, #23: determine auxiliary variable part 1 for computation
	# of copy number of ssms
	# c #21: dssm_aux_1_cn_j_k - dssm_j_k - dssm_a_j_k - dssm_b_j_k <= 0
	#	When more complicated CN can be modeled, the unphased state does not exist.
	# c #22: dssm_aux_1_cn_j_k - phi_k <= 0
	# c #23: dssm_aux_1_cn_j_k - phi_k - dssm_j_k - dssm_a_j_k - dssm_a_j_k >= -1
	#	When more complicated CN can be modeled, the unphased state does not exist.
	# c #23,5, version 1 : dssm_aux_1_cn_j_0 + ... = 0
	# c #23,5, version 2: dssm_aux_1_cn_j_0 = 0
	def constraint_ssm_aux_1_cn(self):
		#21
		self.constraint_ssm_aux_1_cn_dssms()
		#22
		self.constraint_ssm_aux_1_cn_phi()
		#23
		self.constraint_ssm_aux_1_cn_phi_dssms()
		#23,5
		self.constraint_ssm_aux_1_cn_not_in_normal()

	# c #21: dssm_aux_1_cn_j_k - dssm_j_k - dssm_a_j_k - dssm_b_j_k <= 0
	def constraint_ssm_aux_1_cn_dssms(self):
		entries = (self.sublin_num - 1) * self.ssm_num
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(self.create_rownames_constraint_ssm_aux_1_cn_subprobs("ssm_aux_1_cn_dssms"))
		if self.simple_CN_changes:
			self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][j], 
				self.my_colnames_dssm_index[i][j], 
				self.my_colnames_dssm_a_index[i][j], 
				self.my_colnames_dssm_b_index[i][j]], [1.0, -1.0, -1.0, -1.0]]
				for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)])
		else:
			self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][j], 
				self.my_colnames_dssm_a_index[i][j], 
				self.my_colnames_dssm_b_index[i][j]], [1.0, -1.0, -1.0]]
				for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)])

	# c #22: dssm_aux_1_cn_j_k - phi_k <= 0 
	def constraint_ssm_aux_1_cn_phi(self):
		entries = (self.sublin_num - 1) * self.ssm_num
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(self.create_rownames_constraint_ssm_aux_1_cn_subprobs("ssm_aux_1_cn_phi"))
		self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][j], self.my_phis_index[j]], [1.0, -1.0]]
			for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)])

	# c #23: dssm_aux_1_cn_j_k - phi_k - dssm_j_k - dssm_a_j_k - dssm_a_j_k >= -1
	def constraint_ssm_aux_1_cn_phi_dssms(self):
		entries = (self.sublin_num - 1) * self.ssm_num
		self.my_rhs.extend([-1.0] * entries)
		self.my_sense.extend(["G"] * entries)
		self.my_rownames.extend(self.create_rownames_constraint_ssm_aux_1_cn_subprobs("ssm_aux_1_cn_phi_dssms"))
		if self.simple_CN_changes:
			self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][j], 
				self.my_phis_index[j], self.my_colnames_dssm_index[i][j],
				self.my_colnames_dssm_a_index[i][j], self.my_colnames_dssm_b_index[i][j]], 
				[1.0, -1.0, -1.0, -1.0, -1.0]]
				for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)])
		else:
			self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][j], self.my_phis_index[j],
				self.my_colnames_dssm_a_index[i][j], self.my_colnames_dssm_b_index[i][j]], 
				[1.0, -1.0, -1.0, -1.0]]
				for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)])
	
	# c #23,5, version 2: dssm_aux_1_cn_j_0 = 0
	def constraint_ssm_aux_1_cn_not_in_normal(self):
		self.my_rhs.extend([0.0] * self.ssm_num)
		self.my_sense.extend(["E"] * self.ssm_num)
		self.my_rownames.extend(self.create_rownames_one_index("ssm_aux_1_cn_not_in_normal", self.ssm_num))
		self.my_rows.extend([[[self.my_colnames_dssm_aux_1_cn_index[i][0]], 
			[1.0]] for i in xrange(self.ssm_num)])

	def create_rownames_constraint_ssm_aux_1_cn_subprobs(self, name):
		return ["{0}_{1}_{2}".format(name, i, j) for i in xrange(self.ssm_num) for j in range(1, self.sublin_num)]
	
	# constraints #23.6, #23.7, #23.8: determine auxiliary variable part 1.5 for computation
	# c #23.6: dssm_aux_15_cn_p1_j_k - phi_k <= 0, same for b
	# c #23.7: dssm_aux_15_cn_p1_j_k - dssm_infl_cnv_same_lineage_a_p1_j_k, same for b
	# c #23.8: dssm_aux_15_cn_p1_j_k - phi_k - dssm_infl_cnv_same_lineage_a_p1_j_k >= -1, same for b
	def constraint_ssm_aux_15_cn(self):
		# c #23.6
		self.constraint_ssm_aux_15_cn_phi()
		# c #23.7
		self.constraint_ssm_aux_15_cn_dssm_infl()
		# c #23.8
		self.constraint_ssm_aux_15_cn_phi_dssm_infl()

	def constraint_ssm_aux_15_cn_phi(self):
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["ssm_aux_15_cn_phi_{0}_p1_{1}_{2}".format(p, j, k)
			for p in ["a", "b"] for j in xrange(self.ssm_num) for k in range(1, self.sublin_num)])
		dsmm_aux_15_cn = [self.my_colnames_dssm_aux_15_cn_a_p1_index, 
			self.my_colnames_dssm_aux_15_cn_b_p1_index]
		self.my_rows.extend([[[dsmm_aux_15_cn[p][j][k], self.phi_start_index + k + 1], [1.0, -1.0]]
			for p in range(cons.PHASE_NUMBER)
			for j in xrange(self.ssm_num)
			for k in range(self.sublin_num-1)])

	def constraint_ssm_aux_15_cn_dssm_infl(self):
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_rhs.extend([0.0] * entries)
		self.my_sense.extend(["L"] * entries)
		self.my_rownames.extend(["ssm_aux_15_cn_dssm_infl_{0}_p1_{1}_{2}".format(p, j, k)
			for p in ["a", "b"] for j in xrange(self.ssm_num) for k in range(1, self.sublin_num)])
		dsmm_aux_15_cn = [self.my_colnames_dssm_aux_15_cn_a_p1_index,
			self.my_colnames_dssm_aux_15_cn_b_p1_index]
		dssm_infl = [self.my_colnames_dssm_infl_cnv_same_lineage_a_index,
			self.my_colnames_dssm_infl_cnv_same_lineage_b_index]
		self.my_rows.extend([[[dsmm_aux_15_cn[p][j][k], dssm_infl[p][j][k]], [1.0, -1.0]]
			for p in range(cons.PHASE_NUMBER)
			for j in xrange(self.ssm_num) 
			for k in range(self.sublin_num-1)])

	def constraint_ssm_aux_15_cn_phi_dssm_infl(self):
		entries = self.ssm_num * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_rhs.extend([-1.0] * entries)
		self.my_sense.extend(["G"] * entries)
		self.my_rownames.extend(["ssm_aux_15_cn_phi_dssm_infl_{0}_p1_{1}_{2}".format(p, j, k)
			for p in ["a", "b"] for j in xrange(self.ssm_num) for k in range(1, self.sublin_num)])
		dsmm_aux_15_cn = [self.my_colnames_dssm_aux_15_cn_a_p1_index,
			self.my_colnames_dssm_aux_15_cn_b_p1_index]
		dssm_infl = [self.my_colnames_dssm_infl_cnv_same_lineage_a_index, 
			self.my_colnames_dssm_infl_cnv_same_lineage_b_index] 
		self.my_rows.extend([[[dsmm_aux_15_cn[p][j][k], self.phi_start_index + k + 1,
			dssm_infl[p][j][k]], [1.0, -1.0, -1.0]]
			for p in range(cons.PHASE_NUMBER)
			for j in xrange(self.ssm_num)
			for k in range(self.sublin_num-1)])

	# constraints #24, #25, #26, #27, #28: determine auxiliary variable part 2 for computation 
	# c #24: dssm_aux_2_cn_a_p1_j_k_k' - phi_k' <= 0, same for b, same for m1
	# c #25: dssm_aux_2_cn_a_p1_j_k_k' - z_k_k' <= 0, same for b, same for m1
	# c #26: dssm_aux_2_cn_a_p1_j_k_k' - dssm_a_j_k <= 0, same for b, same for m1
	# c #27: dssm_aux_2_cn_a_p1_j_k_k' - dc_a_p1_binary_i_k' <= 0, same for b, similar for m1
	# c #28: dssm_aux_2_cn_a_p1_j_k_k' - dssm_a_j_k - dc_a_p1_binary_i_k' - phi_k' -z_k_k' >=  - 3,
	#	same for b, similar for m1
	# c #28.5 version 2: dssm_aux_2_cn_a_p1_j_k_k' = 0, where k == 0 || k' >= k, same for b, 
	def constraint_ssm_aux_2_cn(self, ssm_list):
		get_seg_index = self.get_seg_index
		for i in range(self.ssm_num):
			seg_index = get_seg_index(ssm_list, i)
			for j in range(1, self.sublin_num):
				for k in range(j+1, self.sublin_num):
					#24
					self.constraint_ssm_aux_2_cn_phi(self.my_colnames_dssm_aux_2_cn_a_p1_index, 
						"a", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_phi(self.my_colnames_dssm_aux_2_cn_b_p1_index,
						"b", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_phi(self.my_colnames_dssm_aux_2_cn_a_m1_index, 
						"a", i, j, k, "m1")
					self.constraint_ssm_aux_2_cn_phi(self.my_colnames_dssm_aux_2_cn_b_m1_index, 
						"b", i, j, k, "m1")
					#25
					self.constraint_ssm_aux_2_cn_z(self.my_colnames_dssm_aux_2_cn_a_p1_index, 
						"a", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_z(self.my_colnames_dssm_aux_2_cn_b_p1_index, 
						"b", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_z(self.my_colnames_dssm_aux_2_cn_a_m1_index, 
						"a", i, j, k, "m1")
					self.constraint_ssm_aux_2_cn_z(self.my_colnames_dssm_aux_2_cn_b_m1_index, 
						"b", i, j, k, "m1")
					#26
					self.constraint_ssm_aux_2_cn_dssm(self.my_colnames_dssm_aux_2_cn_a_p1_index, 
						self.my_colnames_dssm_a_index, "a", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_dssm(self.my_colnames_dssm_aux_2_cn_b_p1_index, 
						self.my_colnames_dssm_b_index, "b", i, j, k, "p1")
					self.constraint_ssm_aux_2_cn_dssm(self.my_colnames_dssm_aux_2_cn_a_m1_index, 
						self.my_colnames_dssm_a_index, "a", i, j, k, "m1")
					self.constraint_ssm_aux_2_cn_dssm(self.my_colnames_dssm_aux_2_cn_b_m1_index, 
						self.my_colnames_dssm_b_index, "b", i, j, k, "m1")
					#27
					self.constraint_ssm_aux_2_cn_dc_p1_binary(
						self.my_colnames_dssm_aux_2_cn_a_p1_index,
						self.my_colnames_dc_a_p1_binary_index, "a", i, j, k, seg_index, "p1")
					self.constraint_ssm_aux_2_cn_dc_p1_binary(
						self.my_colnames_dssm_aux_2_cn_b_p1_index,
						self.my_colnames_dc_b_p1_binary_index, "b", i, j, k, seg_index, "p1")
					self.constraint_ssm_aux_2_cn_dc_p1_binary(
						self.my_colnames_dssm_aux_2_cn_a_m1_index,
						self.my_colnames_dc_a_m1_binary_index, "a", i, j, k, seg_index, "m1")
					self.constraint_ssm_aux_2_cn_dc_p1_binary(
						self.my_colnames_dssm_aux_2_cn_b_m1_index,
						self.my_colnames_dc_b_m1_binary_index, "b", i, j, k, seg_index, "m1")
					#28
					self.constraint_ssm_aux_2_cn_three("a", i, j, k, seg_index, 
						self.my_colnames_dssm_aux_2_cn_a_p1_index, 
						self.my_colnames_dssm_a_index,
						self.my_colnames_dc_a_p1_binary_index, "p1")
					self.constraint_ssm_aux_2_cn_three("b", i, j, k, seg_index,
						self.my_colnames_dssm_aux_2_cn_b_p1_index, 
						self.my_colnames_dssm_b_index,
						self.my_colnames_dc_b_p1_binary_index, "p1")
					self.constraint_ssm_aux_2_cn_three("a", i, j, k, seg_index, 
						self.my_colnames_dssm_aux_2_cn_a_m1_index, 
						self.my_colnames_dssm_a_index,
						self.my_colnames_dc_a_m1_binary_index, "m1")
					self.constraint_ssm_aux_2_cn_three("b", i, j, k, seg_index,
						self.my_colnames_dssm_aux_2_cn_b_m1_index, 
						self.my_colnames_dssm_b_index,
						self.my_colnames_dc_b_m1_binary_index, "m1")
		# 28.5
		self.constraint_ssm_aux_2_cn_valid_variables()

	# helping method for constraint 24
	def constraint_ssm_aux_2_cn_phi(self, dssm_a_b_index, a_or_b, i, j, k, state):
		name = "ssm_aux_2_cn_{0}_{4}_phi_{1}_{2}_{3}".format(a_or_b, i, j, k, state)
		self.create_constraint_a_minus_b_le_zero(
			dssm_a_b_index[i][j][k], self.my_phis_index[k], name)

	# helping method for constraint 25
	def constraint_ssm_aux_2_cn_z(self, dssm_a_b_index, a_or_b, i, j, k, state):
		name = "ssm_aux_2_cn_{0}_{4}_z_{1}_{2}_{3}".format(a_or_b, i, j, k, state)
		self.create_constraint_a_minus_b_le_zero(
			dssm_a_b_index[i][j][k], self.my_colnames_z_index[j][k], name)

	# helping method for constraint 26
	def constraint_ssm_aux_2_cn_dssm(self, dssm_a_b_index, dssm_matrix_index, a_or_b, i, j, k, state):
		name = "ssm_aux_2_cn_{0}_{4}_dssm_{1}_{2}_{3}".format(a_or_b, i, j, k, state)
		self.create_constraint_a_minus_b_le_zero(
			 dssm_a_b_index[i][j][k], dssm_matrix_index[i][j], name)

	# helping method for constraint 27
	def constraint_ssm_aux_2_cn_dc_p1_binary(self, dssm_a_b_index, dc_binary_index, 
		a_or_b, i, j, k, seg_index, state):
		name = "ssm_aux_2_cn_{0}_dc_{4}_binary_{1}_{2}_{3}".format(a_or_b, i, j, k, state)
		self.create_constraint_a_minus_b_le_zero(
			 dssm_a_b_index[i][j][k], dc_binary_index[seg_index][k], name)

	# helping method for constraint 28
	def constraint_ssm_aux_2_cn_three(self, a_or_b, i, j, k, seg_index, dssm_aux_2_cn_index,
		dssm_index, dc_binary_index, state):
		c_value = [1.0, -1.0, -1.0, -1.0, -1.0]
		name = "ssm_aux_2_cn_{0}_{4}_three_{1}_{2}_{3}".format(a_or_b, i, j, k, state)
		row = [dssm_aux_2_cn_index[i][j][k], dssm_index[i][j], dc_binary_index[seg_index][k], 
			self.my_phis_index[k], self.my_colnames_z_index[j][k]]
		self.create_constraint(-3.0, "G", row, c_value, name)

	# constraint 28.5
	def constraint_ssm_aux_2_cn_valid_variables(self):
		for i in range(self.ssm_num):
			for j in range(self.sublin_num):
				for k in range(self.sublin_num):
					if j == 0:
						self.constraint_ssm_aux_2_cn_valid_variables_over_indices(i, j, k)
					elif j >= k:
						self.constraint_ssm_aux_2_cn_valid_variables_over_indices(i, j, k)

	# creates constraints for constraint 28.5
	def constraint_ssm_aux_2_cn_valid_variables_over_indices(self, i, j, k):
		name_a_p1 = "ssm_aux_2_cn_a_p1_valid_variables_%d_%d_%d" % (i, j, k)
		name_b_p1 = "ssm_aux_2_cn_b_p1_valid_variables_%d_%d_%d" % (i, j, k)
		name_a_m1 = "ssm_aux_2_cn_a_m1_valid_variables_%d_%d_%d" % (i, j, k)
		name_b_m1 = "ssm_aux_2_cn_b_m1_valid_variables_%d_%d_%d" % (i, j, k)
		self.create_constraint(0.0, "E", [self.my_colnames_dssm_aux_2_cn_a_p1_index[i][j][k]], [1.0], name_a_p1)
		self.create_constraint(0.0, "E", [self.my_colnames_dssm_aux_2_cn_b_p1_index[i][j][k]], [1.0], name_b_p1)
		self.create_constraint(0.0, "E", [self.my_colnames_dssm_aux_2_cn_a_m1_index[i][j][k]], [1.0], name_a_m1)
		self.create_constraint(0.0, "E", [self.my_colnames_dssm_aux_2_cn_b_m1_index[i][j][k]], [1.0], name_b_m1)

	# constraint #29: Set CN of SSMs and CN of segment equal to observed frequency of SSMs
	# ssm_a_j_j' * ssm_x_j_j' * cn_i(j) + ... - dssm_aux_1_cn_j_k 
	# - dssm_aux_2_cn_a_j_k_k' - dssm_aux_2_cn_b_j_k_k' = 0
	def constraint_ssm_value_spline_frequency(self, ssm_list):
		self.my_rhs.extend([0.0] * self.ssm_num)
		self.my_sense.extend(["E"] * self.ssm_num)
		self.my_rownames.extend(self.create_rownames_one_index("ssm_frequency", self.ssm_num))

		get_seg_index = self.get_seg_index
		self.my_rows.extend([[self.my_colnames_ssm_index[i] + 
			self.my_colnames_dssm_aux_1_cn_index[i].tolist() + 
			self.my_colnames_dssm_aux_15_cn_a_p1_index[i].tolist() +
			self.my_colnames_dssm_aux_15_cn_b_p1_index[i].tolist() +
			self.flatten_list(self.my_colnames_dssm_aux_2_cn_a_p1_index[i].tolist()) + 
			self.flatten_list(self.my_colnames_dssm_aux_2_cn_b_p1_index[i].tolist()) +
			self.flatten_list(self.my_colnames_dssm_aux_2_cn_a_m1_index[i].tolist()) +
			self.flatten_list(self.my_colnames_dssm_aux_2_cn_b_m1_index[i].tolist()),
			[x * self.cn[get_seg_index(ssm_list, i)] for x in self.ssm_splines[i].get_knots()] + 
			[-1.0] * (self.sublin_num + (self.sublin_num-1) * cons.PHASE_NUMBER 
			+ self.sublin_num * self.sublin_num * cons.PHASE_NUMBER) + 
			[1.0] * (self.sublin_num * self.sublin_num * cons.PHASE_NUMBER)]
			for i in xrange(self.ssm_num)])

	# creates constraint in form: 1.0 a - 1.0 b <= 0
	def create_constraint_a_minus_b_le_zero(self, var_a, var_b, item_name):
		c_value = [1.0, -1.0]
		row = [var_a, var_b]
		self.create_constraint(0.0, "L", row, c_value, item_name)

	# creates constraint with basic features rhs, sense, variables and values
	def create_constraint(self, item_rhs, item_sense, item_vars, item_values, item_name):
		self.my_rhs.append(item_rhs)
		self.my_sense.append(item_sense)
		self.my_rows.append([item_vars, item_values])
		self.my_rownames.append(item_name)

	# falltens 2d list
	def flatten_list(self, list):
		return [item for sublist in list for item in sublist]
	
	def flatten_3d_list(self, list):
		list = self.flatten_list(list)
		list = self.flatten_list(list)
		return list

	# adds constraints to my_row array for CPLEX
	# variables are contained in 1D array and combined with values that are the same
	# for all variables
	def add_constraints_mult_indices_same_values(self, i1, i2, my_rows, c_vars, c_values):
		for i in range(i1 * i2):
			my_rows.append([c_vars[i], c_values])

	# constraint #30
	# fixes the values of the variables dc_a_p1_binary, dc_b_p1_binary, dc_a_m1_binary, dc_b_m1_binary
	# dc_a_p1_binary_i_k = fixed value, either 0 or 1, for all i and k, for all variables
	def constraint_fix_dc_binary(self, fixed_values, unfixed_start, unfixed_stop, fixed_cnv_indices=[]):
		self.start_index_constraint_fix_dc_binary = len(self.my_rhs)
		fixed_values_per_matrix = len(fixed_values) / 4
		# for variables dc_a_p1_binary, take only first quater of fixed values
		self.help_constraint_fixed_values(fixed_values[0 : fixed_values_per_matrix], 
			unfixed_start, unfixed_stop, self.delta_c_entry_num, self.sublin_num,
			self.dc_binary_index_start_p1, 0, self.my_colnames_dc_a_p1_binary_index, 
			mutation_indices=fixed_cnv_indices)
		# for variables dc_b_p1_binary, take only second quater of fixed values
		self.help_constraint_fixed_values(
			fixed_values[fixed_values_per_matrix : fixed_values_per_matrix * 2], 
			unfixed_start, unfixed_stop, self.delta_c_entry_num, self.sublin_num,
			self.dc_binary_index_start_p1 + self.delta_c_entry_num, 
			0, self.my_colnames_dc_b_p1_binary_index, mutation_indices=fixed_cnv_indices)
		# for variables dc_a_m1_binary, take only third quater of fixed values
		self.help_constraint_fixed_values(
			fixed_values[fixed_values_per_matrix * 2 : fixed_values_per_matrix * 3], 
			unfixed_start, unfixed_stop, self.delta_c_entry_num, self.sublin_num,
			self.dc_binary_index_start_m1, 0, self.my_colnames_dc_a_m1_binary_index,
			mutation_indices=fixed_cnv_indices)
		# for variables dc_b_m1_binary, take only forth quater of fixed values
		self.help_constraint_fixed_values(
			fixed_values[fixed_values_per_matrix * 3 : fixed_values_per_matrix * 4], 
			unfixed_start, unfixed_stop, self.delta_c_entry_num, self.sublin_num,
			self.dc_binary_index_start_m1 + self.delta_c_entry_num,
			0, self.my_colnames_dc_b_m1_binary_index, mutation_indices=fixed_cnv_indices)

	# fixes CNV and SSM mutations
	# list with mutations to be fixed is given, all variables for the mutations are fixed
	def constraint_fix_dc_binary_or_SSMs(self, fixed_mutations, mutation_type):
		# get number of variables
		# mutations are CNVs
		if mutation_type == cons.CNV:
			var_num = 4
			(a_p1, b_p1, a_m1, b_m1) = (0, 1, 2, 3)
			my_colnames_to_fix = [self.my_colnames_dc_a_p1_binary_index, 
				self.my_colnames_dc_b_p1_binary_index, self.my_colnames_dc_a_m1_binary_index,
				self.my_colnames_dc_b_m1_binary_index]
			mut_name = "cnv"
			colname_names = ["a_p1", "b_p1", "a_m1", "b_m1"]
			self.start_index_constraint_fix_dc_binary = len(self.my_rhs)
		# mutations are SSMs
		else:
			mut_name = "ssm"
			self.start_index_constraint_fix_dssm = len(self.my_rhs)
			# simple CN changes are used
			if self.simple_CN_changes == True:
				var_num = 3
				my_colnames_to_fix = [self.my_colnames_dssm_a_index, self.my_colnames_dssm_b_index,
					self.my_colnames_dssm_index]
				colname_names = ["a", "b", "unphased"]
			# more complex CN changes are allowed
			else:
				var_num = 2
				my_colnames_to_fix = [self.my_colnames_dssm_a_index, self.my_colnames_dssm_b_index]
				colname_names = ["a", "b"]

		# iterate over all fixed segments / SSMs
		for entry in fixed_mutations:

			# create lists with fixation values
			fixation_lists  = [[0] * self.sublin_num for _ in xrange(var_num)]

			# go through list with fixation values und set entry in fixation_list accordingly
			mut_index = entry[0]
			# for CNVs
			if mutation_type == cons.CNV:
				for cn_changes in entry[1]:
					(lineage, change, phase) = cn_changes
					# no CNV indicated with change 0
					if change == 0:
						continue
					# determin, which fixation list gets updated
					elif change == 1:
						if phase == cons.A:
							update = a_p1
						elif phase == cons.B:
							update = b_p1
					elif change == -1:
						if phase == cons.A:
							update = a_m1
						elif phase == cons.B:
							update = b_m1
					else:
						raise oe.MyException("Fixation for this change is not defined")
					# update fixation list
					fixation_lists[update][lineage] = 1
			# for SSMs
			else:
				lineage = entry[1]
				phase = entry[2]
				# in case of more complex CN changes, check phase and eventually change to A
				if self.simple_CN_changes == False:
					if phase == cons.UNPHASED:
						phase = cons.A
				# update fixation list
				fixation_lists[phase][lineage] = 1

			# fix all entries to values in list
			rhs = self.flatten_list(fixation_lists)
			self.my_rhs.extend(rhs)
			self.my_sense.extend(["E"] * len(rhs))
			self.my_rownames.extend(["fixed_constraint_{0}_{1}_{2}_{3}".format(mut_name, colname_names[i],
				mut_index, k) for i in xrange(var_num) for k in xrange(self.sublin_num)])
			self.my_rows.extend([[[my_colnames_to_fix[i][mut_index][k]], [1.0]] 
				for i in xrange(var_num) for k in xrange(self.sublin_num)])

		# get end index of fixed constraints
		if mutation_type == cons.CNV:
			self.end_index_constraint_fix_dc_binary = len(self.my_rhs)
		else:
			self.end_index_constraint_fix_dssm = len(self.my_rhs)
			
	# constraint #30.1
	# fixes the average CN of a segment
	# dc_a_p1_float_i_k + ... + dc_b_p1_float_i_k + ... - dc_a_m1_float_i_k - ...
	#	- dc_b_m1_float_i_k - ... = fixed average cn - 2, for 0 <= k <= K - 1
	def constraint_fix_avg_cn(self, fixed_values, unfixed_start, unfixed_stop):
		number_constraints = 0
		# all segments are fixed
		if unfixed_start == - 1:
			number_constraints = self.seg_num
		else:
			number_constraints = self.seg_num - (unfixed_stop - unfixed_start + 1)

		self.my_rhs.extend([x - 2 for x in fixed_values])
		self.my_sense.extend(["E"] * number_constraints)

		# all segments are fixed
		if unfixed_start == -1:
			self.my_rownames.extend(["fixed_avg_cn_{0}".format(i) 
				for i in xrange(self.seg_num)]) 
			self.my_rows.extend(self.constraint_fix_avg_cn_constraint(0, self.seg_num))
		else:
			self.my_rownames.extend(["fixed_avg_cn_{0}".format(i) for i in xrange(0, unfixed_start)])
			self.my_rownames.extend(["fixed_avg_cn_{0}".format(i) for i in xrange(unfixed_stop + 1, 
				self.seg_num)])
			self.my_rows.extend(self.constraint_fix_avg_cn_constraint(0, unfixed_start))
			self.my_rows.extend(self.constraint_fix_avg_cn_constraint(unfixed_stop + 1, 
				self.seg_num))
	
	def constraint_fix_avg_cn_constraint(self, start, stop):
		return [[self.my_colnames_dc_a_p1_float_index[i].tolist() 
			+ self.my_colnames_dc_b_p1_float_index[i].tolist()
			+ self.my_colnames_dc_a_m1_float_index[i].tolist() 
			+ self.my_colnames_dc_b_m1_float_index[i].tolist(),
			[1.0] * (self.sublin_num * cons.PHASE_NUMBER) 
			+ [-1.0] * (self.sublin_num * cons.PHASE_NUMBER)]
			for i in xrange(start, stop)]

	# constraint #30.1
	# for one single segment
	def constraint_fix_single_avg_cn(self, fixed_value, seg_index):
		self.start_index_constraint_fix_single_avg_cn = len(self.my_rhs)
		self.my_rhs.extend([fixed_value - 2])
		self.my_sense.extend(["E"])
		self.my_rownames.extend(["fixed_single_avg_cn_seg_{0}".format(seg_index)])
		self.my_rows.extend([[self.my_colnames_dc_a_p1_float_index[seg_index].tolist()
			+ self.my_colnames_dc_b_p1_float_index[seg_index].tolist()
			+ self.my_colnames_dc_a_m1_float_index[seg_index].tolist()
			+ self.my_colnames_dc_b_m1_float_index[seg_index].tolist(),
			[1.0] * (self.sublin_num * cons.PHASE_NUMBER) 
			+ [-1.0] * (self.sublin_num * cons.PHASE_NUMBER)]])

	# constraint #31
	# fixes the values of the variables dsnp, dsnp_a, dsnp_b
	# dsnp_l = fixed value, either 0 or 1, for all l, for all variables
	def constraint_fix_dsnp(self, fixed_values, unfixed_start, unfixed_stop):
		if not self.simple_CN_changes:
			raise(oe.MyException("Not supported for more complicated CN changes."))	
			
		self.start_index_constraint_fix_dsnp = len(self.my_rhs)
		fixed_values_per_matrix = len(fixed_values) / 3
		# for variables dsnp, take only first third of fixed values
		self.help_constraint_fixed_values(fixed_values[0:fixed_values_per_matrix], 
			unfixed_start, unfixed_stop, self.snp_num, 1, self.dsnp_start_index)
		# for variables dsnp_a, take only second third of fixed values
		self.help_constraint_fixed_values(
			fixed_values[fixed_values_per_matrix : fixed_values_per_matrix * 2], unfixed_start, 
			unfixed_stop, self.snp_num, 1, self.dsnp_start_index + self.snp_num)
		# for variables dsnp_b, take only third third of fixed values
		self.help_constraint_fixed_values(
			fixed_values[fixed_values_per_matrix * 2: fixed_values_per_matrix * 3], unfixed_start, 
			unfixed_stop, self.snp_num, 1, self.dsnp_start_index + self.snp_num * 2)
	
	# constraint #32
	# fixes the values of the variables dssm, dssm_a, dssm_b
	# dssm_j_k = fixed value, either 0 or 1, for all j and k, for all variables
	def constraint_fix_dssm(self, fixed_values, unfixed_start, unfixed_stop):
		self.start_index_constraint_fix_dssm = len(self.my_rhs)
		# consider three phases when only simple CN changes are allowed
		if self.simple_CN_changes:
			fixed_values_per_matrix = len(fixed_values) / 3
			# for variables dssm, take only first third of fixed values
			self.help_constraint_fixed_values(fixed_values[0 : fixed_values_per_matrix], 
				unfixed_start, unfixed_stop, 
				self.delta_s_ssm_num, self.sublin_num, self.dssm_start_index,
				0, self.my_colnames_dssm_index)
			# for variables dssm_a, take only second third of fixed values
			self.help_constraint_fixed_values(
				fixed_values[fixed_values_per_matrix : fixed_values_per_matrix * 2], 
				unfixed_start, unfixed_stop, self.delta_s_ssm_num, self.sublin_num,
				self.dssm_start_index + self.delta_s_ssm_num,
				0, self.my_colnames_dssm_a_index)
			# for variables dssm_b, take only third third of fixed values
			self.help_constraint_fixed_values(
				fixed_values[fixed_values_per_matrix * 2 : fixed_values_per_matrix * 3], 
				unfixed_start, unfixed_stop, self.delta_s_ssm_num, self.sublin_num,
				self.dssm_start_index + self.delta_s_ssm_num * 2,
				0, self.my_colnames_dssm_b_index)
		# consider only two phases when more complicated CN changes are allowed
		else:
			# fixed file contains three matrices: unphased, A and B
			# but for non-simple CN changes I only work work with phase A and B
			# thus, the two matrices unphased and B need to be combined
			fixed_values_per_matrix = len(fixed_values) / 3
			matrix_unphased = fixed_values[0 : fixed_values_per_matrix]
			tmp_matrix_A = fixed_values[fixed_values_per_matrix : fixed_values_per_matrix * 2]
			matrix_A = [matrix_unphased[i] + tmp_matrix_A[i] for i in xrange(len(tmp_matrix_A))]
			matrix_B = fixed_values[fixed_values_per_matrix * 2 : fixed_values_per_matrix * 3]

			# for variables dssm, take only first half of fixed values
			self.help_constraint_fixed_values(matrix_A, 
				unfixed_start, unfixed_stop, 
				self.delta_s_ssm_num, self.sublin_num, self.dssm_start_index,
				0, self.my_colnames_dssm_a_index)
			# for variables dssm_a, take only second half of fixed values
			self.help_constraint_fixed_values(
				matrix_B, 
				unfixed_start, unfixed_stop, self.delta_s_ssm_num, self.sublin_num,
				self.dssm_start_index + self.delta_s_ssm_num,
				0, self.my_colnames_dssm_b_index)

	# constraint helping to create fixed value
	# constraint-type: value x = fixed value
	def help_constraint_fixed_values(self, fixed_values, unfixed_start, unfixed_stop, number_variables, 
		numbers_per_row, index_start_colnames, index_start_vars, my_colnames_indices,
		phis=False, mutation_indices=[]):

		# when indices of mutations are given that should be fixed another function is used
		if mutation_indices != []:
			self.help_constraint_fixed_values_with_indices(fixed_values, mutation_indices,
				numbers_per_row, index_start_colnames, index_start_vars, my_colnames_indices)
			return

		# computes number of unfixed variables
		number_unfixed = -1
		# no variable is unfixed
		if unfixed_start == -1:
			number_unfixed = 0
		# some variables are fixed
		else:
			number_unfixed = (unfixed_stop - unfixed_start + 1) * numbers_per_row

		number_constraints = number_variables - number_unfixed
		if number_constraints != len(fixed_values):
			raise(oe.MyException("Fixed values has different length than number of constraints."))	

		self.my_sense.extend(["E"] * number_constraints)
		self.my_rhs.extend(fixed_values)

		# pick variables that should be fixed
		fixed_variables_names = []
		fixed_variables_vars = []
		# no variable is unfixed
		if unfixed_start == -1:
			fixed_variables_names = (self.my_colnames[
				index_start_colnames : index_start_colnames + number_variables])
			fixed_variables_vars = (my_colnames_indices.flatten().tolist()[
				index_start_vars : index_start_vars + number_variables])
		# some variables are fixed
		else:
			# because fixation of phis doesn't start at index 0 but 1
			# (because normal lineage frequency is always 1 and thus does not need
			# to be fixed) we decrease unfixed_start and unfixed_stop to get
			# right variables
			if phis:
				unfixed_start -= 1
				unfixed_stop -= 1
			fixed_variables_names = (
				self.my_colnames[index_start_colnames : 
				index_start_colnames + numbers_per_row * unfixed_start])
			fixed_variables_names.extend(self.my_colnames[
				index_start_colnames + numbers_per_row * unfixed_start +
				number_unfixed : index_start_colnames + number_variables])
			fixed_variables_vars = (
				my_colnames_indices.flatten().tolist()[
					index_start_vars : index_start_vars + numbers_per_row * unfixed_start])
			fixed_variables_vars.extend(my_colnames_indices.flatten().tolist()[
				index_start_vars + numbers_per_row * unfixed_start +
				number_unfixed : index_start_vars + number_variables])

		self.my_rownames.extend(["fixed_{0}".format(fixed_variables_names[i]) 
			for i in xrange(number_constraints)])
		self.my_rows.extend([[[fixed_variables_vars[i]], [1.0]] 
			for i in xrange(number_constraints)])

	def help_constraint_fixed_values_with_indices(self, fixed_values, mutation_indices,
		numbers_per_row, index_start_colnames, index_start_vars, my_colnames_indices):

		# set sens and RHS
		number_fixed = len(fixed_values)
		self.my_sense.extend(["E"] * number_fixed)
		self.my_rhs.extend(fixed_values)

		# get names and indices of variables that should be fixed
		fixed_variables_names = []
		fixed_variables_vars = []
		for index in mutation_indices:
			fixed_variables_names.extend(self.my_colnames[
				index_start_colnames + index * numbers_per_row :
				index_start_colnames + (index +1) * numbers_per_row])
			fixed_variables_vars.extend(my_colnames_indices[index].tolist())

		# append names and indices of variables to lists
		self.my_rownames.extend(["fixed_{0}".format(fixed_variables_names[i])
			for i in xrange(number_fixed)])
		self.my_rows.extend([[[fixed_variables_vars[i]], [1.0]]
			for i in xrange(number_fixed)])

	# constraint #33
	# fixes the values of the variables z_k_k' for 0 < k < K-1, for k < k' < K
	# z_k_k' = fixed value, either 0 or 1
	def constraint_fix_z_matrix(self, fixed_values, unfixed_start, unfixed_stop):
		self.z_matrix_is_fixed = True

		number_constraints = self.untrivial_z_entries_num
		self.my_sense.extend(["E"] * number_constraints)
		self.my_rhs.extend(fixed_values)
		variable_names = ([self.my_colnames[self.z_index_start + (k * self.sublin_num + k_prime)] 
			for k in range(1, self.sublin_num) for k_prime in range(k + 1, self.sublin_num)])
		variable_indices = self.get_z_matrix_indices()
		self.my_rownames.extend(["fixed_{0}".format(variable_names[i]) 
			for i in xrange(number_constraints)])
		self.my_rows.extend([[[variable_indices[i]], [1.0]] for i in xrange(number_constraints)])

		return variable_indices

	def get_z_matrix_indices(self):
		return ([self.my_colnames_z_index[k][k_prime] 
			for k in range(1, self.sublin_num) for k_prime in range(k + 1, self.sublin_num)])

	# constraint #34
	# fixes the values of the sublinear frequencies, phi_1, ..., phi_K-1
	# phi_k = fixed value, between 0 and 1, for all 1 <= k < K-1
	def constraint_fix_phi(self, fixed_values, unfixed_start, unfixed_stop):
		self.help_constraint_fixed_values(fixed_values, unfixed_start, unfixed_stop,
			self.sublin_num - 1, 1, self.phi_start_index + 1,
			1, self.my_phis_index, phis=True)

	# constraint #35
	# Applies the lineage divergence rule when the Z matrix is fixed and a list with direct descendant
	# lineages for the lineages is provided
	# phi_k - phi_y' - phi_y'' - ... >= 0, for 0 <= k < K - 1
	# where phi_y are direct descendant of phi_k
	def constraint_lineage_divergence_z_fixed(self, direct_descendants_for_constraints):
		entries = len(direct_descendants_for_constraints)
		if entries > 0:
			self.my_sense.extend(["G"] * entries)
			self.my_rhs.extend([0.0] * entries)
			self.my_rownames.extend(["lineage_divergence_z_fixed_phi_{0}".format(
				direct_descendants_for_constraints[i][0]) for i in range(entries)])
			variable_names = []
			variable_indices = []
			for i in range(entries):
				variable_names_entry = []
				variable_indices_entry = []
				for j in range(len(direct_descendants_for_constraints[i])):
					variable_names_entry.append(self.my_colnames[self.phi_start_index 
						+ direct_descendants_for_constraints[i][j]])
					variable_indices_entry.append(self.my_phis_index[
						direct_descendants_for_constraints[i][j]])
				variable_names.append(variable_names_entry)
				variable_indices.append(variable_indices_entry)
			self.my_rows.extend([[variable_indices[i], [1.0] + [-1.0] * (len(
				direct_descendants_for_constraints[i]) - 1)] for i in range(entries)])

	# constraint #36
	# SSMs that are clustered together are assigned to the same lineage list 
	#	that stores SSM indices per cluster
	# ssm_indices_per_cluster: [seg_unit][cluster][SSM_index]
	# dssm_i0_k = dssm_i_k for i > i0 in SSM_index for k >= 1 for all dssms
	def constraint_clustered_ssms(self, ssm_indices_per_cluster):
		entries = sum(len(cluster) - 1 for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			if len(cluster) > 1) * (self.sublin_num - 1)
		if not self.simple_CN_changes:
			entries *= 2
		else:
			entries *= 3

		self.my_sense.extend(["E"] * entries)
		self.my_rhs.extend([0.0] * entries)
		if self.simple_CN_changes:
			self.my_rows.extend(self.constraint_clustered_ssms_dssms(ssm_indices_per_cluster,
				self.my_colnames_dssm_index))
			self.my_rownames.extend(self.constraint_clustered_ssms_dssms_rownames(ssm_indices_per_cluster))
		self.my_rows.extend(self.constraint_clustered_ssms_dssms(ssm_indices_per_cluster,
			self.my_colnames_dssm_a_index))
		self.my_rownames.extend(self.constraint_clustered_ssms_dssms_rownames(ssm_indices_per_cluster))
		self.my_rows.extend(self.constraint_clustered_ssms_dssms(ssm_indices_per_cluster,
			self.my_colnames_dssm_b_index))
		self.my_rownames.extend(self.constraint_clustered_ssms_dssms_rownames(ssm_indices_per_cluster))

	def constraint_clustered_ssms_dssms(self, ssm_indices_per_cluster, dssm_index):
		return [[[dssm_index[cluster[0]][k], dssm_index[cluster[i]][k]], [1.0, -1.0]]
			#for i in xrange(1, len(cluster)) for k in xrange(1, self.sublin_num)
			for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			for i in xrange(1, len(cluster)) for k in xrange(1, self.sublin_num)]
			#for ssm_indices in cluster for cluster in seg_unit for seg_unit in ssm_indices_per_cluster]

	def constraint_clustered_ssms_dssms_rownames(self, ssm_indices_per_cluster):
		return ["constraint_clustered_ssms_x"
			for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			for i in xrange(1, len(cluster)) for k in xrange(1, self.sublin_num)]

	# constraint #37
	# SSMs thar are clustered together are either all or not influenced by a CN change
	# in the same lineage they occur
	# ssm_indices_per_cluster: [seg_unit][cluster][SSM_index]
	# dssm_infl_cnv_same_lineage_a_p1_i0_k = dssm_infl_cnv_same_lineage_a_p1_i_k for i > i0 index in SSM_index
	#	for k >= 1 for all dssms, for other b as well
	def constraint_clustered_ssms_infl_cnv_same_lineage(self, ssm_indices_per_cluster):
		entries = sum(len(cluster) - 1 for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			if len(cluster) > 1) * (self.sublin_num - 1) * cons.PHASE_NUMBER
		self.my_sense.extend(["E"] * entries)
		self.my_rhs.extend([0.0] * entries)
		dssm_infl_cnv_same_lineage = [self.my_colnames_dssm_infl_cnv_same_lineage_a_index,
			self.my_colnames_dssm_infl_cnv_same_lineage_b_index]
		self.my_rownames.extend(["constraint_clustered_ssms_infl_cnv_same_lineage_x"
			for p in range(cons.PHASE_NUMBER)
			for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			for i in xrange(1, len(cluster)) for k in xrange(self.sublin_num-1)])
		self.my_rows.extend([[[dssm_infl_cnv_same_lineage[p][cluster[0]][k],
			dssm_infl_cnv_same_lineage[p][cluster[i]][k]], [1.0, -1.0]]
			for p in range(cons.PHASE_NUMBER)
			for seg_unit in ssm_indices_per_cluster for cluster in seg_unit
			for i in xrange(1, len(cluster)) for k in xrange(self.sublin_num-1)])
	
	# constraints #38, #39 and #40
	# to model whether lineage k' is a child of lineage k
	def constraint_define_children(self):
		self.constraint_child_ancestral_relation()
		self.constraint_child_other_parents()
		self.constraint_child_ge()

	# constraint #38
	# child_k_k' - z_k_k' <= 0 for 0 <= k < k' < K
	def constraint_child_ancestral_relation(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0.0] * entries)
		self.my_rownames.extend(["constraint_child_ancestral_relation_{0}_{1}".format(k, k_prime)
			for k in xrange(0, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		z_indices = [self.my_colnames_z_index[k][k_prime] for k in xrange(0, self.sublin_num-1)
			for k_prime in xrange(k+1, self.sublin_num)]
		self.my_rows.extend([[[self.my_colnames_children_index[i], z_indices[i]], [1.0, -1.0]]
			for i in xrange(entries)])

	# constraint #39
	# child_k_k' + z_k^{circ}_k' <= 1 for 0 <= k < k^{circ} < k' < K
	def constraint_child_other_parents(self):
		entries = sum([k_prime - k - 1 for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([1.0] * entries)
		self.my_rownames.extend(["constraint_child_{0}_{1}_other_parents_{2}".format(k, k_prime, k_circ)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)
			for k_circ in xrange(k+1, k_prime)])
		self.my_rows.extend([[[self.my_colnames_children_index_friendly_form[k][k_prime], 
			self.my_colnames_z_index[k_circ][k_prime]], [1.0, 1.0]]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)
			for k_circ in xrange(k+1, k_prime)])

	# constraint #40
	# child_k_k' - z_k_k' + \sum k^{circ} z_k^{circ} >= 0
	def constraint_child_ge(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["G"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_child_ge_{0}_{1}".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

		# get all indices of z_k^{circ}_k' for all children
		middle_ancestors = [[[] for _ in xrange(self.sublin_num)] for a in xrange(self.sublin_num)]
		for k in xrange(self.sublin_num-1):
			for k_prime in xrange(k+1, self.sublin_num):
				for k_circ in xrange(k+1, k_prime):
					middle_ancestors[k][k_prime].append(self.my_colnames_z_index[k_circ][k_prime])

		self.my_rows.extend([[[self.my_colnames_children_index_friendly_form[k][k_prime],
			self.my_colnames_z_index[k][k_prime]] + middle_ancestors[k][k_prime], 
			[1.0, -1.0] + [1.0] * len(middle_ancestors[k][k_prime])]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraints #41, #42 and #43
	# three constraints to model that the frequency of lineage k' is given if it is a child of k, 0 otherwise
	def constraint_child_frequency(self):
		self.constraint_child_freq_phi()
		self.constraint_child_freq_child()
		self.constraint_child_freq_phi_child()

	# contsraint #41
	# child_k_k'_freq - phi_k' <= 0 for 0 <= k < k' < K
	def constraint_child_freq_phi(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_child_{0}_{1}_freq_phi".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_children_freq_index_friendly_form[k][k_prime],
			self.my_phis_index[k_prime]], [1.0, -1.0]] for k in xrange(self.sublin_num-1)
			for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #42
	# child_k_k'_freq - child_k_k' <= 0 for 0 <= k < k' < K
	def constraint_child_freq_child(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_child_freq_{0}_{1}_child".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_children_freq_index_friendly_form[k][k_prime],
			self.my_colnames_children_index_friendly_form[k][k_prime]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #43
	# child_k_k'_freq - phi_k' - child_k_k' >= -1 for 0 <= k < k' < K
	def constraint_child_freq_phi_child(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["G"] * entries)
		self.my_rhs.extend([-1] * entries)
		self.my_rownames.extend(["constraint_child_freq_{0}_{1}_phi_child".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_children_freq_index_friendly_form[k][k_prime],
			self.my_phis_index[k_prime], self.my_colnames_children_index_friendly_form[k][k_prime]], 
			[1.0, -1.0, -1.0]] 
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
	
	# constraints #44, #45 and #46
	# three constraints to model that the frequency of lineage k is given if it is the parent of k', 0 otherwise
	def constraint_parent_frequency(self):
		self.constraint_parent_frequency_phi()
		self.constraint_parent_frequency_child()
		self.constraint_parent_frequency_phi_child()

	# constraint #44
	# parent_k_k'_freq - phi_k <= 0 for 0 <= k < k' < K
	def constraint_parent_frequency_phi(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_parent_{0}_{1}_frequency_phi".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_parent_freq_index_friendly_form[k][k_prime],
			self.my_phis_index[k]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #45
	# parent_k_k'_freq - child_k_k' <= 0 for 0 <= k < k' < K
	def constraint_parent_frequency_child(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_parent_{0}_{1}_frequency_child".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_parent_freq_index_friendly_form[k][k_prime],
			self.my_colnames_children_index_friendly_form[k][k_prime]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #46
	# parent_k_k'_freq - phi_k - child_k_k' >= -1 for 0 <= k < k' < K
	def constraint_parent_frequency_phi_child(self):
		entries = model.get_number_of_untrivial_z_entries(self.sublin_num) + self.sublin_num - 1
		self.my_sense.extend(["G"] * entries)
		self.my_rhs.extend([-1] * entries)
		self.my_rownames.extend(["constraint_parent_{0}_{1}_frequency_phi_child".format(k, k_prime)
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_parent_freq_index_friendly_form[k][k_prime],
			self.my_phis_index[k], self.my_colnames_children_index_friendly_form[k][k_prime]], 
			[1.0, -1.0, -1.0]]
			for k in xrange(self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraints #47, #48, #49 and #50
	# Four constraints to model that the frequency of lineage k^{bullet} is given when 
	#	it is a sibling of lineage k^{circ} and both are children of k, 0 otherwise
	def constraint_sibling_frequency(self):
		self.constraint_sibling_frequency_child_circ()
		self.constraint_sibling_frequency_child_bullet()
		self.constraint_sibling_frequency_phi()
		self.constraint_sibling_frequency_ge_all()

	# constraint #47
	# sibling_k_k^{circ}_k^{bullet}_freq - child_k_k^{circ} <= 0 
	# for 0 <= k < K - 2, k < k^{circ} < K, k < b^{bullet} < K, k^{circ} != k^{bullet}
	def constraint_sibling_frequency_child_circ(self):
		my_rownames = ["constraint_sibling_{0}_{1}_{2}_frequency_child_circ".format(k, k_circ, k_bullet)
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet]

		entries = len(my_rownames)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(my_rownames)
		self.my_rows.extend([[[self.my_colnames_sibling_freq_index_friendly_form[k][k_circ][k_bullet],
			self.my_colnames_children_index_friendly_form[k][k_circ]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet])

	# constraint #48
	# sibling_k_k^{circ}_k^{bullet}_freq - child_k_k^{bullet} <= 0
	def constraint_sibling_frequency_child_bullet(self):
		my_rownames = ["constraint_sibling_{0}_{1}_{2}_frequency_child_bullet".format(k, k_circ, k_bullet)
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet]

		entries = len(my_rownames)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(my_rownames)
		self.my_rows.extend([[[self.my_colnames_sibling_freq_index_friendly_form[k][k_circ][k_bullet],
			self.my_colnames_children_index_friendly_form[k][k_bullet]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet])

	# constraint #49
	# sibling_k_k^{circ}_k^{bullet}_freq - phi_k^{bullet} <= 0
	def constraint_sibling_frequency_phi(self):
		my_rownames = ["constraint_sibling_{0}_{1}_{2}_frequency_phi".format(k, k_circ, k_bullet)
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet]

		entries = len(my_rownames)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(my_rownames)
		self.my_rows.extend([[[self.my_colnames_sibling_freq_index_friendly_form[k][k_circ][k_bullet],
			self.my_phis_index[k_bullet]], [1.0, -1.0]]
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet])

	# constraint #50
	# sibling_k_k^{circ}_k^{bullet}_freq - phi_k^{bullet} - child_k_k^{circ} - child_k_k^{bullet} >= -2
	def constraint_sibling_frequency_ge_all(self):
		my_rownames = ["constraint_sibling_{0}_{1}_{2}_frequency_ge_all".format(k, k_circ, k_bullet)
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet]

		entries = len(my_rownames)
		self.my_sense.extend(["G"] * entries)
		self.my_rhs.extend([-2] * entries)
		self.my_rownames.extend(my_rownames)
		self.my_rows.extend([[[self.my_colnames_sibling_freq_index_friendly_form[k][k_circ][k_bullet],
			self.my_phis_index[k_bullet], self.my_colnames_children_index_friendly_form[k][k_circ],
			self.my_colnames_children_index_friendly_form[k][k_bullet]], [1.0, -1.0, -1.0, -1.0]]
			for k in xrange(self.sublin_num-2) for k_circ in xrange(k+1, self.sublin_num)
			for k_bullet in xrange(k+1, self.sublin_num) if k_circ != k_bullet])

	# constraints #51 and #52
	# Two constraints to model value of variable chf_m_pf_LDRa_k_k'
	def constraint_chf_m_pf_LDRa(self):
		self.constraint_chf_m_pf_LDRa_lower_bound()
		self.constraint_chf_m_pf_LDRa_upper_bound()

	# constraint #51
	# 0.00001 * LDR_active_k_k' - chf_m_pf_LDRa_k_k' <= 0 for 1 <= k < k' < K
	def constraint_chf_m_pf_LDRa_lower_bound(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_chf_m_pf_LDRa_lower_bound_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_ldr_active_index_friendly_form[k][k_prime],
			self.my_colnames_chf_m_pf_LDRa_index_friendly_form[k][k_prime]], [0.00001, -1.0]]
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #52
	# chf_m_pf_LDRa_k_k' - LDR_active_k_k' <= 0 for 1 <= k < k' < K
	def constraint_chf_m_pf_LDRa_upper_bound(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_chf_m_pf_LDRa_upper_bound_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_chf_m_pf_LDRa_index_friendly_form[k][k_prime],
			self.my_colnames_ldr_active_index_friendly_form[k][k_prime]], [1.0, -1.0]]
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #53
	# The lineage divergence rule is inactive (LDR_inactive_k_k') if it is not active
	# LDR_inactive_k_k' + LDR_active_k_k' = 1
	def constraint_ldr_inactive(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["E"] * entries)
		self.my_rhs.extend([1] * entries)
		self.my_rownames.extend(["constraint_ldr_inactive_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_ldr_inactive_index_friendly_form[k][k_prime],
			self.my_colnames_ldr_active_index_friendly_form[k][k_prime]], [1.0, 1.0]]
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #54
	# Constraint to model the value of chf_m_pf_LDRi_k_k'
	# - LDR_inactive_k_k' - chf_m_pf_LDRi_k_k' <= 0
	def constraint_chf_m_pf_LDRi(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_chf_m_pf_LDRi_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_ldr_inactive_index_friendly_form[k][k_prime],
			self.my_colnames_chf_m_pf_LDRi_index_friendly_form[k][k_prime]], [-1.0, -1.0]]
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraints #55, #56 and # 57
	# lineage divergence rule constraints
	def constraint_ldr_active(self):
		self.constraint_ldr_active_long()
		self.constraint_ldr_active_short()
		self.constraint_ldr_active_child()

	# constraint #55
	# child_freq_minus_par_freq_k_k' - child_k_k'_freq - phi_k 
	# - sum_{k^{bullet} | k^{bullet} < k^{circ}} sum_{k^{circ} | k^{circ} < k} sibling_k^{bullet}_k_k^{circ}_freq
	# - sum_{k* | k*<k} sum_{k'' | k<k''<K} sibling_k*_k_k''_freq
	# + sum_{k* | k* < k} parent_k*_k_freq = 0	for 1 <= k < k' < K 
	def constraint_ldr_active_long(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["E"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_ldr_active_long_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		big_siblings_freq = [[self.my_colnames_sibling_freq_index_friendly_form[k_bullet][k][k_circ] 
			for k_circ in xrange(k) for k_bullet in xrange(k_circ)]
			for k in xrange(1, self.sublin_num-1)] 
		small_siblings_freq = [[[self.my_colnames_sibling_freq_index_friendly_form[k_star][k][k_prime_prime]
			for k_star in xrange(k) for k_prime_prime in xrange(k+1, self.sublin_num)
			if k_prime != k_prime_prime]
			for k_prime in xrange(k+1, self.sublin_num)]
			for k in xrange(1, self.sublin_num-1)]
		parents_freq = [[self.my_colnames_parent_freq_index_friendly_form[k_star][k]
			for k_star in xrange(k)]
			for k in xrange(1,self.sublin_num-1)]
		self.my_rows.extend([[[self.my_colnames_child_freq_minus_par_freq_index_friendly_form[k][k_prime],
			self.my_colnames_children_freq_index_friendly_form[k][k_prime],
			self.my_phis_index[k]] +  
			big_siblings_freq[k-1] +  small_siblings_freq[k-1][k_prime-k-1] + parents_freq[k-1], 
			[1.0, -1.0, -1.0] + [-1] * len(big_siblings_freq[k-1]) 
			+ [-1] * len(small_siblings_freq[k-1][k_prime-k-1]) + [1.0] * len(parents_freq[k-1])]
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #56
	# child_freq_minus_par_freq_k_k' - chf_m_pf_LDRa_k_k' - chf_m_pf_LDRi_k_k' = 0  for 1 <= k < k' < K
	def constraint_ldr_active_short(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["E"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_ldr_active_short_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_child_freq_minus_par_freq_index_friendly_form[k][k_prime],
			self.my_colnames_chf_m_pf_LDRa_index_friendly_form[k][k_prime],
			self.my_colnames_chf_m_pf_LDRi_index_friendly_form[k][k_prime]], [1.0, -1.0, -1.0]]
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #57
	# LDR_active_k_k' - child_k_k' <= 0  for 1 <= k < k' < K
	def constraint_ldr_active_child(self):
		entries =  model.get_number_of_untrivial_z_entries(self.sublin_num)
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_ldr_active_child_{0}_{1}".format(k, k_prime)
			for k in xrange(1,self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])
		self.my_rows.extend([[[self.my_colnames_ldr_active_index_friendly_form[k][k_prime],
			self.my_colnames_children_index_friendly_form[k][k_prime]], [1.0, -1.0]]
			for k in xrange(1, self.sublin_num-1) for k_prime in xrange(k+1, self.sublin_num)])

	# constraint #58
	# sum_k' child_k_k'_freq - phi_k <= 0
	def constraint_ldr(self):
		entries = self.sublin_num - 2
		self.my_sense.extend(["L"] * entries)
		self.my_rhs.extend([0] * entries)
		self.my_rownames.extend(["constraint_ldr_{0}".format(k) for k in xrange(self.sublin_num-2)])
		self.my_rows.extend([[self.my_colnames_children_freq_index_friendly_form[k][k+1:] +
			[self.my_phis_index[k]], [1.0] * (self.sublin_num-k-1) + [-1.0]]
			for k in xrange(self.sublin_num-2)])

	#################################################################################################
	################################# warm start in CPLEX     #######################################

	def create_warm_starts(self, warm_start_dc_binary=None, warm_start_dsnp=None, warm_start_dssm=None,
		warm_start_freqs=None, z_list=None, fixed_z_matrix_indices=None):
		z_matrix = None
		if z_list is not None:
			z_matrix = model.create_z_as_maxtrix_w_values(self.sublin_num, z_list)
		if warm_start_dc_binary is not None:
			logging.debug("CNAs warm start.")
			# try to flatten list, if not possible, list is already flat
			try:
				warm_start_dc_binary = warm_start_dc_binary.flatten().tolist()
			except AttributeError:
				pass
			self.add_values_for_warm_start(warm_start_dc_binary, 
				self.dc_binary_index_start_p1)
			# if Z matrix is given, also values for dc_descendant are used for
			# warm start
			# following part is not needed, doesn't give improvement in terms of
			# speed and memory usage
			#if z_matrix is not None and self.sublin_num > 2:
			#	dc_descendant_values = self.get_values_for_dc_descendant(
			#		warm_start_dc_binary, z_matrix)
			#	self.add_values_for_warm_start(dc_descendant_values,
			#		self.dc_descdendant_start_index)
			#	dc_ancestral_values = self.get_values_for_dc_ancestral(
			#		warm_start_dc_binary, z_matrix)
			#	self.add_values_for_warm_start(dc_ancestral_values,
			#		self.dc_ancestral_start_index)
		if warm_start_dsnp is not None:
			warm_start_dsnp = warm_start_dsnp.flatten().tolist() 
			self.add_values_for_warm_start(warm_start_dsnp, self.dsnp_start_index)
		if warm_start_dssm is not None:
			logging.debug("SSM warm start.")
			# try to flatten list, if not possible, list is already flat
			try:
				warm_start_dssm = warm_start_dssm.flatten().tolist()
			except AttributeError:
				pass
			self.add_values_for_warm_start(warm_start_dssm, self.dssm_start_index)
		if warm_start_freqs is not None:
			logging.debug("Lineage frequency warm start.")
			self.add_values_for_warm_start(warm_start_freqs, self.phi_start_index, freqs=True)
		if fixed_z_matrix_indices is not None:
			logging.debug("Z warm start.")
			self.add_values_for_warm_start(z_list, 0, all_indices=fixed_z_matrix_indices)

	# given values in a matrix and the start index of the variable set, creates two lists that are
	# needed to use the values of the variables as warm start in CPLEX
	def add_values_for_warm_start(self, values, start_index, freqs=False, all_indices=None):
		# prepare for adding
		num_variables = len(values)
		if freqs == True:
			indices = [start_index + i for i in xrange(num_variables)]
		elif all_indices is not None:
			indices = all_indices
		else:
			indices = [x for x in xrange(start_index, start_index + num_variables)]
		effort = self.my_prob.MIP_starts.effort_level.repair

		# adds new MIP start
		if self.my_prob.MIP_starts.get_num() == 0:
			self.my_prob.MIP_starts.add([indices, values], effort)
		else:
			old_indices = self.my_prob.MIP_starts.get_starts()[0][0].ind
			old_values = self.my_prob.MIP_starts.get_starts()[0][0].val
			values = old_values + values
			indices = old_indices + indices
			self.my_prob.MIP_starts.change(0, [indices, values], effort)

	def add_values_for_complete_warm_start(self, values, indices):
		# if nothing should be added
		if values is None:
			return
		# deletes previous warm starts
		self.my_prob.MIP_starts.delete()
		# creates new effort level and adds warm start
		effort = self.my_prob.MIP_starts.effort_level.auto
		self.my_prob.MIP_starts.add([indices, values], self.my_prob.MIP_starts.effort_level.repair)

	# following part is not needed, doesn't give improvement in terms of
	# speed and memory usage
	## computes values for dc_descendant, given the values in the Delta C matrix
	## and in the Z matrix
	#def get_values_for_dc_descendant(self, dc_binary_matrix, z_matrix):
	#	dc_descendant_values = [self.value_dc_descendant_ancestral(dc_binary_matrix[m][i][k_prime],
	#		z_matrix[k][k_prime]) for m in range(cons.PHASE_NUMBER * self.cnv_state_num)
	#		for i in xrange(self.seg_num) for k in range(1, self.sublin_num -1)
	#		for k_prime in range(k+1, self.sublin_num)]
	#	return dc_descendant_values

	#def get_values_for_dc_ancestral(self, dc_binary_matrix, z_matrix):
	#	dc_ancestral_values = [self.value_dc_descendant_ancestral(dc_binary_matrix[m][i][k],
	#		z_matrix[k][k_prime]) for m in range(self.cnv_state_num, 
	#		cons.PHASE_NUMBER * self.cnv_state_num) for i in xrange(self.seg_num)
	#		for k_prime in range(2, self.sublin_num)
	#		for k in range(1, k_prime)]
	#	return dc_ancestral_values

	## constraints #11.5, #11.6, #11.7 as a function
	## also constraints #11.8, #11.91, #11.91 as a function
	#def value_dc_descendant_ancestral(self, dc_binary, z):
	#	if dc_binary == 0:
	#		return 0
	#	if z == 0:
	#		return 0
	#	return 1


	################################# warm start in CPLEX     #######################################
	#################################################################################################

	def create_2d_index_array(self, start_index, dim1, dim2):
		return np.arange(start_index, start_index + dim1 * dim2).reshape(dim1, dim2)

	def create_3d_index_array(self, start_index, dim1, dim2, dim3):
		return np.arange(start_index, start_index + dim1 * dim2 * dim3).reshape(dim1, dim2, dim3)
