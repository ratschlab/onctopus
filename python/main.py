#!/usr/bin/env python

import log_pdf
import optimization
import model
import sys
import snp_ssm
import segment
import lineage
import onctopus_io as oio
import constants as cons
import time as time_module
import exceptions_onctopus as eo
import argparse
import logging
import constants as cons
import clustered_mutation_optimization as cluster_mod
import numpy as np
import copy
import os
import data_simulation
import random
import cPickle
from cplex.exceptions import CplexError

# Function that reads a segment and an SSM file and writes all the SSMs belonging to segments without CN changes
# to a new file. While doing so, their chromosome and position information is changed. Also, a new segment file
# is written that consists only of one normal segment.
def create_ssm_seg_file_as_of_one_normal_seg(segment_file, ssm_file, normal_seg_indices_file,
	output_segment_file, output_ssm_file):
	# read segment and ssm file
	(seg_list, ssm_list) = model.create_segment_and_mutation_lists(segment_file, 
		None, ssm_file, True)

	# read normal segment file
	normal_seg_indices = []
	with open(normal_seg_indices_file) as f:
		for line in f:
			normal_seg_indices = map(int, line.rstrip("\n").split("\t"))

	# get indices of SSMs of normal segments
	ssm_indices_of_segment = model.choose_ssm_subset(normal_seg_indices, ssm_list)

	# change chromosome and position of SSMs and write to file
	oio.raise_if_file_exists(output_ssm_file)
	with open(output_ssm_file, "w") as f:
		for i in xrange(len(ssm_indices_of_segment)):
			ssm_index = ssm_indices_of_segment[i]
			f.write("{0}\t{1}\t{2}\t{3}\n".format(1, i, ssm_list[ssm_index].variant_count, 
				ssm_list[ssm_index].ref_count))
		
	# create new segment and write to file
	oio.raise_if_file_exists(output_segment_file)
	with open(output_segment_file, "w") as f:
		f.write("1\t0\t{0}\t1.0\t0.1\t1.0\t0.1".format(len(ssm_indices_of_segment)))

# reads a segment file and computes the standard error of the CN estimates based on the length of the segments
# segment_file: file with segments
# ssm_file: file with SSMs, are needed if average read count is not given
def compute_standard_error_of_segments(segment_file, ssm_file, median_read_count=-1):
	# read files
	(seg_list, ssm_list) = model.create_segment_and_mutation_lists(segment_file, None, ssm_file, True)
	ssm_count = model.count_number_of_ssms_per_segment(seg_list, ssm_list)

	# if average read count is not given, it is calculated from the read count of all SSMs
	if median_read_count == -1:
		median_read_count = np.median([ssm.variant_count + ssm.ref_count for ssm in ssm_list])
	
	compute_standard_error_of_segments_without_files(seg_list, median_read_count, ssm_count)

	# write segments with new standard error to file again
	new_seg_file = "{0}_new_standard_errors".format(segment_file)
	oio.raise_if_file_exists(new_seg_file)
	oio.write_segment_file_allele_specific_from_segments(seg_list, new_seg_file)

# computes the standard error of CN estimates for segments
def compute_standard_error_of_segments_without_files(seg_list, median_read_count, ssm_count):
	# compute standrad error per segment
	overdispersion = 1000
	# create data simulation object that can calculate the standard error
	my_ds = data_simulation.Data_Simulation()
	my_ds.noise = True
	my_ds.mass = int(round(median_read_count / 2))
	my_ds.overdispersion = True
	my_ds.coverage_overdispersion = overdispersion

	for i, seg in enumerate(seg_list):
		# compute heterozygous SNP number
		standard_max_num = 3000
		# get maximal number of SNPs, at least as high as number of SSMs per segment
		max_snp_num = max(standard_max_num, ssm_count[i])
		seg_length = seg.end - seg.start + 1
		het_snp_num = min(max(1,int(round(7.0/10000.0 * seg_length))), max_snp_num)
		# compute standard error for A
		seg.standard_error_A = my_ds.compute_cn_standrad_error_as(seg.given_cn_A, het_snp_num, [])
		seg.standard_error_B = my_ds.compute_cn_standrad_error_as(seg.given_cn_B, het_snp_num, [])

# heuristic that optimizes each segment separately and then combines the Z-matrices of
# all optimization to finally optimize segmentwise with fixed Z-matrix
#TODO deal with Z-matrix ambiguity and transitivity in Z-matrix, see notes for this!
def heuristic_segmentwise(segment_file=None, ssm_file=None, lineage_number=None,
	output_file=None,
	time=None, threads=None, probing=None, emph_switch=None, coef_reduc=None, 
	mipgap=None, symmetry=None, strategy_file=None, workmem=None, workdir=None, 
	treememory=None, emphasis_memory=None, nodeselect=None,
	cplex_log_file=None, number_spline_points=None,
	phi_file=None, fixed_phi_list=[],
	unfixed_phi_start=-1, unfixed_phi_stop=-1, seg_list=[], ssm_list=[],
	simple_CN_changes=None, max_x_CN_changes=None, only_one_loss=None, 
	only_gains_losses_LOH=None, cn_weight=None, z_trans_weight=None, kmeans=True, agglo_ward=False):
	
	oio.raise_if_file_exists(output_file)

	# if file with fixed phi values is given, file is read
	if phi_file:
		(fixed_phi_list, unfixed_phi_start, unfixed_phi_stop) = oio.read_fixed_value_file(
			fixed_phi_file)

	# if segment and SSM files are given, read them
	if segment_file and ssm_file:
		(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg, 
			None, input_ssm, allele_specific=True)

	# divide segment and SSM lists segmentwise
	seg_indices_list = [i for i in xrange(len(seg_list))]
	ssm_indices_per_segment = model.choose_ssm_subset(seg_indices_list, ssm_list, normal_segments=False)
	ssms_segmentwise = [[] for i in xrange(len(seg_list))]
	ssms_segmentwise = [ssms_segmentwise[i].append(ssm_list[j]) for i in xrange(len(seg_list))
		for j in xrange(len(ssm_indices_per_segment[i]))]

	# call onctopus optimization for each segment separately
	# here, the Z-matrix is part of the optimization
	lineages_segwise = [go_onctopus(None, None, None, lineage_number, None, time,
		threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
		strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect, 
		cplex_log_file, number_spline_points,
		unfixed_phi_start=unfixed_phi_start, unfixed_phi_stop=unfixed_phi_stop,
		fixed_phi_list=fixed_phi_list, allele_specific=True, 
		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
		cn_weight=cn_weight, z_trans_weight=z_trans_weight, seg_list=seg_list, ssm_list=ssm_list,
		write_output_to_disk=Falsei, kmeans=kmeans, agglo_ward=agglo_ward, take_time=False)[0]]

	# TODO
	# do something to find a combined Z-matrix
	# format of Z-matrix: list of lists, first line has a list for lineage 1 (first not normal lineage),
	#	second for lineage 2, and so forth
	# each list has one entry less then the list before
	z_matrix = None

	# call onctopus optimization again for each segment separately
	# this time the Z-matrix is fixed
	lineages_segwise_fixed_Z = [go_onctopus(None, None, None, lineage_number, None, time,
		threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
		strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect, 
		cplex_log_file, number_spline_points,
		unfixed_phi_start=unfixed_phi_start, unfixed_phi_stop=unfixed_phi_stop,
		fixed_phi_list=fixed_phi_list, allele_specific=True, 
		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
		cn_weight=cn_weight, z_trans_weight=z_trans_weight, seg_list=seg_list, ssm_list=ssm_list,
		fixed_z_matrix_file=z_matrix, write_output_to_disk=False, kmeans=kmeans, agglo_ward=agglo_ward,
		take_time=False)[0]]

	# TODO
	# combine segmentwise lineage to one lineage
	my_lineages = None

	# TODO for Linda for later
	# compute BIC
	bic = None

	# write combined lineage to file
	oio.write_lineages_to_result_file(output_file, my_lineages)


# heuristic, where phi and Z matrix are given and each segment is purely segment-wise
#	optimized
# ssm_list_of_segments: is list of list with SSMs, are grouped in sublists based on to which
# 	segment they belong
# tested via test_clustered_mutation_optimization.py
#TODO deal with Z-matrix ambiguity and transitivity in Z-matrix, see notes for this!
#TODO deal with lineage divergence rule
def heuristic_fixed_phi_pure_segmentwise(segment_list, ssm_list_of_segments, sublin_num,
	fixed_phis, fixed_z, threads, number_spline_points,
	cplex_log_file, allele_specific=True, simple_CN_changes=False, max_x_CN_changes=2,
	only_one_loss=False, only_gains_losses_LOH=True, cn_weight=0.0, z_trans_weight=0.00001,
	ssm_indices_per_cluster=None):

	if not allele_specific:
		raise eo.MyException("not done for non-allele-specific")

	cplex_obj_list = []

	# create list of direct descendants of lineages
	z_matrix = []
	direct_descendant = []
	direct_descendants_for_constraints = []
	if fixed_z is not None:
		# triangle shape, last row missing
		z_matrix = model.parse_fixed_z_matrix_list_to_matrix(fixed_z, sublin_num)
		direct_descendant = model.get_direct_descendants(z_matrix, sublin_num)
		direct_descendants_for_constraints = model.parse_direct_descendants_for_constraints(
			direct_descendant)

		# check whether fixed Z matrix can work with fixed phis, if not, raise error
		if not model.z_matrix_phis_feasible(direct_descendant, fixed_phis):
			raise eo.ZMatrixPhisInfeasibleException(
				"Z matrix {0} is infeasible with fixed phis"
				"{1}".format(fixed_z, fixed_phis))

	for i in xrange(len(segment_list)):
		# create splines
		(seg_spline_A, seg_spline_B, ssm_splines) = model.create_segment_and_mutation_splines(
			[segment_list[i]], [], ssm_list_of_segments[i], number_spline_points,
			allele_specific=True)

		# do optmization
		current_ssm_indices_per_cluster = None
		if ssm_indices_per_cluster is not None:
			current_ssm_indices_per_cluster = [ssm_indices_per_cluster[i]]
		cplex_obj = optimization.Optimization_with_CPLEX([], [], ssm_splines, 
			allele_specific=allele_specific, seg_splines_A=seg_spline_A, 
			seg_splines_B=seg_spline_B, simple_CN_changes=simple_CN_changes, 
			max_x_CN_changes=max_x_CN_changes, only_one_loss=only_one_loss,
			only_gains_losses_LOH=only_gains_losses_LOH, cn_weight=cn_weight,
			z_trans_weight=z_trans_weight, single_segment=True)
		# set up parameters
		cplex_obj.threads = threads
		cplex_obj.cplex_log_file = cplex_log_file
		# start optimization
		cplex_obj.opt_with_CPLEX(sublin_num, [], ssm_list_of_segments[i], [segment_list[i]],
			fixed_z_matrix=fixed_z, fixed_phi=fixed_phis, 
			direct_descendants_for_constraints=direct_descendants_for_constraints,
			ssm_indices_per_cluster=current_ssm_indices_per_cluster)

		cplex_obj_list.append(cplex_obj)

	return cplex_obj_list

# starts the iterative heuristic by letting the optimization run once for all segments 
# for a time initial_time and then
# taking this result as a start for the iterative rounds
# TODO: introduce shorten of lineages (when frequency is 0) at right position!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def start_iterative_heuristic_all_in(input_seg, input_ssm, sublin_num, out_results,
	initial_time,
	seg_time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry, strategy_file, 
	workmem, workdir, treememory, emphasis_memory, nodeselect, number_spline_points, 
	max_rounds, epsilon, cplex_log_file, numeric_logging_info,
	test_run=False, write_output_to_disk=True,
	simple_CN_changes=True, max_x_CN_changes=-1, only_one_loss=True, only_gains_losses_LOH=True,
	cn_weight=0, z_trans_weight=0.00001, lineage_divergence_rule=True, normal_seg_indices=[],
	fixed_phi_file=None, cluster_SSM=False, kmeans=True, agglo_ward=False, cluster_num_param=-1,
	review_ambiguous_relations=True, dont_break_z_symmetry=False, overdispersion_parameter=1000):
	#TODO remove only_one_loss and only_gains_losses_LOH --> use some default values

	# creating file names
	out_results_initial = out_results + "_initial"
	out_results_heuristic = out_results + "_heuristic"
	# check if output file exists
	if not test_run and write_output_to_disk:
		oio.raise_if_file_exists(out_results_initial) 
		oio.raise_if_file_exists(out_results_heuristic) 

	# set parameters for clustering in go_onctopus
	if cluster_SSM:
		use_super_SSMs = True
	else:
		use_super_SSMs = False
	# first run to get start solution, this is needed to fix variables with first results
	# ambiguous relations are not reviewed because optimization is not done after first run
	returned_values = go_onctopus(input_seg, None, input_ssm,
		sublin_num, out_results_initial, initial_time, 
		threads, probing, emph_switch, coef_reduc, mipgap, symmetry, 
		strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect,
		cplex_log_file, number_spline_points, 
		test_run=test_run, write_output_to_disk=write_output_to_disk,
		allele_specific=True,
		simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
		only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
		cn_weight=cn_weight, z_trans_weight=z_trans_weight, 
		lineage_divergence_rule=lineage_divergence_rule,
		normal_seg_indices=normal_seg_indices, review_ambiguous_relations=False, z_matrix_list=None,
		heuristic1=True, fixed_phi_file=fixed_phi_file, cluster_SSM=cluster_SSM,
		use_super_SSMs=use_super_SSMs, kmeans=kmeans, agglo_ward=agglo_ward, cluster_num_param=cluster_num_param,
		dont_break_z_symmetry=dont_break_z_symmetry, take_time=False)
	# unpack returned values depending on whether SSMs where clustered or not	
	superSSMs = None
	if cluster_SSM == False:
		initial_lineages, cplex_obj, seg_list, ssm_list, seg_splines_A, seg_splines_B, ssm_splines = returned_values
	else:
		(initial_lineages, cplex_obj, seg_list, ssm_list, seg_splines_A, seg_splines_B, ssm_splines, superSSMs,
			superSSM_hash) = returned_values
		
	initial_objective = cplex_obj.my_prob.solution.get_objective_value()

	# if optimal result was found after normal optimization, segmentwise approach is not needed!
	cplex_status = cplex_obj.my_prob.solution.status[cplex_obj.my_prob.solution.get_status()]
	if ((cplex_status == "MIP_optimal" or cplex_status == "optimal_tolerance") and test_run == False):
		logging.info("Optimal result is found. Iteration over segments is not needed.")
		end_lineages = initial_lineages
		end_objective = initial_objective
		used_rounds = 0
	else:
		# call heuristic
		(lineages_after_heuristic, used_rounds, heuristic_objective) = (
			iterative_heuristic_all_in(sublin_num, seg_list, ssm_list, cplex_obj,
			seg_splines_A, seg_splines_B, ssm_splines,
			seg_time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry, strategy_file,
			workmem, workdir, treememory, emphasis_memory, nodeselect, number_spline_points,
			max_rounds, epsilon, numeric_logging_info, cplex_log_file, initial_objective,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cn_weight=cn_weight, z_trans_weight=z_trans_weight, 
			normal_seg_indices=normal_seg_indices, superSSMs=superSSMs,
			dont_break_z_symmetry=dont_break_z_symmetry))

		logging.info("Heuristic done after {0} rounds (here count from 1, otherwise 0). "
			"Objective of heuristic is {1}, compared to {2} of initial run.".
			format(used_rounds, heuristic_objective, initial_objective))

		end_lineages = lineages_after_heuristic
		end_objective = heuristic_objective

	# if SSMs were clustered, the reconstruction is done with fixed variables and declustered SSMs
	if cluster_SSM == True:
		out_results_declustered = out_results + "_declustered"
		if cplex_log_file is not None:
			cplex_log_file_pref = "".join(cplex_log_file.split(".")[0:-2])
			cplex_log_file_declustered = cplex_log_file_prefix + "_declustered.cplex.log"
		else: 
			cplex_log_file_declustered = None
		(my_lineages, cplex_obj, rec_wo_super_success) = reconstruct_with_non_clustered_SSMs(cplex_obj, 
			None, superSSMs, seg_list, superSSM_hash, out_results_declustered,
			sublin_num, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
			strategy_file, workmem, workdir, emphasis_memory, nodeselect,
			cplex_log_file_declustered, number_spline_points, allele_specific=True,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cn_weight=cn_weight, z_trans_weight=z_trans_weight,
			ssm_list=ssm_list, lineage_divergence_rule=lineage_divergence_rule)

		end_lineages = my_lineages
		end_objective = cplex_obj.my_prob.solution.get_objective_value()
		logging.info("Objective on declustered SSMs is {0}".format(end_objective))

	# compute LLH
	llh = log_pdf.compute_llh(end_lineages, ssm_list, seg_list, overdispersion_parameter)
	logging.info("LLH: {0}".format(llh))
	# compute and print BIC
	#bic = model.get_BIC(len(seg_list), sublin_num, len(ssm_list), 0, True,
	#	end_objective)
	#logging.info("BIC: {0}".format(bic))

	# if ambiguous relations should be reviewed
	if review_ambiguous_relations == True:
		z_matrix_list, new_lineage_list, lin_div_rule_feasibility = review_ambiguous_relations_function(
			my_lineages=end_lineages, z_matrix_list=None, seg_list=seg_list, 
			write_output_to_disk=write_output_to_disk, out_results=out_results_heuristic, test_run=test_run, 
			new_lineage_list=None, lin_div_rule_feasibility=None)

		return (end_lineages, used_rounds, end_objective, initial_lineages,
			initial_objective, z_matrix_list, new_lineage_list, lin_div_rule_feasibility, cluster_SSM)

	else:
		return (end_lineages, used_rounds, end_objective, initial_lineages, initial_objective, cluster_SSM)

# all variables except the ones in one segment are fixed
# optimization is done in several rounds through all segments
# optimization stops after a specific number of rounds
# or when objective doesn't improve anymore
def iterative_heuristic_all_in(sublin_nums, seg_list, ssm_list, cplex_obj,
	seg_splines_A, seg_splines_B, ssm_splines,
	time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry, strategy_file, 
	workmem, workdir, treememory, emphasis_memory, nodeselect, number_spline_points, 
	max_rounds, epsilon, numeric_logging_info, cplex_log_file, prev_objective,
	simple_CN_changes=True, max_x_CN_changes=-1, only_one_loss=True,
	only_gains_losses_LOH=True, cn_weight=0, z_trans_weight=0.00001, 
	normal_seg_indices=[], superSSMs=None, dont_break_z_symmetry=False):

	# get number of segments
	seg_num = len(seg_list)
	# depending on whether superSSMs are used, chose the right list with SSMs
	if superSSMs is None:
		ssm_list_to_use = ssm_list
	else:
		ssm_list_to_use = superSSMs
	# for each segment, get indices of SSMs that appear on it
	ssms_indices_on_seg = model.get_mutation_index_list_for_segments(ssm_list_to_use, seg_num)
	# get indices of variables in optimization
	variable_indices = [i for i in xrange(len(cplex_obj.my_colnames))]

	# set up parameters for segment-wise run
	cplex_obj.set_CPLEX_parameters(time, threads, probing, emph_switch, coef_reduc, 
		mipgap, symmetry, strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect,
		cplex_log_file)

	# cplex log file prefix
	if cplex_log_file is not None:
		cplex_log_file_prefix = "".join(cplex_log_file.split(".")[0:-2])
		if not os.path.exists(cplex_log_file_prefix):
			os.makedirs(cplex_log_file_prefix)
	else:
		cplex_log_file_prefix = None
	
	# optimize in rounds for each segment
	for round_number in xrange(max_rounds):
		objective = 0
		for seg_index in xrange(seg_num+1):
			if seg_index < seg_num:
				logging.info("Optimization in round {0} for segment {1}".format(round_number,
					seg_index))
			else:
				logging.info("Optimization in round {0} for all segments, unfixed".format(round_number))
			
			# new log file
			if cplex_log_file_prefix is not None:
				new_cplex_log_file = "{0}/{1}_{2}.cplex.log".format(cplex_log_file_prefix, 
					round_number, seg_index)
				cplex_obj.cplex_log_file = new_cplex_log_file

			# get result of last run
			# I only need CNVs and SSMs here, lineage relationships are not important, that's why
			# I don't need to check for transitivity and I don't need to break the Z symmetry
			my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, None, 
				ssm_list_to_use, seg_list, check_z_transitivity=False,
				dont_break_z_symmetry=True)
			# for each segment, fix CNVs and SSMs
			# for last iteration of round don't fix any variables but optimize over all variables
			if seg_index < seg_num:
				# fix CNVs
				fixed_cnvs = model.create_fixed_CNV_data_new(my_lineages, seg_num)
				# remove CNV of current segment, unless they are fixed to be normal
				if seg_index not in normal_seg_indices:
					del fixed_cnvs[seg_index]
				# fix SSMs
				fixed_ssms = model.create_fixed_SSM_data_new(my_lineages)
				# remove SSMs of current segment
				try:
					del fixed_ssms[ssms_indices_on_seg[seg_index][0]:ssms_indices_on_seg[seg_index][-1]+1]
				# segment contains no SSMs
				except IndexError:
					pass
				# create specific fixed constraints
				cplex_obj.constraint_fix_dc_binary_or_SSMs(fixed_cnvs, cons.CNV)
				cplex_obj.constraint_fix_dc_binary_or_SSMs(fixed_ssms, cons.SSM)

			# no logging
			logging.basicConfig(level=logging.NOTSET)

			# start optimization with warm start
			variable_results = cplex_obj.my_prob.solution.get_values()
			try:
				cplex_obj.start_CPLEX(warm_start_values=variable_results, warm_start_indices=variable_indices)
			except CplexError, exc:
				oio.write_lineages_to_result_file("{0}/{1}_{2}_prev_result".format(cplex_log_file_prefix, 
					round_number, seg_index),
					my_lineages, test=True)
				logging.warning("No solution could be found for segment {0} in round {1}. "
					"Now doubling time and trying again.".format(seg_index, round_number))
				# new log file name
				cplex_obj.cplex_log_file = "{0}/{1}_{2}_try_again.cplex.log".format(cplex_log_file_prefix,
					round_number, seg_index)
				# do again with doubled time
				cplex_obj.time = 2 * time
				cplex_obj.start_CPLEX(warm_start_values=variable_results, warm_start_indices=variable_indices)
				# set time back to usual time
				cplex_obj.time = time
			objective = cplex_obj.my_prob.solution.get_objective_value()

			# only when variables were fixed the constraints need to be removed afterwards
			if seg_index < seg_num:
				# remove fixed constraints
				# remove constraints for SSMs first because they have higher indices
				# and thus don't change the indices of the other constraints
				cplex_obj.remove_specific_constraints(cplex_obj.start_index_constraint_fix_dssm,
					cplex_obj.end_index_constraint_fix_dssm)
				cplex_obj.remove_specific_constraints(cplex_obj.start_index_constraint_fix_dc_binary,
					cplex_obj.end_index_constraint_fix_dc_binary)
			# last iteration of round, where no variables are fixed
			else:
				if cplex_log_file_prefix is not None:
					# current result is written to file
					my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, None,
						ssm_list_to_use, seg_list, check_z_transitivity=True,
						dont_break_z_symmetry=dont_break_z_symmetry)
					oio.write_lineages_to_result_file("{0}/{1}_{2}_result".format(cplex_log_file_prefix, 
						round_number, seg_index),
						my_lineages, test=True)

		# set logging level back to standard
		logging.basicConfig(level=numeric_logging_info)
		logging.info("round:{0},segment:{1},objective:{2}".format(round_number,
			seg_index, objective))

		# check for unfixed round whether CPLEX status is optimal or near optimal
		cplex_status = cplex_obj.my_prob.solution.status[cplex_obj.my_prob.solution.get_status()]
		if (cplex_status == "MIP_optimal" or cplex_status == "optimal_tolerance"):
			logging.info("Optimal result is found.")
			break
				

		# objective of this round is close enough to objective of previous round
		if abs(prev_objective - objective) <= epsilon:
			logging.info("Objective after {0} rounds doesn't improve more than {1}. (Here count from 1, "
				"otherwise 0.)".format(
				round_number+1, epsilon))
			logging.info("Stop optimization.")
			break

		# set prev_objective to current objective
		prev_objective = objective

	used_rounds = round_number + 1
	logging.info("Optimization done after {0} rounds. (Here count from 1, otherwise 0.)".format(round_number+1))

	return (my_lineages, used_rounds, objective)

# after superSSMs were used, reconstruction with non-clustered SSMs is computed
# for this, all variables expect the lineage frequencies are fixed
def reconstruct_with_non_clustered_SSMs(cplex_obj, snp_list, superSSMs, seg_list, superSSM_hash, out_results,
	num, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
	strategy_file, workmem, workdir, emphasis_memory, nodeselect, cplex_log_file, number_spline_points,
	fixed_cnv_file=None, fixed_avg_cn_file=None, fixed_snp_file=None, fixed_ssm_file=None,
	fixed_z_matrix_file=None, fixed_phi_file=None, test_run=False, fixed_phi_list=None, unfixed_phi_start=-1,
	unfixed_phi_stop=-1, fixed_avg_cn_list=None, unfixed_avg_cn_start=-1, unfixed_avg_cn_stop=-1,
	warm_start_dc_binary=None, warm_start_dsnp=None, warm_start_dssm=None, warm_start_freqs=None,
	allele_specific=True,
	simple_CN_changes=True, max_x_CN_changes=-1, only_one_loss=False, only_gains_losses_LOH=True,
	cn_weight=0, z_trans_weight=0.00001, ssm_list=None, lineage_divergence_rule=True,
	dont_break_z_symmetry=False, take_time=False):

	rec_wo_super_success = False
	my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, snp_list, superSSMs, 
		seg_list, dont_break_z_symmetry=dont_break_z_symmetry)
	# transform superSSMs in lineages to individual SSMs
	model.replace_superSSMs_in_lineages(my_lineages, superSSM_hash)
	# create fixed files, write to disk
	tmp_name = oio.create_temp_rand_name(prefix=out_results)
	fixed_cnv_file = "{0}.{1}".format(tmp_name, cons.CNV)
	fixed_ssm_file = "{0}.{1}".format(tmp_name, cons.SSM)
	fixed_z_matrix_file = "{0}.{1}".format(tmp_name, cons.Z)
	# create other cplex log file
	if cplex_log_file is not None:
		new_cplex_log_file = "{0}.no_superSSMS.cplex.log".format(".".join(cplex_log_file.split(".")[0:-2]))
	else:
		new_cplex_log_file = None
	#TODO use new fixed entries, then temporary file don't have to be written
	oio.create_some_fixed_files(my_lineages, fixed_cnv_file=fixed_cnv_file, 
		fixed_ssm_file=fixed_ssm_file, fixed_z_matrix_file=fixed_z_matrix_file)
	# call onctopus again with fixed files, overwrite corresponding variables
	logging.info("Call Onctopus with fixed values expect lineage frequencies.")
	fixed_run_time = 200
	fixed_run_treememory = 1000
	try:
		(my_lineages, cplex_obj) = go_onctopus(None, None, None, num, None, 
			fixed_run_time,
			threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
			strategy_file, workmem, workdir, fixed_run_treememory, emphasis_memory, nodeselect,
			new_cplex_log_file, number_spline_points,
			fixed_cnv_file=fixed_cnv_file, fixed_avg_cn_file=fixed_avg_cn_file,
			fixed_snp_file=fixed_snp_file, fixed_ssm_file=fixed_ssm_file,
			fixed_z_matrix_file=fixed_z_matrix_file, 
			fixed_phi_file=fixed_phi_file, test_run=test_run, write_output_to_disk=False,
			fixed_phi_list=fixed_phi_list, unfixed_phi_start=unfixed_phi_start,
			unfixed_phi_stop=unfixed_phi_stop, fixed_avg_cn_list=fixed_avg_cn_list,
			unfixed_avg_cn_start=unfixed_avg_cn_start, unfixed_avg_cn_stop=unfixed_avg_cn_stop,
			warm_start_dc_binary=warm_start_dc_binary, warm_start_dsnp=warm_start_dsnp,
			warm_start_dssm=warm_start_dssm, warm_start_freqs=warm_start_freqs,
			allele_specific=allele_specific,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cn_weight=cn_weight, z_trans_weight=z_trans_weight, cluster_SSM=False,
			seg_list=seg_list, ssm_list=ssm_list, use_super_SSMs=False,
			lineage_divergence_rule=lineage_divergence_rule, review_ambiguous_relations=False,
			take_time=take_time)
		logging.debug("reconstruct_with_non_clustered_SSMs: successfull")
		logging.info("Finished Onctopus with fixed values.")
		rec_wo_super_success = True
	except CplexError, exc:
		# optimization with fixed values was not possible
		logging.debug("reconstruct_with_non_clustered_SSMs: failed")
		# assign lineage indices to SSMs
		model.assign_lin_index_to_ssms(my_lineages)
		# write cplex_obj as pickle to file
		#pickle_file = "{0}.cplex_obj.pkl"".".join(cplex_log_file.split(".")[0:-2])
		#with open(pickle_file, "wb") as f:
		#	cPickle.dump(cplex_obj, f, -1)
		# check why optimization was not possible
		if new_cplex_log_file is not None:
			infeasible_status = False
			# log file is read to get answer why optimization didn't work
			with open(new_cplex_log_file, "r") as f:
				for line in f:
					if "infeasible" in line:
						infeasible_status = True
						break

			# reason for failure is printed
			if infeasible_status == True:
				logging.info("Optimization with fixed values is infeasible.")
			else:
				logging.info("Optimization with fixed values not possible. Probably because of time limit. "
					"Check file {0}.".format(new_cplex_log_file))
		# if no log file exists, reason for failure can't be found
		else:
			logging.info("Optimization with fixed values not possible.")

		logging.info("Reconstruction contains declustered SSMs and frequency values computed from first optimization.")

	# removed fixed files
	oio.remove_some_fixed_files(fixed_cnv_file=fixed_cnv_file, fixed_ssm_file=fixed_ssm_file,
		fixed_z_matrix_file=fixed_z_matrix_file)

	return my_lineages, cplex_obj, rec_wo_super_success

# main function that reads in input from file and does all the rest
# z_matrix_list: stores different Z-matrices after review of ambiguous relations
def go_onctopus(input_seg, input_snp, input_ssm, num, out_results, time,
	threads, probing, emph_switch, coef_reduc, mipgap, symmetry, 
	strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect,
	cplex_log_file, number_spline_points,
	fixed_cnv_file=None, fixed_avg_cn_file=None, fixed_snp_file=None, fixed_ssm_file=None, 
	fixed_z_matrix_file=None, unfixed_z_matrix_start=-1, unfixed_z_matrix_stop=-1,
	fixed_z_matrix_list=None,
	fixed_phi_file=None, test_run=False, write_output_to_disk=True,
	fixed_phi_list=None, unfixed_phi_start=-1, unfixed_phi_stop=-1,
	fixed_avg_cn_list=None, unfixed_avg_cn_start=-1, unfixed_avg_cn_stop=-1,
	warm_start_dc_binary=None, warm_start_dsnp=None, warm_start_dssm=None, warm_start_freqs=None,
	fixed_cnv_list=None, unfixed_cnv_start=-1, unfixed_cnv_stop=-1,
	fixed_snp_list=None, unfixed_snp_start=-1, unfixed_snp_stop=-1,
	fixed_ssm_list=None, unfixed_ssm_start=-1, unfixed_ssm_stop=-1,
	fixed_cnv_list_new = None, fixed_ssm_list_new = None,
	allele_specific=False,
	simple_CN_changes=True, max_x_CN_changes=-1, only_one_loss=True, only_gains_losses_LOH=True,
	cn_weight=0, z_trans_weight=0.00001, cluster_SSM=False,
	seg_list=[], ssm_list=[], use_super_SSMs=False, lineage_divergence_rule=True,
	normal_seg_indices=[], combine_normal_segments=False,
	review_ambiguous_relations=True, z_matrix_list=None, heuristic1=False,
	kmeans=True, agglo_ward=False, cluster_num_param=-1, dont_break_z_symmetry=False, take_time=True,
	overdispersion_parameter=1000,
	warm_start_z_matrix_file=None, warm_start_z_matrix_list=None,
	warm_start_dc_binary_file=None, warm_start_dssm_file=None, warm_start_freqs_file=None,
	warm_start_solution=None, start_freq=None, start_Z=None, start_SSMs=None, start_CNVs=None):

	# parameter checking
	if fixed_cnv_file is not None and normal_seg_indices != []:
		raise eo.ParameterException("Parameters fixed_cnv_file and normal_seg_indices_file are not allowed to "
			"set together in one run.")
	if fixed_cnv_list is not None and normal_seg_indices != []:
		raise eo.ParameterException("Parameters fixed_cnv_list and normal_seg_indices_file are not allowed to "
			"set together in one run.")
	
	# start with logging
	logging.info("Starting Onctopus...")

	# check if output file exists
	if not test_run and write_output_to_disk:
		oio.raise_if_file_exists(out_results) 
	
	# define variables
	sublin_nums = num
	cplex_obj_list = []
	new_lineage_list = None
	lin_div_rule_feasibility = None

	# create segment and mutation lists and splines	
	seg_splines = seg_splines_A = seg_splines_B = snp_list = snp_splines = []
	ssm_splines = []
	if allele_specific:
		if seg_list == [] and ssm_list == []:
			(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg, 
				input_snp, input_ssm, True)
		# if normal segments should be combined to a large one
		if combine_normal_segments and len(normal_seg_indices) > 1:
			logging.debug("Normal segments get combined to one segment.")
			# continue working with seg_list, ssm_list which are actually changed
			(seg_list, ssm_list, normal_seg_indices, original_seg_list, original_ssm_list, 
				ssm_normal_changed, ssm_normal, original_normal_seg_indices, original_seg_index) = (
				model.combine_normal_segments_and_ssms(seg_list, ssm_list, normal_seg_indices))
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, snp_list, ssm_list, number_spline_points, True)
	else:
		(seg_list, snp_list, ssm_list) = (model.create_segment_and_mutation_lists(
			input_seg, input_snp, input_ssm, False))
		(seg_splines, snp_splines, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, snp_list, ssm_list, number_spline_points, False)
	
	######
	# fixation

	# if fixed files for variables are given, read them
	direct_descendants_for_constraints = []

	# CNVs are fixed here
	# list that already contains the fixed cnvs is not given
	if fixed_cnv_list is None:
		# if file with fixation information is given, it is used
		if fixed_cnv_file:
			if combine_normal_segments == True:
				eo.MyException("Fixation of CNAs and combination of normal segments not implemented yet.")
			(fixed_cnv_list, unfixed_cnv_start, unfixed_cnv_stop) = oio.read_fixed_value_file(
				fixed_cnv_file)
	# both list and file for CNV fixation are given, information in file is used
	elif fixed_cnv_file:
		if combine_normal_segments == True:
			eo.MyException("Fixation of CNAs and combination of normal segments not implemented yet.")
		logging.warning("Both list with values and file with values for fixation of " +
			"CNVs exist. Using file now.")
		(fixed_cnv_list, unfixed_cnv_start, unfixed_cnv_stop) = oio.read_fixed_value_file(
			fixed_cnv_file)
	# only fixed_cnv_list list is given
	else:
		if combine_normal_segments == True:
			eo.MyException("Fixation of CNAs and combination of normal segments not implemented yet.")
	# if only list with fixed CNVs is given, it is used automatically
	#
	# use warm start info of CNVs
	if warm_start_dc_binary_file is not None:
		if warm_start_dc_binary is not None:
			logging.warning("Both list with values and file with values for warm start of CNVs is given."
				" Using file now.")
		if combine_normal_segments == True:
			eo.MyException("Warm start of CNAs and combination of normal segments not implemented yet.")
		warm_start_dc_binary, tmp1, tmp2 = oio.read_fixed_value_file(warm_start_dc_binary_file)
		# a warm start needs to be provided for all values
		if len(warm_start_dc_binary) != num * len(seg_list) * 4:
			eo.MyException("Warm start needs to be provided for all CNVs.")
	#
	# if list with normal segments are given, the information is processed
	if normal_seg_indices != []:
		fixed_cnv_list_new = model.create_fixed_values_new_for_normal_segments(normal_seg_indices, fixed_cnv_list_new)

	# SNPs are fixed here
	if fixed_snp_list is None and not allele_specific:
		if fixed_snp_file:
			(fixed_snp_list, unfixed_snp_start, unfixed_snp_stop) = oio.read_fixed_value_file(
				fixed_snp_file)
	elif fixed_snp_file and not allele_specific:
		logging.warning("Both list with values and file with values for fixation of " +
			"SNPs exist. Choose only one option.")
		(fixed_snp_list, unfixed_snp_start, unfixed_snp_stop) = oio.read_fixed_value_file(
			fixed_snp_file)

	# SSMs are fixed here
	if fixed_ssm_list is None:
		if fixed_ssm_file:
			if combine_normal_segments == True:
				eo.MyException("Fixation of SSMs and combination of normal segments not implemented yet.")
			(fixed_ssm_list, unfixed_ssm_start, unfixed_ssm_stop) = oio.read_fixed_value_file(
				fixed_ssm_file)
	elif fixed_ssm_file:
		if combine_normal_segments == True:
			eo.MyException("Fixation of SSMs and combination of normal segments not implemented yet.")
		logging.warning("Both list with values and file with values for fixation of " +
			"SSMs exist. Choose only one option.")
		(fixed_ssm_list, unfixed_ssm_start, unfixed_ssm_stop) = oio.read_fixed_value_file(
			fixed_ssm_file)
	else:
		if combine_normal_segments == True:
			eo.MyException("Fixation of SSMs and combination of normal segments not implemented yet.")
	#
	# warm start for SSM is provided from file
	if warm_start_dssm_file is not None:
		if warm_start_dssm is not None:
			logging.warning("Both list with values and file with values for warm start of SSM is given."
				" Using file.")
		warm_start_dssm, tmp1, tmp2 = oio.read_fixed_value_file(warm_start_dssm_file)
		if combine_normal_segments == True:
			eo.MyException("Warm start of SSMs and combination of normal segments not implemented yet.")
		# a warm start needs to be provided for all values
		if len(warm_start_dssm) != num * len(ssm_list) * 3:
			eo.MyException("Warm start needs to be provided for all SSMs.")


		
	# fixed Z-matrix
	if fixed_z_matrix_file:
		(fixed_z_matrix_list, unfixed_z_matrix_start, unfixed_z_matrix_stop) = (
			oio.read_fixed_value_file(fixed_z_matrix_file))
		# if Z matrix is given and lineage divergence rule should be used, 
		# create list of direct descendants of lineages
		if lineage_divergence_rule:
			# triangle shape, last row missing
			z_matrix = model.parse_fixed_z_matrix_list_to_matrix(fixed_z_matrix_list, sublin_nums)
			direct_descendant = model.get_direct_descendants(z_matrix, sublin_nums)
			direct_descendants_for_constraints = model.parse_direct_descendants_for_constraints(
				direct_descendant)
	# if Z-matrix is fixed, it doesn't need to be given as warm start
	if fixed_z_matrix_file and warm_start_z_matrix_file:
		warm_start_z_matrix_file = None
	if fixed_z_matrix_list and warm_start_z_matrix_list:
		warm_start_z_matrix_list = None
	# Z-matrix for warm start
	if warm_start_z_matrix_file:
		(warm_start_z_matrix_list, warm_start_z_matrix_start, warm_start_z_matrix_stop) = (
			oio.read_fixed_value_file(warm_start_z_matrix_file))

	# if no list with average copy numbers to fix exists, it is checked whether a file with
	# fixation information for the average copy numbers is given
	if fixed_avg_cn_list is None:
		if fixed_avg_cn_file:
			if combine_normal_segments == True:
				eo.MyException("Fixation of average CN and combination of normal segments not implemented yet.")
			(fixed_avg_cn_list, unfixed_avg_cn_start, unfixed_avg_cn_stop) = (
				oio.read_fixed_value_file(fixed_avg_cn_file))
	elif fixed_avg_cn_file:
		if combine_normal_segments == True:
			eo.MyException("Fixation of average CN and combination of normal segments not implemented yet.")
		logging.warning("Both list with values and file with values for fixation of " +
			"average copy number exist. Choose only one option.")
		(fixed_avg_cn_list, unfixed_avg_cn_start, unfixed_avg_cn_stop) = (
			oio.read_fixed_value_file(fixed_avg_cn_file)) 
	else:
		if combine_normal_segments == True:
			eo.MyException("Fixation of average CN and combination of normal segments not implemented yet.")

	# if no list with phi values to fix exists, it is checked whether a file with
	# fixation information for the phis is given
	if fixed_phi_list is None: 
		if fixed_phi_file:
			(fixed_phi_list, unfixed_phi_start, unfixed_phi_stop) = oio.read_fixed_value_file(
				fixed_phi_file)
	elif fixed_phi_file:
		logging.warning("Both list with values and file with values for fixation of phis "+
			"exist. Using only file.")
		(fixed_phi_list, unfixed_phi_start, unfixed_phi_stop) = oio.read_fixed_value_file( 
			fixed_phi_file)
	# if phis are fixed, check whether they can lead to compatible result
	if fixed_phi_list is not None and allele_specific == True:
		phi_compatible_result = model.check_phi_lineage_and_vaf_compatibility(fixed_phi_list, ssm_list, seg_list, num)
		if phi_compatible_result == False:
			oio.print_status_value("MIP_infeasible", -float("inf"))
			oio.print_llh(-float("inf"))
			logging.info("Not possible to find a reconstruction because fixed phi and low copy number"
				" ci leads to VAF > 1. Not possible. Reconstruction with 2 lineages is not"
				" possible.")
			raise eo.FixPhiIncompatibleException("Reconstruction with 2 lineages is not possible.")
	# use warm start for phis
	if warm_start_freqs_file is not None:
		if warm_start_freqs is not None:
			logging.warning("Both list with values and file with values for warm start of phis are given."
				" Using file.")
		warm_start_freqs, tmp1, tmp2 = oio.read_fixed_value_file(warm_start_freqs_file)
		# warm start need to be given for all phis
		if len(warm_start_freqs) != num - 1 :
			eo.MyException("All valus for warm start of phis need to be given.")
			
	# get values of previous solution
	if warm_start_solution is not None:
		# read previous reconstruction
		prev_reconstruction = oio.read_result_file(warm_start_solution)
		# check that as many lineages are given as lineages should be inferred
		# append dummy lineages until the number of lineages equals the number of inferred lineages
		while len(prev_reconstruction) < sublin_nums:
			prev_reconstruction.append(lineage.Lineage([], 0.0, [], [], [], [], [], [], [], []))
		# get values to fix
		start_freq = model.create_fixed_file(prev_reconstruction, cons.FREQ)[0]
		start_Z = model.create_fixed_file(prev_reconstruction, cons.Z)[0]
		start_SSMs = model.create_fixed_file(prev_reconstruction, cons.SSM)[0]
		start_CNVs = model.create_fixed_file(prev_reconstruction, cons.CNV)[0]

	# fixation
	######

	# SSM can get clustered based on their frequency
	ssm_indices_per_cluster_per_seg = None
	if cluster_SSM:
		logging.info("SSMs are clustered.")
		# parameter with which the number of clusters is computed is checked 
		# and eventually updated
		# list with entries from 0:len(seg_list)-1
		seg_for_clustering_indices = [i for i in xrange(len(seg_list))]
		# for each segment the same number of clusters is used
		if cluster_num_param == -1:
			cluster_num_param = cons.CLUSTER_NUM_PARAM
		cluster_num = [cluster_num_param for i in xrange(len(seg_list))]
		## tests with different numbers of CN changes showed, that performance increases with
		#	number of clusters independant of the number of CN changes
		# 	so I won't use the adaption anymore
		## use less clusters on segments that contain no CNVs
		## TODO if I decide to use adaption of cluster numbers again, adapt clustering also an fixed_cnv_list_new
		#if fixed_cnv_list:
		#	model.adapt_cluster_num_to_fixed_CNV_num(unfixed_cnv_start, unfixed_cnv_stop, 
		#		fixed_cnv_list, cluster_num, num, len(seg_list), 
		#		normal_seg_indices=normal_seg_indices)
		# the given number of clusters per segment is multipled with the number of non-normal lineages
		cluster_num = [x * (sublin_nums - 1)  for x in cluster_num]

		cluster_labels_per_seg = []
		(ssm_indices_per_cluster_per_seg, ssm_objects_of_segment_per_seg) = (
			cluster_mod.choose_SSMs_cluster_create_indices(seg_for_clustering_indices, 
			ssm_list, [], cluster_num, cluster_labels_per_seg, kmeans=kmeans, agglo_ward=agglo_ward))
		# superSSMs should be used
		if use_super_SSMs:
			superSSMs, superSSM_hash = (model.create_superSSMs(ssm_indices_per_cluster_per_seg,
				ssm_objects_of_segment_per_seg))
			(seg_splines_A, seg_splines_B, superSSM_splines) = (
				model.create_segment_and_mutation_splines(seg_list, snp_list, 
				superSSMs, number_spline_points, True))
				
	# depending on whether superSSMs should be used or not decide what variables are given to CPLEX
	if use_super_SSMs:
		ssm_list_for_cplex = superSSMs
		ssm_splines_for_cplex = superSSM_splines
		ssm_indices_per_cluster_per_seg_for_cplex = None
	else:
		ssm_list_for_cplex = ssm_list
		ssm_splines_for_cplex = ssm_splines
		ssm_indices_per_cluster_per_seg_for_cplex = ssm_indices_per_cluster_per_seg

	# take start time
	start_time = time_module.time()
	# do optmization
	cplex_obj = optimization.Optimization_with_CPLEX(seg_splines,
		snp_splines, ssm_splines_for_cplex, allele_specific, seg_splines_A, seg_splines_B,
		simple_CN_changes=simple_CN_changes, 
		max_x_CN_changes=max_x_CN_changes, only_one_loss=only_one_loss,
		only_gains_losses_LOH=only_gains_losses_LOH, cn_weight=cn_weight, z_trans_weight=z_trans_weight,
		use_lineage_divergence_rule=lineage_divergence_rule)
	# set up parameters
	cplex_obj.set_CPLEX_parameters(time, threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
		strategy_file, workmem, workdir, treememory, emphasis_memory, nodeselect, cplex_log_file)
	# start optimization
	cplex_obj.opt_with_CPLEX(sublin_nums, snp_list, ssm_list_for_cplex, seg_list,
		fixed_cnv=fixed_cnv_list, unfixed_cnv_start=unfixed_cnv_start, 
		unfixed_cnv_stop=unfixed_cnv_stop,
		fixed_avg_cn=fixed_avg_cn_list, unfixed_avg_cn_start=unfixed_avg_cn_start, 
		unfixed_avg_cn_stop=unfixed_avg_cn_stop,
		fixed_snp=fixed_snp_list, unfixed_snp_start=unfixed_snp_start, 
		unfixed_snp_stop=unfixed_snp_stop,
		fixed_ssm=fixed_ssm_list, unfixed_ssm_start=unfixed_ssm_start, 
		unfixed_ssm_stop=unfixed_ssm_stop,
		fixed_z_matrix=fixed_z_matrix_list, unfixed_z_matrix_start=unfixed_z_matrix_start, 
		unfixed_z_matrix_stop=unfixed_z_matrix_stop, 
		fixed_phi=fixed_phi_list, unfixed_phi_start=unfixed_phi_start, 
		unfixed_phi_stop=unfixed_phi_stop, 
		direct_descendants_for_constraints=direct_descendants_for_constraints,
		warm_start_dc_binary=warm_start_dc_binary, warm_start_dsnp=warm_start_dsnp, 
		warm_start_dssm=warm_start_dssm, warm_start_freqs=warm_start_freqs,
		ssm_indices_per_cluster=ssm_indices_per_cluster_per_seg_for_cplex,
		fixed_cnv_indices=normal_seg_indices, fixed_cnv_list_new=fixed_cnv_list_new,
		fixed_ssm_list_new=fixed_ssm_list_new, dont_break_z_symmetry=dont_break_z_symmetry,
		warm_start_z_matrix_list=warm_start_z_matrix_list,
		start_freq=start_freq, start_Z=start_Z, start_SSMs=start_SSMs, start_CNVs=start_CNVs)

	# if superSSMs were used, some things need to be done
	# this is not the case when heuristic1 is used
	rec_wo_super_success = None
	if use_super_SSMs == True and heuristic1 == False:
		inferred_freqs = model.get_frequencies_from_CPLEX(cplex_obj)

		(my_lineages, cplex_obj, rec_wo_super_success) = reconstruct_with_non_clustered_SSMs(cplex_obj, 
			snp_list, superSSMs, seg_list, superSSM_hash, out_results,
			num,
			threads, probing, emph_switch, coef_reduc, mipgap, symmetry,
			strategy_file, workmem, workdir, emphasis_memory, nodeselect,
			cplex_log_file, number_spline_points,
			fixed_cnv_file=fixed_cnv_file, fixed_avg_cn_file=fixed_avg_cn_file,
			fixed_snp_file=fixed_snp_file, fixed_ssm_file=fixed_ssm_file,
			fixed_z_matrix_file=fixed_z_matrix_file, 
			fixed_phi_file=fixed_phi_file, test_run=test_run,
			fixed_phi_list=fixed_phi_list, unfixed_phi_start=unfixed_phi_start,
			unfixed_phi_stop=unfixed_phi_stop, fixed_avg_cn_list=fixed_avg_cn_list,
			unfixed_avg_cn_start=unfixed_avg_cn_start, unfixed_avg_cn_stop=unfixed_avg_cn_stop,
			warm_start_dc_binary=warm_start_dc_binary, warm_start_dsnp=warm_start_dsnp,
			warm_start_dssm=warm_start_dssm, warm_start_freqs=inferred_freqs, 
			allele_specific=allele_specific,
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=max_x_CN_changes,
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cn_weight=cn_weight, z_trans_weight=z_trans_weight,
			ssm_list=ssm_list, lineage_divergence_rule=lineage_divergence_rule,
			dont_break_z_symmetry=dont_break_z_symmetry,
			take_time=False)

	end_time = time_module.time()
	if take_time == True:
		logging.info("#Time for optimization: {0}".format(end_time - start_time))

	# TODO: check and then# delete, z matrix should be fine because of new constraints
	# save import parts of solution in sublineage objects
	# before: check if Z matrix has to be refined
	#best_opt.save_part_of_solution_in_class()
	#model.refine_z_matrix(best_opt, ssm_list)

	#print best_opt.solution_z

	# check again if output file exists
	if not test_run and write_output_to_disk:
		oio.raise_if_file_exists(out_results) 

	# get lineages from CPLEX object if no superSSMs are used or heuristic1 is used
	# if superSSMs are used, lineages are already there because of function reconstruct_with_non_clustered_SSMs
	if use_super_SSMs == False or heuristic1 == True:
		my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, snp_list, ssm_list, seg_list,
			dont_break_z_symmetry=dont_break_z_symmetry)

	# if normal segments were combined, rewrite lineages
	if combine_normal_segments == True and len(original_normal_seg_indices) > 1:
		# normal segments and SSMs are set to their original positions
		seg_list, ssm_list, normal_seg_indices, new_seg_list, new_ssm_list, new_normal_seg_indices, ssm_normal_changed, ssm_normal = (
			model.decombine_normal_segments_and_ssms(my_lineages, seg_list, ssm_list, normal_seg_indices, 
			original_seg_list, original_ssm_list, ssm_normal_changed, ssm_normal, original_normal_seg_indices,
			original_seg_index))

	# if a lineage has a frequency of ~ 0, it is removed from the reconstruction
	try:
		model.shorten_lineages(my_lineages, z_matrix_list)
	# if lineage with frequency of ~ 0 has unallowed mutationsi or Z-matrix is not None, 
	# error is raised after printing reconstruction to file
	except (eo.LineageWith0FreqMutations, eo.ZMatrixNotNone) as e:
		# if out results shouldn't usually be written to file, create another filename
		if out_results is None:
			if cplex_log_file is not None:
				out_results = "{0}.results".format(cplex_log_file)
			else:
				out_results = "{0}_results".format(random.random())
				logging.debug("Reconstruction printed to {0}".format(out_results))
		# print reconstruction to file
		oio.write_lineages_to_result_file(out_results, my_lineages, test=test_run)
		raise e

	# print reconstruction to file
	if write_output_to_disk:
		oio.write_lineages_to_result_file(out_results, my_lineages, test=test_run)

	# if initial optimization was called by heuristic1, more information is returned
	if heuristic1 == True:
		# no clustering was used and normal segments weren't combined
		if use_super_SSMs == False and combine_normal_segments == False:
			return (my_lineages, cplex_obj, seg_list, ssm_list, seg_splines_A, seg_splines_B, ssm_splines)
		# clustering was used and normal segments weren't combined
		elif use_super_SSMs == True and combine_normal_segments == False:
			return (my_lineages, cplex_obj, seg_list, ssm_list, seg_splines_A, seg_splines_B, ssm_splines,
				superSSMs, superSSM_hash)
		#TODO without and with clustering when normal segments were combined
		else:
			eo.MyException("Not yet implemented")

	# compute LLH
	if allele_specific == True:
		llh = log_pdf.compute_llh(my_lineages, ssm_list, seg_list, overdispersion_parameter)
		mdl = model.compute_MDL(my_lineages, len(seg_list), len(ssm_list), llh)
		if use_super_SSMs == False or rec_wo_super_success == False:
			oio.print_llh(llh)
			oio.print_mdl(mdl)
	# TODO: use len(my_lineages) instead of num
	# TODO: used LLH insteadt of objective
	# compute and print BIC
	#bic = model.get_BIC(len(seg_list), num, len(ssm_list), len(snp_list), allele_specific,
	#	cplex_obj.my_prob.solution.get_objective_value())
	#logging.info("BIC: {0}".format(bic))


	# if ambiguous lineage relations shouldn't be used, function is done here
	if review_ambiguous_relations == False:
		return (my_lineages, cplex_obj)

	z_matrix_list, new_lineage_list, lin_div_rule_feasibility = review_ambiguous_relations_function(my_lineages, 
		z_matrix_list, seg_list, write_output_to_disk, out_results, test_run, new_lineage_list, lin_div_rule_feasibility)

	return (my_lineages, cplex_obj, z_matrix_list, new_lineage_list)

def review_ambiguous_relations_function(my_lineages, z_matrix_list, seg_list, write_output_to_disk, out_results, test_run,
	new_lineage_list, lin_div_rule_feasibility):
	tmp_lineages = copy.deepcopy(my_lineages)
	# review all ambiguous lineage relations
	if z_matrix_list is None:
		my_lineages, z_matrix_list, new_lineage_list, lin_div_rule_feasibility = model.get_all_possible_z_matrices_with_lineages(
			my_lineages, len(seg_list))
	
	# if original lineages list was updated because of Z-transitivity rules in model, rewrite them
	if tmp_lineages != my_lineages and write_output_to_disk:
		oio.write_lineages_to_result_file(out_results, my_lineages, test=True)

	# print Z-matrices to file
	# print lineage reconstructions to file
	if write_output_to_disk == True:
		for i in xrange(len(z_matrix_list)):
			oio.write_matrix_to_file(z_matrix_list[i].tolist(), "{0}_{1}.zmatrix".format(out_results, i+1), 
				test=test_run)
			oio.write_lineages_to_result_file("{0}.reconstruction_{1}".format(out_results, i+1), 
				new_lineage_list[i], test=test_run)

	return z_matrix_list, new_lineage_list, lin_div_rule_feasibility

def call_review_ambiguous_relation(out_results, input_seg, input_ssm):
	# read result file
	my_lineages = oio.read_result_file(out_results)

	# read seg list
	(seg_list, ssm_list) = model.create_segment_and_mutation_lists(input_seg,
		None, input_ssm, True)

	z_matrix_list = None
	write_output_to_disk = True
	test_run = False
	new_lineage_list = None
	lin_div_rule_feasibility = None

	# review ambiguous relationships
	logging.debug("start to review ambiguous relationships")
	review_ambiguous_relations_function(my_lineages, z_matrix_list, seg_list, write_output_to_disk, out_results, 
		test_run, new_lineage_list, lin_div_rule_feasibility)

## chooses the parameter with which the number of clusters is computed
## depends on whether this parameter was already set and if a maximum number of
## CN changes is used
#def choose_cluster_num_param(cluster_num_param, max_x_CN_changes):
#	# only max_x_CN_changes is given
#	if max_x_CN_changes > 0 and cluster_num_param <= 0:
#		return max_x_CN_changes + 1
#	# only cluster_num_param is given
#	if cluster_num_param > 0 and max_x_CN_changes <= 0:
#		return cluster_num_param
#	# both parameters unset
#	if cluster_num_param == -1 and max_x_CN_changes == -1:
#		return cons.CLUSTER_NUM_PARAM
#	# more clusters migth be needed because I have more possible CN changes
#	# per segment
#	if max_x_CN_changes > cluster_num_param:
#		logging.info("Setting cluster_num_param to {0} because otherwise not enough"
#			" clusters might exist.".format(max_x_CN_changes + 1))
#		return max_x_CN_changes + 1
#	# maybe too many clusters, but is fine
#	if cluster_num_param > max_x_CN_changes:
#		return cluster_num_param
#	if cluster_num_param == max_x_CN_changes: 
#		logging.info("Setting cluster_num_param to {0} because otherwise not enough"
#			" clusters might exist.".format(max_x_CN_changes + 1))
#		return max_x_CN_changes + 1
#
#	return cons.CLUSTER_NUM_PARAM

if __name__ == '__main__':
	#(program, seg_num, snp_num, ssm_num, sublins, runtime) = sys.argv
	#print seg_num, snp_num, ssm_num, sublins
	#test_runtime(int(seg_num), int(snp_num), int(ssm_num), int(sublins), int(runtime))

	# create argumentparser
	parser = argparse.ArgumentParser()
	# add commandline arguments "--" for optional arguments. All other arguments are positional and required.
	parser.add_argument("input_segment", help = "input segment file")
	parser.add_argument("input_ssm", help = "input ssm file")
	parser.add_argument("num", type = int, help = "number of lineages")
	parser.add_argument("out_results", help = "output file")
        parser.add_argument("--input_snp_file", default=None, type=str, help =
                "input snp file (required when allele_specific=False)")
	parser.add_argument("--time", default = 1e+75, type = float, help = "TODO")
	parser.add_argument("--threads", default = 1, type = int, help = "TODO")
	parser.add_argument("--probing", default = 0, type = int, help = "TODO")
	parser.add_argument("--emph_switch", default = 0, type = int, help = "TODO")
	parser.add_argument("--coef_reduc", default = -1, type = int, help = "TODO")
	parser.add_argument("--mipgap", default = 1e-04, type = float, help = "TODO")
	parser.add_argument("--symmetry", default = 4, type = int, help = "TODO")
	parser.add_argument("--strategy_file", default = 1, type = int, help = "TODO")
	parser.add_argument("--workmem", default = 128, type = float, help = "TODO")
	parser.add_argument("--workdir", default = "/scratch", type = str, help = "TODO")
	parser.add_argument("--treememory", default = 1e+75, type = float, help = "TODO")
	parser.add_argument("--emphasis_memory", default = 0, type = int, help = "TODO")
	parser.add_argument("--nodeselect", default = 1, type = int, help = "TODO")
	parser.add_argument("--fixed_cnv_file", default = None, help = "TODO")
	parser.add_argument("--fixed_avg_cn_file", default = None, help = "TODO")
	parser.add_argument("--fixed_snp_file", default = None, help = "TODO")
	parser.add_argument("--fixed_ssm_file", default = None, help = "TODO")
	parser.add_argument("--fixed_z_matrix_file", default = None, help = "TODO")
	parser.add_argument("--warm_start_z_matrix_file", default = None, help = "TODO")
	parser.add_argument("--warm_start_dc_binary_file", default = None, help = "TODO")
	parser.add_argument("--warm_start_dssm_file", default = None, help = "TODO")
	parser.add_argument("--warm_start_freqs_file", default = None, help = "TODO")
	parser.add_argument("--warm_start_solution", default = None, help = "TODO")
	parser.add_argument("--fixed_phi_file", default = None, help = "TODO")
	parser.add_argument("--log", default = "DEBUG", type = str, help = "TODO")
	parser.add_argument("--log_file", type = str, help = "TODO")
	parser.add_argument("--cplex_log_file", type = str, help = "TODO")
	parser.add_argument("--number_spline_points", default = 50, type = int)
	parser.add_argument("--heuristic_max_rounds", type = int, help = "Maximal rounds of heuristic")
	parser.add_argument("--heuristic_epsilon", type = float, help = "Accepted distance"
		" between last and current objective")
	parser.add_argument("--heuristic_initial_time", type = int, help = "Time in secs for "
		"initial run to get first version of fixed variables")
	parser.add_argument("--heuristic_seg_time", type = int, help = "tic_initial_time")
	parser.add_argument("--allele_specific", default = None, help = "TODO")
	parser.add_argument("--simple_CN_changes", default = None, help = "Whether simple "
		"CN changes should be used or not.")
	parser.add_argument("--max_x_CN_changes", default = -1, type = int, help = "How many "
		"CN changes are allowed to appear per segment.")
	parser.add_argument("--only_one_loss", default = None, help = "If only one loss of each "
		"chromatid per segment is allowed.")
	parser.add_argument("--only_gains_losses_LOH", default = None, help = "If only gains or "
		"losses or LOH is allowed per segment.")
	parser.add_argument("--cn_weight", default = 0, help = "Weight with which CN are multiplied "
		"in objective. Prefers solutions with less CN variations.")
	parser.add_argument("--z_trans_weight", default = 0.00001, help = "Weight with variables of transitizity "
		"in Z-matrix are multiplied, so that solutions are prevented that have unnecessary 1's in "
		"Z-matrix because of transitivity rule.")
	parser.add_argument("--cluster_SSM", help = "If set to True, SSMs get clustered "
		"based on their frequency.")
	parser.add_argument("--k_means_clustering", action='store_true')
	parser.add_argument("--agglo_ward_clustering", action='store_true')
	parser.add_argument("--cluster_num_param", type = int, help = "Used to determine "
		"the number of clusters.")
	parser.add_argument("--use_super_SSMs", action='store_true')
	parser.add_argument("--lineage_divergence_rule", action='store_true')
	parser.add_argument("--normal_seg_indices_file", default = None, help = "TODO")
	parser.add_argument("--dont_break_z_symmetry", action='store_true')
	parser.add_argument("--dont_review_ambiguous_relations", action='store_true')
	parser.add_argument("--combine_normal_segments", action='store_true')
	parser.add_argument("--overdispersion_parameter", type = int, default=1000)
	parser.add_argument("--review_ambiguous_relation_arg", action='store_true')

	# parse commandline input into object
	args = parser.parse_args()

	# if allele specific CNs are used or not
	allele_specific = False
	if args.allele_specific is not None:
		allele_specific = True

	# check if input files exist
	oio.raise_if_file_not_exists(args.input_segment)
	if not allele_specific and args.heuristic_initial_time is None:
            try:
		oio.raise_if_file_not_exists(args.input_snp_file)
            except:
                sys.stderr.write('Error: No input_snp file specified or file ' + \
                        'not existent. Exiting\n')
                parser.print_help()
                exit(1)

	oio.raise_if_file_not_exists(args.input_ssm)

	# check if output file already exists
	if args.review_ambiguous_relation_arg == False:
		oio.raise_if_file_exists(args.out_results)

	# threads cannot be smaller than 1
	if args.threads < 1:
		args.threads = 1

	# test if logging info is set correctly
	log_file_name = None
	if args.log_file is None:
		if args.review_ambiguous_relation_arg == False:
			log_file_name = "{0}.log".format(args.out_results)
		else:
			log_file_name = "{0}.ambiguity_review.log".format(args.out_results)
	elif not args.log_file.endswith(".log"):
		raise eo.UnallowedNameException("The filename of the log file must end with the extention \".log\".")
	else:
		log_file_name = args.log_file

	numeric_logging_info = getattr(logging, args.log.upper(), None)
	if not isinstance(numeric_logging_info, int):
		raise ValueError('Invalid log level: %s' % loglevel)
	logging.basicConfig(filename=log_file_name, filemode='w', level=numeric_logging_info)
	
	# test name of cplex log file, or set the name
	cplex_log_file = None
	if args.cplex_log_file is None:
		cplex_log_file = "{0}.cplex.log".format(args.out_results)
	elif not args.cplex_log_file.endswith(".cplex.log"):
		raise eo.UnallowedNameException("The filename of the log file must end with the extention" +
			"\".cplex.log\".")
	else:
		cplex_log_file = args.cplex_log_file

	if args.number_spline_points < 2:
		raise ValueError('At least 2 points for the spline need to be given.')

	# parse arguments for CN changes
	simple_CN_changes = True
	if args.simple_CN_changes is not None:
		if args.simple_CN_changes == "True":
			pass
		elif args.simple_CN_changes == "False":
			simple_CN_changes = False
		else:
			raise ValueError('Can\'t understand input for simple_CN_changes. Must be boolean.')
	only_one_loss = True
	if args.only_one_loss is not None:
		if args.only_one_loss == "True":
			pass
		elif args.only_one_loss == "False":
			only_one_loss = False
		else:
			raise ValueError('Can\'t understand input for \'only_one_loss\'. Must be boolean.')
	only_gains_losses_LOH = True
	if args.only_gains_losses_LOH is not None:
		if args.only_gains_losses_LOH == "True":
			pass
		elif args.only_gains_losses_LOH == "False":
			only_gains_losses_LOH = False
		else:
			raise ValueError('Can\'t understand input for \'only_gains_losses_LOH\'. Must be boolean.')
	cn_weight = float(args.cn_weight)
	if cn_weight < 0:
		raise ValueError('CN weight must be positive.')
	z_trans_weight = float(args.z_trans_weight)
	if z_trans_weight < 0:
		raise ValueError('Weight for Z-matrix transitivity variables must be positive.')

	#TODO: more checks for other parameters

	cluster_SSM = False
	if args.cluster_SSM is not None:
		cluster_SSM = oio.str_to_bool(args.cluster_SSM)

	# check which clustering algorithm should be used
	if cluster_SSM == True:
		if args.k_means_clustering:
			kmeans = True
			logging.info("Using k-means clustering.")
		else:
			kmeans = False
		if args.agglo_ward_clustering:
			agglo_ward = True
			logging.info("Using agglomerative ward clustering.")
		else:
			agglo_ward = False
		# check that one algorithm is choosen
		if kmeans == False and agglo_ward == False:
			kmeans = True
			logging.info("Choosing k-means as clustering algorithm.")
	else:
		kmeans = False
		agglo_ward = False

	if args.use_super_SSMs and not cluster_SSM:
		raise ValueError('SSMs must be clustered if superSSMs should be used.')
	
	if args.fixed_cnv_file is not None and args.normal_seg_indices_file is not None:
		raise eo.ParameterException("Parameters fixed_cnv_file and normal_seg_indices_file are not allowed to "
			"be set together in one run.")

	normal_seg_indices = []
	if args.normal_seg_indices_file is not None:
		logging.info("Reading file with normal segment indices {0}".format(args.normal_seg_indices_file))
		with open(args.normal_seg_indices_file) as f:
			for line in f:
				normal_seg_indices = map(int, line.rstrip("\n").split("\t"))

	cluster_num_param = -1
	if args.cluster_num_param is not None:
		if int(args.cluster_num_param) == 0:
			raise ValueError("cluster_num_param must be greater 0.")
		cluster_num_param = int(args.cluster_num_param)

	review_ambiguous_relations = True
	if args.dont_review_ambiguous_relations == True:
		review_ambiguous_relations = False
		
	# initial time for heuristic is set, so heuristic should be used
	if args.heuristic_initial_time is not None:
		if args.heuristic_max_rounds is None or args.heuristic_epsilon is None:
			raise ValueError('heuristic_max_rounds and heuristic_epsilon need to be set'+
				'if heuristic should be applied.')
                start_iterative_heuristic_all_in(args.input_segment,
                        args.input_ssm, args.num,
                        args.out_results, args.heuristic_initial_time,
                        args.heuristic_seg_time, args.threads, args.probing,
                        args.emph_switch, args.coef_reduc, args.mipgap,
                        args.symmetry, args.strategy_file, args.workmem,
                        args.workdir, args.treememory, args.emphasis_memory,
                        args.nodeselect, args.number_spline_points,
                        args.heuristic_max_rounds,
                        args.heuristic_epsilon, cplex_log_file,
                        numeric_logging_info,
                        simple_CN_changes=simple_CN_changes,
                        max_x_CN_changes=args.max_x_CN_changes,
                        only_one_loss=only_one_loss,
                        only_gains_losses_LOH=only_gains_losses_LOH,
                        cn_weight=cn_weight, z_trans_weight=z_trans_weight, 
			lineage_divergence_rule=args.lineage_divergence_rule,
			normal_seg_indices=normal_seg_indices, fixed_phi_file=args.fixed_phi_file,
			cluster_SSM=cluster_SSM, kmeans=kmeans, agglo_ward=agglo_ward,
			cluster_num_param=cluster_num_param, review_ambiguous_relations=review_ambiguous_relations,
			dont_break_z_symmetry=args.dont_break_z_symmetry, 
			overdispersion_parameter=args.overdispersion_parameter) 
	# only ambiguous relationships should be computed
	elif args.review_ambiguous_relation_arg == True:
		call_review_ambiguous_relation(args.out_results, args.input_segment, args.input_ssm)
	else:
		# commandline arguments are referenced by args.<argumentname>
                go_onctopus(args.input_segment, args.input_snp_file,
                        args.input_ssm, args.num, args.out_results, args.time,
                        args.threads, args.probing, args.emph_switch,
                        args.coef_reduc, args.mipgap, args.symmetry,
                        args.strategy_file, args.workmem, args.workdir,
                        args.treememory, args.emphasis_memory, args.nodeselect,
                        cplex_log_file, args.number_spline_points,
			fixed_cnv_file=args.fixed_cnv_file, fixed_avg_cn_file=args.fixed_avg_cn_file,
			fixed_snp_file=args.fixed_snp_file, fixed_ssm_file=args.fixed_ssm_file,
			fixed_z_matrix_file=args.fixed_z_matrix_file, fixed_phi_file=args.fixed_phi_file,
                        allele_specific=allele_specific,
                        simple_CN_changes=simple_CN_changes,
                        max_x_CN_changes=args.max_x_CN_changes,
                        only_one_loss=only_one_loss,
                        only_gains_losses_LOH=only_gains_losses_LOH,
                        cn_weight=cn_weight, z_trans_weight=z_trans_weight, cluster_SSM=cluster_SSM,
			use_super_SSMs=args.use_super_SSMs,
			lineage_divergence_rule=args.lineage_divergence_rule,
			normal_seg_indices=normal_seg_indices, combine_normal_segments=args.combine_normal_segments,
			kmeans=kmeans, agglo_ward=agglo_ward,
			cluster_num_param=cluster_num_param, review_ambiguous_relations=review_ambiguous_relations,
			dont_break_z_symmetry=args.dont_break_z_symmetry,
			overdispersion_parameter=args.overdispersion_parameter,
			warm_start_z_matrix_file=args.warm_start_z_matrix_file,
			warm_start_dc_binary_file=args.warm_start_dc_binary_file,
			warm_start_dssm_file=args.warm_start_dssm_file, warm_start_freqs_file=args.warm_start_freqs_file,
			warm_start_solution=args.warm_start_solution)

	#parser.add_argument("--dont_break_z_symmetry", action='store_true')
	#parser.add_argument("--dont_review_ambiguous_relations", action='store_true')
	# review_ambiguous_relations

	# (prog, input_seg, input_snp, input_ssm, num, out_results, time, threads, 
	# 	probing, emph_switch, coef_reduc, mipgap, symmetry, 
	# 	strategy_file, workmem, treememory, emphasis_memory,
	# 	fixed_cnv_file,
	# 	fixed_snp_file, fixed_ssm_file, fixed_z_matrix_file, fixed_phi_file) = sys.argv
	# fixed_cnv_file = oio.str_possibly_to_none(fixed_cnv_file)
	# fixed_snp_file = oio.str_possibly_to_none(fixed_snp_file)
	# fixed_ssm_file = oio.str_possibly_to_none(fixed_ssm_file)
	# fixed_z_matrix_file = oio.str_possibly_to_none(fixed_z_matrix_file)
	# fixed_phi_file = oio.str_possibly_to_none(fixed_phi_file)
	# go_onctopus(input_seg, input_snp, input_ssm, int(num), out_results, 
	# 	int(time), int(threads), int(probing), int(emph_switch), 
	# 	int(coef_reduc), float(mipgap), int(symmetry), 
	# 	int(strategy_file), float(workmem), float(treememory), int(emphasis_memory), 
	# 	fixed_cnv_file, fixed_snp_file,
	# 	fixed_ssm_file, fixed_z_matrix_file, fixed_phi_file)

	
