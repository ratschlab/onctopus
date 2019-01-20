#!/usr/bin/env python

import onctopus_io as oio
import model
import exceptions_onctopus as eo
import optimization
import logging
import argparse
import time
import main
import numpy as np
import exceptions_onctopus as eo
import constants as cons

def start_cluster_heuristic_fixed_phi_pure_segmentwise(ssm_input_file, seg_input_file,
	sublin_num, normal_seg_indices, cluster_num,
	allele_specific=True, number_spline_points=50,
	test=False, threads=1, write_result_file=False, result_file_name=None,
	simple_CN_changes=False, max_x_CN_changes=2, only_one_loss=False, 
	only_gains_losses_LOH=True, cn_weight=0.00001, kmeans=True, agglo_ward=False):

	overdispersion = 1000

	if allele_specific == False:
		raise eo.MyException("not done for non-allele-specific")
	
	# read in data
	(seg_list, ssm_list) = model.create_segment_and_mutation_lists(seg_input_file, 
		[], ssm_input_file, allele_specific=True)
	seg_num = len(seg_list)
	ssm_num = len(ssm_list)
	snp_num = 0

	# optimize for normal segments
	(lineages_normal_segs, cplex_obj, bic, ssm_indices_of_segment, cluster_labels) = (
		cluster_ssms_normal_segs_optimization(ssm_input_file, seg_input_file,
		sublin_num, normal_seg_indices,
		write_result_file=False, result_file_name=None, test=test, threads=threads, kmeans=kmeans,
		agglo_ward=agglo_ward))
	fixed_phis = model.get_phis_from_lineages(lineages_normal_segs)

	# create lists that contain segment information
	lineages_per_seg = [lineages_normal_segs]
	cplex_obj_list = [cplex_obj]
	ssm_indices_of_segment_per_seg = [ssm_indices_of_segment]
	cluster_labels_per_seg = [cluster_labels]
	ssm_objects_of_segment_per_seg = []

	# get indices of segments that have CN changes, also get only these segment
	seg_w_cn_change_indices = model.get_seg_indices_w_cn_changes(seg_num, normal_seg_indices)
	choosen_segments = model.choose_seg_subset(seg_w_cn_change_indices, seg_list)
	# choose SSMs that belong to single segments
	# cluster SSMs and get indices of SSMs that are clustered together
	cluster_num_list = [cluster_num for i in xrange(len(seg_w_cn_change_indices))]
	(ssm_indices_per_cluster_per_seg, ssm_objects_of_segment_per_seg) = choose_SSMs_cluster_create_indices(
		seg_w_cn_change_indices, ssm_list, ssm_indices_of_segment_per_seg, cluster_num_list,
		cluster_labels_per_seg, kmeans=kmeans, agglo_ward=agglo_ward)

	# Z matrices
	different_z_matrices = model.get_fixed_z_matrices(sublin_num)

	# optimize for each fixed Z matrix, only store run with best LLH
	best_cplex_obj_list = None
	best_llh = -float("inf")
	best_z_matrices = []
	for current_z_matrix in different_z_matrices:

		# call heuristic
		cplex_log_file = result_file_name + ".cplex.log"
		#cplex_log_file = None
		cplex_obj_list_2 = []
		# when only two lineages exist, the Z matrix does not need to be fixed
		if sublin_num == 2:
			cplex_obj_list_2 = main.heuristic_fixed_phi_pure_segmentwise(choosen_segments, 
				ssm_objects_of_segment_per_seg[1:],
				sublin_num, fixed_phis, None, threads, number_spline_points, cplex_log_file, 
				allele_specific=allele_specific, simple_CN_changes=simple_CN_changes, 
				max_x_CN_changes=max_x_CN_changes, only_one_loss=only_one_loss, 
				only_gains_losses_LOH=only_gains_losses_LOH, cn_weight=cn_weight,
				ssm_indices_per_cluster=ssm_indices_per_cluster_per_seg)
		else:
			try:
				cplex_obj_list_2 = main.heuristic_fixed_phi_pure_segmentwise(
					choosen_segments, 
					ssm_objects_of_segment_per_seg[1:],
					sublin_num, fixed_phis, current_z_matrix, threads, 
					number_spline_points, 
					cplex_log_file, 
					allele_specific=allele_specific, 
					simple_CN_changes=simple_CN_changes, 
					max_x_CN_changes=max_x_CN_changes, only_one_loss=only_one_loss, 
					only_gains_losses_LOH=only_gains_losses_LOH, cn_weight=cn_weight,
					ssm_indices_per_cluster=ssm_indices_per_cluster_per_seg)
			# fixed Z matrix and fixed phis are infeasible
			# continue with next Z matrix
			except eo.ZMatrixPhisInfeasibleException, exc:
				logging.info(exc)
				continue

		# get LLH
		llh = sum(cplex_obj.my_prob.solution.get_objective_value() for cplex_obj
			in cplex_obj_list_2)

		#logging.info("LLH for z matrix {0} for all non-normal segments is {1}.".format(
		#	current_z_matrix, llh))

		if llh > best_llh:
			best_llh = llh
			best_cplex_obj_list = cplex_obj_list_2
			best_z_matrices = [current_z_matrix]
		elif llh == best_llh:
			best_z_matrices.append(current_z_matrix)

	# do for best LLH, combine results, compute LLH
	cplex_obj_list.extend(best_cplex_obj_list)
	llh = sum(cplex_obj.my_prob.solution.get_objective_value() for cplex_obj in cplex_obj_list)
	lineages_per_seg.extend(model.get_lineages_ob_from_CPLEX_results(best_cplex_obj_list[i], [], 
		ssm_objects_of_segment_per_seg[i+1], [choosen_segments[i]], single_segment=True) 
		for i in range(len(best_cplex_obj_list)))
	combined_lineages = model.combine_lineages_lists_fixed_phi_z(lineages_per_seg)
	# compute BIC
	bic = model.get_BIC(seg_num, sublin_num, ssm_num, snp_num, allele_specific, llh)

	if write_result_file and  result_file_name is not None:
		oio.write_lineages_to_result_file(result_file_name, combined_lineages, test=test)

	logging.info("LLH for all individual mutations: {0}\nBIC: {1}\nZ-matrix:{2}".format(
		llh, bic, best_z_matrices[0]))
	logging.info("Z-matrices that reach the same LLH: {0}".format(best_z_matrices[1:]))


# seg_w_cn_change_indices: indices of segments that are used here
# ssm_list: list with SSM objects
# ssm_indices_of_segment_per_seg: indices of SSMs in ssm_list per segment
# cluster_num: number of clusters per segment
# cluster_labels_per_seg: list with cluster labels of previous runs, can be empty
# return: ssm_indices_per_cluster_per_seg: 3D list, 1D: indices of SSMs that belong to one cluster,
#	2D: elements of one segment, indices in lists of 2D list correspond to indices of 2D list 
#	of ssm_objects_of_segment_per_seg
# return: ssm_objects_of_segment_per_seg: 2D list with SSMs that belong to one segment
def choose_SSMs_cluster_create_indices(seg_w_cn_change_indices, ssm_list, ssm_indices_of_segment_per_seg,
	cluster_num, cluster_labels_per_seg, kmeans=True, agglo_ward=False):

	# if cluster_labels_per_seg already contains a first entry (from normal segment clustering)
	# don't use it
	cluster_labels_per_seg_empty = True
	if len(cluster_labels_per_seg) == 1:
		cluster_labels_per_seg_empty = False
	elif len(cluster_labels_per_seg) > 1:
		raise eo.MyExecption("Clustering not right, more segments already labeled"
			"than expected.")
	# choose SSMs that belong to these segments
	ssm_indices_of_segments_w_cn_change = model.choose_ssm_subset(seg_w_cn_change_indices, ssm_list,
		normal_segments=False)
	# get SSM objects of the indices for SSMs in new segments
	ssm_objects_of_new_segments_per_seg = ([[ssm_list[i] for i in list_per_seg]
		for list_per_seg in ssm_indices_of_segments_w_cn_change])
	# get SSM objects of the old segment
	ssm_objects_of_old_segment_per_seg = ([[ssm_list[i] for i in list_per_seg]
		for list_per_seg in ssm_indices_of_segment_per_seg])
	# combine SSM objects of old and new segments
	ssm_objects_of_segment_per_seg = (ssm_objects_of_old_segment_per_seg 
		+ ssm_objects_of_new_segments_per_seg)
	
	# cluster SSMs segment-wise
	for i in xrange(len(seg_w_cn_change_indices)):
		# create numpy array with VAFs of SSMs
		segment_VAFs = np.array([float(my_ssm.variant_count) / 
			(my_ssm.variant_count + my_ssm.ref_count)
			for my_ssm in ssm_objects_of_new_segments_per_seg[i]]).reshape(
			len(ssm_objects_of_new_segments_per_seg[i]), 1)
		# cluster SSMs based on VAF with kmeans
		cluster_labels = model.cluster_VAFs(segment_VAFs, cluster_num[i], kmeans=kmeans, agglo_ward=agglo_ward)
		cluster_labels_per_seg.append(cluster_labels)
	
	# if cluster_labels_per_seg already contains a first entry (from normal segment clustering)
	# don't use it
	tmp_cluster_labels_per_seg = cluster_labels_per_seg
	if not cluster_labels_per_seg_empty:
		tmp_cluster_labels_per_seg = cluster_labels_per_seg[1:]

	# create SSM indices per cluster per segment
	ssm_indices_per_cluster_per_seg = model.create_ssm_indices_per_cluster(
		tmp_cluster_labels_per_seg, cluster_num)

	return (ssm_indices_per_cluster_per_seg, ssm_objects_of_segment_per_seg)

def cluster_ssms_normal_segs_optimization(ssm_input_file, seg_input_file,
	sublin_num, normal_seg_indices,
	allele_specific=True, number_spline_points=50,
	write_result_file=False, result_file_name=None, test=False,
	threads=1, kmeans=True, agglo_ward=False):

	overdispersion = 1000

	cluster_num_normal = sublin_num - 1

	if write_result_file and result_file_name is None:
		logging.warning("Result file and log files won't be written because name is not set.")
	if not test and write_result_file:
		oio.raise_if_file_exists(result_file_name)		

	if allele_specific == False:
		raise eo.MyException("not done for non-allele-specific")

	logging.info("ssm_input_file: {0}\nseg_input_file: {1}\nsublin_num: {2}\n"
		"normal_seg_indices: {3}\nallele_specific: {4}\nresult_file_name: {5}\n"
		.format(ssm_input_file, seg_input_file, sublin_num, normal_seg_indices,
		allele_specific, result_file_name))
	
	# read in data
	(seg_list, ssm_list) = model.create_segment_and_mutation_lists(seg_input_file, 
		[], ssm_input_file, allele_specific=True)
	seg_num = len(seg_list)
	ssm_num = len(ssm_list)
	snp_num = 0
	
	# choose subset of SSMs
	ssm_indices_of_segment = model.choose_ssm_subset(normal_seg_indices, ssm_list)
	ssm_objects_of_segment = [ssm_list[i] for i in ssm_indices_of_segment]
	# choose subset of segments
	chosen_segments = model.choose_seg_subset(normal_seg_indices, seg_list)
	# create numpy array with VAFs of SSMs
	#segment_VAFs = model.create_VAF_array_of_index_SSMs(ssm_list, ssm_indices_of_segment)
	segment_VAFs = np.array([float(my_ssm.variant_count) / (my_ssm.variant_count + my_ssm.ref_count)
		for my_ssm in ssm_objects_of_segment]).reshape(len(ssm_objects_of_segment), 1)
	# cluster SSMs based on VAF with kmeans
	logging.debug("Cluster SSMs...")
	start_clustering = time.time()
	cluster_labels = model.cluster_VAFs(segment_VAFs, cluster_num_normal, kmeans=kmeans, agglo_ward=agglo_ward)
	end_clustering = time.time()
	logging.debug("Time for Clustering: {0}".format(end_clustering-start_clustering))
	# create ssm indices per cluster
	ssm_indices_per_cluster = model.create_ssm_indices_per_cluster([cluster_labels], [cluster_num_normal])

	# create splines
	(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
		chosen_segments, [], ssm_objects_of_segment, number_spline_points, 
		allele_specific=True, overdispersion=overdispersion)

	# create optimization object
	# single_segment is set to be true because seg_index in optimization doesn't matter because
	#	all segments have a normal CN
	#	and when single_segment is not set, the segment indices of the SSMs would have
	#	to be converted to lower numbers because they can be higher than the actual
	#	index than the segment in the use chosen_segment list
	cplex_obj = optimization.Optimization_with_CPLEX([], [], ssm_splines, allele_specific=True,
		seg_splines_A=seg_splines_A, seg_splines_B=seg_splines_B, single_segment=True)
	cplex_obj.set_other_parameter(sublin_num, [], ssm_objects_of_segment, chosen_segments)

	if write_result_file:
		cplex_obj.cplex_log_file = result_file_name + ".cplex.log"
	else:
		cplex_obj.cplex_log_file = None
	cplex_obj.threads = threads
	# create assignment for normal segments
	# all segments have a normal copy number, so no copy number changes are allowed
	fixed_cn_normal_list = [0] * cplex_obj.cnv_aux_all_states_one_type_entry_num
	# create fixed  Z matrix values
	# because no copy number changes occur, we don't get information on branching and thus
	# can use a linear phylogeny for the inference
	z_matrix_list = [1] * model.get_number_of_untrivial_z_entries(sublin_num)
	z_matrix = model.parse_fixed_z_matrix_list_to_matrix(z_matrix_list, sublin_num)
	direct_descendant = model.get_direct_descendants(z_matrix, sublin_num)
	direct_descendants_for_constraints = model.parse_direct_descendants_for_constraints(
		direct_descendant)
	# start optimization
	cplex_obj.opt_with_CPLEX(sublin_num, [], ssm_objects_of_segment, chosen_segments, 
		fixed_cnv=fixed_cn_normal_list,	fixed_z_matrix=z_matrix_list, 
		direct_descendants_for_constraints=direct_descendants_for_constraints,
		ssm_indices_per_cluster=ssm_indices_per_cluster)

	# create lineage object
	my_lineages = model.get_lineages_ob_from_CPLEX_results(cplex_obj, [], ssm_objects_of_segment, chosen_segments)
	# write lineage file
	if write_result_file and result_file_name is not None:
		oio.write_lineages_to_result_file(result_file_name, my_lineages, test=test)
			
	# compute BIC
	parameter_num = model.get_parameter_num(seg_num, sublin_num, ssm_num, snp_num)
	sample_size = model.get_sample_size(seg_num, ssm_num, snp_num, allele_specific)
	bic = model.compute_BIC(cplex_obj.my_prob.solution.get_objective_value(), parameter_num, sample_size)
	logging.info("LLH for all individual mutations: {0}\nBIC: {1}".format(
		cplex_obj.my_prob.solution.get_objective_value(), bic))

	return (my_lineages, cplex_obj, bic, ssm_indices_of_segment, cluster_labels)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--threads", default = 1, type = int, help = "TODO")
	parser.add_argument("--ssm_input_file", default = None, help = "TODO", required=True)
	parser.add_argument("--seg_input_file", default = None, help = "TODO", required=True)
	parser.add_argument("--sublin_num", default = None, type = int, help = "TODO", required=True)
	parser.add_argument("--normal_seg_indices_file", default = None, help = "TODO", required=True)
	#parser.add_argument("--cluster_num", default = None, type = int, help = "TODO")
	#parser.add_argument("--cluster_num_param", default = None, type = int, help = "TODO")
	#parser.add_argument("--cluster_num_normal", default = None, type = int, help = "TODO")
	parser.add_argument("--allele_specific", help = "TODO")
	parser.add_argument("--number_spline_points", default = 50, type = int, help = "TODO")
	parser.add_argument("--write_result_file", default = None, help = "TODO")
	parser.add_argument("--result_file_name", default = None, help = "TODO")
	parser.add_argument("--test", help = "TODO")
	parser.add_argument("--cluster_ssms_normal_segs_optimization", default = None, help = "TODO")
	parser.add_argument("--simple_CN_changes", help = "TODO")
	parser.add_argument("--max_x_CN_changes", default = 2, type = int, help = "TODO")
	parser.add_argument("--only_one_loss", help = "TODO")
	parser.add_argument("--only_gains_losses_LOH", help = "TODO")
	parser.add_argument("--cn_weight", default = 0.00001, help = "TODO")
	parser.add_argument("--use_start_cluster_heuristic_fixed_phi_pure_segmentwise", 
		default = None, help = "TODO")
	parser.add_argument("--kmeans_clustering", action='store_true')
	parser.add_argument("--agglo_ward_clustering", action='store_true')

	args = parser.parse_args()

	normal_seg_indices = []
	if args.normal_seg_indices_file is not None:
		with open(args.normal_seg_indices_file) as f:
			for line in f:
				normal_seg_indices = map(int, line.rstrip("\n").split("\t"))

	allele_specific = True
	if args.allele_specific:
		allele_specific = oio.str_to_bool(args.allele_specific)
	write_result_file = False
	if args.write_result_file:
		write_result_file = oio.str_to_bool(args.write_result_file)
	test = False
	if args.test:
		test = oio.str_to_bool(args.test)
	simple_CN_changes = False
	if args.simple_CN_changes:
		simple_CN_changes = oio.str_to_bool(args.simple_CN_changes)
	only_one_loss = False
	if args.only_one_loss:
		only_one_loss = oio.str_to_bool(args.only_one_loss)
	only_gains_losses_LOH = True
	if args.only_gains_losses_LOH:
		only_gains_losses_LOH = oio.str_to_bool(args.only_gains_losses_LOH)
	use_cluster_ssms_normal_segs_optimization = False
	if args.cluster_ssms_normal_segs_optimization:
		use_cluster_ssms_normal_segs_optimization = oio.str_to_bool(
			args.cluster_ssms_normal_segs_optimization)
	use_start_cluster_heuristic_fixed_phi_pure_segmentwise = False
	if args.use_start_cluster_heuristic_fixed_phi_pure_segmentwise:
		use_start_cluster_heuristic_fixed_phi_pure_segmentwise = (
			oio.str_to_bool(args.use_start_cluster_heuristic_fixed_phi_pure_segmentwise))

	numeric_logging_info = getattr(logging, "DEBUG".upper(), None)
	logging.basicConfig(filename=args.result_file_name+".log",
		filemode='w', level=numeric_logging_info)

	if use_cluster_ssms_normal_segs_optimization:
		cluster_ssms_normal_segs_optimization(args.ssm_input_file, args.seg_input_file,
			args.sublin_num, normal_seg_indices,
			allele_specific=allele_specific, number_spline_points=args.number_spline_points,
			write_result_file=write_result_file, result_file_name=args.result_file_name,
			test=test, threads=args.threads, kmeans=args.kmeans_clustering,
			agglo_ward=args.agglo_ward_clustering)
	elif use_start_cluster_heuristic_fixed_phi_pure_segmentwise:
		cluster_num = cons.CLUSTER_NUM_PARAM
		start_cluster_heuristic_fixed_phi_pure_segmentwise(args.ssm_input_file, args.seg_input_file,
			args.sublin_num, normal_seg_indices, cluster_num,
			allele_specific=allele_specific, number_spline_points=args.number_spline_points,
			test=test, threads=args.threads, write_result_file=write_result_file,
			result_file_name=args.result_file_name, 
			simple_CN_changes=simple_CN_changes, max_x_CN_changes=int(args.max_x_CN_changes),
			only_one_loss=only_one_loss, only_gains_losses_LOH=only_gains_losses_LOH,
			cn_weight=float(args.cn_weight), kmeans=args.kmeans_clustering, agglo_ward=args.agglo_ward_clustering)

