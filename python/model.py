import sys
import exceptions_onctopus as eo
import math
import lineage
import onctopus_io as oio
import numpy as np
import copy
import evaluation
import constants as cons
import operator
import cnv
import logging
import log_pdf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import snp_ssm
import segment
import copy
import main

# given lineages with SSMs, assigns the index of the lineage to the SSM
def assign_lin_index_to_ssms(my_lineages):
	for i, my_lin in enumerate(my_lineages):
		assign_lin_index_to_ssms_phased(my_lin.ssms, i)
		assign_lin_index_to_ssms_phased(my_lin.ssms_a, i)
		assign_lin_index_to_ssms_phased(my_lin.ssms_b, i)

def assign_lin_index_to_ssms_phased(ssm_list, lin_index):
	for my_ssm in ssm_list:
		my_ssm.lineage = lin_index

# checks the frequency of the lineages, when a lineage has a frequency ~ 0, it is removed and the others are updated
# my_lineages: list with lineages
# z_matrix_list: list with Z-matrix, when Onctopus was just started, this list is not defined, the following
#	function only works when it is not defined
def shorten_lineages(my_lineages, z_matrix_list):
	
	parameter_lin_num = len(my_lineages)

	# z_matrix_list needs to be None
	if z_matrix_list is not None:
		raise eo.ZMatrixNotNone("Z-matrix should be 'None' here, function used at wrong position.")

	# iterate through lineage list, start at the end
	for i in xrange(len(my_lineages)-1, -1, -1):
		# check value of lineage frequency
		# if too small, remove
		if my_lineages[i].freq < cons.EPSILON_FREQUENCY:
			# if lineage frequency is close enough to epsilon lineage frequency, it's okay
			if abs(my_lineages[i].freq - cons.EPSILON_FREQUENCY) < 0.0000000001:
				continue
			# get mutation number
			mutation_count = count_mutation_number(my_lineages[i])
			# lineage is not allowed to have mutations when it has such a low frequency
			if mutation_count > 0:
				raise eo.LineageWith0FreqMutations("Lineage {0} with frequency of {1} has {2} "
					"mutation(s) assigned.".format(i, my_lineages[i].freq, mutation_count))

			# last entry of lineage list is removed
			my_lineages.pop()
			# lineages with lower index are checked wether they were ancestors of the one that got removed
			shorten_sublineages(my_lineages, i)

		# if lineage frequency is high enough, no other lineages need to be checked
		else:
			break

	logging.info("Parameter for lineage number: {0}".format(parameter_lin_num))
	logging.info("Used lineage number: {0}".format(len(my_lineages)))

# after lineage i was removed from lineage list my_lineages, sublineages of other lineages are checked
# if i is contained in it, it is removed
def shorten_sublineages(my_lineages, i):
	for lin in my_lineages:
		for index in xrange(len(lin.sublins) - 1, -1, -1):
			if lin.sublins[index] == i:
				lin.sublins.pop(index)
				break
			if lin.sublins[index] < i:
				break

# counts the number of all mutations in a lineage
def count_mutation_number(lin):
	return len(lin.cnvs_a) + len(lin.cnvs_b) + len(lin.ssms) + len(lin.ssms_a) + len(lin.ssms_b)

# counts the number of SSMs that appear in a lineage segment with CN gains
def count_ssms_in_lineage_segments_with_cn_gains(my_lins):
	count = 0
	for my_lin in my_lins:
		count += count_ssms_in_lineage_segments_with_cn_gains_phase(my_lin.ssms_a, my_lin.cnvs_a)
		count += count_ssms_in_lineage_segments_with_cn_gains_phase(my_lin.ssms_b, my_lin.cnvs_b)
	return count

# counts the number of SSMs that appear in a phased lineage segment with CN gains
def count_ssms_in_lineage_segments_with_cn_gains_phase(ssms_phased, cnvs_phased):
	count = 0
	cnv_list_index = 0
	# no CN changes exist in that lineage and phase
	if len(cnvs_phased) == 0:
		return 0
	for my_ssm in ssms_phased:
		# increase segment index of CNVs when it's not equal to segment index of SSM
		while cnv_list_index < len(cnvs_phased) and my_ssm.seg_index > cnvs_phased[cnv_list_index].seg_index:
			cnv_list_index += 1
		# no CN change with same segment index exists
		if cnv_list_index == len(cnvs_phased):
			break
		# SSM is on the same segment the CN change
		if my_ssm.seg_index == cnvs_phased[cnv_list_index].seg_index:
			# if CN change is a gain
			if cnvs_phased[cnv_list_index].change == 1:
				count += 1
	return count
			
# counts lineages that have a frequency at least as high as the minimal allowed value
def count_lineages_based_on_freq(lins):
	lin_num = 0
	for my_lin in lins:
		# frequency is higher than minimal allowed one
		if my_lin.freq > cons.EPSILON_FREQUENCY:
			lin_num += 1
		else:
			# lineage frequency is only sliiiightly lower than minimal allowed value
			if abs(my_lin.freq - cons.EPSILON_FREQUENCY) < 0.0000000001:
				lin_num += 1
	return lin_num

# returns the bits used to send the parental information for each lineage k >= 2
# \sum_{k=2}^{K-1} log(k)
def get_parent_bits(lin_num):
	return sum([np.log(i) for i in xrange(2,lin_num)])
	
# computes different numbers of segments and CNAs
def get_number_of_different_segments_and_CNAs(my_lins, seg_num):
	# number of CNA-free segments
	seg_num_CNA_free = len(my_lins[0].cnvs_a)

	# number of segments with CNAs
	seg_num_w_CNAs = seg_num - seg_num_CNA_free

	# number of CNAs
	cna_count = 0
	for i in xrange(1, len(my_lins)):
		cna_count += len(my_lins[i].cnvs_a) + len(my_lins[i].cnvs_b)

	# number of "additional" CNAs
	additional_cna_count = cna_count - seg_num_w_CNAs

	return seg_num_CNA_free, seg_num_w_CNAs, cna_count, additional_cna_count

# computes the number of SSMs that appear on segments with CNAs
def get_number_SSMs_on_segs_w_CNAs(my_lins, seg_num):
	# indices of segments that contain at least one CNA
	segs_w_CNAs = get_segments_w_CNAs(my_lins, seg_num)

	# compare SSM segment indices with these in segment list
	ssm_count = 0
	for i in xrange(1, len(my_lins)):
		# phased SSMs only appear on segments with CNAs
		ssm_count += len(my_lins[i].ssms_a) + len(my_lins[i].ssms_b)
		# check which unphased SSMs also appear on segments with CNAs
		unphased_ssms = my_lins[i].ssms
		ssm_index = 0
		current_seg_index = 0
		while(ssm_index < len(unphased_ssms) and current_seg_index < len(segs_w_CNAs)):
			# SSM appears on segment with CNA
			if unphased_ssms[ssm_index].seg_index == segs_w_CNAs[current_seg_index]:
				ssm_count += 1
				ssm_index += 1
			# segment index of SSM is lower than the one of segment with CNA
			# SSM appears on segment without CNA
			elif unphased_ssms[ssm_index].seg_index < segs_w_CNAs[current_seg_index]:
				ssm_index += 1
			# segment index of SSM is higher than the one of segment with CNA
			# maybe SSM appears on other segment with CNAs
			elif unphased_ssms[ssm_index].seg_index > segs_w_CNAs[current_seg_index]:
				current_seg_index += 1
	return ssm_count

# returns the indices of segments that contain at least one CNA
def get_segments_w_CNAs(my_lins, seg_num):
	# compute indices of segments that contain CNAs
	segs_w_CNAs = [i for i in xrange(seg_num)]
	# remove indices of CNAs that are CNA-free
	current_seg_index = 0
	cna_free_index = 0
	cna_free_segments = my_lins[0].cnvs_a
	while(current_seg_index < len(segs_w_CNAs) and cna_free_index < len(cna_free_segments)):
		# current segment is CNA-free
		if segs_w_CNAs[current_seg_index] == cna_free_segments[cna_free_index].seg_index:
			# remove segment, check next segments with next CNA-free segments
			segs_w_CNAs.pop(current_seg_index)
			cna_free_index += 1
		# current segment has lower index than CNA-free segment
		elif segs_w_CNAs[current_seg_index] < cna_free_segments[cna_free_index].seg_index:
			# increase segment index
			current_seg_index += 1
	return segs_w_CNAs

def compute_MDL(my_lins, seg_num, ssm_number, llh):

	# count frequencies > 0
	inf_lin_num = count_lineages_based_on_freq(my_lins)

	# count inferred CNAs and number of phased SSMs
	seg_num_CNA_free, seg_num_w_CNAs, cna_count, additional_cna_count = get_number_of_different_segments_and_CNAs(
		my_lins, seg_num)
	additional_cna_count_part = 0
	if additional_cna_count > 0:
		additional_cna_count_part = 2 * np.log(additional_cna_count)

	# get SSMs in segments with CNAs
	ssms_w_cna = get_number_SSMs_on_segs_w_CNAs(my_lins, seg_num)

	# get parent bits
	parent_bits = get_parent_bits(inf_lin_num)

	# count ssms in lineage segment of CN gain
	ssms_w_gain = count_ssms_in_lineage_segments_with_cn_gains(my_lins)

	# compute MDL
	if inf_lin_num > 1:
		mdl = llh - ((2 * np.log(inf_lin_num-1)) + 1 + additional_cna_count_part
			+ 1 + ((seg_num_CNA_free + cna_count) * np.log(inf_lin_num))
			+ ((seg_num_CNA_free + cna_count) * np.log(2)) + (cna_count * np.log(2))
			+ (cna_count * np.log(2))
			+ (ssm_number * np.log(inf_lin_num-1)) + (ssms_w_cna * np.log(3))
			+ (ssms_w_gain * np.log(2)) + ((inf_lin_num-1) * np.log(1000))
			+ parent_bits)
	else:
		mdl = llh - (1 + additional_cna_count_part
			+ 1 + ((seg_num_CNA_free + cna_count) * np.log(inf_lin_num))
			+ ((seg_num_CNA_free + cna_count) * np.log(2)) + (cna_count * np.log(2))
			+ (cna_count * np.log(2))
			+ (ssms_w_cna * np.log(3))
			+ (ssms_w_gain * np.log(2)) 
			+ parent_bits)

	return mdl

# checks whether it is possible that a reconstruction with the fixed phi value exists
def check_phi_lineage_and_vaf_compatibility(fixed_phi_list, ssm_list, seg_list, num):
	# result is possible if there are more than 2 lineages
	if num > 2:
		return True
	
	# otherwise it needs to be checked for all segments that contain SSMs that the VAF of SSMs is not larger than 1
	seg_index = -1
	for my_ssm in ssm_list:
		if my_ssm.seg_index != seg_index:
			seg_index = my_ssm.seg_index
			ci = seg_list[seg_index].given_cn_A + seg_list[seg_index].given_cn_B
			vaf = fixed_phi_list[0] / ci
			if vaf > 1:
				return False
	
	return True

# SSMs in lineages are set to original positions 
def decombine_normal_segments_and_ssms(my_lineages, new_seg_list, new_ssm_list, new_normal_seg_indices, seg_list, ssm_list,
	ssm_normal_changed, ssm_normal, normal_seg_indices, original_seg_index):
	
	# deep copy new SSM lists 
	ssm_normal_changed_copied = copy.deepcopy(ssm_normal_changed)
	new_ssm_list_copied = copy.deepcopy(new_ssm_list)
	new_seg_list_copied = copy.deepcopy(new_seg_list)

	# set chromosome and position of normal SSMs back to their original values
	for i in xrange(len(ssm_normal)):
		ssm_normal_changed[i].chr = ssm_normal[i].chr
		ssm_normal_changed[i].pos = ssm_normal[i].pos

	# sort SSMs in new_seg_list now that normal SSMs have their original values
	new_ssm_list = sort_snps_ssms(new_ssm_list)
	# update index of SSMs
	assign_muts_to_segments(seg_list, new_ssm_list)

	# sort SSMs of lineages and update CNV indices
	for i in xrange(len(my_lineages)):
		my_lineages[i].ssms = sort_snps_ssms(my_lineages[i].ssms)
		my_lineages[i].ssms_a = sort_snps_ssms(my_lineages[i].ssms_a)
		my_lineages[i].ssms_b = sort_snps_ssms(my_lineages[i].ssms_b)
		update_indices(my_lineages[i].cnvs_a, original_seg_index)
		update_indices(my_lineages[i].cnvs_b, original_seg_index)

	# update normal lineage with segments without CN changes
	my_cnvs = []
	for i in normal_seg_indices:
		no_cnv = cnv.CNV(0, i, seg_list[i].chr, seg_list[i].start, seg_list[i].end)
		no_cnv.phase = cons.A
		my_cnvs.append(no_cnv)
	my_lineages[0].cnvs_a = sort_segments(my_cnvs + my_lineages[0].cnvs_a[1:])

	return (seg_list, new_ssm_list, normal_seg_indices, new_seg_list, new_ssm_list_copied, new_normal_seg_indices,
		ssm_normal_changed_copied, ssm_normal)

def update_indices(cnv_list, original_seg_index):
	for my_cnv in cnv_list:
		my_cnv.seg_index = original_seg_index[my_cnv.seg_index]


# given a list with segments and SSMs and indices of normal segments, 
# combines normal segments to one segment and assigns their SSMs to them
# returns reordered list of segments and SSMs
def combine_normal_segments_and_ssms(seg_list, ssm_list, normal_seg_indices):

	# combine normal segments, get new list with combined normal and other segments
	median_read_count = np.median([ssm.variant_count + ssm.ref_count 
		for ssm in ssm_list])
	new_seg_list, original_seg_index = combine_normal_segments_to_new_list(seg_list, normal_seg_indices)

	# combine SSMs of normal segments, get new list with combined normal and other SSMs
	new_ssm_list, ssm_normal_changed, ssm_normal = combine_normal_ssms_to_new_list(ssm_list,
		normal_seg_indices, new_seg_list)

	# compute standard error of normal segment
	main.compute_standard_error_of_segments_without_files([new_seg_list[0]], median_read_count, [len(ssm_normal)])

	new_normal_seg_indices = [0]

	return (new_seg_list, new_ssm_list, new_normal_seg_indices, seg_list, ssm_list, 
		ssm_normal_changed, ssm_normal, normal_seg_indices, original_seg_index)

# assigns SSM that appear on segments without CN change to first segment, which is the normal one
# chromosome and position of these SSMs are changed
# all SSMs get assign a new segment index, based on segment position of new segment list
def combine_normal_ssms_to_new_list(ssm_list, normal_seg_indices, new_seg_list):
	ssm_normal = []
	ssm_normal_changed = []
	ssm_not_normal = []

	# iterate through all SSMs
	current_normal_index = 0
	for i, ssm in enumerate(ssm_list):
		# SSM appears on a segment without CN changes
		if current_normal_index < len(normal_seg_indices) and ssm.seg_index == normal_seg_indices[current_normal_index]:
			ssm_normal.append(ssm)
		# SSM appears on next segment that has no CN changes
		elif (current_normal_index+1 < len(normal_seg_indices) 
			and ssm.seg_index == normal_seg_indices[current_normal_index+1]):
			ssm_normal.append(ssm)
			current_normal_index += 1
		# SSM appears on segment without CN changes
		else:
			ssm_not_normal.append(ssm)

	# copy normal SSMs and change chromosome and position
	ssm_normal_changed = copy.deepcopy(ssm_normal)
	for i, ssm in enumerate(ssm_normal_changed):
		ssm.chr = -1
		ssm.pos = i+1

	# change segment indices of SSMs
	new_ssm_list = ssm_normal_changed + ssm_not_normal
	assign_muts_to_segments(new_seg_list, new_ssm_list)

	return new_ssm_list, ssm_normal_changed, ssm_normal

# combine normal segment in first segment, appends all other segments afterwards
def combine_normal_segments_to_new_list(seg_list, normal_seg_indices):
	# create normal segment and put into new list
	normal_segment = segment.Segment_allele_specific(-1, 1, 0, 1.0, 0, 1.0, 0)
	normal_segment.index = 0
	new_seg_list = [normal_segment]
	original_seg_index = [-1] * len(seg_list)

	# iterate through all segments
	current_normal_index = 0
	non_normal_index = 1
	for i, seg in enumerate(seg_list):
		# segment has no CN change
		if current_normal_index < len(normal_seg_indices) and i == normal_seg_indices[current_normal_index]:
			# increase length of normal segment by length of current segment
			seg_length = seg.end - seg.start + 1
			normal_segment.end = normal_segment.end + seg_length
			# increase index
			current_normal_index += 1
		# segment has CN change
		else:
			copied_seg = copy.deepcopy(seg)
			copied_seg.index = non_normal_index
			new_seg_list.append(copied_seg)

			original_seg_index[non_normal_index] = i

			non_normal_index += 1

	return new_seg_list, original_seg_index

# checks for each present lineage relationship whether a CN influence exists
# if not, relationship is removed
def check_CN_influence(z_matrix, lineages):

	lin_num = len(lineages)
	
	# get hashs with CN changes and SSM appearances
	CN_changes_hash, SSMs_hash = create_CN_changes_and_SSM_hash_for_LDR(lineages)

	# iterate through Z-matrix
	for k in xrange(1, lin_num-1):
		for k_prime in xrange(k+1, lin_num):
			# check CN influecne, if a 1 is in Z-matrix
			if z_matrix[k][k_prime] == 1:
				# if no CN influence is present, relationship is removed
				if not is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash):
					z_matrix[k][k_prime] = 0

# checks whether a CN change in lineage k_prime influences the VAF of an SSM in lineage k
# this is the case, when one CN change in k_prime lies on the same segment and phase as one SSM in k
def is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash):
	# check all segments in which k_prime contains CN changes
	try:
		for seg in CN_changes_hash[k_prime].keys():
			# check all phases of the CN changes
			for phase in CN_changes_hash[k_prime][seg]:
				try:
					# if an SSM in k appears on the same segment with the same phase,
					# there is an influence
					if phase in SSMs_hash[k][seg]:
						return True
				# there is no SSM on the segment in lineage k
				except KeyError:
					pass
	# lineage k_prime doesn't contain any CN changes
	except:
		return False

	# when no CN influence was found, there is non
	return False

# creates a hash with the CN changes and one with SSMs
# CN_changes hash: [lineage][segment] = [list with phases]
# SSM hash: [lineage][segment] = [list with phases]
def create_CN_changes_and_SSM_hash_for_LDR(lineages):
	# create empty hashes
	CN_changes = {}
	SSMs = {}

	# iterate through all lineages
	for lin_index, my_lin in enumerate(lineages):
		# don't consider normal lineage
		if lin_index == 0:
			continue

		# add CNVs
		add_CN_changes_to_hash(CN_changes, my_lin.cnvs_a, cons.A, lin_index)
		add_CN_changes_to_hash(CN_changes, my_lin.cnvs_b, cons.B, lin_index)

		# add SSMs
		add_SSM_appearence_to_hash(SSMs, my_lin.ssms_a, cons.A, lin_index)
		add_SSM_appearence_to_hash(SSMs, my_lin.ssms_b, cons.B, lin_index)

	return CN_changes, SSMs

# for each lineage, an entry is added for the first SSM of one phase in a segment
def add_SSM_appearence_to_hash(SSMs_hash, ssms, phase, lin_index):
	for ssm in ssms:
		try:
			# it's sufficient that the entry is done for the first SSM with the phase on the segment
			if phase in SSMs_hash[lin_index][ssm.seg_index]:
				continue
			SSMs_hash[lin_index][ssm.seg_index].append(phase)
		except KeyError:
			try:
				SSMs_hash[lin_index][ssm.seg_index] = [phase]
			except KeyError:
				SSMs_hash[lin_index] = {}
				SSMs_hash[lin_index][ssm.seg_index] = [phase]

# adds the CNVs in a list to the hash
def add_CN_changes_to_hash(CN_changes_hash, cnvs, phase, lin_index):
	for cnv in cnvs:
		try:
			if phase in CN_changes_hash[lin_index][cnv.seg_index]:
				raise eo.MyException("Only one CNV in this lineage in this segment should have this phase.")
			CN_changes_hash[lin_index][cnv.seg_index].append(phase)
		except KeyError:
			try:
				CN_changes_hash[lin_index][cnv.seg_index] = [phase]
			except KeyError:
				CN_changes_hash[lin_index] = {}
				CN_changes_hash[lin_index][cnv.seg_index] = [phase]
				




# SSMs after reconstruction don't have information about the variant and reference count
# this information is taken here from the SSM list that is used as input for the optimization
# list_counts: list with SSMs which are used as input, has variant and reference count information
# list_reconstruction: list with SSMs after reconstruction, has lineage information and maybe also phase and
#	CN influence in same lineage
def combine_ssm_lists_with_different_info(list_counts, list_reconstruction):

	# lists have to have the same content, thus the same length
	if len(list_counts) != len(list_reconstruction):
		raise eo.MyException("SSM lists need to have the same content!")

	# all SSMs are processed
	for i in xrange(len(list_counts)):
		# SSMs at same index position have to describe the same SSM, thus the chromosome
		# and position need to be equal
		if not list_counts[i] == list_reconstruction[i]:
			raise eo.MyException("SSMs need to have same chromosome and position!")

		# variant and reference count values are copied
		list_reconstruction[i].variant_count = list_counts[i].variant_count
		list_reconstruction[i].ref_count = list_counts[i].ref_count

	return list_reconstruction

# creates a list with SSMs from the lineages, sorted by chromosome and position
def build_ssm_list_from_lineages(lineages):
	ssm_list = []

	# append all SSMs to list
	for my_lin in lineages:
		ssm_list.extend(my_lin.ssms + my_lin.ssms_a + my_lin.ssms_b)

	# sort SSMs according to chromosome and position
	ssm_list = sorted(ssm_list, key=lambda x: (x.chr, x.pos))

	return ssm_list


# after the optimization, the Z-matrix can be refined 
#	if the Z-matrix gets forked, also new lineage reconstructions are created
#	the lineage divergence rule is also checked and new lineage reconstructions might be created as well
# my_lineages: list with all lineages from the reconstruction
# seg_num: number of segments
def get_all_possible_z_matrices_with_lineages(my_lineages, seg_num):

	# get Z-matrix
	z_matrix = get_Z_matrix(my_lineages)[0]
	# only keep CN influence 1's
	logging.debug("check_CN_influence")
	check_CN_influence(z_matrix, my_lineages)
	# get number of 0's in Z-matrix
	zero_count = get_0_number_in_z_matrix(z_matrix)

	# complete Z-matrix is checked for consistency
	# also, triplets in the Z-matrix are created and can be updated
	lineage_num = len(my_lineages)
	logging.debug("check_and_update_complete_Z_matrix_from_matrix")
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(
		z_matrix, zero_count, lineage_num)

	# update lineages based on direct update of Z-matrix because of the transitivity rule
	logging.debug("update_linage_relations_based_on_z_matrix")
	update_linage_relations_based_on_z_matrix(my_lineages, z_matrix)
	# post anaylsis of Z-matrix
	logging.debug("post_analysis_Z_matrix")
	z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list = (
		post_analysis_Z_matrix(my_lineages, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
		get_CNVs=True))
	# look for LDR
	logging.debug("get_all_matrices_fulfilling_LDR")
	zmcos = create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list)
	zmcos = get_all_matrices_fulfilling_LDR(zmcos, my_lineages)
	# find absent ADRs necessary because of sum rule
	find_absent_adrs_necessary_sumrule(zmcos, my_lineages)

	# create new lineages that fit the new Z-matrices
	# z_matrix_list contains numpy arrays now
	logging.debug("adapt_lineages_after_Z_matrix_update")
	z_matrix_list = [zmco.z_matrix for zmco in zmcos]
	present_ssms_list = [zmco.present_ssms for zmco in zmcos]
	my_lineages, new_lineage_list = adapt_lineages_after_Z_matrix_update(my_lineages, z_matrix_fst_rnd, z_matrix_list, 
		origin_present_ssms, present_ssms_list)

	# feasibility check for lineage divergence rule
	logging.debug("post_opt_lineage_divergence_rule_feasibility_check")
	lin_div_rule_feasibility = post_opt_lineage_divergence_rule_feasibility_check(z_matrix_list, my_lineages)

	return my_lineages, z_matrix_list, new_lineage_list, lin_div_rule_feasibility

def find_absent_adrs_necessary_sumrule(zmcos, my_lineages):
	lin_num = len(my_lineages)
	# just needed as dummy value
	zero_count = lin_num * lin_num

	# check all reconstructions
	for reci in xrange(len(zmcos)):
		zmco = zmcos[reci]
		# check all relevant lineage relationship that are ambiguous
		for k in xrange(0, lin_num-2):
			for k_prime in xrange(k+1, lin_num-1):
				if zmco.z_matrix[k][k_prime] == 0:
					# get children
					children = get_children(zmco.z_matrix, k)
					
					# check sum rule
					if my_lineages[k].freq < (sum([my_lineages[chil].freq for chil in children])
						+ my_lineages[k_prime].freq):

						# copy current reconstruction
						zmco_copy = copy.deepcopy(zmco)
						# set lineages k and k' in ADR
						zmco_copy.z_matrix[k][k_prime] = 1
						# move unphased SSMs if necessary
						move_unphased_SSMs_if_necessary(k, k_prime, zmco_copy.present_ssms,
							zmco_copy.CNVs, zmco_copy.matrix_after_first_round, 1)
						# transitivity update
						update_Z_matrix_iteratively(zmco_copy.z_matrix, zero_count, 
							zmco_copy.triplet_xys, zmco_copy.triplet_ysx, zmco_copy.triplet_xsy,
							(k, k_prime), zmco_copy.present_ssms, zmco_copy.CNVs,
							zmco_copy.matrix_after_first_round)

						# try whether reconstruction can fulfill sum rule
						zmco_copy_list = [zmco_copy]
						zmco_copy_list = get_all_matrices_fulfilling_LDR(zmco_copy_list, 
							my_lineages, start_k=k)

						# if reconstruction cannot fulfil sum rule
						if len(zmco_copy_list) == 0:
							zmco.z_matrix[k][k_prime] = -1
							update_Z_matrix_iteratively(zmco.z_matrix, zero_count, zmco.triplet_xys,
								zmco.triplet_ysx, zmco.triplet_xsy, (k, k_prime),
								zmco.present_ssms, zmco.CNVs, zmco.matrix_after_first_round)

def create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, CNVs, triplets_list):
	zmcos = []
	for i in xrange(len(z_matrix_list)):
		zmco = Z_Matrix_Co(z_matrix=z_matrix_list[i], triplet_xys=triplets_list[i][0], 
			triplet_ysx=triplets_list[i][1], triplet_xsy=triplets_list[i][2], 
			present_ssms=present_ssms_list[i], CNVs=CNVs, matrix_after_first_round=z_matrix_fst_rnd)
		zmcos.append(zmco)
	return zmcos

# zmcos: list of objects that contain Z-matrices, xys-triplets and present ssm list
def get_all_matrices_fulfilling_LDR(zmcos, lineages, start_k=0):
	lin_num = len(zmcos[0].z_matrix)
	# zero_count isn't needed, just as dummy value
	zero_count = lin_num * lin_num

	# k = K-2 doesn't need to get checked because it can only have one child and thus it need to fulfill LDR
	for k in xrange(start_k, lin_num-2):
		logging.debug("k={0}".format(k))

		# new list in which valid and new Z-matrices and co gets written
		new_zmcos = []

		# iterate through whole list of Z-matrices
		logging.debug("len(zmcos) = {0}".format(len(zmcos)))
		while len(zmcos) != 0:

			# get last Z-matrix and co
			zmco = zmcos.pop()
			# get children of lineage k
			children = get_children(zmco.z_matrix, k)

			# if lineage k has only up to 1 child, the LDR must be fulfilled
			if len(children) <= 1:
				# current Z-matrix and co get stored to new list
				if not zmatrix_co_already_in_list(zmco, new_zmcos):
					new_zmcos.append(zmco)
				continue

			# check whether LDR is fulfilled
			freq_k = lineages[k].freq
			freq_children = sum([lineages[k_prime].freq for k_prime in children])
			if freq_k >= freq_children or (abs(freq_k - freq_children) <= 0.0000000001):
				# current Z-matrix and co get stored to new list
				if not zmatrix_co_already_in_list(zmco, new_zmcos):
					new_zmcos.append(zmco)
				continue

			# LDR is not fulfilled
			# get all possible children combinations
			combinations = get_all_possible_children_combinations(zmco.z_matrix, children)

			# if there exist no valid combinations, this Z-matrix can't fulfill the LDR, thus it is discarded
			if len(combinations) == 0:
				continue

			# check all possible combinations
			for comb in combinations:
				# copy Z-matrix and co
				new_zmco = copy.deepcopy(zmco)
				# make k'' a child of k'
				(k_prime, k_prime_prime) = (comb[0], comb[1])
				new_zmco.z_matrix[k_prime][k_prime_prime] = 1
				# move unphased SSMs if necessary
				move_unphased_SSMs_if_necessary(k_prime, k_prime_prime, new_zmco.present_ssms,
					new_zmco.CNVs, new_zmco.matrix_after_first_round, 1)
				# transitivity update
				update_Z_matrix_iteratively(new_zmco.z_matrix, zero_count, new_zmco.triplet_xys, 
					new_zmco.triplet_ysx, new_zmco.triplet_xsy, (k_prime, k_prime_prime),
					new_zmco.present_ssms, new_zmco.CNVs, new_zmco.matrix_after_first_round)

				# check whether LDR is fulfilled now, with k having one child less
				freq_less_children = freq_children - lineages[k_prime_prime].freq
				if freq_k >= freq_less_children:
					# if new Z-matrix isn't yet in list, add it
					if not zmatrix_co_already_in_list(new_zmco, new_zmcos):
						new_zmcos.append(new_zmco)
					continue

				# LDR is still not fulfilled
				# if current Z-matrix isn't in list, add it, so that it gets checked again
				if not zmatrix_co_already_in_list(new_zmco, zmcos):
					zmcos.append(new_zmco)

		# make new list to current list
		zmcos = new_zmcos

	return zmcos

# for a list of children, all possible combinations are returned, were one child becomes the child of another child
def get_all_possible_children_combinations(z_matrix, children):
	combinations = []
	number_children = len(children)

	for c1 in xrange(0, number_children-1):
		k_prime = children[c1]
		for c2 in xrange(c1+1, number_children):
			k_prime_prime = children[c2]
			# k'' is not allowed to be a descendant of k
			if z_matrix[k_prime][k_prime_prime] == -1:
				continue
			if z_matrix[k_prime][k_prime_prime] == 1:
				raise eo.MyException("Not possible that k'' is already a descendant of k'!")

			# possible that k'' becomes a child of k'
			combinations.append((k_prime, k_prime_prime))

	return combinations

# gets all children of a lineage k given the Z-matrix
def get_children(z_matrix, k):
	lin_num = len(z_matrix)
	children = []

	# check all potential descendant of k
	for k_prime in xrange(k+1, lin_num):
		# k isn't an ancestor of k'
		if z_matrix[k][k_prime] != 1:
			continue
		
		k_prime_potential_child = True
		# check all lineages that could be between k and k'
		for k_star in xrange(k+1, k_prime):
			# k' has another ancestor k*>k, thus k can't be the parent
			if z_matrix[k_star][k_prime] == 1:
				k_prime_potential_child = False
				break

		# no other ancestor of k' was found, thus k is the parent
		if k_prime_potential_child == True:
			children.append(k_prime)

	return children


# counts the number of 0's in the Z-matrix
def get_0_number_in_z_matrix(z_matrix):
	zero_count = 0
	for row in z_matrix:
		for entry in row:
			if entry == 0:
				zero_count += 1
	return zero_count

def check_and_update_complete_Z_matrix_from_matrix(z_matrix, zero_count, lineage_num):
	# 3 hashed to store triplets that contain 0s
	triplet_xys = {}
	triplet_ysx = {}
	triplet_xsy = {}

	# iterate through Z-matrix
	for x in xrange(lineage_num-3, 0, -1):
		for y in xrange(lineage_num-2, x, -1):
			for s in xrange(lineage_num-1, y, -1):
				# update Z triplet
				changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
					z_matrix[x][y], z_matrix[y][s], z_matrix[x][s]))
				# triplet contains at least one 0
				if triplet_zeros > 0:
					# add triplets to hash
					update_triplet_hash(triplet_xys, x, y, s)
					update_triplet_hash(triplet_ysx, y, s, x)
					update_triplet_hash(triplet_xsy, x, s, y)
				# triplet was changed
				if changed == True:
					zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field, 
						v_x, v_y, v_s, x, y, s, triplet_xys, triplet_ysx, triplet_xsy)

	return zero_count, triplet_xys, triplet_ysx, triplet_xsy

# two lineages are set in an ancestor-descendant relation
#	independant of the mutations they harbour
#	no phasing of SSMs is done here
# z_matrix: Z-matrix
# k, k_prime: indices of field in Z-matrix that should be set to 1
# my_lineages: list with lineages
# seg_num: number of segments
def activate_ancestor_descendant_relation(z_matrix, k, k_prime, my_lineages, seg_num):
	lineage_num = len(z_matrix)

	# some checks
	if k > k_prime:
		raise eo.MyException("k needs to be smaller than k_prime!")
	if z_matrix[k][k_prime] != 0:
		raise eo.MyException("Entry in Z-matrix needs to be 0, otherwise it can't be changed!")

	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segment once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
		ssm_infl_cnv_same_lineage)

	# next function needs zero_count but the parameter isn't important here
	# make something up that won't lead to problems
	zero_count = lineage_num * lineage_num

	# get triplets
	z_matrix_new = copy.deepcopy(z_matrix)
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(
		z_matrix_new, zero_count, lineage_num)

	# old and new Z-matrix need to be the same, otherwise it's possible that the old Z-matrix wasn't
	# updated and forked already
	if z_matrix != z_matrix_new:
		raise eo.MyException("Both matrices need to be equal")

	# activate ancestor-descendant relation in matrix
	z_matrix[k][k_prime] = 1
	# update SSM phasing
	present_ssms_new = copy.deepcopy(present_ssms)
	move_unphased_SSMs_if_necessary(k, k_prime, present_ssms_new, CNVs, z_matrix_new, 1)
	# check for updates in matrix because of the new a.-d. relation
	update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, (k, k_prime),
		present_ssms=present_ssms_new, CNVs=CNVs, matrix_after_first_round=z_matrix_new)

	# create new lineage object with newly phased SSMs, according to matrix update
	new_lineages = create_updates_lineages(my_lineages, 0, [z_matrix], present_ssms, [present_ssms_new])

	return True, new_lineages

# when a triplet was changed, things get updated:
#	zero count is decreased, Z-matrix is updated, it is checked, whether triplets which contain the
#	changed field can be updated as well
# z_matrix: the Z-matrix
# zero_count: number of 0 entires in the Z-matrix
# changed_field: the field in the triplet that was changed
# v_x, v_y, v_s: values of triplet entries
# x, y, s: indices of triplet entries
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# matrix_after_first_round: Z-matrix after the first round
# can raise exception ZInconsistence when updating of the Z_triplet let to inconsistance
def update_after_tiplet_change(z_matrix, zero_count, changed_field, v_x, v_y, v_s, x, y, s,
	triplet_xys, triplet_ysx, triplet_xsy, present_ssms=None, CNVs=None, matrix_after_first_round=None):
	# update number of 0's
	# each change decreases the number of 0's by one
	zero_count -= 1

	# get indices of pair that was influenced by change
	if changed_field == cons.X:
		index_pair = (x, y)
		new_z_entry = v_x
	elif changed_field == cons.Y:
		index_pair = (y, s)
		new_z_entry = v_y
	else:
		index_pair = (x, s)
		new_z_entry = v_s
	# update entry in Z-matrix
	z_matrix[index_pair[0]][index_pair[1]] = new_z_entry

	# if phasing should and has to be considered, consider it
	if present_ssms is not None and new_z_entry == 1:
		i = index_pair[0]
		i_prime = index_pair[1]
		phasing_allows_relation(i, i_prime, matrix_after_first_round, present_ssms, CNVs, new_z_entry)
		move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, CNVs, matrix_after_first_round, new_z_entry)

	# see whether previous triplets need to be updated
	try:
		zero_count = update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
			index_pair, present_ssms, CNVs, matrix_after_first_round)
	except eo.ZInconsistence as e:
		raise e

	return zero_count

# checks if triplets in which the changed entry is involved also need to be updated
# checks this iteratively by checking also all triplets that are influenced by the first change
# z_matrix: the Z-matrix
# zero_count: number of 0 entires in the Z-matrix
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
# index_pair: indices of the element in the triplet that was changed
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# matrix_after_first_round: Z-matrix after the first round
# can raise exception ZInconsistence when updating of the Z_triplet let to inconsistance
def update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, index_pair,
	present_ssms=None, CNVs=None, matrix_after_first_round=None):
	first_index = index_pair[0]
	second_index = index_pair[1]
	# check whether lowest triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the xy position
		for s in sorted(triplet_xys[first_index][second_index].keys(), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[first_index][second_index], z_matrix[second_index][s], z_matrix[first_index][s]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=first_index,
					y=second_index, s=s)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not  None:
					if changed_field == cons.X:
						i = first_index
						i_prime = second_index
						value = v_x
					elif changed_field == cons.Y:
						i = second_index
						i_prime = s
						value = v_y
					elif changed_field == cons.S:
						i = first_index
						i_prime = s
						value = v_s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=first_index, y=second_index, s=s, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy)

	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	# check whether middle triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the xs position
		for y in sorted(triplet_xsy[first_index][second_index].keys(), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[first_index][y], z_matrix[y][second_index], z_matrix[first_index][second_index]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=first_index,
					y=y, s=second_index)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not None:
					if changed_field == cons.X:
						i = first_index
						i_prime = y
						value = v_x
					elif changed_field == cons.Y:
						i = y
						i_prime = second_index
						value = v_y
					elif changed_field == cons.S:
						i = first_index
						i_prime = second_index
						value = s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=first_index, y=y, s=second_index, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy)
	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	# check whether highest triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the ys position
		for x in sorted(triplet_ysx[first_index][second_index].keys(), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[x][first_index], z_matrix[first_index][second_index], z_matrix[x][second_index]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=x,
					y=first_index, s=second_index)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not None:
					if changed_field == cons.X:
						i = x
						i_prime = first_index
						value = v_x
					if changed_field == cons.Y:
						i = first_index
						i_prime = second_index
						value = v_y
					if changed_field == cons.S:
						i = x
						i_prime = second_index
						value = v_s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=x, y=first_index, s=second_index, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy)
	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	return zero_count


# removes a triplet entry from all three hashes
def remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x, y, s):
	remove_triplet_from_hash(triplet_xys, x, y, s)
	remove_triplet_from_hash(triplet_ysx, y, s, x)
	remove_triplet_from_hash(triplet_xsy, x, s, y)

# removes an triplet entry from the hash
def remove_triplet_from_hash(t_hash, i_1, i_2, i_3):
	# removes triplet
	try:
		del t_hash[i_1][i_2][i_3]
		# if first two indices aren't part of another triplet, delete hash for second index
		if len(t_hash[i_1][i_2].keys()) == 0:
			del t_hash[i_1][i_2]
			# if first index isn't part of another triplet, delete it from hash
			if len(t_hash[i_1].keys()) == 0:
				del t_hash[i_1]
	# triplet was already removed
	except KeyError:
		pass

# updates the given hash
# i_1 - i_3 are the indices of the lineages
# lineage indices are given in the order of the hash name
# hash contains an entry, if triplet contains at least one 0
def update_triplet_hash(t_hash, i_1, i_2, i_3):
	try:
		t_hash[i_1][i_2][i_3] = True
	except KeyError:
		try:
			t_hash[i_1][i_2] = {}
			t_hash[i_1][i_2][i_3] = True
		except KeyError:
			t_hash[i_1] = {}
			t_hash[i_1][i_2] = {}
			t_hash[i_1][i_2][i_3] = True


# x, y and s define the values of the corresponding fields in a triplet of the Z-matrix
# Z[x][y] = value x
# Z[y][s] = value y
# Z[x][s] = value s
# returns if something was changed, which field was changed, how many 0 are still in triplet
#	and the values of the three fields
# can raise Exception: ZInconsistence
def update_Z_triplet(x, y, s):
	if x == 0:
		if y == 0:
			# case 7
			if s == 0:
				return False, "", 3, x, y, s
			# case 6b
			elif s == 1:
				return False, "", 2, x, y, s
			# case 8b
			elif s == -1:
				return False, "", 2, x, y, s
		elif y == 1:
			# case 6c
			if s == 0:
				return False, "", 2, x, y, s
			# case 4a
			elif s == 1:
				return True, cons.X, 0, 1, y, s
			# case 10b
			elif s == -1:
				return True, cons.X, 0, -1, y, s
		elif y == -1:
			# case 8c
			if s == 0:
				return False, "", 2, x, y, s
			# case 10a
			elif s == 1:
				return False, "", 1, x, y, s
			# case 9a
			elif s == -1:
				return False, "", 1, x, y, s
	elif x == 1:
		if y == 0:
			# case 6a
			if s == 0:
				return False, "", 2, x, y, s
			# case 4c
			elif s == 1:
				return False, "", 1, x, y, s
			# case 10e
			elif s == -1:
				return True, cons.Y, 0, x, -1, s
		elif y == 1:
			# case 4b
			if s == 0:
				return True, cons.S, 0, x, y, 1
			# case 1
			elif s == 1:
				return False, "", 0, x, y, s
			# case 2b
			elif s == -1:
				raise eo.ZInconsistence("Inconsistent case: x=1, y=1, s=-1.")
		elif y == -1:
			# case 10c
			if s == 0:
				return False, "", 1, x, y, s
			# case 2c
			elif s == 1:
				return False, "", 0, x, y, s
			# case 3a
			elif s == -1:
				return False, "", 0, x, y, s
	elif x == -1:
		if y == 0:
			# case 8a
			if s == 0:
				return False, "", 2, x, y, s
			# case 10f
			elif s == 1:
				return True, cons.Y, 0, x, -1, s
			# case 9c
			elif s == -1:
				return False, "", 1, x, y, s
		elif y == 1:
			# case 10d
			if s == 0:
				return True, cons.S, 0, x, y, -1
			# case 2a
			elif s == 1:
				raise eo.ZInconsistence("Inconsistent case: x=-1, y=1, s=1.")
			# case 3c
			elif s == -1:
				return False, "", 0, x, y, s
		elif y == -1:
			# case 9b
			if s == 0:
				return False, "", 1, x, y, s
			# case 3b
			elif s == 1:
				return False, "", 0, x, y, s
			# case 5
			elif s == -1:
				return False, "", 0, x, y, s

# given the lineages of the results, the encoded Z matrix is refined
# values that need to be 0 and values that are 0s but can be 1s are derived
# my_lineages: list with lineages
# seg_num: number of segments
# z_matrix: Z-matrix
# zero_count: number of 0's in matrix
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
def post_analysis_Z_matrix(my_lineages, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
	get_CNVs=False):

	lineage_num = len(my_lineages)

	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []

	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
		ssm_infl_cnv_same_lineage)

	# copy present_ssm list for later
	origin_present_ssms = copy.deepcopy(present_ssms)

	# check SSM phasing as it can be unnecessary because ADRs were removed before
	change_unnecessary_phasing(len(my_lineages), CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num)

	# iterate through all segments the first time and
	# only check the simple cases, that are directly to decide in the first run
	for seg_index in xrange(seg_num):

		# CN-free segment, check next segment
		if gain_num[seg_index] == 0 and loss_num[seg_index] == 0:
			continue

		# check now different combinations of CNVs together with SSMs
		zero_count = check_1a_CN_LOSS(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_1c_CN_loss(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_1d_2c_CN_losses(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_2f_CN_gains(gain_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_2h_LOH(loss_num[seg_index], gain_num[seg_index], CNVs[seg_index], z_matrix, 
			zero_count, present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_2i_phased_changes(CNVs[seg_index], z_matrix, zero_count, present_ssms[seg_index],
			triplet_xys, triplet_ysx, triplet_xsy)
		if zero_count == 0:
			break
		zero_count = check_1f_2d_2g_2j_losses_gains(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.LOSS)
		if zero_count == 0:
			break
		zero_count = check_1f_2d_2g_2j_losses_gains(gain_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.GAIN)
		if zero_count == 0:
			break

	# iterate through all segments a second time and check the hard cases now
	# that are not easy to resolve and might result in multiple Z-matrices
	# matrix after first round of analysis
	z_matrix_fst_rnd = copy.deepcopy(z_matrix) 
	# lists for Z matrix and other variables are created
	z_matrix_list = [np.asarray(z_matrix)]
	triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
	present_ssms_list = [present_ssms]
	for seg_index in xrange(seg_num):
		check_1f_2d_2g_2j_losses_gains(loss_num[seg_index], CNVs[seg_index], None, None, 
			None,  None, None, None,
			first_run=False, mutations=cons.LOSS, z_matrix_fst_rnd=z_matrix_fst_rnd,
			z_matrix_list=z_matrix_list, triplets_list=triplets_list, present_ssms_list=present_ssms_list,
			seg_index=seg_index, CNVs_all=CNVs)
		check_1f_2d_2g_2j_losses_gains(gain_num[seg_index], CNVs[seg_index], None, None, 
			None, None, None, None,
			first_run=False, mutations=cons.GAIN, z_matrix_fst_rnd=z_matrix_fst_rnd,
			z_matrix_list=z_matrix_list, triplets_list=triplets_list, present_ssms_list=present_ssms_list,
			seg_index=seg_index, CNVs_all=CNVs)
	
	if get_CNVs == False:
		return z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list
	else:
		return z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list

# if lineages are based on ground truth datasets and were not reconstructed, SSM phasing needs
# to be checked
# CNVs: list with CN changes
# present_ssms: lists with phasing of SSMs
# ssm_infl_cnv_same_lineage: list storing whether SSMs are influenced by CN gains in same lineage
def change_unnecessary_phasing(lin_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num):
	# check all cancerous lineages
	for k in xrange(1, lin_num):
		# get descendants
		descendants = [k_prime for k_prime in xrange(k+1, lin_num) if z_matrix[k][k_prime] == 1]
		# get ancestors, normal lineage does not have to be considered here
		ancestors = [k_star for k_star in xrange(1, k) if z_matrix[k_star][k] == 1]
		
		# check all segments
		for seg_index in xrange(seg_num):
			# get CN changes of segment
			CNV_seg = CNVs[seg_index]

			keep_phased_A = False
			keep_phased_B = False

			# only if lineage k has phased SSMs in segment i, the phasing needs to be checked
			if present_ssms[seg_index][cons.A][k] == False and present_ssms[seg_index][cons.B][k] == False:
				continue

			# if ancestors or k have CN loss, phasing is kept
			if do_ancestors_have_CN_loss(ancestors + [k], CNV_seg) == True:
				keep_phased_A = True
				keep_phased_B = True
			# if descendants have CN change, phasing is kept
			elif do_descendants_have_CN_change(descendants, CNV_seg) == True:
				keep_phased_A = True
				keep_phased_B = True
			# if current lineage has CN gains and influenced SSMs, phasing is kept
			else:
				keep_phased_A, keep_phased_B = is_CN_gain_in_k(k, CNV_seg, 
					ssm_infl_cnv_same_lineage[seg_index])

			# eventually change phasing
			if keep_phased_A == False:
				present_ssms[seg_index][cons.A][k] = False
				present_ssms[seg_index][cons.UNPHASED][k] = True
			if keep_phased_B == False:
				present_ssms[seg_index][cons.B][k] = False
				present_ssms[seg_index][cons.UNPHASED][k] = True

# checks whether ancestors of k and k have a CN loss in any allele
# ancok: ancestors and lineage k itself
# CNV_seg: CNVs of current segment
def do_ancestors_have_CN_loss(ancok, CNV_seg):
	CN_loss_present = False

	# check all relevant lineages
	for a in ancok:
		# CN loss on A?
		try:
			if CNV_seg[cons.LOSS][cons.A][a] is not None:
				CN_loss_present = True
				break
		except KeyError:
			pass
		# CN loss on B?
		try:
			if CNV_seg[cons.LOSS][cons.B][a] is not None:
				CN_loss_present = True
				break
		except KeyError:
			pass

	return CN_loss_present

# checks whether descendants have a CN change in any allele
# des: descendant
# CNV_seg: CNVs of current segment
def do_descendants_have_CN_change(des, CNV_seg):
	CN_present = False
	# iterates through all descendants, CN changes and phases
	for d in des:
		for c in [cons.GAIN, cons.LOSS]:
			for p in [cons.A, cons.B]:
				try:
					if CNV_seg[c][p][d] is not None:
						CN_present = True
						break
				except KeyError:
					pass
			if CN_present == True:
				break
		if CN_present == True:
			break
	return CN_present

# checks whether lineage k itself contains CN gains and whether it contains SSMs that are influenced by these gains
# lineage k
# CNV_seg: CNVs of current segment
# ssm_infl_cnv_same_lin_i_k: list whether SSMs are influences by CN changes in same lineage or not
def is_CN_gain_in_k(k, CNV_seg, ssm_infl_cnv_same_lin_i):
	keep_phased_A = False
	keep_phased_B = False
	CN_gains_A_B = [False, False]

	# check if CN gains are present in k
	for p in [cons.A, cons.B]:
		try:
			if CNV_seg[cons.GAIN][p][k] is not None:
				CN_gains_A_B[p] = True
		except KeyError:
			pass
	
	# CN gain in phase A
	if CN_gains_A_B[cons.A] == True:
		# if A has SSMs that are influenced by gain
		if ssm_infl_cnv_same_lin_i[cons.A][k] == True:
			keep_phased_A = True
	# CN gain in phase B
	if CN_gains_A_B[cons.B] == True:
		# if B has SSMs that are influenced by gain
		if ssm_infl_cnv_same_lin_i[cons.B][k] == True:
			keep_phased_B = True

	return keep_phased_A, keep_phased_B

# my_lineages: list with lineages objects how they are after the optimization
# z_matrix_fst_rnd: Z-matrix after first round of updates, where only -1's are introduced
# z_matrix_list: list with all Z-matrices after second round of updates
# origin_present_ssms: list with phasing of SSMs before any update
# present_ssms_list: lists with phasing of SSMs after update of Z-matrix
def adapt_lineages_after_Z_matrix_update(my_lineages, z_matrix_fst_rnd, z_matrix_list, origin_present_ssms, 
	present_ssms_list):

	# if after checking hard cases and LDR the Z-matrix didn't change, the lineages don't have to be updated
	if len(z_matrix_list) == 1:
		if (np.array_equal(np.asarray(z_matrix_fst_rnd), z_matrix_list[0])):
			return my_lineages, [my_lineages]

	#TODO go here when ground truth simulated data is used
	# 	because it contains only phased SSMs, the SSM phasing in the lineages might need to be updated
	#	altough the Z-matrices don't differ, but is not so important right now because I don't use this
	#	information currently
	# copy lineages for each Z-matrix, update the sublineages and the phasing
	new_lineages_list = [create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, present_ssms_list)
		for i in xrange(len(z_matrix_list))]

	# update  my_lineages
	my_lineages = new_lineages_list[0]

	return my_lineages, new_lineages_list

# creates new lineages with updated sublineages and phased SSMs
# my_lineages: lineages from after optimization
# i: index of the current Z-matrix
# z_matrix_list: list with all Z-matrices after second round of updates
# origin_present_ssms: list with phasing of SSMs before any update
# present_ssms_list: lists with phasing of SSMs after update of Z-matrix
def create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, present_ssms_list):
	new_lineages = copy.deepcopy(my_lineages)
	# update the sublineages
	update_sublineages_after_Z_matrix_update(new_lineages, z_matrix_list[i])
	# update the phasing
	update_SSM_phasing_after_Z_matrix_update(new_lineages, origin_present_ssms, present_ssms_list[i])

	return new_lineages

# current_lineages: lineages that might need to be modified
# origin_present_ssms: list with phasing of SSMs before any update
# current_ssms_list: lists with phasing of SSMs after update of Z-matrix
#	3D list: [segment][A, B, unphased][lineage]
def update_SSM_phasing_after_Z_matrix_update(current_lineages, origin_present_ssms, current_ssms_list):
	seg_num = len(origin_present_ssms)
	lin_num = len(current_lineages)

	# get the SSMs orderd in lists by their segment indices
	ssms_per_segments = get_ssms_per_segments(current_lineages, seg_num)

	# compare all lineages
	for lin_index in xrange(lin_num):
		# compare all segments
		for seg_index in xrange(seg_num):
			# check all cases
			# original: A: true, B: true, unphased: false
			if (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: true, unphased: false --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
						pass
				# A: true, B: false, unphased: true --> B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
				# A: true, B: false, unphased: false --> B -> A
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.B)
				# A: false, B: true, unphased: true --> A -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
				# A: false, B: true, unphased: false --> A -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.A)
				# A: false, B: false, unphased: true --> A & B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
			# original: A: true, B: false, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: true, B: false, unphased: true --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				# A: true, B: false, unphased: false --> unphased -> A
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.UNPHASED)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: true, B: false, unphased: false
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: false, unphased: false --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					pass
				# A: false, B: true, unphased: false --> A -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.A)
				# A: false, B: false, unphased: true --> A -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: true, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: false, B: true, unphased: true --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				# A: false, B: true, unphased: false --> unphased -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.UNPHASED)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: true, unphased: false
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: false, unphased: false --> B -> A
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.B)
				# A: false, B: true, unphased: false --> nothing
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					pass
				# A: false, B: false, unphased: true --> B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: false, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: true, B: false, unphased: false --> unphased -> A
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.UNPHASED)
				# A: false, B: true, unphased: false --> unphased -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.UNPHASED)
				# A: false, B: false, unphased: true --> nothing
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				else:
					raise eo.MyException("This SSM moving case is not possible.")

		# update phasing of current lineage
		current_lineages[lin_index].ssms = get_updated_SSM_list(lin_index, cons.UNPHASED,
			ssms_per_segments)
		current_lineages[lin_index].ssms_a = get_updated_SSM_list(lin_index, cons.A,
			ssms_per_segments)
		current_lineages[lin_index].ssms_b = get_updated_SSM_list(lin_index, cons.B,
			ssms_per_segments)

# flattens the list
# lin_index: index of the lineage
# phase: current phase
# ssms_per_segments: list in which SSMs are put in lists with their segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
def get_updated_SSM_list(lin_index, phase, ssms_per_segments):
	return [j for i in ssms_per_segments[lin_index][phase] for j in i]

# changes the phasing of the SSMs in their segment lists
# ssms_per_segments: list in which SSMs are put in lists with their segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
# seg_index: index of the current segment
# lin_index: index of the current lineage
# new_phase: phase to which SSMs should be assigned
# old phase: phase in which SSMs where before
def move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, new_phase, old_phase):
	sort_ssms = False
	# if lineage already has SSMs in the phase in the current segment, SSMs need to be sorted afterwards
	if len(ssms_per_segments[lin_index][new_phase][seg_index]) > 0:
		sort_ssms = True

	# swap SSMs from old to new phase
	ssms_per_segments[lin_index][new_phase][seg_index] += ssms_per_segments[lin_index][old_phase][seg_index]
	ssms_per_segments[lin_index][old_phase][seg_index] = []

	# SSMs get sorted
	if sort_ssms:
		#TODO
		ssms_per_segments[lin_index][new_phase][seg_index] = sorted(ssms_per_segments[lin_index][new_phase][seg_index],
			key=lambda x: (x.chr, x.pos))
	
# for all lineages, assigns all SSMs per phase to a list with its segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
# my_lineages: lineages
# seg_num: number of segments
def get_ssms_per_segments(my_lineages, seg_num):

	lin_num = len(my_lineages)
	
	# create list
	ssms_per_segments = [[] for _ in xrange(lin_num)]

	# fill list for all lineages
	for i in xrange(lin_num):
		current_lin = my_lineages[i]
		# create lists for each phase
		ssms_per_segments[i] = [[], [], []]
		ssms_per_segments[i][cons.A] = get_ssms_per_segments_lineage_phase(current_lin.ssms_a, seg_num)
		ssms_per_segments[i][cons.B] = get_ssms_per_segments_lineage_phase(current_lin.ssms_b, seg_num)
		ssms_per_segments[i][cons.UNPHASED] = get_ssms_per_segments_lineage_phase(current_lin.ssms, seg_num)

	return ssms_per_segments
	
# assigns all SSMs to a list with its segment index
# my_ssms: list with all SSMs of the lineage
# seg_num: number of segments
def get_ssms_per_segments_lineage_phase(my_ssms, seg_num):
	# create list with empty list for each segment
	ssms_per_segment_tmp = [[] for _ in xrange(seg_num)]

	# append each SSM to the segment list with its segment index
	for ssm in my_ssms:
		ssms_per_segment_tmp[ssm.seg_index].append(ssm)

	return ssms_per_segment_tmp



# updates the sublineages of each lineage according to the entries in the Z-matrix
# current_lineages: lineages that might need to be modified
# current_z_matrix: Z-matrix
def update_sublineages_after_Z_matrix_update(current_lineages, current_z_matrix):
	for i, lineage_relation in enumerate(current_z_matrix):
		# if lineage i is ancestor of lineage j, the entry lineage_relation[j] is 1
		# and the index of j will be given to the sublineages list of lineage i
		current_lineages[i].sublins = np.where(lineage_relation == 1)[0].tolist()
	

# iterates through all segments, to get the number of CN gains and losses, the CNVs itself and in which
#	phases SSMs appear, all info per segment
# present_ssms: 3D list: [segment][lineage][A, B, unphased]
# ssm_infl_cnv_same_lineage 3D list: [segment][lineage][A, B]
# evaluation_param: if function is called from evaluation function
def get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
	ssm_infl_cnv_same_lineage, evaluation_param=False):

	cnvs_a_index = [0] * lineage_num
	cnvs_b_index = [0] * lineage_num
	ssms_a_index = [0] * lineage_num
	ssms_b_index = [0] * lineage_num
	ssms_index = [0] * lineage_num

	for seg_index in xrange(seg_num):

		# set temporary variables to 0
		tmp_gain_num = 0
		tmp_loss_num = 0
		tmp_CNVs = {}
		# store where SSMs appear in the lineages, whether they are phased to A, B or unphased
		tmp_present_ssms = [[False] * lineage_num for _ in xrange(3)]
		tmp_ssm_infl_cnv_same_lineage = [[False] * lineage_num for _ in xrange(2)]

		# go through all lineages to get current CN changes and to see where SSMs occur
		for lin_index in xrange(1, lineage_num):
			# look at CN changes
			tmp_gain_num, tmp_loss_num = add_CN_change_to_hash(my_lineages, lin_index, seg_index,
				tmp_CNVs, tmp_gain_num, tmp_loss_num, cons.A, cnvs_a_index)
			tmp_gain_num, tmp_loss_num = add_CN_change_to_hash(my_lineages, lin_index, seg_index,
				tmp_CNVs, tmp_gain_num, tmp_loss_num, cons.B, cnvs_b_index)
			# only check for LOH if the current function is not called for evaluation
			if evaluation_param == False:
				is_it_LOH(tmp_gain_num, tmp_loss_num, tmp_CNVs)

			# look what kind of SSMs appear
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.A, ssms_a_index, tmp_ssm_infl_cnv_same_lineage)
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.B, ssms_b_index, tmp_ssm_infl_cnv_same_lineage)
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.UNPHASED, ssms_index)

		gain_num.append(tmp_gain_num)
		loss_num.append(tmp_loss_num)
		CNVs.append(tmp_CNVs)
		present_ssms.append(tmp_present_ssms)
		ssm_infl_cnv_same_lineage.append(tmp_ssm_infl_cnv_same_lineage)

# checks whether the lineage divergence rule is fullfilled
def post_opt_lineage_divergence_rule_feasibility_check(z_matrix_list, my_lineages):
	# frequencies of non-normal lineages
	non_normal_freqs = [my_lineages[i].freq for i in xrange(1, len(my_lineages))]

	feasibility = []

	for i, z_matrix in enumerate(z_matrix_list):
		# direct descendants of each lineage are computed
		direct_descendant = get_direct_descendants(z_matrix, len(my_lineages))

		# feasibility of matrix concerning the lineage divergence rule is computed
		feas = z_matrix_phis_feasible(direct_descendant, non_normal_freqs)

		message = "Z-matrix {0} is".format(i)
		if feas == False:
			message = "{0} not".format(message)
		message = "{0} feasible with the lineage divergence rule".format(message)
		logging.info(message)
		
		feasibility.append(feas)

	return feasibility

# case 1f) two losses, different alleles and lineages
# 	check for SSMs in downstream lineages
# case 2d) check for SSMs in upstream lineages when there are losses before
# case 2g) check for SSMs in upstream lineages when there are gains
# case 2j) check for SSMs in lineages between when two losses happen
# first run only produces result if relation between lineages is known
# second run then checks for cases where relation is not known
# z_matrix_fst_rnd: matrix after first round of analysis, is not defined in first round, only in second round
# triplets_list: list with triplet lists, each entry is [triplet_xys, triplet_ysx, triplet_xsy]
# present_ssms_list: list where entries are copies of "present_ssms", is a 3D list where for each segment 
#	and lineage the phasing of SSMs is stored
# seg_index: segment index
# CNVs_all: contains CNV list/hash for all segments
def check_1f_2d_2g_2j_losses_gains(spec_mut_num, CNVs, z_matrix, zero_count, present_ssms, 
	triplet_xys, triplet_ysx, triplet_xsy, first_run=True, mutations=cons.LOSS,
	z_matrix_fst_rnd=None, z_matrix_list=None, triplets_list=None, present_ssms_list=None, seg_index=None,
	CNVs_all=None):
	# at least 2 CN changes of the specific mutation type
	if spec_mut_num < 2:
		return zero_count

	# both phases need to be affected
	if len(CNVs[mutations].keys()) != 2:
		return zero_count

	if present_ssms:
		lin_num = len(present_ssms[cons.UNPHASED])
	else:
		lin_num = len(present_ssms_list[0][0][cons.UNPHASED])

	# check all pairs between the alleles
	for lin_A in CNVs[mutations][cons.A].keys():
		for lin_B in CNVs[mutations][cons.B].keys():
			# don't check for equal lineages
			if lin_A == lin_B:
				continue
			k_prime, k_prime_prime = sorted([lin_A, lin_B])

			# first run (first round)
			if first_run:

				# the lineages are in an ancestor-descendant relation
				if z_matrix[k_prime][k_prime_prime] == 1:
					if mutations == cons.LOSS:
						# check all lower lineages (case 1f)
						for low_lin in xrange(k_prime_prime+1, lin_num):
							if present_ssms[cons.UNPHASED][low_lin]:
								zero_count, old_z_status = update_z_matrix_first_round_m1(
									z_matrix,
									zero_count, k_prime_prime, low_lin,
									triplet_xys, triplet_ysx, triplet_xsy)
								if old_z_status == cons.Z_ONE:
									raise eo.MyException("This shouldn't happen. "
										"Both alleles were deleted, there "
										"can't be more SSMs in this lineage.")
								if zero_count == 0:
									return zero_count
						# check all lineage between (case 2j)
						for mid_lin in xrange(k_prime+1, k_prime_prime):
							if present_ssms[cons.UNPHASED][mid_lin]:
								zero_count, old_z_status_p = update_z_matrix_first_round_m1(
									z_matrix,
									zero_count, k_prime, mid_lin,
									triplet_xys, triplet_ysx, triplet_xsy)
								zero_count, old_z_status_pp = update_z_matrix_first_round_m1(
									z_matrix,
									zero_count, mid_lin, k_prime_prime,
									triplet_xys, triplet_ysx, triplet_xsy)
								# if middle lineage has unphased SSMs, it's not possible
								# that is was in any relation with the other lineage
								if old_z_status_p + old_z_status_pp >= 1:
									raise eo.MyException("Not possible that middle "
										"lineage is in any ancestor-descendant "
										"relation with k' or k''.")
								if zero_count == 0:
									return zero_count

					# check all higher lineages (case 2d and case 2g)
					for high_lin in xrange(1, k_prime):
						if present_ssms[cons.UNPHASED][high_lin]:
							# update relations of to k_prime AND k_prime_prime
							zero_count, old_z_status_p = update_z_matrix_first_round_m1(z_matrix, 
								zero_count, high_lin, k_prime, 
								triplet_xys, triplet_ysx, triplet_xsy)
							zero_count, old_z_status_pp = update_z_matrix_first_round_m1(z_matrix,
								zero_count, high_lin, k_prime_prime, 
								triplet_xys, triplet_ysx, triplet_xsy)
							# if higher lineage has unphased SSMs it can't be in an
							# ancestor-descendant relation to k' and/or k''
							if old_z_status_p + old_z_status_pp >= 1:
								raise eo.MyException("Not possible that higher"
									" lineage is in any relation to k' or k''")
							if zero_count == 0:
								return zero_count



			# second run (round)
			if first_run == False:
				# only for CN losses
				if mutations == cons.LOSS:
					# 1f
					for low_lin in xrange(k_prime_prime+1, lin_num):
						# low_lin needs to have SSMs in segment
						if not has_SSMs_in_segment(present_ssms_list, low_lin, seg_index):
							continue
						fork_z_matrix(z_matrix_fst_rnd, z_matrix_list, k_prime, k_prime_prime,
							low_lin, cons.HC_1_F, triplets_list, present_ssms_list, CNVs_all)
					# 2j
					for mid_lin in xrange(k_prime+1, k_prime_prime):
						# mid_lin needs to have SSMs in segment
						if not has_SSMs_in_segment(present_ssms_list, mid_lin, seg_index):
							continue
						fork_z_matrix(z_matrix_fst_rnd, z_matrix_list, k_prime, mid_lin,
							k_prime_prime, cons.HC_2_J, triplets_list, present_ssms_list, CNVs_all)
				# 2d and 2g
				for high_lin in xrange(1, k_prime):
					# high_lin needs to have SSMs in segment
					if not has_SSMs_in_segment(present_ssms_list, high_lin, seg_index):
						continue
					fork_z_matrix(z_matrix_fst_rnd, z_matrix_list, high_lin, k_prime,
						 k_prime_prime, cons.HC_2_D, triplets_list, present_ssms_list, CNVs_all)
			
	return zero_count

# given a lineage index and a segment index, it's determined whether the lineage has SSMs in this segment
# present_ssms_list: list where entries are copies of "present_ssms", is a 3D list where for each segment
#	and lineage the phasing of SSMs is stored
# lin_index: index of lineage which is checked
# seg_index: index of segment which is checked
def has_SSMs_in_segment(present_ssms_list, lin_index, seg_index):
	# as only the phasing but not the presence of SSMs can change for different Z-matrices,
	# it is sufficient to check the first entry in the present_ssms_list which belongs to the
	# first Z-matrix
	# if lineage has some SSMs in segment in some phase or unphased, it has SSMs
	has_SSMs = (present_ssms_list[0][seg_index][cons.A][lin_index]
		or present_ssms_list[0][seg_index][cons.B][lin_index]
		or present_ssms_list[0][seg_index][cons.UNPHASED][lin_index])
	return has_SSMs


# matrix_after_first_round: Z-matrix after the first round
# z_matrix_list: list with Z-matrices in second round
# k, k_prime, k_prime_prime: indices of lineage, k < k_prime < k_prime_prime
# hard_case: hard_case, can be 1f, 2j, 2d or 2g
# triplets_list: list with triplet lists, each entry is [triplet_xys, triplet_ysx, triplet_xsy]
# present_ssms_list: list where entries are copies of "present_ssms", is a 3D list where for each segment
#      and lineage the phasing of SSMs is stored
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
def fork_z_matrix(matrix_after_first_round, z_matrix_list, k, k_prime, k_prime_prime, hard_case,
	triplets_list, present_ssms_list, CNVs):

	# get number of Z-matrices in list
	number_used_z_matrices = 0
	z_matrix_number = len(z_matrix_list)

	# depending on hard case, get possible forking scenarios
	if hard_case == cons.HC_1_F:
		forking_scenarios = get_forking_scenarios_1f(matrix_after_first_round, k, k_prime, k_prime_prime)
	elif hard_case == cons.HC_2_J:
		forking_scenarios = get_forking_scenarios_2j(matrix_after_first_round, k, k_prime, k_prime_prime)
	# 2G does not need to be checked separately because it's equal to 2D
	elif hard_case == cons.HC_2_D:
		forking_scenarios = get_forking_scenarios_2d_2g(matrix_after_first_round, k, k_prime, k_prime_prime)

	# when no forking cases exist, nothing has to be done here
	if len(forking_scenarios) == 0:
		return

	# forking scenarios exist
	# iterate through all of them for all Z-matrices in list
	while number_used_z_matrices < z_matrix_number:
		# when at least one relation is unknown, iterate through all possible scenarios
		for scenario in forking_scenarios:
			try:
				new_matrix, new_triplet_xys, new_triplet_ysx, new_triplet_xsy, new_present_ssms = (
					create_new_Z_matrix(matrix_after_first_round, z_matrix_list[0],
					triplets_list[0][0], triplets_list[0][1],
					triplets_list[0][2], scenario[0], scenario[1], scenario[2],
					k, k_prime, k_prime_prime, present_ssms_list[0],
					CNVs))
				# if the newly created matrix new_matrix can already be found in the list
				# of all matrices (because of the hard cases 1f and 2j)
				# new_matrix is not stored but discarded
				if new_matrix_already_in_list(new_matrix, z_matrix_list):
					continue
				# append new Z-matrix, triplets and phased SSM information to lists
				z_matrix_list.append(new_matrix)
				triplets_list.append([new_triplet_xys, new_triplet_ysx, new_triplet_xsy])
				present_ssms_list.append(new_present_ssms)
			# this scenario can't be used to fork this current Z-matrix
			except eo.ZUpdateNotPossible:
				pass

		# after the first Z-matrix in the list was processed, delete it
		# either, it could be forked to other scenarios, thus "lives" now in them
		# or it didn't need to be forked because it was already complete and was appended to the end of the list
		# or it couldn't be forked, thus is incompatible
		z_matrix_list.pop(0)
		triplets_list.pop(0)
		present_ssms_list.pop(0)

		# Z-matrix list cannot be empty, one scenario must always be possible
		if len(z_matrix_list) == 0:
			raise eo.MyException("Not possible that Z-matrix list is empty!")

		# increase counter of used Z-matrices
		number_used_z_matrices += 1

# compares the newly created matrix with all other matrices in the list
#	if the newly created matrix equals one of the old ones, True is returned
# new_matrix: the newly created matrix
# z_matrix_list: the list with all Z-matrices that were already created and that are different from each other
def new_matrix_already_in_list(new_matrix, z_matrix_list):
	# first element of matrix list is not used for comparison because this is the matrix,
	# the new matrix was created from and which will be deleted later on
	# so it's allowed that the new matrix equals this current matrix
	for old_matrix in z_matrix_list[1:]:
		if np.array_equal(new_matrix, old_matrix):
			return True
	return False

# compares object Z-matrix & Co, whether Z-matrix is already in list
def zmatrix_co_already_in_list(new_zmco, my_list):
	for entry in my_list:
		if np.array_equal(entry.z_matrix, new_zmco.z_matrix):
			return True
	return False

def get_forking_scenarios_2j(matrix_after_first_round, k, k_prime, k_prime_prime):
	# 4 forking scenarios, values for x, y and s
	a = [-1, -1, -1]
	b = [1, -1, 0]
	c = [0, -1, 1]
	d = [-1, 1, -1]

	# get values in Z-matrix after first round
	x = matrix_after_first_round[k][k_prime]
	y = matrix_after_first_round[k_prime][k_prime_prime]
	s = matrix_after_first_round[k][k_prime_prime]

	# find correct forking scenarios
	if x == 0:
		if s == 0:
			if y == 0:
				return [a, b, c, d]
		elif s == 1:
			if y == 0:
				raise eo.MyException("This should never happen because of catched easy case.")
	elif x == -1:
		if s == 1:
			if y == 0:
				raise eo.MyException("This should never happen because of transitivity rule.")

	return []

def get_forking_scenarios_2d_2g(matrix_after_first_round, k, k_prime, k_prime_prime):
	# 4 forking scenarios, values for x, y and s
	a = [-1, -1, -1]
	b = [1, -1, -1]
	c = [-1, -1, 1]
	d = [-1, 1, -1]

	# get values in Z-matrix after first round
	x = matrix_after_first_round[k][k_prime]
	y = matrix_after_first_round[k_prime][k_prime_prime]
	s = matrix_after_first_round[k][k_prime_prime]

	# find correct forking scenarios
	if x == 0:
		if s == 0:
			if y == 0:
				return [a, b, c, d]
			elif y == 1:
				raise eo.MyException("This should never happen because of catched easy case.")
			elif y == -1:
				return [a, b, c]

	return []

def get_forking_scenarios_1f(matrix_after_first_round, k, k_prime, k_prime_prime):
	# 4 forking scenarios, values for x, y and s
	a = [-1, -1, -1]
	b = [1, -1, 0]
	c = [0, -1, 1]
	d = [-1, 1, -1]

	# get values in Z-matrix after first round
	x = matrix_after_first_round[k][k_prime]
	y = matrix_after_first_round[k_prime][k_prime_prime]
	s = matrix_after_first_round[k][k_prime_prime]

	# find correct forking scenarios
	if x == 0:
		if s == 0:
			if y == 0:
				# STAYS
				return [a, b, c, d]
	elif x == 1:
		if s == 0:
			if y == 0:
				raise eo.MyException("This should never happen because of earlier applied easy case.")
		if s == -1:
			if y == 0:
				raise eo.MyException("This should never happen because of transitivity rule.")

	return []

# can raise eo.ZUpdateNotPossible exception
# matrix_after_first_round: Z-matrix after the first round
# current_matrix: current Z-matrix that should be updated
# triplet_xys, triplet_ysx, triplet_xsy: triplets with 0 entries
# x, y, s: values to which should be updated
# k, k_prime, k_prime_prime: indices of lineages, k < k_prime < k_prime_prime
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
def create_new_Z_matrix(matrix_after_first_round, current_matrix, triplet_xys, triplet_ysx, triplet_xsy, x, y, s,
	k, k_prime, k_prime_prime, present_ssms, CNVs):
	# deep copy current_matrix, the triplets and present_ssms
	new_matrix = copy.deepcopy(current_matrix)
	new_triplet_xys = copy.deepcopy(triplet_xys)
	new_triplet_ysx = copy.deepcopy(triplet_ysx)
	new_triplet_xsy = copy.deepcopy(triplet_xsy)
	new_present_ssms = copy.deepcopy(present_ssms)

	# try to update the Z-matrix with the three values x, y and s
	# possible that eo.ZUpdateNotPossible exception is raised
	update_new_Z_matrix(x, cons.X, k, k_prime, k, k_prime, k_prime_prime,
		new_matrix, matrix_after_first_round, new_present_ssms,
		CNVs, new_triplet_xys, new_triplet_ysx, new_triplet_xsy)
	update_new_Z_matrix(y, cons.Y, k_prime, k_prime_prime, k, k_prime, k_prime_prime,
		new_matrix, matrix_after_first_round, new_present_ssms,
		CNVs, new_triplet_xys, new_triplet_ysx, new_triplet_xsy)
	update_new_Z_matrix(s, cons.S, k, k_prime_prime, k, k_prime, k_prime_prime,
		new_matrix, matrix_after_first_round, new_present_ssms,
		CNVs, new_triplet_xys, new_triplet_ysx, new_triplet_xsy)

	# return new Z-matrix and other new values at end
	return new_matrix, new_triplet_xys, new_triplet_ysx, new_triplet_xsy, new_present_ssms

# updates one field in the Z-matrix if possible
# can raise: eo.ZUpdateNotPossible
# value: value to which field Z_k_k_prime should be updated
# changed_field: whether x, y or s should be updated
# i, i_prime: lineage indices, i < i_prime, indices of field that should be updated
# k, k_prime, k_prime_prime: lineage indices, k < k_prime < k_prime_prime
# current_matrix: current Z-matrix that should be updated
# matrix_after_first_round: Z-matrix after the first round
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# triplet_xys, triplet_ysx, triplet_xsy: triplets with 0 entries
def update_new_Z_matrix(value, changed_field, i, i_prime, k, k_prime, k_prime_prime, current_matrix, 
	matrix_after_first_round, present_ssms, CNVs, triplet_xys, triplet_ysx, triplet_xsy):
	# if value is 0 or current matrix already has value of value, nothing has to be done
	if value == 0 or current_matrix[i][i_prime] == value:
		return True
	# otherwise, the value is not 0

	# if the current field is not 0 and also not equal to the value to which it should be updated,
	# the update is not possible
	if current_matrix[i][i_prime] != 0 and current_matrix[i][i_prime] != value:
		raise eo.ZUpdateNotPossible("Z-Matrix can't be updated.")
	# otherwise the update is theoretically possible

	# if the two lineages should be set in an ancestror-descendant relation it needs to be checked
	# whether already phased SSMs allow this
	if value == 1:
		try:
			phasing_allows_relation(i, i_prime, matrix_after_first_round, present_ssms, CNVs, value)
			move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, CNVs, matrix_after_first_round, value)
		except eo.ADRelationNotPossible:
			raise eo.ZUpdateNotPossible("Update not possible because of already phased SSMs.")
	
	# update the Z-matrix
	try:
		# zero_count is not needed, that's why the number of entries in the Z-Matrix is used instead
		entry_num = len(current_matrix) * len(current_matrix)

		#TODO QUESTION:
		# Do I really have to test this? I only check the update of one single triplet, however the
		# value in field i, i' can be involved in many triplets!
		# Does it give me anything to check whether the update is possible for the triplet with the indices
		# of the lineags?
		v_x, v_y, v_s = get_values_to_update(value, changed_field, current_matrix, k, k_prime, k_prime_prime)
		changed, changed_field_because_of_update, triplet_zeros, v_x, v_y, v_s = update_Z_triplet(v_x, v_y, v_s)

		# if no exception was thrown, matrix can be updated
		current_matrix[i][i_prime] = value

		update_Z_matrix_iteratively(z_matrix=current_matrix, zero_count=entry_num, triplet_xys=triplet_xys, 
			triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, index_pair=(i, i_prime),
			present_ssms=present_ssms, CNVs=CNVs, matrix_after_first_round=matrix_after_first_round)

		#TODO QUESTION:
		# Do I really have to call the update function again? Isn't it called for all affected triplets in
		# update_Z_matrix_iteratively?
		# application of transitivoty rule
		# one more field gets updated
		if changed == True:
			zero_count = update_after_tiplet_change(z_matrix=current_matrix, zero_count=entry_num, 
				changed_field=changed_field_because_of_update, 	v_x=v_x, v_y=v_y, v_s=v_s, 
				x=k, y=k_prime, s=k_prime_prime, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, 
				triplet_xsy=triplet_xsy, present_ssms=present_ssms, CNVs=CNVs, 
				matrix_after_first_round=matrix_after_first_round)

	except eo.ZInconsistence:
		raise eo.ZUpdateNotPossible("Update not possible because of Z-matrix inconsistence.")
	except eo.ADRelationNotPossible:
		raise eo.ZUpdateNotPossible("Update not possible because of phased SSMs in update.")
	
	return True

# returns the three values, to which a triplet defined through the three lineage indices should be updated
# value: value to which should be updated
# changed_field: position of triplet that should be updated, triplet defined through k, k_prime, k_prime_prime
# current_matrix
# k, k_prime, k_prime_prime: indices of lineages
def get_values_to_update(value, changed_field, current_matrix, k, k_prime, k_prime_prime):
	if changed_field == cons.X:
		return value, current_matrix[k_prime][k_prime_prime], current_matrix[k][k_prime_prime]
	if changed_field == cons.Y:
		return current_matrix[k][k_prime], value, current_matrix[k][k_prime_prime]
	if changed_field == cons.S:
		return current_matrix[k][k_prime], current_matrix[k_prime][k_prime_prime], value

# checks whether an ancestor-descendant relation between the lineage is allowed
# can throw exception: eo.ADRelationNotPossible
# k, k_prime: lineage indices, k < k_prime
# matrix_after_first_round: Z-matrix after the first round
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# value: value to which the field [k,k_prime] in the current matrix should be updated
def phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, value):
	# only if value to which the field [k,k_prime] in the current matrix should be updated is 1
	# phasing should be considered
	if value != 1:
		raise eo.MyException("Phasing shoudn't be considered here!")

	# if ancestor-descendant relation between lineages was already given after first round, 
	# nothing needs to be done
	if matrix_after_first_round[k][k_prime] == 1:
		return True

	# check all segments whether the SSMs and CN changes in the two lineages allow an
	# ancestor-descendant relation
	for seg_index in xrange(len(present_ssms)):
		phasing_allows_relation_per_allele_lineage(k, k_prime, present_ssms, CNVs, cons.A, seg_index)
		phasing_allows_relation_per_allele_lineage(k, k_prime, present_ssms, CNVs, cons.B, seg_index)
		phasing_allows_relation_per_allele_lineage(k_prime, k, present_ssms, CNVs, cons.A, seg_index)
		phasing_allows_relation_per_allele_lineage(k_prime, k, present_ssms, CNVs, cons.B, seg_index)
	
	# if exception was not thrown before, an ancestor-descendant relation between the lineages is possible
	return True

# checks whether an ancestor-descendant relation between the lineage is allowed
# when the SSMs are phased to the same phase then the CN changes, no relation is allowed
# otherwise it is
# can throw exception: eo.ADRelationNotPossible
# lineage_ssm: index of lineage with SSMs
# lineage_cnv: index of lineage with CN changes
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
#	list[segments]:hash[loss, gain][A, B][lineage index]={cnv}
# phase: phase that's considered here
# seg_index: index of the current segment that is checked
def phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, present_ssms, CNVs, phase, seg_index):
	# if lineage_ssm has no SSMs phased to allele 'phase', ancestor-descendant relation between lineages is possible
	if not present_ssms[seg_index][phase][lineage_ssm]:
		return True

	# check if phasing allows an ancestor-descendant relation between lineages
	# dependant on which lineage would be an ancestor
	#
	# lineage_ssm would be the ancestor, would be influenced by gains and losses
	if lineage_ssm < lineage_cnv:
		try:
			if lineage_cnv in CNVs[seg_index][cons.GAIN][phase].keys():
				raise eo.ADRelationNotPossible("Gain in lineage {0} forbids ancestor-descendant relation.".format(
					lineage_cnv))
		except KeyError:
			pass
		try:
			if lineage_cnv in CNVs[seg_index][cons.LOSS][phase].keys():
				raise eo.ADRelationNotPossible("Loss in lineage {0} forbids ancestor-descendant relation.".format(
					lineage_cnv))
		except KeyError:
			pass
	# lineage_ssm would be the descendant, would only be influenced by losses
	else:
		try:
			if lineage_cnv in CNVs[seg_index][cons.LOSS][phase].keys():
				raise eo.ADRelationNotPossible("Loss in lineage {0} forbids ancestor-descendant relation.".format(
					lineage_cnv))
		except KeyError:
			pass
	
	return True

# k, k_prime: indices of lineages
#	k < k_prime
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: hash with all CNVs
# matrix_after_first_round: Z-matrix after the first round
# value: value to which the field [k, k_prime] should be updated
def move_unphased_SSMs_if_necessary(k, k_prime, present_ssms, CNVs, matrix_after_first_round, value):
	# only if value is 1, SSMs can be moved
	if value != 1:
		raise eo.MyException("Value is not 1, so SSMs should not be moved!")

	# if lineage k and k_prime are already in an ancestor-descendant relation after the first round,
	# they were set in this relation in the optimization, 
	# thus the phasing of all SSMs was already considered
	if matrix_after_first_round[k][k_prime] == 1:
		return

	# it is checked whether the CNVs in lineage k_prime have an influence on the SSMs in lineage k
	# and vice versa
	unphased_checking(k, k_prime, present_ssms, CNVs)
	unphased_checking(k_prime, k, present_ssms, CNVs)
		
# if lineage with SSMs has unphased SSMs, all segments with unphased SSMs are checked
#	whether they are influenced by CN change in other lineage
# lineage_ssm: index of lineage that has unphased SSMs in a segment
# lineage_cnv: index of lineage that has CNVs in same segment
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
#	list[segments]:hash[loss, gain][A, B][lineage index]={cnv}
def unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs):
	# if lineage with SSMs doesn't have unphased SSMs, the function is quit
	unphased_ssms = [present_ssms[i][cons.UNPHASED][lineage_ssm] for i in xrange(len(present_ssms))]
	if sum(unphased_ssms) == 0:
		return

	# all segments are checked
	for seg_index in xrange(len(present_ssms)):
		# if lineage doesn't have unphased SSMs for this segment, nothing needs to be done
		if present_ssms[seg_index][cons.UNPHASED][lineage_ssm] == False:
			continue

		try:
			# CNVsof lineage are derived
			loss_a, loss_b, gain_a, gain_b = get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		except eo.no_CNVs:
			# when lineage has no CN changes, nothing needs to be done for the segment
			continue

		# otherwise lineage has one CN change on one allele
		# mutation and phase of CNVs is determined
		mutation = cons.LOSS
		phase = cons.A
		if gain_a or gain_b:
			mutation = cons.GAIN
		if loss_b or gain_b:
			phase = cons.B
		# LOH of ancestral lineage is possible, than mutations in SSM lineage have to be phased to the 
		# allele that's not deleted
		if gain_a and loss_b:
			mutation = cons.LOSS
			phase = cons.B
		# if CN change influences the SSMs in the other lineage, the SSMs will be moved to another phase
		cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)

# gets CNVs of lineage
#	and checks whether multiple CN changes appear
# lineage_cnv: index of lineage with potential CNVs
# CNVs: hash with all CNVs
# seg_index: index of current segment
def get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm):
	# Do different kinds of CNVs happen in the lineage that's checked for CN changes?
	loss_a = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.LOSS, cons.A)
	loss_b = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.LOSS, cons.B)
	gain_a = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.GAIN, cons.A)
	gain_b = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.GAIN, cons.B)
	# when no CNVs appear, nothing needs to be done
	if sum([loss_a, loss_b, gain_a, gain_b]) == 0:
		raise eo.no_CNVs("No CN changes in segment {0} of lineage {1}".format(
			seg_index, lineage_cnv))
	# not possible that descendant lineage has losses or gains on both alleles
	if (lineage_ssm < lineage_cnv) and ((loss_a and loss_b == True) or (gain_a and gain_b == True)):
		raise eo.MyException("When lineage {0} has CNVs on both alleles, there cannot be "
			"an ancestor-descendant relation with lineage {1}.".format(
			lineage_cnv, lineage_ssm))
	# not possible that ancestral lineage has losses on both alleles
	if (lineage_ssm > lineage_cnv) and (loss_a and loss_b == True):
		raise eo.MyException("When lineage {0} has CNVs on both alleles, there cannot be "
			"an ancestor-descendant relation with lineage {1}.".format(
			lineage_cnv, lineage_ssm))
	# not possible that descendant lineage has LOH
	if (lineage_ssm < lineage_cnv) and ((loss_a and gain_b == True) or (loss_b and gain_a == True)):
		raise eo.MyException("In case of LOH in lineage {0}, there shouldn't be any checking "
			"for hard cases!".format(lineage_cnv))
	
	return loss_a, loss_b, gain_a, gain_b

# checks whether the current lineage has a CNV of the given type in the given phase
# lineage_cnv: index of lineage with potential CNVs
# CNVs: hash with all CNVs
# seg_index: index of current segment
# mutation_type: Does the current lineage have a CNV of this kind?
# phase: Does the current lineage have a CNV in this phase?
def has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase):
	try:
		if lineage_cnv in CNVs[seg_index][mutation_type][phase].keys():
			return True
		else:
			return False
	except KeyError:
		return False
		

# if the CN change in one lineage could influence SSMs in another linages, the SSMs are moved to the
#	other allele
# lineage_ssm: index of lineage that has unphased SSMs in a segment
# lineage_cnv: index of lineage that has CNVs in same segment
# present_ssms: 3D list with booleans: [segment][lineage][A,B,unphased]
# seg_index: index of segment in which the SSMs are considered
# mutation: type of CN change: either loss or gain
# phase: phase to which CN change belongs
def cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase):
	# lineage with SSMs is ancestor of lineage with CN changes
	if lineage_ssm < lineage_cnv:
		# mutation is a loss or gain
		if mutation == cons.LOSS or mutation == cons.GAIN:
			# unphased SSMs of lineage_ssm are moved to the other phase of the CN change of lineage_cnv
			move_unphased_SSMs(present_ssms, seg_index, lineage_ssm, other_phase(phase))
	# lineage with SSMs is desendant of lineage with CN changes
	else:
		# SSMs of decendant lineage can ony be influenced by CN loss
		if mutation == cons.LOSS:
			# unphased SSMs of lineage_ssm are moved to the other phase of the CN change of lineage_cnv
			move_unphased_SSMs(present_ssms, seg_index, lineage_ssm, other_phase(phase))

# changes the phase of the unphased SSMs in a segment of a lineage
# present_ssms: 3D list with booleans: [segment][lineage][A,B,unphased]
# seg_index: index of segment in which the SSMs are considered
# current_lin: index of lineage who's SSM phases should be changed
# phase: phase to which unphased SSMs should be changed
def move_unphased_SSMs(present_ssms, seg_index, current_lin, phase):
	# check that no SSMs are phased to the phase already that will be influenced by the CN change
	if present_ssms[seg_index][other_phase(phase)][current_lin] == True:
		raise eo.MyException("Some SSMs are already phased to phase that will be influenced by the "
			"CN change. This should not happen!")
	# phases of unphased SSMs are updated
	present_ssms[seg_index][phase][current_lin] = True
	present_ssms[seg_index][cons.UNPHASED][current_lin] = False

# given a phase, the other phase is returned
def other_phase(phase):
	if phase == cons.A:
		return cons.B
	return cons.A

# case 2i) considers all changes that can happen somewhere, SSMs in upstream lineages are influenced
def check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	for changes in CNVs.keys():
		for phases in CNVs[changes].keys():
			for my_lin in CNVs[changes][phases]:
				# check all higher lineages whether they have phased SSMs
				for higher_lin in xrange(1, my_lin):
					if (present_ssms[phases][higher_lin]):
						zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, 
							zero_count, higher_lin, my_lin,
							triplet_xys, triplet_ysx, triplet_xsy)
						if zero_count == 0:
							return zero_count 
	return zero_count 

# case 2h) LOH in a lineage, influence on SSMs in upstream lineage
def check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	# both loss_num and gain_num need to be 1
	if loss_num != 1 or gain_num != 1:
		return zero_count

	# check whether the lineage with the gain also contains the loss
	gain_phase = CNVs[cons.GAIN].keys()[0]
	loss_phase = cons.B
	if gain_phase == loss_phase:
		loss_phase = cons.A
	gain_lin = CNVs[cons.GAIN][gain_phase].keys()[0]
	if gain_lin not in CNVs[cons.LOSS][loss_phase].keys():
		raise eo.MyException("Unallowed CN state")

	# check for upstream lineages
	for higher_lin in xrange(1, gain_lin):
		if (present_ssms[cons.A][higher_lin] or present_ssms[cons.B][higher_lin]
			or present_ssms[cons.UNPHASED][higher_lin]):
				zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
					higher_lin, gain_lin, triplet_xys, triplet_ysx, triplet_xsy)
				if zero_count == 0:
					return zero_count
	return zero_count

# case 2f) two gains on different alleles in the same lineage, influence SSMs in upstream lineages
def check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	# needs at least two CN gains
	if gain_num < 2:
		return zero_count

	# check whether gains are contained on both alleles
	if len(CNVs[cons.GAIN]) < 2:
		return zero_count

	lin_num = len(present_ssms[0])

	# check whether a lineage has two gains
	# it's enough to check whether the lineage that gains an additional B-allele also gains
	# the A-allele
	for my_lin in CNVs[cons.GAIN][cons.B].keys():
		if my_lin in CNVs[cons.GAIN][cons.A]:
			# check whether higher lineages have unphased SSMs 
			for higher_lin in xrange(1, my_lin):
				if (present_ssms[cons.UNPHASED][higher_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
						higher_lin, my_lin, triplet_xys, triplet_ysx, triplet_xsy)
					if zero_count == 0:
						return zero_count
	return zero_count


# case 1d) two losses in same lineage, no SSMs in downstream lineages
# case 2d) two losses in same lineage, influence on SSMs in upstream lineages
def check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	# needs at least two CN losses
	if loss_num < 2:
		return zero_count

	# check whether losses are contained on both alleles
	try:
		if len(CNVs[cons.LOSS][cons.A].keys()) == 0 or len(CNVs[cons.LOSS][cons.B].keys()) == 0:
			return zero_count
	except KeyError:
		return zero_count

	lin_num = len(present_ssms[0])

	# check whether a lineage has two losses
	# it's enough to check whether the lineage that loose the A-allele also lost the
	# B-allele
	for my_lin in CNVs[cons.LOSS][cons.A].keys():
		if my_lin in CNVs[cons.LOSS][cons.B]:
			# check all lower lineages, it's not possible that they have an SSMs at all
			for lower_lin in xrange(my_lin+1, lin_num):
				# if lower lineage has unphased SSMs it can't be the child
				if (present_ssms[cons.UNPHASED][lower_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, 
						zero_count, my_lin, lower_lin,
						triplet_xys, triplet_ysx, triplet_xsy)
					# check for errors in results
					if old_z_status == cons.Z_ONE:
						raise eo.MyException("This shouldn't happen, "
							"when both alleles are deleted there can't "
							"be SSMs.")
					if zero_count == 0:
						return zero_count
			# check upstream lineages, if they have unphased SSMs and are not in a relation with
			# the current lineage, a relation would change the likelihood, better not to
			# allow it
			for higher_lin in xrange(1, my_lin):
				if (present_ssms[cons.UNPHASED][higher_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
						higher_lin, my_lin, triplet_xys, triplet_ysx, triplet_xsy)
					if zero_count == 0:
						return zero_count
	return zero_count

# case 1c): downstream SSM can be on a deleted allele
def check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	# needs at least one CN loss
	if loss_num == 0:
		return zero_count

	# check for A allele
	zero_count = check_1c_CN_loss_phase(loss_num, CNVs, z_matrix, zero_count, cons.A, present_ssms,
		triplet_xys, triplet_ysx, triplet_xsy)
	# all 0 entries in the Z matrix were checked and changed
	if zero_count == 0:
		return zero_count
	# check for B allele
	zero_count = check_1c_CN_loss_phase(loss_num, CNVs, z_matrix, zero_count, cons.B, present_ssms,
		triplet_xys, triplet_ysx, triplet_xsy)

	return zero_count

def check_1c_CN_loss_phase(loss_num, CNVs, z_matrix, zero_count, phase, present_ssms, triplet_xys, triplet_ysx, triplet_xsy):
	# CN loss needs to affect a lineage
	try:
		affected_lineages = sorted(CNVs[cons.LOSS][phase].keys())
	except KeyError:
		return zero_count

	lin_num = len(present_ssms[0])
	# check for each affected lineage and all lower lineages
	for lin_index in affected_lineages:
		for lower_lin in xrange(lin_index+1, lin_num):
			if present_ssms[phase][lower_lin]:
				zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count, 
					lin_index, lower_lin, triplet_xys, triplet_ysx, triplet_xsy)
				# check for errors in results
				if old_z_status == cons.Z_ONE:
					raise eo.MyException("This shouldn't happen, deleted allele can't "
						"have SSMs.")
				# all 0 entries in the Z matrix were checked and changed
				if zero_count == 0:
					return zero_count
	return zero_count

# case 1a): one allele can only be deleted once
def check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy):
	# needs more than one CN loss
	if loss_num <= 1:
		return zero_count

	# check for A allele
	zero_count = check_1a_CN_LOSS_phase(cons.A, CNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)
	# all 0 entries in the Z matrix were checked and changed
	if zero_count == 0:
		return zero_count
	# check for B allele
	zero_count = check_1a_CN_LOSS_phase(cons.B, CNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	return zero_count
	
def check_1a_CN_LOSS_phase(phase, CNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy):
	# if CN loss affects lineages in this phase
	try:
		affected_lineages = sorted(CNVs[cons.LOSS][phase].keys())
	except KeyError:
		return zero_count
	# more than one lineage needs to be affected
	lin_num = len(affected_lineages)
	if lin_num <= 1:
		return zero_count
	# check all pairs of lineages
	
	for i in xrange(lin_num):
		for j in xrange(i+1, lin_num):
			lin_high = affected_lineages[i]
			lin_low = affected_lineages[j]
			zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count, lin_high, lin_low,
				triplet_xys, triplet_ysx, triplet_xsy)
			# check for errors in results
			if old_z_status == cons.Z_ONE:
				raise eo.MyException("This shouldn't happen, deleted allele can't be"
					" deleted twice.")
			# all 0 entries in the Z matrix were checked and changed
			if zero_count == 0:
				return zero_count
	return zero_count

# updates the Z matrix, only values of 0s are changed to -1s
# lin_high: lineage with higher frequency, actually smaller index
# lin_low: lineage with lower frequency, actually higher index
def update_z_matrix_first_round_m1(z_matrix, zero_count, lin_high, lin_low, triplet_xys, triplet_ysx, triplet_xsy):
	# if current entry is 1, nothing needs to be done
	if z_matrix[lin_high][lin_low] == cons.Z_ONE:
		return zero_count, cons.Z_ONE
	# if current entry is -1, also nothing needs to be done
	if z_matrix[lin_high][lin_low] == cons.Z_MINUSONE:
		return zero_count, cons.Z_MINUSONE
	# if current entry is 0, it needs to be changes to -1
	if z_matrix[lin_high][lin_low] == cons.Z_ZERO:
		z_matrix[lin_high][lin_low] = cons.Z_MINUSONE
		zero_count -= 1

		# check whether other entries in Z matrix could be updated iteratively because of triplet changes
		zero_count = update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
			(lin_high, lin_low))

		return zero_count, cons.Z_ZERO

# iterates through the SSMs of a lineage for a given segment
# finds out in which phases SSMs are present
def get_present_ssms(present_ssms, lin_index, my_lineages, seg_index, phase, ssms_index_list, 
	ssm_infl_cnv_same_lineage=None):

	current_ssms_index = ssms_index_list[lin_index]

	# get all ssms
	my_lin = my_lineages[lin_index]
	if phase == cons.A:
		my_ssms = my_lin.ssms_a
	elif phase == cons.B:
		my_ssms = my_lin.ssms_b
	else:
		my_ssms = my_lin.ssms

	# if lineage has no SSMs, function is stopped
	if my_ssms is None:
		return

	# see if SSMs are present in current segment
	if current_ssms_index < len(my_ssms) and my_ssms[current_ssms_index].seg_index == seg_index:
		present_ssms[phase][lin_index] = True
		# increase current_ssms_index until it points to the next segment
		while ((current_ssms_index < len(my_ssms) and 
			my_ssms[current_ssms_index].seg_index == seg_index)):
				# if SSM is influened by gain in same lineage, store this information
				if my_ssms[current_ssms_index].infl_cnv_same_lin == True:
					ssm_infl_cnv_same_lineage[phase][lin_index] = True
				current_ssms_index += 1
		ssms_index_list[lin_index] = current_ssms_index
		

# given the number of gains and losses it is checked whether LOH is present
def is_it_LOH(gain_num, loss_num, CNVs):
	# positive and negative CN change
	if gain_num > 0 and loss_num > 0:
		# exactly one gain and one loss
		if gain_num == 1 and loss_num == 1:
			key_gain = CNVs[cons.GAIN].keys()[0]
			key_loss = CNVs[cons.LOSS].keys()[0]
			# lineages of loss and gain are equal
			if CNVs[cons.GAIN][key_gain].keys() == CNVs[cons.LOSS][key_loss].keys():
				return True
			else:
				raise eo.NotProperLOH("Different lineages of loss and gain.")
		else:
			raise eo.NotProperLOH("More CN changes than allowed!")
	else:
		return False

# given a lineage and the index where to start to look for CNVs, 
# CNVs are inserted to the hash and the number of variations is updated
def add_CN_change_to_hash(my_lineages, lin_index, seg_index, CNVs, gain_num, loss_num, phase, cnv_index_list):
	# get current CNV
	cnv_index = cnv_index_list[lin_index]
	try:
		if phase == cons.A:
			my_cnv = my_lineages[lin_index].cnvs_a[cnv_index]
		else:
			my_cnv = my_lineages[lin_index].cnvs_b[cnv_index]
	# if last CNV in list was already processed
	except IndexError:
		return gain_num, loss_num
	# if no CNV exists for this lineage
	except TypeError:
		return gain_num, loss_num

	# check if current CNVs belongs to the current segment
	while my_cnv.seg_index == seg_index:
		# CN gain
		if my_cnv.change > 0:
			gain_num += my_cnv.change
		# CN loss
		elif my_cnv.change < 0:
			loss_num += (my_cnv.change * -1) 
		else:
			raise eo.MyException("Unparsed CN change!")
		# insert CN to hash at right position
		CNVs_insert(CNVs, phase, lin_index, my_cnv)
		# update CNV index as CNV was used
		cnv_index_list[lin_index] += 1
		cnv_index += 1
		try:
			if phase == cons.A:
				my_cnv = my_lineages[lin_index].cnvs_a[cnv_index]
			else:
				my_cnv = my_lineages[lin_index].cnvs_b[cnv_index]
		# if last CNV in list was already processed
		except IndexError:
			return gain_num, loss_num

	return gain_num, loss_num

# inserts the copy number variation at the right position in the has
def CNVs_insert(CNVs, phase, lin_index, my_cnv):
	try:
		if lin_index in CNVs[my_cnv.change][phase]:
			CNVs[my_cnv.change][phase][lin_index] = [CNVs[my_cnv.change][phase][lin_index]] + [my_cnv]
		else:
			CNVs[my_cnv.change][phase][lin_index] = my_cnv
	except KeyError:
		try:
			CNVs[my_cnv.change][phase] = {}
			CNVs[my_cnv.change][phase][lin_index] = my_cnv
		except KeyError:
			CNVs[my_cnv.change] = {}
			CNVs[my_cnv.change][phase] = {}
			CNVs[my_cnv.change][phase][lin_index] = my_cnv

def get_Z_matrix(my_lineages):
	
	# create empty Z matrix
	z_matrix = [[0] * len(my_lineages) for _ in xrange(len(my_lineages))]

	# fill with 1s
	one_count = 0
	for x in xrange(len(my_lineages)):
		for y in my_lineages[x].sublins:
			z_matrix[x][y] = 1
			one_count += 1
	# first row shouldn't count for the ones
	one_count = one_count - len(my_lineages) + 1
	# compute number of relevant 0s
	zero_count = get_number_of_untrivial_z_entries(len(my_lineages)) - one_count

	# fill diagonal and lower half with -1s as there can be no relations
	for x in xrange(len(my_lineages)):
		for y in xrange(0,x+1):
			z_matrix[x][y] = -1

	return z_matrix, zero_count

# based on the indicated relations in the Z-matrix, the descendants of the lineages
# are updated
def update_linage_relations_based_on_z_matrix(my_lineages, z_matrix):

	lin_num = len(my_lineages)

	# check all lineage but the normal and the last one
	for k in xrange(1, lin_num-1):
		# get indices of all lineages the current lineage is an ancestor of
		descendants = [k_prime for k_prime in xrange(k, lin_num) if z_matrix[k][k_prime] == 1]
		# update descendants of current lineage
		my_lineages[k].sublins = descendants

# when normal segments should be fixed so that they can't have a CN change
#	 their change is set to 0
def create_fixed_values_for_normal_segments(normal_seg_indices, lin_num):
	# there are 4 Delta C matrices, a_p1, b_p1, a_m1, b_m1 this is why we need
	#	to multiply with 4 here
	return [0] * 4 * len(normal_seg_indices) * lin_num

# normal_seg_indices: list with indices of segments that don't contain a CNV
# fixed_cnv_list_new: list with segments who's CNVs are fixed
# function creates fixation entries for segments without CN changes and combines them with entries
#	of already fixed CN changes (if this list exists)
def create_fixed_values_new_for_normal_segments(normal_seg_indices, fixed_cnv_list_new):

	logging.info("Fixed values for normal segments are created.")

	# if no fixed cnvs exist before
	if fixed_cnv_list_new is None:
		fixed_cnv_list_new = []

	# sort list with with fixed CNVs according to index of CNVs
	sorted_cnvs = sorted(fixed_cnv_list_new, key = lambda x: x[0])
	length_fixed_cnvs = len(sorted_cnvs)
	# sort normal indices
	sorted_seg_indices = sorted(normal_seg_indices)
	length_normal = len(sorted_seg_indices)

	# stores position in list with indices of normal segments
	seg_index = 0

	# iterate through already fixed segments
	for i in xrange(length_fixed_cnvs):
		# when not all normal segments are processed and the index of the current fixed CNV is higher than
		#	the index of the current normal segmentm this normal segment is added to the list
		#	of fixed segments
		while seg_index < length_normal and sorted_cnvs[i][0] > sorted_seg_indices[seg_index]:
			# fixed entry for segment without CNV is created and added to fixation list
			sorted_cnvs.append([sorted_seg_indices[seg_index], [[0, 0, cons.A]]])
			seg_index += 1
		# if all segments without CNVs are processed, we can stop
		if seg_index == length_normal:
			break

	# if there are still segments without CNVs, they are added to the fixation list
	sorted_cnvs.extend([[sorted_seg_indices[j], [[0, 0, cons.A]]] for j in xrange(seg_index, length_normal)])

	return sorted(sorted_cnvs, key = lambda x: x[0])
				


# adapts the cluster number for individual segments
# for segments without CNVs, (#lineages - 1) is used
#	for other the constant parameter of cluster numbers
# unfixed_cnv_start: start segment of segments with unfixed values
# unfixed_cnv_stop: last segment of segments with unfixed values 
# fixed_cnv_list: list with fixed values for CNVs
# cluster_num: array with cluster number parameters
# lin_num: number of lineages
# seg_num: number of segments
# normal_seg_indices: list with indices of segments that have a normal CN
def adapt_cluster_num_to_fixed_CNV_num(unfixed_cnv_start, unfixed_cnv_stop, fixed_cnv_list, cluster_num,
		lin_num, seg_num, normal_seg_indices=[]):

	# segment copy numbers are fixed by giving index numbers of normal segments
	if normal_seg_indices != []:
		for i in normal_seg_indices:
			cluster_num[i] = 1
		return
	
	# segment copy numbers are fixed by giving start and stop positions of unfixed areas
	#
	# get fixed segment indices
	seg_indices = [x for x in range(seg_num)]
	# check that there is really a start and a stop index
	# either start and stop position are not given or they are given
	if unfixed_cnv_start == -1 and unfixed_cnv_stop == -1:
		fixed_segment_indices = seg_indices
	elif unfixed_cnv_start == 1 or unfixed_cnv_stop == -1:
		raise eo.MyException("Wrong header in fixed file. Not possible that only one possition "
			"is not given.")
	else:
		fixed_segment_indices = seg_indices[ : unfixed_cnv_start] + seg_indices[unfixed_cnv_stop + 1 : ]
	# restructure list with fixed CNVs
	cnv_state_matrices = 4
	fixed_cnv_array = None
	try:
		fixed_cnv_array = np.array(fixed_cnv_list).reshape(cnv_state_matrices,
			len(fixed_segment_indices), lin_num)
	except ValueError as ex:
		raise eo.MyException("Fixed Array can't be created. Probably the number of used"
			"lineages in fixation file is different than the number used for rhis run.")
	# get number of fixed CNVs per segment and set as cluster number for segment
	for i in xrange(len(fixed_segment_indices)):
		number_cnvs = (np.count_nonzero(fixed_cnv_array[0][i])
			+ np.count_nonzero(fixed_cnv_array[1][i])
			+ np.count_nonzero(fixed_cnv_array[2][i])
			+ np.count_nonzero(fixed_cnv_array[3][i]))
		if number_cnvs == 0:
			cluster_num[fixed_segment_indices[i]] = 1

def get_phis_from_lineages(lineages):
	return [lineages[i].freq for i in range(1, len(lineages))]

def get_fixed_z_matrices(sublin_num):
	if sublin_num == 2:
		return [-1]
	if sublin_num == 3:
		return [[0], [1]]
	if sublin_num == 4:
		return [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]] 
	if sublin_num == 5:
		binary = [0, 1]
		return [[binary[a], binary[b], binary[c], binary[d], binary[e], binary[f]]
			for a in range(2) for b in range(2) for c in range(2) for d in range(2)
			for e in range(2) for f in range(2)]
	if sublin_num == 6:
		binary = [0, 1]
		return [[binary[one], binary[two], binary[three], binary[four], binary[five],
			binary[six], binary[seven], binary[eight], binary[nine], binary[ten]]
			for one in range(2) for two in range(2) for three in range(2)
			for four in range(2) for five in range(2) for six in range(2)
			for seven in range(2) for eight in range(2) for nine in range(2) 
			for ten in range(2)]
	if sublin_num == 7:
		binary = [0, 1]
		return [[binary[one], binary[two], binary[three], binary[four], binary[five],
			binary[six], binary[seven], binary[eight], binary[nine], binary[ten],
			binary[eleven], binary[twelve], binary[thirteen], binary[fourteen], 
			binary[fifteen]]
			for one in range(2) for two in range(2) for three in range(2)
			for four in range(2) for five in range(2) for six in range(2)
			for seven in range(2) for eight in range(2) for nine  in range(2) 
			for ten in range(2) for eleven in range(2) for twelve in range(2) 
			for thirteen in range(2) for fourteen in range(2) for fifteen in range(2)]
	raise eo.MyExecption("Function not defined for {0} sublineages.".format(sublin_num))

def combine_lineages_lists_fixed_phi_z(lineages_list):
	sublin_num = len(lineages_list[0])

	# create empty lists for lineage object
	cnvs_a = [[] for i in range(sublin_num)]
	cnvs_b = [[] for i in range(sublin_num)]
	snps = [[] for i in range(sublin_num)]
	snps_a = [[] for i in range(sublin_num)]
	snps_b = [[] for i in range(sublin_num)]
	ssms = [[] for i in range(sublin_num)]
	ssms_a = [[] for i in range(sublin_num)]
	ssms_b = [[] for i in range(sublin_num)]

	# append content of lineages to new lists
	for my_lineages in lineages_list:
		for i in range(sublin_num):
			cnvs_a[i].extend(my_lineages[i].cnvs_a)
			cnvs_b[i].extend(my_lineages[i].cnvs_b)
			snps[i].extend(my_lineages[i].snps)
			snps_a[i].extend(my_lineages[i].snps_a)
			snps_b[i].extend(my_lineages[i].snps_b)
			ssms[i].extend(my_lineages[i].ssms)
			ssms_a[i].extend(my_lineages[i].ssms_a)
			ssms_b[i].extend(my_lineages[i].ssms_b)

	# create new lineage list with combinded content
	my_new_lineages = []
	for i in range(sublin_num):
		# frequency can be taken from element 0 in list as this describes the lineages
		# in the normal list that are used to calculate the frequencies
		# sublineage must not be taken from element 0 because here a linear phylogeny
		# is always assumed, take sublineages from element 1 because this contains
		# structure that was calculated in optimization
		my_new_lineages.append(lineage.Lineage(lineages_list[1][i].sublins, 
			lineages_list[0][i].freq,
			cnvs_a[i], cnvs_b[i], snps[i], snps_a[i], snps_b[i], ssms[i], 
			ssms_a[i], ssms_b[i]))

	return my_new_lineages



# returns the indices of the segments with CN changes, given the indices of 
# the segments with normal CN
def get_seg_indices_w_cn_changes(seg_num, normal_seg_indices):
	return [i for i in xrange(seg_num) if i not in normal_seg_indices]

# when lineage assignments were created for super SSMs, the lineages
# are now assigned to the normal SSMs that belong to the super SSMs
# return list has normal SSMs and their lineage assignment
def get_lin_ass_for_ssms_w_ssms_from_super_ssms(lineage_assignment_for_ssms_w_ssms,
	super_ssms, cluster_labels, ssms_of_segment, ssm_list):

	# assign cluster label to super SSMs
	super_ssms_w_cluster_labels = []
	for i in xrange(len(super_ssms)):
		super_ssms_w_cluster_labels.append([super_ssms[i], i])
	# sort via chromosome and position of super SSMs
	super_ssms_w_cluster_labels_ordered = sorted(super_ssms_w_cluster_labels,
		key=lambda x: (x[0].chr, x[0].pos))

	# assign lineage label to cluster label (cluster index)
	lineage_for_super_ssms = [0] * len(lineage_assignment_for_ssms_w_ssms)
	for i in xrange(len(lineage_assignment_for_ssms_w_ssms)):
		lineage_index = lineage_assignment_for_ssms_w_ssms[i][1]
		cluster_index = super_ssms_w_cluster_labels_ordered[i][1]
		lineage_for_super_ssms[cluster_index] = lineage_index

	# create lineage assignment for normal ssms
	lineage_assignment_for_normal_ssms_w_ssms = []
	for i in xrange(len(ssms_of_segment)):
		lineage_assignment_for_normal_ssms_w_ssms.append(
			[ssm_list[ssms_of_segment[i]], lineage_for_super_ssms[cluster_labels[i]]])

	return lineage_assignment_for_normal_ssms_w_ssms

# gets lineage assignments for a subset of ssms
def get_lineage_assignment_for_subset_of_ssms(lineage_assignment_for_ssms_w_ssms, ssm_subset):
	lineage_assignments = []

	i = 0
	j = 0
	while(i < len(lineage_assignment_for_ssms_w_ssms) and j < len(ssm_subset)):
		# when both SSMs are equal, lineage assignment is added to list
		if lineage_assignment_for_ssms_w_ssms[i][0] == ssm_subset[j]:
			lineage_assignments.append(lineage_assignment_for_ssms_w_ssms[i][1])
			i += 1
			j += 1
		# SSM i has smaller chr and position than SSM j
		# thus, it is not in the subset and can be skipped
		elif lineage_assignment_for_ssms_w_ssms[i][0] < ssm_subset[j]:
			i += 1
		# SSM j has smaller chr and position than SSM i
		# thus it was not found in the complete list with SSMs
		# error
		else:
			raise eo.SSMNotFoundException("SSM {0} was not found in get_lineage_assignment"
				"_for_subset_of_ssms".format(j))

	# error if not all SSMs in subset are found
	if j != len(ssm_subset):
		raise eo.SSMNotFoundException("Last SSM was not found in get_lineage_assignment_for_subset_of_ssms")

	return lineage_assignments

# creates a list, where each SSM is contained together with the index of the lineage it is assigned to
def get_lineage_assignment_for_ssms_w_ssms(lineages):
	lineage_assignment_for_ssms_w_ssms = []
	
	# go through lineages, not the first one because this is the
	# normal lineage and no SSMs are assigned here
	for i in range(1,len(lineages)):
		# go through all three SSM lists per lineage
		lineage_assignment_for_ssms_w_ssms.extend(get_lineage_assignment_for_ssms_w_ssms_per_lin_per_list(
			lineages[i].ssms, i))
		lineage_assignment_for_ssms_w_ssms.extend(get_lineage_assignment_for_ssms_w_ssms_per_lin_per_list(
			lineages[i].ssms_a, i))
		lineage_assignment_for_ssms_w_ssms.extend(get_lineage_assignment_for_ssms_w_ssms_per_lin_per_list(
			lineages[i].ssms_b, i))

	# sorting
	return sorted(lineage_assignment_for_ssms_w_ssms, key=lambda x: (x[0].chr, x[0].pos))

# helping function for get_lineage_assignment_for_ssms_w_ssms
def get_lineage_assignment_for_ssms_w_ssms_per_lin_per_list(ssms_phased, lineage_index):
	lineage_assignment_for_ssms_w_ssms = []
	
	for ssm in ssms_phased:
		lineage_assignment_for_ssms_w_ssms.append([ssm, lineage_index])

	return lineage_assignment_for_ssms_w_ssms

def get_BIC(seg_num, sublin_num, ssm_num, snp_num, allele_specific, llh, version=11):
	parameter_num = get_parameter_num(seg_num, sublin_num, ssm_num, snp_num, version=version)
	sample_size = get_sample_size(seg_num, ssm_num, snp_num, allele_specific)
	return compute_BIC(llh, parameter_num, sample_size)

def compute_BIC(loglikelihood, parameter_num, sample_size):
	# Quaid said that I should work with LLH
	# likelihood = np.exp(loglikelihood)
	# return (-2 * likelihood + (parameter_num * np.log(sample_size)))
	return (2 * loglikelihood - (parameter_num * np.log(sample_size)))

# tested in test_optimization
def get_parameter_num(seg_num, sublin_num, ssm_num, snp_num, version=11):
	if version == 1:
		return sublin_num - 1
	if version == 2:
		return 2 * (sublin_num - 1)
	if version == 3:
		return (sublin_num - 1) * (sublin_num - 1)
	if version == 4:
		return get_number_of_untrivial_z_entries(sublin_num)
	if version == 5:
		return sublin_num - 1 + ((sublin_num - 1) * (sublin_num - 1))
	if version == 6:
		return (sublin_num - 1) + get_number_of_untrivial_z_entries(sublin_num)

# tested in test_optimization
def get_sample_size(seg_num, ssm_num, snp_num, allele_specific):
	sample_size = ssm_num + snp_num
	if allele_specific:
		return sample_size + 2 * seg_num
	else:	
		return sample_size + seg_num

# ssm_indices_per_cluster_per_seg: 3D list, 1D: indices of SSMs that belong to one cluster,
#       2D: elements of one segment, indices in lists of 2D list correspond to indices of 2D list
#       of ssm_objects_of_segment_per_seg
#	ssm_objects_of_segment_per_seg: 2D list with SSMs that belong to one segment
def create_superSSMs(ssm_indices_per_cluster_per_seg, ssm_objects_of_segment_per_seg):
	if len(ssm_indices_per_cluster_per_seg) != len(ssm_objects_of_segment_per_seg):
		raise eo.MyException("ssm_indices_per_cluster_per_seg and ssm_objects_of_segment_per_seg "
			"must have the same length to create superSSMs")
	superSSM_hash = {}
	create_superSSMs_help_local = create_superSSMs_help
	superSSMs = [create_superSSMs_help_local(ssm_indices_per_cluster_per_seg[i][j], 
		ssm_objects_of_segment_per_seg[i], superSSM_hash) 
		for i in xrange(len(ssm_indices_per_cluster_per_seg))
		for j in xrange(len(ssm_indices_per_cluster_per_seg[i]))]
	# remove empty elements
	superSSMs = [x for x in superSSMs if x]
	superSSMs.sort(key=lambda x: (x.chr, x.pos))
	return superSSMs, superSSM_hash
	
# cluster_indices: indices of SSMs that belong to one cluster
# my_ssms: SSMs that are in one segment, indices correspond to SSM positions in this list
def create_superSSMs_help(cluster_indices, my_ssms, superSSM_hash):
	# when no SSMs are assigned to a cluster, None is returned
	if cluster_indices == []:
		return None
	for i in xrange(len(cluster_indices)):
		my_ssm = my_ssms[cluster_indices[i]]
		# create superSSM for first SSM with all its attributes
		if i == 0:
			superSSM = copy.deepcopy(my_ssm)
			try:
				superSSM_hash[superSSM.chr][superSSM.pos] = [my_ssm]
			except KeyError:
				superSSM_hash[superSSM.chr] = {}
				superSSM_hash[superSSM.chr][superSSM.pos] = [my_ssm]
		# for all other SSMs add their counts to the superSSM
		else:
			superSSM.variant_count += my_ssm.variant_count
			superSSM.ref_count += my_ssm.ref_count
			superSSM_hash[superSSM.chr][superSSM.pos].append(my_ssm)
	return superSSM

# Given a list of lineages, the superSSMs in this lineage are replaced by the corresponding
# individual SSMs
def replace_superSSMs_in_lineages(my_lineages, superSSM_hash):
	for my_lin in my_lineages:
		my_lin.ssms = replace_superSSMs(my_lin.ssms, superSSM_hash)
		my_lin.ssms_a = replace_superSSMs(my_lin.ssms_a, superSSM_hash)
		my_lin.ssms_b = replace_superSSMs(my_lin.ssms_b, superSSM_hash)

# superSSMs: list of superSSMs
# superSSM_hash: hash that contains SSMs that correspod to superSSM, keys: 1. chromosome, 2. position
def replace_superSSMs(superSSMs, superSSM_hash):
	indi_ssms = [superSSM_hash[superSSM.chr][superSSM.pos] for superSSM in superSSMs]
	# flatten list
	return [item for sublist in indi_ssms for item in sublist]
	
# cluster_labels_list: array list that assigns each SSM in a segment unit a cluster label
# cluster_num_list: number of clusters per segment
# ssm_indices_per_cluster: [seg_unit][cluster][SSM indices]
def create_ssm_indices_per_cluster(cluster_labels_list, cluster_num_list):
	ssm_indices_per_cluster = []
	# go through all segment units
	for seg_index in xrange(len(cluster_labels_list)):
		seg_unit = cluster_labels_list[seg_index]
		# counter for how many SSMs are assigned to one cluster id
		counter = [0 for i in xrange(cluster_num_list[seg_index])]
		# big list with arrays to store SSM indices
		ssm_indices_of_one_cluster = [[0] * (len(seg_unit)) for i in xrange(
			cluster_num_list[seg_index])]
		# look at each SSM cluster assignment
		for i, cluster_assignment in enumerate(seg_unit):
			cluster_id = cluster_assignment
			# store index of SSM at position of cluster index
			ssm_indices_of_one_cluster[cluster_id][counter[cluster_id]] = i
			# increase counter for this cluster id
			counter[cluster_id] += 1
		# shorten SSM indices to right size
		ssm_indices_of_one_cluster = [ssm_indices_of_one_cluster[i][:counter[i]]
			for i in xrange(cluster_num_list[seg_index])]
		# append to larger list
		ssm_indices_per_cluster.append(ssm_indices_of_one_cluster)
	
	return ssm_indices_per_cluster

# applies kmeans on VAFs in segment_VAFs with cluster_num clusters
def cluster_VAFs(segment_VAFs, cluster_num, kmeans=True, agglo_ward=False):
	# adapting number of clusters
	if cluster_num > len(segment_VAFs):
		cluster_num = len(segment_VAFs)
	# no clustering when there are no SSMs
	if len(segment_VAFs) == 0:
		return []
	# checking which clustering algorithm to use
	if kmeans == True and agglo_ward == True:
		raise oe.MyException("Please choose only one clustering method.")
	if kmeans == False and agglo_ward == False:
		raise oe.MyException("Please choose a clustering algorithm.")
	if kmeans:
		kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(segment_VAFs)
		return kmeans.labels_
	if agglo_ward:
		cluster_result = AgglomerativeClustering(n_clusters=cluster_num, linkage="ward").fit(segment_VAFs)
		return cluster_result.labels_

# creates a subset of segments whose indices are in list of segmentIDs
def choose_seg_subset(segids, seg_list):
	return [seg for seg in seg_list if seg.index in segids]

# segids: list with segmentIDs
# ssm_list: list with SSM objects
# indices of SSMs are chosen that belong to segments in segids
def choose_ssm_subset(segids, ssm_list, normal_segments=True):
	ssms_of_segment = []
	ssms_of_segments_per_seg = []

	current_seg_index = 0
	current_seg = segids[current_seg_index]
	ssm_seg = -1
	for i in xrange(len(ssm_list)):
		ssm_seg = ssm_list[i].seg_index
		# if SSM doesn't belong to current segment but previous segment
		# skip SSM
		if ssm_seg < current_seg:
			continue
		# if SSM belongs to current segment
		# add SSM to list
		elif ssm_seg == current_seg:
			ssms_of_segment.append(i)
		# if SSM doesn't belong to current segment but following segment
		elif ssm_seg > current_seg:
			# if last segment was already processed but SSM list is not empty yet,
			# add last SSM IDs to list
			if current_seg_index == len(segids) - 1:
				ssms_of_segments_per_seg.append(ssms_of_segment)
				break
			# as long as SSM belongs to following segment and last segment in
			# segids is not processed
			while ssm_seg > current_seg and current_seg_index < len(segids) - 1:
				# if segments with CN changes are processed, SSMs of previous segment
				# are stored separately
				if not normal_segments:
					ssms_of_segments_per_seg.append(ssms_of_segment)
					ssms_of_segment = []
				# go to next segment in segids
				current_seg_index += 1
				current_seg = segids[current_seg_index]
			# check if SSM belongs to current segment now after segment got
			# potentially updated
			if ssm_seg == current_seg:
				ssms_of_segment.append(i)
			# last segment was already processed but SSM list is not
			# stop process
			elif ssm_seg > current_seg:
				break

	# if last SSMs are in last segments and segments with CN change are processed
	# SSMs are added to list
	if not normal_segments and ssm_seg == current_seg:
		ssms_of_segments_per_seg.append(ssms_of_segment)

	# if segments with CN changes are processed and all SSMs are considered but there are still segments left,
	# add empty lists for these segments
	if not normal_segments and current_seg_index < segids[-1]:
		empty_segs_num = len(segids) - 1 - current_seg_index 
		ssms_of_segments_per_seg.extend([[]] * empty_segs_num)

	if normal_segments:
		return ssms_of_segment
	else:
		if len(ssms_of_segments_per_seg) != len(segids):
			raise oe.MyException("Choosing subset of SSMs for clustering didn't work. Number of processed"
				" segments is different than it should be.")
		return ssms_of_segments_per_seg

# when cnvs, ssms or snps should be fixed, we can write them in a file
# and specify which mutation shall be unfixed
# this function creates a list that contains all the mutation indices
# that should be *fixed*
def create_fixed_indices(unfixed_start, unfixed_stop, mut_num):
	# all mutations are fixed, no mutation is unfixed
	if unfixed_start == -1:
		return [i for i in xrange(mut_num)]

	# some mutations are unfixed, these unfixed mutation are all
	# adjacent index-vice
	fixed_indices = []
	for i in xrange(mut_num):
		if i < unfixed_start or i > unfixed_stop:
			fixed_indices.append(i)
	return fixed_indices

# given fixed values of the Z matrix in a list, creates a matrix of
# dimension sublin_num x sublin_num, where the entries correspond
# to the values
def create_z_as_maxtrix_w_values(sublin_num, values):
	# create Z-matrix with the right dimension
	z_matrix = np.zeros((sublin_num, sublin_num))

	# first row is 1, expect first entry
	z_matrix[0, 1:] = 1

	# set other entries according to values
	index = 0
	for i in range(1,sublin_num-1):
		for j in range(i+1, sublin_num):
			z_matrix[i][j] = values[index]
			index += 1

	return z_matrix

# counts the number of SSMs per segment
def count_number_of_ssms_per_segment(seg_list, ssm_list):
	# empty list is created
	count_list = []
	
	seg_index = 0
	ssm_num = 0

	# each SSM is processed
	for ssm in ssm_list:
		# if SSM belongs to current segment, its count is increased
		if ssm.seg_index == seg_index:
			ssm_num += 1
		# SSM doesn't belong to current segment
		else:
			# number of SSMs of previous segment is appended to list
			count_list.append(ssm_num)
			# seg index is increased
			seg_index += 1
			# number of segment is set to 0 again
			ssm_num = 0
			# as long as the segment index still doesn't match the index of the SSM
			while seg_index != ssm.seg_index:
				# SSM number of 0 is appended to list
				count_list.append(ssm_num)
				# seg index is increased
				seg_index += 1
			# SSM count is set to 1 because SSM matches segment
			ssm_num = 1

	# last number of SSMs is append to list
	count_list.append(ssm_num)

	# when last segments don't have SSMs, append 0 count to list
	while len(count_list) < len(seg_list):
		count_list.append(0)

	return count_list

def create_segment_and_mutation_lists(input_seg, input_snp, input_ssm,
	allele_specific=False):

	seg_list = []
	snp_list = []
	ssm_list = []

	# read files
	# in allele_specific case, SNPs are not needed
	if allele_specific:
		logging.info("Reading input files \"{0}\" and \"{1}\"...".format(
			input_seg, input_ssm))
	else:
		logging.info("Reading input files \"{0}\", \"{1}\" and \"{2}\"...".format(
			input_seg, input_snp, input_ssm))
		snp_list = oio.read_snp_ssm_file(input_snp, cons.SNP)
	seg_list = oio.read_segment_file(input_seg, allele_specific)
	ssm_list = oio.read_snp_ssm_file(input_ssm, cons.SSM)

	logging.info("Adjusting data for optimization...")

	# sort lists
	seg_list = sort_segments(seg_list)
	add_segment_index(seg_list)
	if not allele_specific:
		snp_list = sort_snps_ssms(snp_list)
	ssm_list = sort_snps_ssms(ssm_list)

	# assign single mutations to segments
	if not allele_specific:
		assign_muts_to_segments(seg_list, snp_list)
	assign_muts_to_segments(seg_list, ssm_list)

	if allele_specific:
		return (seg_list, ssm_list)
	else:
		return (seg_list, snp_list, ssm_list)


def create_segment_and_mutation_splines(seg_list, snp_list, ssm_list,  number_spline_points,
	allele_specific=False, overdispersion=1000):
	seg_splines = []
	snp_splines = []
	ssm_splines = []

	#TODO: seg_points, snp_points, ssm_points somehow equivalent with number_spline_points
	seg_overdispersion = overdispersion
	seg_points = 1000
	snp_overdispersion = overdispersion
	snp_points = 1000
	ssm_overdispersion = overdispersion
	ssm_points = 1000

	logging.info("Creating splines...")
	logging.info("... with parameter n_splinepoints = {0} ...".format(number_spline_points))

	# create splines
	if allele_specific:
		(seg_splines_A, seg_splines_B) = log_pdf.compute_piecewise_linear_for_seg_list_allele_specific(
			seg_list, seg_points, number_spline_points)
	else:
		seg_splines = log_pdf.compute_piecewise_linear_for_seg_list(seg_list, seg_overdispersion,
			seg_points, number_spline_points)
		snp_splines = log_pdf.compute_piecewise_linear_for_snp_ssm_list(snp_list, 
			snp_overdispersion, snp_points, number_spline_points)
	ssm_splines = log_pdf.compute_piecewise_linear_for_snp_ssm_list(ssm_list,
		ssm_overdispersion, ssm_points, number_spline_points)

	if allele_specific:
		return (seg_splines_A, seg_splines_B, ssm_splines)
	else:
		return (seg_splines, snp_splines, ssm_splines)

# create a fixed value file for CNVs, SNPs, SSMs, Z and frequencies
def create_fixed_file(lineages, file_type, cn_state_num=2, test=False, result_file_name=None, output_file=None):
	# check for file type and get corresponding data
	if file_type == cons.CNV:
		(output_matrices, row_num, column_num, info) = create_fixed_CNV_data(lineages, cn_state_num, 
			result_file_name)
		output = output_matrices.flatten().tolist()
	elif file_type == cons.SNP:
		(output_matrices, row_num, column_num, info) = create_fixed_SNPs_data(lineages, result_file_name)
		output = output_matrices.flatten().tolist()
	elif file_type == cons.SSM:
		(output_matrices, row_num, column_num, info) = create_fixed_SSMs_data(lineages, result_file_name)
		output = output_matrices.flatten().tolist()
	elif file_type == cons.Z:
		(output, row_num, column_num, info) = create_fixed_Z_data(lineages, result_file_name)
	elif file_type == cons.FREQ:
		(output, row_num, column_num, info) = create_fixed_frequencies_data(lineages, result_file_name)	
	# if not a valid file type return false, maybe raise an error or warning
	else:
		return False

	# write data to file and report success afterwards
	if output_file is not None:
		oio.write_fixed_value_file(output, output_file, row_num, column_num, info, file_type=file_type, 
			test=test)	 
	return (output, row_num, column_num, info)

# check for duplicates in a list containing tuples, regarding the first two values
# check for segments where multiple CNV occur, for example +1 on one chromatid and -1 on the other
def check_duplicates(chr_pos_list):
	prev = (-1,-1)
	num_dupl = 0
	duplicate_list = []
	for entry in chr_pos_list:
		# 0: chromosome, 1: start position
		if entry[0] == prev[0] and entry[1] == prev[1]:
			duplicate_list.append(prev)
			num_dupl += 1
		prev = entry
	return (duplicate_list, num_dupl)

# create fixed value files(CNVs, SNPs, SSMs) with an unfixed segment for all segments
def create_fixed_segments_all_but_one(lineages, output_prefix, cn_state_num=2, test=False, 
	result_file_name=None):

	snps = []
	ssms = []
	cnvs = []
	# generate segment list and mutation lists
	for lineage in lineages:
		cnvs.extend(lineage.cnvs_a)
		cnvs.extend(lineage.cnvs_b)
		snps.extend(lineage.snps)
		snps.extend(lineage.snps_a)
		snps.extend(lineage.snps_b)
		ssms.extend(lineage.ssms)
		ssms.extend(lineage.ssms_a)
		ssms.extend(lineage.ssms_b)

	# list sorting
	sorted_snps = sort_snps_ssms(snps)
	sorted_ssms = sort_snps_ssms(ssms)

	# generate all fixed data
	(data_cnv_matrices, rows_cnv, columns_cnv, info_cnv) = create_fixed_CNV_data(
		lineages, cn_state_num, result_file_name)
	data_cnv = data_cnv_matrices.flatten().tolist()
	(data_snp_matrices, rows_snp, columns_snp, info_snp) = create_fixed_SNPs_data(
		lineages, result_file_name)
	data_snp = data_snp_matrices.flatten().tolist()
	(data_ssm_matrices, rows_ssm, columns_ssm, info_ssm) = create_fixed_SSMs_data(
		lineages, result_file_name)
	data_ssm = data_ssm_matrices.flatten().tolist()
	
	# get start and stop row in the data of mutations
	segment = 0
	num_segments = get_number_of_segments(cnvs)
	snp_start = 0
	snp_stop = 0
	ssm_start = 0
	ssm_stop = 0
	output_file_CNV = "{0}_CNV_unfixed_segment_".format(output_prefix)
	output_file_SNP = "{0}_SNP_unfixed_segment_".format(output_prefix)
	output_file_SSM = "{0}_SSM_unfixed_segment_".format(output_prefix)
	
	while (segment < num_segments):
		# compute end of intervall of SNPs and SSM that belong to the current segment
		for snp in sorted_snps[snp_start:]:
			if snp.seg_index == segment:
				snp_stop += 1
			else:
				break
		for ssm in sorted_ssms[ssm_start:]:
			if ssm.seg_index == segment:
				ssm_stop += 1
			else:
				break
		# write data with unfixed segments to files, snp_stop and ssm_stop get decreased by one, 
		# because they are the index of the first row of the next segment
		oio.write_fixed_value_file(data_cnv, output_file_CNV + str(segment), rows_cnv, columns_cnv, 
			info_cnv, segment, segment, segment, cons.CNV, test)
		oio.write_fixed_value_file(data_snp, output_file_SNP + str(segment), rows_snp, columns_snp, 
			info_snp, segment, snp_start, snp_stop-1, cons.SNP, test)
		oio.write_fixed_value_file(data_ssm, output_file_SSM + str(segment), rows_ssm, columns_ssm, 
			info_ssm, segment, ssm_start, ssm_stop-1, cons.SSM, test)
		
		# Preparations for next iteration
		segment += 1
		snp_start = snp_stop
		ssm_start = ssm_stop

# creates list with information to which CNVs the segments should be fixed
def create_fixed_CNV_data_new(lineages, seg_num):
	# create initial list
	fixed_cnv_list = [[i, []] for i in xrange(seg_num)]

	# iterate through all lineages and create list with entries for fixation
	for i, my_lin in enumerate(lineages):
		create_fixed_CNV_data_new_phase(fixed_cnv_list, my_lin.cnvs_a, cons.A, i)
		create_fixed_CNV_data_new_phase(fixed_cnv_list, my_lin.cnvs_b, cons.B, i)

	return fixed_cnv_list

def create_fixed_CNV_data_new_phase(fixed_cnv_list, my_cnvs, phase, lin_index):
	[fixed_cnv_list[single_cnv.seg_index][1].append([lin_index, single_cnv.change, phase]) for single_cnv in my_cnvs]

# creates list with information how SSMs should be fixed
def create_fixed_SSM_data_new(lineages):
	# create initial list
	fixed_ssm_list = []

	# iterate through all lineages and create list with entries for fixation
	for i, my_lin in enumerate(lineages):
		fixed_ssm_list.extend(create_fixed_SSM_data_new_phase(my_lin.ssms_a, cons.A, i))
		fixed_ssm_list.extend(create_fixed_SSM_data_new_phase(my_lin.ssms_b, cons.B, i))
		fixed_ssm_list.extend(create_fixed_SSM_data_new_phase(my_lin.ssms, cons.UNPHASED, i))

	# sort fixed SSMs
	sorted_ssms = sorted(fixed_ssm_list, key = lambda x: (x[0][0], x[0][1]))

	# add right indices in the beginng
	new_fixed_ssm_list = [[i, sorted_ssms[i][1], sorted_ssms[i][2]] for i in xrange(len(sorted_ssms))]

	return new_fixed_ssm_list

def create_fixed_SSM_data_new_phase(my_ssms, phase, lin_index):
	return [[[single_ssm.chr, single_ssm.pos], lin_index, phase] for single_ssm in my_ssms]


# function to create fixed value CNV-data
def create_fixed_CNV_data(lineages, cn_state_num, result_file_name=None):
	
	# colums for output
	num_lineages = len(lineages)
	
	# rows for output
	num_cnvs = 0
	cnvs = []
	for lin_num, lineage in enumerate(lineages):
		cnvs_a = lineage.cnvs_a
		cnvs_b = lineage.cnvs_b
		num_cnvs += len(cnvs_a) + len(cnvs_b)
		# save the CNVs as k-tupel(chr, startposition, change, lineage, chromatid: a or b)
		for cnv in cnvs_a:
			cnvs.append((cnv.chr, cnv.start, int(cnv.change), lin_num, cons.A))
		for cnv in cnvs_b:
			cnvs.append((cnv.chr, cnv.start, int(cnv.change), lin_num, cons.B))

	# sort the CNVs by chromosome and then position
	cnvs_sorted = sorted(cnvs, key=operator.itemgetter(0,1))
	
	# check for segments where multiple CNV occur, for example +1 on one chromatid and -1 on the other
	(dupl_list, num_dupl) = check_duplicates(cnvs_sorted)

	# correct the number of cnvs without duplicates
	num_cnvs_no_dupl = num_cnvs - num_dupl

	# create matrices
	output = [0] * (num_cnvs_no_dupl * num_lineages * cn_state_num * cons.PHASE_NUMBER)
	
	matrix = 0
	used_duplicates = 0
	for cnv_index, cnv in enumerate(cnvs_sorted):
		change = cnv[2]
		chromatid = cnv[4]
		if change == 0:
			continue
		if change == 1:
			if chromatid == cons.A:
				matrix = 0
			else:
				matrix = 1
		elif change == -1:
			if chromatid == cons.A:
				matrix = 2
			else:
				matrix = 3

		# put entry in matrix, determined above, in row i(because of the sorted CNVs) 
		# 	and column lineage_number
		output[(matrix * num_lineages * num_cnvs_no_dupl) 
			+ ((cnv_index - used_duplicates) * num_lineages) + cnv[3]] = 1

		# check if a duplicate was used and mark it
		if used_duplicates != num_dupl and cnv == dupl_list[used_duplicates]:
			used_duplicates += 1

	info = "Fixed values for CNVs from result file {0} with {1} lineages, {2} CNVs and {3} states.".format(
		result_file_name, num_lineages, num_cnvs, cn_state_num)

	# (data, number of rows, number of columns, info line)
	output_matrices = np.array(output).reshape(cn_state_num * 2, num_cnvs-num_dupl, num_lineages)
	return (output_matrices, num_cnvs-num_dupl, num_lineages, info)

# function to create fixed value SNP-data
def create_fixed_SNPs_data(lineages, result_file_name=None):

	# colums for output
	num_lineages = len(lineages)
	# rows for output
	num_snps = 0
	snps_list = []

	for lineage in lineages:
		snps = lineage.snps
		snps_a = lineage.snps_a
		snps_b = lineage.snps_b
		num_snps += len(snps) + len(snps_a) + len(snps_b)
		# save the SNPs as k-tupel(chr, startposition, chromatid: unphased, a or b)
		for snp in snps:
			snps_list.append((snp.chr, snp.pos, cons.UNPHASED))
		for snp in snps_a:
			snps_list.append((snp.chr, snp.pos, cons.A))
		for snp in snps_b:
			snps_list.append((snp.chr, snp.pos, cons.B))

	# only 1 column
	output = [0] * (cons.SNP_SSM_PHASING_MATRIX_NUM * num_snps)

	# sort the SNPs by chromosome and then position
	snps_sorted = sorted(snps_list, key=operator.itemgetter(0,1))

	matrix = 0
	for index, snp in enumerate(snps_sorted):
		chromatid = snp[2]
		if chromatid == cons.UNPHASED:
			matrix = 0
		elif chromatid == cons.A:
			matrix = 1
		else:
			matrix = 2
	# entry in Matrix, determined above, in row index(because of the sorted SNPs)
		output[(matrix * num_snps) + index] = 1

	info = "Fixed values for SNPs from result file {0} with {1} lineages and {2} SNPs.".format(
		result_file_name, num_lineages, num_snps)

	# (data, number of rows, number of columns, info line)
	output_matrices = np.array(output).reshape(3, num_snps, 1)
	return (output_matrices, num_snps, 1, info)
	
# function to create fixed value SSM-data
def create_fixed_SSMs_data(lineages, result_file_name=None):

	# colums for output
	num_lineages = len(lineages)
	# rows for output
	num_ssms = 0
	ssms_list = []
		
	for lineage_index, lineage in enumerate(lineages):
		ssms = lineage.ssms
		ssms_a = lineage.ssms_a
		ssms_b = lineage.ssms_b
		num_ssms += len(ssms) + len(ssms_a) + len(ssms_b)
		# save the SSms as k-tupel(chr, startposition, lineage, chroamtid: unassigned, a or b)
		for ssm in ssms:
			ssms_list.append((ssm.chr, ssm.pos, lineage_index, cons.UNPHASED))	
		for ssm in ssms_a:
			ssms_list.append((ssm.chr, ssm.pos, lineage_index, cons.A))
		for ssm in ssms_b:
			ssms_list.append((ssm.chr, ssm.pos, lineage_index, cons.B))

	# 3 matrices with num_ssms X num_lineages
	output = [0] *  (cons.SNP_SSM_PHASING_MATRIX_NUM * num_ssms * num_lineages)

	# sort the SSms by chromosome and then position
	ssms_sorted = sorted(ssms_list, key=operator.itemgetter(0,1))

	matrix = 0
	for ssm_index, ssm in enumerate(ssms_sorted):
		chromatid = ssm[3]
		if chromatid == cons.UNPHASED:
			matrix = 0
		elif chromatid == cons.A:
			matrix = 1
		else:
			matrix = 2
		# entry in matrix, determined above, in row ssm_index(because of the sorted SSMs) and 
		# column lineage_number
		output[(matrix * num_lineages * num_ssms) + (ssm_index * num_lineages) + ssm[2]] = 1

	info = "Fixed values for SSMs from result file {0} with {1} lineages and {2} SSMs.".format(
		result_file_name, num_lineages, num_ssms)
	output_matrices = np.array(output).reshape(3, num_ssms, num_lineages)
	return (output_matrices, num_ssms, num_lineages, info)

# create fixed value Z-data
def create_fixed_Z_data(lineages, result_file_name=None):
	num_lineages = len(lineages)
	columns = num_lineages - 2

	# entries is equal to (n * (n + 1) / 2) (kleiner Gauss)
	output = [0] * (columns * (columns + 1) / 2)

	offset = 0
	# only consider the lineages from 1 to the penultimate one (excluding 0 and the last one)
	for i in range(1, num_lineages - 1):
		lineage = lineages[i]
		for sublin in lineage.sublins:
			# sublin - i - 1 sets the first possible sublineage to 0, etc.
			output[offset + sublin - i - 1] = 1
		# add passed columns for this row
		offset += columns + 1 - i
	
	info = "Fixed values for z-matrix from result file {0} with {1} lineages.".format(
		result_file_name, num_lineages)
	
	return (output, columns, columns, info)
	
# create fixed value data for frequencies
def create_fixed_frequencies_data(lineages, result_file_name=None):
	freq = []
	# get frequencies without lineage 0
	for lineage in lineages[1:]:
		freq.append(lineage.freq)
	info = "Fixed values for frequencies from result file {0} with {1} lineages.".format(
		result_file_name, len(lineages))
	
	return (freq, 1, len(lineages)-1, info)
	
# returns the index of a mutation at a specific index of a segment in the list
# that stores which mutations are assigned to which segments
# if no mutations are assigned to a given segment, -1 is returned
def get_mut_index_from_muts_indices_on_seg(muts_indices_on_seg, seg, index):
	# return -1 for a segment that has no mutation assigned
	if len(muts_indices_on_seg[seg]) == 0:
		return -1
	else:
		return muts_indices_on_seg[seg][index]

# creates a list that stores at the given index of segment the indices of the
# SNPs or SSMs that appear on this segment
def get_mutation_index_list_for_segments(mut_list, seg_num):
	muts_indices_on_seg = [[] for __ in xrange(seg_num)]

	[muts_indices_on_seg[mut.seg_index].append(i) for i, mut in enumerate(mut_list)]
	
	return muts_indices_on_seg
		
#########################################################################################################
# the SSMs of two lineage lists are compared
# only the one that appear in lineages_keep are kept in lineages_other
def compare_SSMs_keep_the_ones_of_one_lineage_list(lineages_other, lineages_keep):
	# create deep copy of other lineage that gets changed
	lineages_a = copy.deepcopy(lineages_other)

	SSM_hash = {}
	# iterate through all lineages but the normal one
	# of the lineage list with the SSMs to keep
	for i in xrange(1, len(lineages_keep)):
		# add SSMs to hash for all SSM lists
		[add_SSM_to_hash(SSM_hash, my_ssm) for my_ssm in lineages_keep[i].ssms]
		[add_SSM_to_hash(SSM_hash, my_ssm) for my_ssm in lineages_keep[i].ssms_a]
		[add_SSM_to_hash(SSM_hash, my_ssm) for my_ssm in lineages_keep[i].ssms_b]

	# iterate through all lineages but the normal one
	# of the other lineage list lineages_a
	for i in xrange(1, len(lineages_a)):
		lineages_a[i].ssms = [my_ssm for my_ssm in lineages_a[i].ssms if is_SSM_in_hash(SSM_hash, my_ssm)]
		lineages_a[i].ssms_a = [my_ssm for my_ssm in lineages_a[i].ssms_a if is_SSM_in_hash(SSM_hash, my_ssm)]
		lineages_a[i].ssms_b = [my_ssm for my_ssm in lineages_a[i].ssms_b if is_SSM_in_hash(SSM_hash, my_ssm)]

	return lineages_a

# returns True if SSM in contained in hash, False otherwise
def is_SSM_in_hash(SSM_hash, my_ssm):
	try:
		in_hash = SSM_hash[my_ssm.chr][my_ssm.pos]
		return in_hash
	except KeyError:
		return False

# SSM is added to hash
def add_SSM_to_hash(SSM_hash, my_ssm):
	try:
		SSM_hash[my_ssm.chr][my_ssm.pos] = True
	except KeyError:
		SSM_hash[my_ssm.chr] = {}
		SSM_hash[my_ssm.chr][my_ssm.pos] = True
#########################################################################################################

def get_lineages_ob_from_CPLEX_results(opt, snp_list, ssm_list, seg_list, single_segment=False,
	check_z_transitivity=True, dont_break_z_symmetry=False):

	my_lineages = [lineage.Lineage([], -1, [], [], [], [], [], [], [], []) for i in range(opt.sublin_num)]

	# add frequencies
	add_frequencies_from_CPLEX(opt, my_lineages)


	# add sublins from CPLEX result to lineages objects
	add_sublins_from_CPLEX(opt, my_lineages, check_z_transitivity=check_z_transitivity,
		dont_break_z_symmetry=dont_break_z_symmetry)

	# SNPs are only used when copy numbers are not modeled allele-specific
	# add SNPs from CPLEX result to lineages objects
	if not opt.allele_specific:
		snp_index = opt.dsnp_start_index
		add_SNPs_per_phase_from_CPLEX(opt, snp_list,  snp_index, my_lineages, cons.UNPHASED)
		snp_a_index = opt.dsnp_start_index + opt.snp_num
		add_SNPs_per_phase_from_CPLEX(opt, snp_list, snp_a_index, my_lineages, cons.A)
		snp_b_index = opt.dsnp_start_index + 2 * opt.snp_num
		add_SNPs_per_phase_from_CPLEX(opt, snp_list, snp_b_index, my_lineages, cons.B)
	
	# get list with ssms
	# when only simple CN changes are allowed, also unphased SNPs exist
	if opt.simple_CN_changes:
		ssm_index = opt.dssm_start_index
		add_SSMs_per_phase_from_CPLEX(opt, ssm_list, ssm_index, my_lineages, cons.UNPHASED,
			single_segment=single_segment)
		ssm_a_index = opt.dssm_start_index + opt.ssm_num * opt.sublin_num
		add_SSMs_per_phase_from_CPLEX(opt, ssm_list, ssm_a_index, my_lineages, cons.A,
			single_segment=single_segment)
		ssm_b_index = opt.dssm_start_index + 2 * opt.ssm_num * opt.sublin_num
		add_SSMs_per_phase_from_CPLEX(opt, ssm_list, ssm_b_index, my_lineages, cons.B,
			single_segment=single_segment)
	else:
		ssm_a_index = opt.dssm_start_index
		add_SSMs_per_phase_from_CPLEX(opt, ssm_list, ssm_a_index, my_lineages, cons.A,
			single_segment=single_segment)
		ssm_b_index = opt.dssm_start_index + opt.ssm_num * opt.sublin_num
		add_SSMs_per_phase_from_CPLEX(opt, ssm_list, ssm_b_index, my_lineages, cons.B,
			single_segment=single_segment)

	# add CNVs
	add_CNV_from_CPLEX(opt, seg_list, opt.dc_binary_index_start_p1, opt.dc_binary_index_start_m1,
		my_lineages)

	return my_lineages

# gets the solution from CPLEX, which frequencies are assigned to the lineages and writes it to the lineages
def add_frequencies_from_CPLEX(opt, my_lineages):
	my_freqs = get_frequencies_from_CPLEX(opt)
	sublin_num = opt.sublin_num
	for i in range(sublin_num):
		my_lineages[i].freq = my_freqs[i]

# gets the solution from CPLEX, which frequencies are assigned to the lineages
def get_frequencies_from_CPLEX(opt):
	lin_index = opt.phi_start_index
	sublin_num = opt.sublin_num
	solution = opt.my_prob.solution.get_values()
	my_freqs = []

	for i in range(sublin_num):
		freq = solution[lin_index + i]
		# in few cases it can happen that CPLEX computes a frequency slightly smaller than 0 
		# or slightly smaller than 1 although
		# I set the lower bound of the frequency to be 0 and the upper bound to be 1
		# if this is the case, set the frequency to 0 or 1 manually
		if freq < 0 and freq <= -0.00001:
			freq = 0
		elif freq > 1 and freq - 1 <= 0.00001:
			freq = 1
		my_freqs.append(freq)

	return my_freqs


# gets the solution from CPLEX, which sublineages are children of other sublineages
def add_sublins_from_CPLEX(opt, my_lineages, check_z_transitivity=True, dont_break_z_symmetry=False):

	lin_index = opt.z_index_start
	sublin_num = opt.sublin_num
	solution = opt.my_prob.solution.get_values()

	# if entry in Z matrix in row i and column j is 1, sublineage i is parent
	# of sublineage j, so we append the value of j in the array of i
	for i in range(sublin_num):
		for j in range(sublin_num):
			# entry in Z-matrix is set to 1
			if round(solution[lin_index + i * sublin_num + j]) == 1:
				# if Z-matrix is fixed, ancestor-descendant relation can be taken directly
				if opt.z_matrix_is_fixed == True:
					my_lineages[i].sublins.append(j)
					continue

				# if i is the index of the normal lineage, all the other lineages are its
				# descendants anyway
				if i == 0:
					my_lineages[i].sublins.append(j)
					continue

				if dont_break_z_symmetry == False:
					# lineage divergence rule is active
					if (opt.use_lineage_divergence_rule == True and
						round(solution[opt.my_colnames_ldr_active_index_friendly_form[i][j]])):
						my_lineages[i].sublins.append(j)
						continue

					# otherwise it needs to be further checked whether an SSM of lineage i is 
					# influenced by a CN change in lineage j 
					# only then an ancestor-descendant relation really holds
					dssm_infl_cnv_a_p1_index = [my_list[i - 1][j - i - 1] 
						for my_list in opt.my_colnames_dssm_infl_cnv_a_p1_index]
					influence_a_p1 = sum(solution[index] for index in dssm_infl_cnv_a_p1_index)	
					dssm_infl_cnv_b_p1_index = [my_list[i - 1][j - i - 1] 
						for my_list in opt.my_colnames_dssm_infl_cnv_b_p1_index]
					influence_b_p1 = sum(solution[index] for index in dssm_infl_cnv_b_p1_index)	
					dssm_infl_cnv_a_m1_index = [my_list[i - 1][j - i - 1] 
						for my_list in opt.my_colnames_dssm_infl_cnv_a_m1_index]
					influence_a_m1 = sum(solution[index] for index in dssm_infl_cnv_a_m1_index)	
					dssm_infl_cnv_b_m1_index = [my_list[i - 1][j - i - 1] 
						for my_list in opt.my_colnames_dssm_infl_cnv_b_m1_index]
					influence_b_m1 = sum(solution[index] for index in dssm_infl_cnv_b_m1_index)	

					# influence by CN changes is given
					if round(influence_a_p1 + influence_b_p1 + influence_a_m1 + influence_b_m1) >= 1.0:
						my_lineages[i].sublins.append(j)
				else:
					my_lineages[i].sublins.append(j)
	
	# transitivity of Z can be checked afterwards
	if check_z_transitivity == False:
		return

	# transitivity rules for Z can direcly be checked
	# check and update Z-matrix
	z_matrix, zero_count = get_Z_matrix(my_lineages)
	lineage_num = len(my_lineages)
	check_and_update_complete_Z_matrix_from_matrix(z_matrix, zero_count, lineage_num)
	# update lineages based on direct update of Z-matrix because of the transitivity rule
	update_linage_relations_based_on_z_matrix(my_lineages, z_matrix)
	


# gets the solution for the CNV from the CPLEX results and writes them in a list
def add_CNV_from_CPLEX(opt, seg_list, start_index_p1_a, start_index_m1_a, my_lineages):
	# makes variables local
	my_colnames = opt.my_colnames
	solution = opt.my_prob.solution.get_values()
	seg_num = opt.seg_num
	sublin_num = opt.sublin_num

	# calculate all possible start indicies for different states
	start_index_p1_b = start_index_p1_a + (seg_num * sublin_num)
	start_index_m1_b = start_index_m1_a + (seg_num * sublin_num)
	
	for i in xrange(seg_num):
		cnv_appeared = False
		my_segment = seg_list[i]
		for j in range(sublin_num):
			offset = i * sublin_num + j
			if round(solution[start_index_p1_a + offset]) == 1:
				my_cnv = cnv.CNV(1, my_segment.index, my_segment.chr, my_segment.start, 
					my_segment.end)
				my_lineages[j].cnvs_a.append(my_cnv)
				cnv_appeared = True
			if round(solution[start_index_p1_b + offset]) == 1:
				my_cnv = cnv.CNV(1, my_segment.index, my_segment.chr, my_segment.start, 
					my_segment.end)
				my_lineages[j].cnvs_b.append(my_cnv)
				cnv_appeared = True
			if round(solution[start_index_m1_a + offset]) == 1:
				my_cnv = cnv.CNV(-1, my_segment.index, my_segment.chr, my_segment.start, 
					my_segment.end)
				my_lineages[j].cnvs_a.append(my_cnv)
				cnv_appeared = True
			if round(solution[start_index_m1_b + offset]) == 1: 
				my_cnv = cnv.CNV(-1, my_segment.index, my_segment.chr, my_segment.start, 
					my_segment.end)
				my_lineages[j].cnvs_b.append(my_cnv)
				cnv_appeared = True
		if cnv_appeared == False:
			my_cnv = cnv.CNV(0, my_segment.index, my_segment.chr, my_segment.start, my_segment.end)
			my_lineages[cons.NORMAL].cnvs_a.append(my_cnv)

########
##### add_SSMs_per_phase_from_CPLEX 

def add_SSMs_per_phase_from_CPLEX(opt, ssm_list, start_index, my_lineages, phase, single_segment=False):
	my_colnames = opt.my_colnames
	solution = opt.my_prob.solution.get_values()

	# variables needed when unphased SSMs have to be calculated from phased SSMs to A
	#	(when more complex CN changes are allowed to be modeled)
	phasing_list = [-1] * opt.sublin_num
	prev_seg_index = -1
	prev_pos = -1
	phis = solution[opt.phi_start_index:opt.phi_start_index+opt.sublin_num]
	dc_des_max_k = opt.sublin_num - 2
	dc_des_anc_num_entries_per_line = get_sum_num_k_for_dc_des(opt.sublin_num, dc_des_max_k)
	dc_des_anc_num_entries_per_matrix = dc_des_anc_num_entries_per_line * opt.seg_num
	dc_des_start_index_per_k = [get_sum_num_k_for_dc_des(opt.sublin_num, k-1) for k in range(opt.sublin_num-1)]
	dc_des_num_entries_per_k = [get_num_k_for_dc_des(opt.sublin_num, k) for k in range(opt.sublin_num-1)]
	dc_ans_min_k = 2
	dc_anc_start_index_per_k = [get_sum_num_k_for_dc_anc(k-1) for k in range(opt.sublin_num)]
	dc_anc_num_entries_per_k = [get_num_k_for_dc_anc(k) for k in range(opt.sublin_num)]
	dc_binary_num_entries_per_matrix = opt.seg_num * opt.sublin_num

	# checking each entry in SSM matrix with given phase
	for i in xrange(opt.ssm_num * opt.sublin_num):
		# SSM is assigned
		if round(solution[start_index + i]) == 1:
			# get index of sublinage to which the SSM is assigned and index of SSM
			split_var = my_colnames[start_index + i].split("_")
			sublin_index = int(split_var[-1])
			ssm_index = int(split_var[-2])

			# check whether SSM was influenced by CN gain in same lineage
			if (round(solution[opt.my_colnames_dssm_infl_cnv_same_lineage_a_index[ssm_index][sublin_index-1]]) == 1
				or round(
				solution[opt.my_colnames_dssm_infl_cnv_same_lineage_b_index[ssm_index][sublin_index-1]]) 
				== 1):
				ssm_list[ssm_index].infl_cnv_same_lin = True
			else:
				ssm_list[ssm_index].infl_cnv_same_lin = False

			# if only simple CN changes are allowed and thus the unphased state exists,
			#	SSMs are directly assigned to their lineages in the given phase
			if opt.simple_CN_changes:
				assign_ssms_to_lineages_phases(my_lineages, sublin_index, ssm_list, ssm_index, phase)
			# if more complex CN states are allowed to be modeled,
			#	the phase needs to be refinded because it could be that the SSM should
			#	actually be unphased
			else:
				current_seg_index = 0
				if not single_segment:
					current_seg_index = ssm_list[ssm_index].seg_index
				# if other segment starts
				if current_seg_index != prev_seg_index:
					phasing_list = [-1] * opt.sublin_num
				prev_seg_index = current_seg_index
				prev_pos = ssm_list[ssm_index].pos

				# if phase wasn't determinded yet for SSM in this segment and lineage
				if phasing_list[sublin_index] == -1:
					# CN change on chromatid A
					change_A = get_CN_change(cons.A, current_seg_index, sublin_index, solution,
						phis, dc_binary_num_entries_per_matrix, 
						dc_des_anc_num_entries_per_matrix,
						dc_des_num_entries_per_k, dc_des_anc_num_entries_per_line,
						dc_des_start_index_per_k,
						dc_des_max_k, dc_ans_min_k, dc_anc_num_entries_per_k,
						dc_anc_start_index_per_k, opt)
					# CN change on chromatid B
					change_B = get_CN_change(cons.B, current_seg_index, sublin_index, solution,
						phis, dc_binary_num_entries_per_matrix, 
						dc_des_anc_num_entries_per_matrix,
						dc_des_num_entries_per_k, dc_des_anc_num_entries_per_line,
						dc_des_start_index_per_k,
						dc_des_max_k, dc_ans_min_k, dc_anc_num_entries_per_k,
						dc_anc_start_index_per_k, opt)
					# if the difference in CN change on both chromatids is equal,
					#	the SSM is unphased
					# if there is no change on A and B, the SSM is unphased
					if change_A == 0 and change_B == 0:
						phasing_list[sublin_index] = cons.UNPHASED
					elif phase == cons.A:
						phasing_list[sublin_index] = cons.A
					else:
						phasing_list[sublin_index] = cons.B

				# get needed indices of variables that stores influence of CN gain in same lineage
				# as SSM and that stores whether other allele is lost in same lineage
				dssm_infl_cnv_same_lineage_phase = opt.my_colnames_dssm_infl_cnv_same_lineage_a_index
				dc_m1_phase_offset = 0
				if phase == cons.B:
					dssm_infl_cnv_same_lineage_phase = (
						opt.my_colnames_dssm_infl_cnv_same_lineage_b_index)
				if other_phase == cons.B:
					dc_m1_phase_offset = dc_binary_num_entries_per_matrix
				# check whether SSM is influenced by CN gain in the same lineage, if so
				# set phase directly
				if round(solution[dssm_infl_cnv_same_lineage_phase
					[ssm_index][sublin_index-1]]) == 1:
					assign_ssms_to_lineages_phases(my_lineages, sublin_index, 
						ssm_list, ssm_index, phase)
				# check whether other allele is deleted in same lineage, if so, keep found phase
				elif (round(solution[opt.dc_binary_index_start_m1 + dc_m1_phase_offset + 
					(current_seg_index * opt.sublin_num) + sublin_index]) == 1):
					assign_ssms_to_lineages_phases(my_lineages, sublin_index, 
						ssm_list, ssm_index, phase)
				# if SSM is not influenced by CN gain in the same lineage,
				# it is assigned to its lineages with the (just) computed phase
				else:
					assign_ssms_to_lineages_phases(my_lineages, sublin_index, ssm_list,
						ssm_index, phasing_list[sublin_index])

# computes the CN change of a chromatid based on a lineage sublin_index, only lineage that are ancestors or
#	deceendants of this lineage influence the CN change
# phase: current phase whose CN change should be computed
# solution: CPLEX results
# opt: optimization object
def get_CN_change(phase, current_seg_index, sublin_index, solution, phis, dc_binary_num_entries_per_matrix, 
	dc_des_anc_num_entries_per_matrix, dc_des_num_entries_per_k, dc_des_anc_num_entries_per_line, 
	dc_des_start_index_per_k, dc_des_max_k, dc_ans_min_k, dc_anc_num_entries_per_k, dc_anc_start_index_per_k,
	opt):

	# offsets are set if phase B is considered
	dc_m1_phase_offset = 0
	dc_des_anc_phase_offset = 0
	if phase == cons.B:
		dc_m1_phase_offset = dc_binary_num_entries_per_matrix
		dc_des_anc_phase_offset = dc_des_anc_num_entries_per_matrix

	# dc_descendant change is comuted
	# change can be positive and negative
	dc_descendant_p1_change = dc_descendant_m1_change = 0
	if sublin_index <= dc_des_max_k:
		for i in range(dc_des_num_entries_per_k[sublin_index]):
			# variable offset + matrix offset + line offset + offset for sublin index + current position
			position_p1 = (opt.dc_descdendant_start_index + dc_des_anc_phase_offset + 
				(current_seg_index * dc_des_anc_num_entries_per_line) + dc_des_start_index_per_k[sublin_index] + i)
			position_m1 = position_p1 + (2 * dc_des_anc_num_entries_per_matrix)
			k_prime = sublin_index + 1 + i
			dc_descendant_p1_change += round(solution[position_p1]) * phis[k_prime]
			dc_descendant_m1_change -= round(solution[position_m1]) * phis[k_prime]

	# dc_ancestral change is computed
	dc_ancestral_change = 0
	if sublin_index  >= dc_ans_min_k:
		for i in range(dc_anc_num_entries_per_k[sublin_index]):
			# variable offset + matrix offset + line offset + offset for sublin index + current position
			position = (opt.dc_ancestral_start_index + dc_des_anc_phase_offset + 
				(current_seg_index * dc_des_anc_num_entries_per_line) + dc_anc_start_index_per_k[sublin_index] + i)
			dc_ancestral_change -= round(solution[position]) * phis[i + 1]

	# dc_m1 change
	dc_m1_value = round(solution[opt.dc_binary_index_start_m1 + dc_m1_phase_offset + 
		(current_seg_index * opt.sublin_num) + sublin_index])
	dc_m1_change = - (dc_m1_value * phis[sublin_index])

	return dc_descendant_p1_change + dc_descendant_m1_change + dc_ancestral_change + dc_m1_change

# summed up number of entries of dc_descendant_a/b_state_i_k with k <= k_max
def get_sum_num_k_for_dc_des(sublin_num, max_k):
	k = 0
	for i in range(1, max_k + 1):
		k += get_num_k_for_dc_des(sublin_num, i)
	return k
				
# number of entries dc_descendant_a/b_state_i_k has
def get_num_k_for_dc_des(sublin_num, k):
	if k < 1:
		return 0
	return sublin_num - 1 - k

# summed up number of entries of dc_ancestral_a/b_state_i_k' with k' <= max_k
def get_sum_num_k_for_dc_anc(max_k_prime):
	k = 0
	for i in range(2, max_k_prime + 1):
		k += get_num_k_for_dc_anc(i)
	return k

# number of entries dc_ancestral_a/b_state_i_k' has
def get_num_k_for_dc_anc(k_prime):
	if k_prime < 2:
		return 0
	return k_prime - 1

def assign_ssms_to_lineages_phases(my_lineages, sublin_index, ssm_list, ssm_index, phase):
	# add lineage index and phase to SSM
	ssm_list[ssm_index].lineage = sublin_index
	ssm_list[ssm_index].phase = phase

	# assign SSM to lineage and phase
	if phase == cons.UNPHASED:
		my_lineages[sublin_index].ssms.append(ssm_list[ssm_index])
	elif phase == cons.A:
		my_lineages[sublin_index].ssms_a.append(ssm_list[ssm_index])
	elif phase == cons.B:
		my_lineages[sublin_index].ssms_b.append(ssm_list[ssm_index])

##### add_SSMs_per_phase_from_CPLEX 
########

def add_SNPs_per_phase_from_CPLEX(opt, snp_list, start_index, my_lineages, phase):

	solution = opt.my_prob.solution.get_values()
	for i in xrange(opt.snp_num):
		# SNP is assigned
		if round(solution[start_index + i]) == 1:
			if phase == cons.A:
				my_lineages[0].snps_a.append(snp_list[i])	
			elif phase == cons.B:
				my_lineages[0].snps_b.append(snp_list[i])	
			elif phase == cons.UNPHASED:
				my_lineages[0].snps.append(snp_list[i])	

# average CN for all segments is computed
def compute_average_cn(lineages, seg_num):

	if seg_num <= 1:
		raise eo.MyException("Segment number has to be at least 1.")

	# create array with standard CNs of 2.0
	average_cn = [2.0] * seg_num

	# all lineages are checked
	for i, my_lin in enumerate(lineages):
		
		# normal lineage does not need to get checked because it does not contain any mutations
		if i == 0:
			continue

		# CN changes of allele A and B change the average CN
		compute_average_cn_for_cnvs(my_lin.cnvs_a, average_cn, my_lin.freq)
		compute_average_cn_for_cnvs(my_lin.cnvs_b, average_cn, my_lin.freq)
	
	return average_cn

# CN allele-specific CN changes change the average CN
def compute_average_cn_for_cnvs(my_cnvs, average_cn, lin_freq):
	# each CN change is checked
	for single_cnv in my_cnvs:
		average_cn[single_cnv.seg_index] += (single_cnv.change * lin_freq)

def compute_average_cn_difference(lineage_1, lineage_2, seg_num):
	ci_1 = compute_average_cn(lineage_1, seg_num)
	ci_2 = compute_average_cn(lineage_2, seg_num)

	differences = [abs(ci_1[i] - ci_2[i]) for i in xrange(seg_num)]
	average_diff = float(sum(differences))/seg_num

	return average_diff

# checks whether a fixed Z matrix is feasible with fixed phis
# direct descendants of a lineage are not allowed to have a higher
#	frequency than the lineage itself
# direct_descendants[lineage][direct descendants] 
def z_matrix_phis_feasible(direct_descendants, fixed_phis):
	# add normal frequency to list
	complete_phis = [1] + fixed_phis

	for i in range(len(direct_descendants)):
		# frequencies of direct descendants
		freq_descendants = sum([complete_phis[index] 
			for index in direct_descendants[i]])
		# frequency of lineage must not be lower than the one
		# of its direct descendants
		if complete_phis[i] < freq_descendants:
			return False

	return True

# parses a list with direct descendant of the format
# direct_descendants[lineage][direct descendants] 
# creates a new list in the format
# list[lineage, direct descendants]
# creates only an entry when a lineage has more than one direct descandant
def parse_direct_descendants_for_constraints(direct_descendants):
	direct_descendants_for_constraints = []
	for i, entry in enumerate(direct_descendants):
		if len(entry) > 1:
			new_entry = [i]
			new_entry.extend(direct_descendants[i])
			direct_descendants_for_constraints.append(new_entry)

	return direct_descendants_for_constraints

# given a Z matrix (triangle shape, last row missing), a list with direct descendants is created
# format of the list:
# tmp_direct_descendant[lineage][direct descendants]
def get_direct_descendants(z_matrix, sublin_num):
	unused_direct_descendants = [True] * sublin_num
	direct_descendant = [[] for __ in range(sublin_num)]

	if is_z_matrix_quadratic(z_matrix):
		quadratic_z_matrix = z_matrix
	else:
		quadratic_z_matrix = get_quadratic_z_matrix(z_matrix)

	# start with the second highest lineage index
	for lineage_index in reversed(xrange(sublin_num - 1)):
		# go through all entries in the Z matrix in this line, start from the end
		# go until diagonal
		for column_index in xrange(lineage_index+1, sublin_num):
			# check if a descendant was found and whether this is a direct descendant
			if (quadratic_z_matrix[lineage_index][column_index] == 1 
				and unused_direct_descendants[column_index]):
					direct_descendant[lineage_index].append(column_index)
					unused_direct_descendants[column_index] = False

	return direct_descendant

# check whether matrix has a quadratic shape
def is_z_matrix_quadratic(z_matrix):
	row_dimension = len(z_matrix)
	entries_last_row = len(z_matrix[-1])
	# if matrix has as many entries in the last row as it has rows, it's said to be quadratic
	if row_dimension == entries_last_row and row_dimension != 1:
		return True
	return False

# given a triangle Z-matrix, where the last row is missing and all entries of the 
# lower left triangle, a quadratic Z-matrix is constructed
def get_quadratic_z_matrix(triangle_z_matrix):
	sublin_num = len(triangle_z_matrix) + 1
	# create quadratic shape with -1 entries
	z_matrix = [[-1] * sublin_num for _ in xrange(sublin_num)]

	# fill entries according to triangle_z_matrix
	column_offset = 1
	for row_index in xrange(len(triangle_z_matrix)):
		for column_index in xrange(len(triangle_z_matrix[row_index])):
			z_matrix[row_index][column_index+column_offset] = triangle_z_matrix[row_index][column_index]
		column_offset += 1

	return z_matrix
		

# gets Z matrix in a flat list where the first row is missing
# parses it into a 2D list and creates the first row for the normal lineage
# triangle shape, last row is missing
def parse_fixed_z_matrix_list_to_matrix(fixed_z_matrix_list, sublin_num):
	z_matrix = [[] for __ in range(sublin_num - 1)]
	# fill first row
	z_matrix[0] = [1] * (sublin_num - 1)
	# fill other row
	fixed_z_matrix_list_index = 0
	for i in range(1, sublin_num - 1):
		
		z_matrix[i].extend(fixed_z_matrix_list[fixed_z_matrix_list_index : fixed_z_matrix_list_index + 
			sublin_num - 1 - i])
		fixed_z_matrix_list_index += sublin_num - 1 - i
	# check for wrong z matrix lists
	if (z_matrix[-1] == []) or (len(z_matrix[-1]) != 1):
		raise eo.MyException("Fixed Z matrix list has wrong format.")
	# return matrix
	return z_matrix

def compare_result_files_and_change_labeling(sim_result_file, true_result_file, new_result_file):
	indices = evaluation.compute_min_bipartite_match(true_result_file, sim_result_file, verbose=False)[0]
	sim_result = oio.read_result_file(sim_result_file)
	new_result = change_labeling_of_lineage(indices, sim_result)
	oio.write_result_file(new_result, new_result_file)

# creates a new true and inferred lineages, where the number of both is equal and the
# lineage ordering of the inferred lineages correspondes to the order of the true lineages
# the ordering information is given with the indices mapping
def change_labeling_of_lineage(indices, old_lin_inferred, old_lin_true):
	
	# true lineages are copied
	new_lin_true = copy.deepcopy(old_lin_true)
	# if inferred reconstruction has more lineages, empty lineages are appended to the true reconstruction
	diff_to_true = len(old_lin_inferred) - len(old_lin_true)
	for i in range(diff_to_true):
		new_lin_true.append(lineage.Lineage([], 0.0, [], [], [], [], [], [], [], []))
		if len(new_lin_true[0].sublins) > 0:
			new_lin_true[0].sublins.append(new_lin_true[0].sublins[-1]+1)
		else:
			new_lin_true[0].sublins = [1]

	# inferred lineages are copied
	new_lin_inferred = copy.deepcopy(old_lin_inferred)
	#check if old_lin_inferred has enough entries and calculate difference
	diff = len(indices) - len(old_lin_inferred)
	#add missing entries
	for _ in range(diff):
		new_lin_inferred.append(lineage.Lineage([], 0.0, [], [], [], [], [], [], [], []))
		if len(new_lin_inferred[0].sublins) > 0:
			new_lin_inferred[0].sublins.append(new_lin_inferred[0].sublins[-1]+1)
		else:
			new_lin_inferred[0].sublins = [1]

	#check if order of inferred reconstruction is already right
	ordered = True
	for (i, j) in indices:
		if i != j:
			ordered = False
			break
	if ordered:
		return (new_lin_inferred, new_lin_true)
	# order old_lin_inferred
	else:
		# create dictionary, which contains mapping between old_lin_inferred lineage_number 
		# and new_lin_inferred lineage_number. Relevant for change of sublineages

		sublin_dict = dict()
		for (change, old) in indices:
			sublin_dict[old] = change

		ordered_lin_inferred = [[] for i in range(len(indices))]
		for (change, new) in indices:
			ordered_lin_inferred[change] = new_lin_inferred[new]
			# rename sublineages
			old_sublins = new_lin_inferred[new].sublins
			new_sublins = []
			for i in old_sublins:
				new_sublins.append(sublin_dict[i])
			ordered_lin_inferred[change].sublins = new_sublins

		return (ordered_lin_inferred, new_lin_true)

		#sublin_dict = dict()
		#for (change, old) in indices:
		#	sublin_dict[old] = change

		#new_lin_inferred = []
		##create new order
		#for (_, old) in indices:
		#	# append copy of old entry
		#	new_lin_inferred.append(copy.deepcopy(old_lin_inferred[old]))
		#	# create new list of sublineages due to changed lineage numbers
		#	new_sublins = []
		#	for sublin in new_lin_inferred[-1].sublins:
		#		new_sublins.append(sublin_dict[sublin])
		#	new_lin_inferred[-1].sublins = new_sublins

		#return new_lin_inferred

# computes the B-allele frequency
def compute_BAF(variance_count, total):
	if variance_count == 0:
		return 0
	if total == 0:
		raise eo.BAFComputationException("Total count is 0 but variant count isn't.")
	if total < variance_count:
		raise eo.BAFComputationException("Variant count is bigger than total count.")
	return float(variance_count)/total

def compute_LogR_normal(count, avg_count):
	return np.log2(count + 0.1) - np.log2(avg_count)

def compute_LogR_tumor(count, count_normal):
	return np.log2(count + 0.1) - np.log2(count_normal + 0.1)

def compute_LogR_star(logr, median_logr):
	return logr - median_logr

def median_list(my_list):
	return np.median(np.array(my_list))

# computes the LogR value (old computation)
def compute_LogR(count, avg_count):
	return np.log2(count + 1) - np.log2(avg_count + 1)

# in the Z-matrix, some entries are trivial as their solution is already given
# these are the entries of the first row and of the lower, left triangle including
#	the diagonal
# this leads to a specific number of entries that don't have a trivial solution
def get_number_of_untrivial_z_entries(sublin_num):
	if sublin_num > 2:
		return (((sublin_num - 1) * (sublin_num - 2)) / 2)
	return 0

# get number of segments from a list of cnvs by incrementing the last segment index
def get_number_of_segments(cnvs):
	cnvs_sorted = sorted(cnvs, key = lambda x: (x.chr, x.start))
	return (cnvs_sorted[-1].seg_index + 1)

# compare results of simulated data and cplex output
# percentage_per_mutation: if True found_ssm_percentage and found_cnv_percentage are calculated on the 
#	basis how many mutations are put into the right lineage in general and not per lineage
def compare_results(lineages_true, lineages_cplex, percentage_per_mutation=False):
	freq_difference = 0
	found_ssm_percentage = 0
	found_cnv_percentage = 0
	ssm_num = 0
	cnv_num = 0
	lineage_num = 0
	ssm_num_found = 0
	cnv_num_found = 0
	lineages_without_normal = len(lineages_true) -1

	for i in range(1, len(lineages_true)):
		# counts number of true ssms in lineage on both chromatids and unphased
		ssms_num_lin = (len(lineages_true[i].ssms) + len(lineages_true[i].ssms_a) + 
			len(lineages_true[i].ssms_b))
		if percentage_per_mutation:
			ssm_num += ssms_num_lin
		# counts number of cnvs in lineage on both chromatids
		cnv_num_lin = len(lineages_true[i].cnvs_a) + len(lineages_true[i].cnvs_b)
		if percentage_per_mutation:
			cnv_num += cnv_num_lin

		# only compare true lineages with cplex lineages if a cplex lineage for the index
		# number exists
		if i < len(lineages_cplex):
			# difference in frequency, if lineage_cplex has frequency bigger 0
			if lineages_cplex[i].freq > 0:
				freq_difference += math.fabs(lineages_true[i].freq - lineages_cplex[i].freq)
				lineage_num += 1

			# difference in found number of ssms
			# counts how many of the true unphased ssms can be found somewhere in the computed lineage
			found_ssms_in_unphased = (check_if_cplex_found_ssms_in_one_list(lineages_true[i].ssms, 
				lineages_cplex[i].ssms)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms, lineages_cplex[i].ssms_a)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms, lineages_cplex[i].ssms_b))
			# counts how many of the true ssms phased to A can be found somewhere 
			# in the computed lineage
			found_ssms_in_a = (check_if_cplex_found_ssms_in_one_list(lineages_true[i].ssms_a, 
				lineages_cplex[i].ssms)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms_a, lineages_cplex[i].ssms_a)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms_a, lineages_cplex[i].ssms_b))
			# counts how many of the true ssms phased to B can be found somewhere 
			# in the computed lineage
			found_ssms_in_b = (check_if_cplex_found_ssms_in_one_list(lineages_true[i].ssms_b, 
				lineages_cplex[i].ssms)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms_b, lineages_cplex[i].ssms_a)
				+ check_if_cplex_found_ssms_in_one_list(
				lineages_true[i].ssms_b, lineages_cplex[i].ssms_b))
			if percentage_per_mutation:
				ssm_num_found = (ssm_num_found + found_ssms_in_unphased + found_ssms_in_a 
					+ found_ssms_in_b)
			else:
				# if there are no ssms in the lineage, all are said to be found
				if ssms_num_lin == 0:
					found_ssm_percentage += 1
				# percentage of found ssms is added to varible
				else:
					found_ssm_percentage += (float(found_ssms_in_unphased + found_ssms_in_a + 
						found_ssms_in_b) / ssms_num_lin)

			# difference in found number of cnvs
			# counts how many cvs phased to A are found somewhere in computed lineage
			found_cnvs_in_a = (check_if_cplex_found_cnvs_in_one_list(lineages_true[i].cnvs_a, 
				lineages_cplex[i].cnvs_a)
				+ check_if_cplex_found_cnvs_in_one_list(
				lineages_true[i].cnvs_a, lineages_cplex[i].cnvs_b))
			# counts how many cnvs phased to B are found somewhere in computed lineage
			found_cnvs_in_b = (check_if_cplex_found_cnvs_in_one_list(lineages_true[i].cnvs_b, 
				lineages_cplex[i].cnvs_a)
				+ check_if_cplex_found_cnvs_in_one_list(
				lineages_true[i].cnvs_b, lineages_cplex[i].cnvs_b))

			if percentage_per_mutation:
				cnv_num_found = cnv_num_found + found_cnvs_in_a + found_cnvs_in_b
			else:
				# if there are no cnvs in the lineage, all are said to be found 
				if cnv_num_lin == 0:
					found_cnv_percentage += 1
				# percentage of found cnvs is added to variable
				else :
					found_cnv_percentage += ((float (found_cnvs_in_a + found_cnvs_in_b) 
						/ cnv_num_lin))

	# mean frequency difference, percentage of ssms and cnvs is computed
	freq_difference = float(freq_difference) / lineage_num
	if percentage_per_mutation:
		if ssm_num == 0:
			found_ssm_percentage = 1
		else:
			found_ssm_percentage = float(ssm_num_found) / ssm_num
		if cnv_num == 0:
			found_cnv_percentage = 1
		else:
			found_cnv_percentage = float(cnv_num_found) / cnv_num
	else:
		found_ssm_percentage = found_ssm_percentage / lineages_without_normal
		found_cnv_percentage = found_cnv_percentage / lineages_without_normal

	#print "freq_difference: {0}, found_ssm_percentage: {1}, found_cnv_percentage{2}".format(freq_difference,
	#	found_ssm_percentage, found_cnv_percentage)

	return (freq_difference, found_ssm_percentage, found_cnv_percentage)

def check_if_cplex_found_ssms_in_one_list(true_ssm_list, cplex_list):
	found_ssms = 0
	for i in range(len(true_ssm_list)):
		for j in range(len(cplex_list)):
			condition = (true_ssm_list[i].chr == cplex_list[j].chr 
				and true_ssm_list[i].pos == cplex_list[j].pos)
			if condition:
				found_ssms += 1
				break
	return found_ssms

def check_if_cplex_found_cnvs_in_one_list(true_cnv_list, cplex_list):
	found_cnvs = 0
	for i in range(len(true_cnv_list)):
		for j in range(len(cplex_list)):
			condition = (true_cnv_list[i].change == cplex_list[j].change
				and true_cnv_list[i].chr == cplex_list[j].chr
				and true_cnv_list[i].start == cplex_list[j].start)
			if condition:
				found_cnvs += 1
				break
	return found_cnvs

# check if number of entries z matrix was changed
def check_number_entries_z_matrix(opt):
	if ((len(opt.my_colnames_z) != opt.sublin_num) and
		(len(opt.my_colnames_z[opt.sublin_num - 1])) != opt.sublin_num):
		print ("ATTENTION in optimization in save_part_of_solution_in_class and following." +
			"Different number of Z variables!")

# see not efrom 26.9.15, problem with z-matrix 
#TODO When I need this function, check whether CPLEX solution should be rounded
def refine_z_matrix(opt, ssm_list):

	# check if number of entries of variables were changed
	check_number_entries_z_matrix(opt)

	solution = opt.my_prob.solution.get_values()

	for i in range(opt.sublin_num * (opt.sublin_num - 1)):
		index = opt.z_index_start + i + opt.sublin_num
		#print opt.my_colnames[index]
		#print opt.solution_z[i]
		if opt.solution_z[i] == 1:
			relation_between_ssm_cnv = False
			split_name = opt.my_colnames[index].split("_")
			ancestor = int(split_name[1])
			descendant = int(split_name[2])
			#print "{0}, {1}, {2}".format(split_name, ancestor, descendant)
			for j in xrange(opt.ssm_num):
				dssm_a_j_anc = solution[opt.dssm_start_index + 
					opt.sublin_num * opt.ssm_num + opt.sublin_num * j +
					ancestor]
				#print dssm_a_j_anc
				dssm_b_j_anc = solution[opt.dssm_start_index + 
					opt.sublin_num * opt.ssm_num * 2 + opt.sublin_num * j +
					ancestor]
				seg_index = ssm_list[j].seg_index
				#print dssm_b_j_anc
				dc_a_p1_bin_dec = solution[opt.dc_binary_index_start_p1 +
					opt.sublin_num * seg_index + descendant]
				#print dc_a_p1_b_dec
				dc_b_p1_bin_dec = solution[opt.dc_binary_index_start_p1 +
					opt.sublin_num * opt.seg_num + 
					opt.sublin_num * seg_index + descendant]
				#print dc_b_p1_b_dec
				relation_between_ssm_cnv = ((dssm_a_j_anc * dc_a_p1_bin_dec) 
					and (dssm_b_j_anc * dc_b_p1_bin_dec))
				if relation_between_ssm_cnv:
					break
			if relation_between_ssm_cnv == False:
				#print "Value will be changed."
				opt.solution_z[i] = 0.0

def sort_segments(segment_list):
	return sorted(segment_list, key = lambda x: (x.chr, x.start))

# each segment is given an index of the position where it appears in a list sorted by its
# chromosome and starting position
def add_segment_index(segment_list):
	for i in xrange(len(segment_list)):
		segment_list[i].index = i

def sort_cnvs(cnv_list):
	return sorted(cnv_list, key = lambda x:(x.chr, x.start, x.end, x.change))

def sort_snps_ssms(mut_list):
	return sorted(mut_list, key = lambda x: (x.chr, x.pos))

# assigns mutations (SNPs and SSMs) to segments they appear on
def assign_muts_to_segments(seg_list, mut_list):
	i = 0
	for mut in mut_list:
		while(i < len(seg_list)):
			if (seg_list[i].chr == mut.chr and
				seg_list[i].start <= mut.pos and
				seg_list[i].end >= mut.pos):
				mut.seg_index = i
				break
			else:
				i = i + 1
		# assignment didn't work
		if i == len(seg_list):
			raise eo.SegmentAssignmentException("Assigning of segments to mutations didn't work.")

# distributes the snps and ssms equally to the seg for testing cases only
def distribute_muts_between_seg_for_testing(mut_num, seg_num, mut_list):
	r = []
	muts_per_seg = int(mut_num / seg_num)
	zeros_before = 0
	zeros_after = seg_num - 1
	for i in range(mut_num):
		r_line = [0] * zeros_before + [1] + [0] * zeros_after
		r.append(r_line)

		# write index to mutation in list
		mut_list[i].seg_index = zeros_before

		if zeros_after == 0:
			zeros_before = 0
			zeros_after = seg_num - 1
		else:
			zeros_before = zeros_before + 1
			zeros_after = zeros_after - 1
	return r


# compute all differences in mutations between two lineages without regard of phasing
def compute_diff_between_lineages(lineage1, lineage2):
	# variabel for an existing lineage, if one lineage is None
	exist_lineage = None
	# difference count
	diff = 0

	# check if one lineage is None and preparation
	if lineage1 is None:
		exist_lineage = lineage2
	elif lineage2 is None:
		exist_lineage = lineage1
	# both lineages exist, count different mutations without phasing
	else:
		# get all cnvs, snps and ssms for each lineage
		cnvs_lin1 = []
		cnvs_lin2 = []
		snps_lin1 = []
		snps_lin2 = []
		ssms_lin1 = []
		ssms_lin2 = []

		cnvs_lin1.extend(lineage1.cnvs_a)
		cnvs_lin1.extend(lineage1.cnvs_b)
		snps_lin1.extend(lineage1.snps)
		snps_lin1.extend(lineage1.snps_a)
		snps_lin1.extend(lineage1.snps_b)
		ssms_lin1.extend(lineage1.ssms)
		ssms_lin1.extend(lineage1.ssms_a)
		ssms_lin1.extend(lineage1.ssms_b)

		cnvs_lin2.extend(lineage2.cnvs_a)
		cnvs_lin2.extend(lineage2.cnvs_b)
		snps_lin2.extend(lineage2.snps)
		snps_lin2.extend(lineage2.snps_a)
		snps_lin2.extend(lineage2.snps_b)
		ssms_lin2.extend(lineage2.ssms)
		ssms_lin2.extend(lineage2.ssms_a)
		ssms_lin2.extend(lineage2.ssms_b)

		# sort all mutations
		cnvs_lin1 = sort_cnvs(cnvs_lin1)
		snps_lin1 = sort_snps_ssms(snps_lin1)
		ssms_lin1 = sort_snps_ssms(ssms_lin1)
		cnvs_lin2 = sort_cnvs(cnvs_lin2)
		snps_lin2 = sort_snps_ssms(snps_lin2)
		ssms_lin2 = sort_snps_ssms(ssms_lin2)

		# check if one mutation list is empty, the remaining mutations are differences
		while cnvs_lin1 != [] and cnvs_lin2 != []:
			# if the mutations are equal, remove both from list. There is no difference
			if cnvs_lin1[0] == cnvs_lin2[0]:
				cnvs_lin1.pop(0)
				cnvs_lin2.pop(0)
			# if they are not equal, remove the lesser element and update the difference by 1
			elif cnvs_lin1[0] < cnvs_lin2[0]:
				diff += 1
				cnvs_lin1.pop(0)
			else:
				diff += 1
				cnvs_lin2.pop(0)
		diff += len(cnvs_lin1) + len(cnvs_lin2)

		# same logic as for cnvs
		while snps_lin1 != [] and snps_lin2 != []:
			if snps_lin1[0] == snps_lin2[0]:
				snps_lin1.pop(0)
				snps_lin2.pop(0)
			elif snps_lin1[0] < snps_lin2[0]:
				diff += 1
				snps_lin1.pop(0)
			else:
				diff += 1
				snps_lin2.pop(0)
		diff += len(snps_lin1) + len(snps_lin2)

		# same logic as for cnvs
		while ssms_lin1 != [] and ssms_lin2 != []:
			if ssms_lin1[0] == ssms_lin2[0]:
				ssms_lin1.pop(0)
				ssms_lin2.pop(0)
			elif ssms_lin1[0] < ssms_lin2[0]:
				diff += 1
				ssms_lin1.pop(0)
			else:
				diff += 1
				ssms_lin2.pop(0)
		diff += len(ssms_lin1) + len(ssms_lin2)

		return diff

	# both lineages are None, there are no differences(diff still 0)
	if exist_lineage is None:
		return diff
	# one lineage exist, count the mutations of that lineage 
	else:
		diff += len(exist_lineage.cnvs_a)
		diff += len(exist_lineage.cnvs_b)
		diff += len(exist_lineage.snps)
		diff += len(exist_lineage.snps_a)
		diff += len(exist_lineage.snps_b)
		diff += len(exist_lineage.ssms)
		diff += len(exist_lineage.ssms_a)
		diff += len(exist_lineage.ssms_b)

		return diff

# control a result file if all mutations are correct and included
def check_result_file(result_file, segment_file, snp_file, ssm_file):
	import constants as cons
	# read all input files
	lineages = oio.read_result_file(result_file)
	segments = oio.read_segment_file(segment_file)
	snps = oio.read_snp_ssm_file(snp_file, cons.SNP)
	ssms = oio.read_snp_ssm_file(ssm_file, cons.SSM)

	num_lineages = len(lineages)
	cnvs = []
	for lineage in lineages:
		cnvs.extend(lineage.cnvs_a)
		cnvs.extend(lineage.cnvs_b)

	# filter all cnvs with change 0
	cnvs = filter(lambda x: x.change != 0, cnvs)
	# sort cnvs and segments
	cnvs = sort_segments(cnvs)
	segments = sort_segments(segments)

	# check if all segments are included in the result_file and associated with the correcht CNVs
	# stores the CNVs contained in the observed segment
	match = []
	# Flag, if a duplicate CNV with correct change was found
	duplicate = False
	# Flag for a duplicate with different change, more than 2 CNVs in the same segment or not the same start- and endposition
	error_duplicate = False
	for segment in segments:
		# set all variables to default
		match = []
		duplicate = False
		error_duplicate = False
		for cnv in cnvs:
			# check if CNV lies in the segment
			if cnv.chr == segment.chr and cnv.start >= segment.start and cnv.end <= segment.end:
				if match == []:
					match.append(cnv)
				# if a match already exists, check if the difference is the change(accepted condition)
				elif cnv.chr == match[0].chr and cnv.start == match[0].start and cnv.end == match[0].end and duplicate == False:
					if cnv.change == 1 and match[0].change == -1:
						duplicate = True
						match.append(cnv)
					elif cnv.change == -1 and match[0].change == 1:
						duplicate = True
						match.append(cnv)
					else:
						error_duplicate = True
						match.append(cnv)
				else:
					error_duplicate = True
		
		# if the segment has no CNV, there is an error
		if match == []:
			print "Segment on chromosome {0} start: {1}, end: {2} from file \"{3}\" has no CNV in result-file \"{4}\"".format(segment.chr, segment.start, segment.end, segment_file, result_file)

		# if an error with duplicates occured, print all information about the segment and the corresponding CNVs
		if error_duplicate == True:
			print "Segment on chromosome {0} start: {1}, end: {2} from file \"{3}\" has wrong CNVs in result-file \"{4}\"".format(segment.chr, segment.start, segment.end, segment_file, result_file)
			for cnv in match:
				print "\tCNV on chromosome {0} start: {1}, end: {2}, change: {3}".format(cnv.chr, cnv.start, cnv.end, cnv.change)


	# check if all SNPs are included in the result_file
	for snp in snps:
		if not (snp in lineages[0].snps or snp in lineages[0].snps_a or snp in lineages[0].snps_b):
			print "SNP on chromosome {0} at position {1} from file \"{2}\" not found in result-file \"{3}\"".format(snp.chr, snp.pos, snp_file, result_file)

	# check if all ssms are included in the result_file
	error = True
	for ssm in ssms:
		error = True
		# check every lineage except 0
		for lineage_num in range (1,num_lineages):
			if ssm in lineages[lineage_num].ssms or ssm in lineages[lineage_num].ssms_a or ssm in lineages[lineage_num].ssms_b:
				error = False
		if error:
			print "SSM on chromosome {0} at position {1} from file \"{2}\" not found in result-file \"{3}\"".format(ssm.chr, ssm.pos, ssm_file, result_file)



class Z_Matrix_Co(object):

	def __init__(self, z_matrix, triplet_xys, triplet_ysx, triplet_xsy, present_ssms, CNVs, matrix_after_first_round):
		self.z_matrix = z_matrix
		self.triplet_xys = triplet_xys
		self.triplet_ysx = triplet_ysx
		self.triplet_xsy = triplet_xsy
		self.present_ssms = present_ssms
		self.CNVs = CNVs
		self.matrix_after_first_round = matrix_after_first_round
