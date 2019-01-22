import onctopus_io as oio
import model
import numpy as np
import scipy as sp
import scipy.sparse as spsp 
import constants as cons
from sklearn.metrics import average_precision_score
import argparse
import exceptions_onctopus as eo
import logging
import time
import sys
import exceptions_onctopus as eo
from os.path import dirname, abspath, join, realpath, isfile, sep, pardir
sys.path.append(dirname(realpath(__file__)) + sep + "parsing")
import cPickle

# AUPRC value for SSMs is computed between an inferred and the true lineage file
def compute_auprc_of_inferred_and_true_result_files(result_file_inferred, result_file_true):
	# read result files, get lineages
	lineages_inferred = oio.read_result_file(result_file_inferred)
	lineages_true = oio.read_result_file(result_file_true)

	# create co-clustering matrix
	co_clustering_matrix_inferred = get_co_clustering_matrix_from_lineages(lineages_inferred)
	co_clustering_matrix_true = get_co_clustering_matrix_from_lineages(lineages_true)

	# compute AUPRC
	auprc = co_clustering_matrices_to_avg_precision_score(co_clustering_matrix_true, 
		co_clustering_matrix_inferred)

	return (auprc, co_clustering_matrix_inferred, co_clustering_matrix_true)

# creates lineage assignements and computes co-clustering matrix
def get_co_clustering_matrix_from_lineages(my_lineages):
	lineage_assignment = np.asarray([x[1] for x in model.get_lineage_assignment_for_ssms_w_ssms(my_lineages)])
	return compute_co_clustering_matrix(lineage_assignment)

# result_file: inferred result file
# seg_file: segment file
# ssm_file: SSM file with all SSMs
# normal_seg_indices: indices of segments without CN changes
# ssm_subset_file: SSM file with SSMs that are used for comparison
#	subset of ssm_file, can also be the same file
# lineage_assignment_file: contains assignment of SSMs to true lineage, 
#	format: one line per SSM, line only contains assigned lineage
# auprc_file: name of file in which AUPRC is written
# test: boolean, indicating whether auprc_file is allowed to be overwritten
# allele_specific: boolean, indicating if the allele-specific modelling was used
#	must be true
# write_to_file: boolean, indicating if the AUPRC should be written to a file
#	would be the file auprc_file
def compute_auprc_of_inferred_lineages_and_true_lineage_assignment(result_file, seg_file, ssm_file, 
	normal_seg_indices, ssm_subset_file, lineage_assignment_file, auprc_file, test,
	allele_specific=True, write_to_file=False):

	if not allele_specific:
		raise eo.MyException("not done for non-allele-specific")

	# read different files
	# get lineages from result file
	logging.debug("read different files")
	start_reading = time.time()
	my_lineages = oio.read_result_file(result_file)
	(_, ssm_list) = model.create_segment_and_mutation_lists(seg_file,
		[], ssm_file, allele_specific)
	ssms_of_segment = model.choose_ssm_subset(normal_seg_indices, ssm_list)
	# ssm_subset is used for evaluation, only these SSMs are considered
	ssm_subset = oio.read_snp_ssm_file(ssm_subset_file, cons.SSM)
	end_reading = time.time()
	logging.debug("time to read: {0}".format(end_reading-start_reading))

	logging.debug("get lineage assignment")
	start_t = time.time()
	# get lineage assignment for SSMs
	# list contains SSMs and their assigned lineage
	lin_ass_ssms_w_ssms = model.get_lineage_assignment_for_ssms_w_ssms(my_lineages)
	# get the pure lineage assignment for the SSMs that belong to the subset of SSMs
	lin_ass_ssms = model.get_lineage_assignment_for_subset_of_ssms(lin_ass_ssms_w_ssms, ssm_subset)
	end_t = time.time()
	logging.debug("time to get assignment: {0}".format(end_t - start_t))

	# get true lineage assignment
	true_lin_ass_ssms = read_true_lineage_assignment(lineage_assignment_file)

	if len(true_lin_ass_ssms) != len(lin_ass_ssms):
		raise eo.MyException("Number of true lineage assignments differ from number"
			" of onctopus lineage assignments.")
	
	logging.debug("build co-clustering matrices")
	start_t = time.time()
	# build co-clustering matrices and compute AUPRC
	inferred_co_clustering_matrix = compute_co_clustering_matrix(np.asarray(lin_ass_ssms))
	true_co_clustering_matrix = compute_co_clustering_matrix(np.asarray(true_lin_ass_ssms))
	end_t = time.time()
	logging.debug("time to build co-clustering matrices {0}".format(end_t - start_t))

	logging.debug("compute AUPRC")
	auprc = co_clustering_matrices_to_avg_precision_score(true_co_clustering_matrix, inferred_co_clustering_matrix)

	# write AUPRC to file
	write_AUPRC_to_file(auprc, auprc_file, test)

	return (auprc, inferred_co_clustering_matrix, true_co_clustering_matrix)

def write_AUPRC_to_file(auprc, auprc_file, test):
	if not test:
		oio.raise_if_file_exists(auprc_file)
	
	with open(auprc_file, 'w') as f:
		f.write("{0}".format(auprc))

def read_true_lineage_assignment(lineage_assignment_file):
	true_lineage_assignment = []
	with open(lineage_assignment_file) as f:
		for line in f:
			lineage_assignment = line.rstrip("\n")
			# if lineage assignment doesn't consist of numbers,
			# characters are used
			try:
				lineage_assignment = int(lineage_assignment)
			except ValueError:
				pass
			true_lineage_assignment.append(lineage_assignment)
	return true_lineage_assignment

# compares two co clustering matrives and returns their average precision score
# true_matrix, inferred_matrix: 2D arrays of the same size
def co_clustering_matrices_to_avg_precision_score(true_matrix, inferred_matrix):
	
	# remove diagonal because this is always one
	ssm_num = true_matrix.shape[0]
	diagonal_indices = np.arange(0, ssm_num*ssm_num, ssm_num+1)
	true_vector = np.delete(true_matrix, diagonal_indices, None)
	inferred_vector = np.delete(inferred_matrix, diagonal_indices, None)

	# special case, where no SSMs cluster together
	no_clustering = np.zeros(ssm_num*ssm_num - ssm_num)
	if np.array_equal(no_clustering, true_vector) and np.array_equal(no_clustering, inferred_vector):
		return 1.0

	# sort vector with inferred values
	sorted_indices = np.argsort(inferred_vector)

	# rearrange elements in vectors according to sorted order in inferred_vector
	true_vector = true_vector[sorted_indices]
	inferred_vector = inferred_vector[sorted_indices]

	# compute average precision score, equivalent to value of area under precision recall curce
	return average_precision_score(true_vector, inferred_vector)

# given a matix, function creates a 1D vector of it
def matrix_to_vector(matrix):
	my_size = matrix.size
	vector = np.asarray(matrix)
	vector = vector.reshape(my_size)
	return vector

# compare two co-clustering matrices and returns precision and recall
# TODO test
def compare_co_clustering_matrices(true_marix, inferred_matrix):
	true_positives = true_negatives = false_positives = false_negatives = 0
	ssm_num = true_marix.shape[0]

	for i in xrange(ssm_num):
		for j in xrange(ssm_num):
			# values on diagonal are not counted as they are equal by construction
			if i != j:
				if true_marix[i][j] == 1:
					if inferred_matrix[i][j] == 1:
						true_positives += 1
					else:
						false_negatives += 1
				else:
					if inferred_matrix[i][j] == 1:
						false_positives += 1
					else:
						true_negatives += 1

	precision = get_precision(true_positives, false_positives)
	recall = get_recall(true_positives, false_negatives)
	return (precision, recall)

def get_precision(true_positives, false_positives):
	return float(true_positives)/(true_positives + false_positives)

def get_recall(true_positives, false_negatives):
	return float(true_positives)/(true_positives + false_negatives)

# creates co-clustering matrix
# is boolean matrix where entry[i][j] is True if SSM i and SSM j belong to the same lineage
# lineage_assignment: array with lineage assignments for the SSMs, order is sorted according
#	to corresponding SSMs
def compute_co_clustering_matrix(lineage_assignment):
	idx1, idx2 = sp.where(lineage_assignment[:, sp.newaxis] == lineage_assignment)
	matrix = spsp.coo_matrix((sp.ones(idx1.shape[0]), (idx1, idx2)), 
		shape=(lineage_assignment.shape[0], lineage_assignment.shape[0])).toarray()
	return matrix

# gets the ADR matrix
def get_adr_matrix(my_lins, my_list=False, z_matrix_list_or_not=None):
	if my_list == False:
		lin_ass = model.get_lineage_assignment_for_ssms_w_ssms(my_lins)
		adr_matrix = compute_adr_matrix(lin_ass, my_lins, z_matrix_list_or_not)
		return adr_matrix
	else:
		# SSM-lineage assignment equal for all reconstructions
		lin_ass = model.get_lineage_assignment_for_ssms_w_ssms(my_lins[0])
		# if necessary, prepare Z-matrix
		if z_matrix_list_or_not is None:
			z_matrix_list_or_not = [None] * len(my_lins)
		# create empty ADR matrix
		adr_tmp = np.zeros((len(lin_ass), len(lin_ass)))
		# for each reconstruction
		for i in xrange(len(my_lins)):
			# add ADR to previous matrixes
			adr_tmp += compute_adr_matrix(lin_ass, my_lins[i], z_matrix_list_or_not[i])
		# normalize combined ADR matrix
		return (adr_tmp / float(len(my_lins))).round(10)

# computes the ancestor-descendant matrix
# entry [i][j] is 1 when SSM i belongs to lineage that is ancestral to lineage of SSM j
# entry [i][j] is 0.5 when Z-matrix is given and lin k and k' are in ambiguous relationship
def compute_adr_matrix(lineage_assignment, my_lins, z_matrix=None):

	mut_num = len(lineage_assignment)

	if z_matrix is None:
		sublin_matrix = create_sublin_matrix(my_lins)
	else:
		sublin_matrix = z_matrix

	# create matrix with 0
	matrix = np.zeros((mut_num, mut_num))

	# iterate through all fields
	for i in xrange(mut_num):
		for j in xrange(mut_num):
			# if mutation j belongs to lineage that is descendant to lineage of mutation i
			if sublin_matrix[lineage_assignment[i][1]][lineage_assignment[j][1]] == 1:
				matrix[i][j] = 1.0
			elif (sublin_matrix[lineage_assignment[i][1]][lineage_assignment[j][1]] == 0 
				and z_matrix is not None):
				matrix[i][j] = 0.5

	return matrix

# creates a matrix where i,j is 1, when lineage i is ancestor of j, 0 otherwise
def create_sublin_matrix(my_lins):

	lin_num = len(my_lins)

	matrix = np.zeros((lin_num,lin_num))

	for i in xrange(lin_num):
		for j in xrange(lin_num):
			if j in my_lins[i].sublins:
				matrix[i][j] = 1

	return matrix


# construct an assignment matrix between two sets of lineages
def construct_assignment_matrix(true_lineages, rec_lineages):
	# get the number of lineages in each set and the maximum number of lineages
	# because the set with fewer lineages will get dummy lineages without mutations
	num_true_line = len(true_lineages)
	num_rec_line = len(rec_lineages)
	max_num_line = max(num_rec_line, num_true_line)

	# init nxn-matrix with all entries 0
	matrix = [[0]*max_num_line for i in xrange(max_num_line)]

	# fill matrix
	# get differences between lineages as cost for each entry
	for row in range(max_num_line):
		for column in range(max_num_line):
			# dummy true_lineage
			if row >= num_true_line:
				matrix[row][column] = (model.compute_diff_between_lineages(None, 
					rec_lineages[column]))
			# dummy rec_lineage
			elif column >= num_rec_line:
				matrix[row][column] = (model.compute_diff_between_lineages(true_lineages[row], 
					None))
			# two lineages
			else:
				matrix[row][column] = (model.compute_diff_between_lineages(true_lineages[row], 
					rec_lineages[column]))

	return matrix

if __name__ == "__main__":
	#import sys
	#(prog, true, rec) = sys.argv
	#compute_min_bipartite_match(true, rec)

	parser = argparse.ArgumentParser()
	parser.add_argument("--compute_auprc", default = None, help = "TODO")
	parser.add_argument("--result_file", default = None, help = "TODO")
	parser.add_argument("--seg_file", default = None, help = "TODO")
	parser.add_argument("--ssm_file", default = None, help = "TODO")
	#parser.add_argument("--super_ssms_file", default = None, help = "TODO")
	#parser.add_argument("--cluster_assignments_file", default = None, help = "TODO")
	parser.add_argument("--normal_seg_indices_file", default = None, help = "TODO")
	parser.add_argument("--ssm_subset_file", default = None, help = "TODO")
	parser.add_argument("--lineage_assignment_file", default = None, help = "TODO")
	parser.add_argument("--auprc_file", default = None, help = "TODO")
	parser.add_argument("--test", default = None, help = "TODO")
	parser.add_argument("--write_to_file", default = None, help = "TODO")

	args = parser.parse_args()

	normal_seg_indices = []
	if args.normal_seg_indices_file is not None:
		with open(args.normal_seg_indices_file) as f:
			for line in f:
				normal_seg_indices = map(int, line.rstrip("\n").split("\t"))
	
	test = False
	if args.test:
		test = oio.str_to_bool(args.test)
	
	compute_auprc = False
	if args.compute_auprc:
		compute_auprc = oio.str_to_bool(args.compute_auprc)

	write_to_file = False
	if args.write_to_file:
		write_to_file = oio.str_to_bool(args.write_to_file)
	
	if compute_auprc:
		#numeric_logging_info = getattr(logging, "DEBUG".upper(), None)
		#logging.basicConfig(filename=args.auprc_file+".log", filemode='w', level=numeric_logging_info)
		compute_auprc_of_inferred_lineages_and_true_lineage_assignment(args.result_file, args.seg_file, 
			args.ssm_file, normal_seg_indices, 
			args.ssm_subset_file, args.lineage_assignment_file, args.auprc_file, test,
			write_to_file=write_to_file)

