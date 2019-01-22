#!/usr/bin/env python

from sys import stdout, stderr, exit
from os.path import isdir, join, basename, isfile
from os import remove
from itertools import chain
from argparse import ArgumentParser

from evaluation import construct_assignment_matrix
from evaluation import get_adr_matrix
from evaluation import get_co_clustering_matrix_from_lineages
from onctopus_io import read_result_file
from model import change_labeling_of_lineage
from model import get_CN_changes_SSM_apperance
from random import random
import subprocess

import exceptions_onctopus as eo
import numpy as np
from copy import deepcopy

def getLineageMapping(munk_inst, true_lin, test_lin):
    """ computes matching between true and inferred lineages"""

    mtrx = construct_assignment_matrix(true_lin, test_lin)
    
    return munk_inst.compute(mtrx) 

#
def get_CN_change_differences(true_lin, inf_lin, seg_num):
	# get CN changes of true reconstruction
	gain_num_true = []
	loss_num_true = []
	tmp_CNVs = []
	tmp_present_ssms = []
	tmp_ssm_infl_cnv_same_lineage = []
	get_CN_changes_SSM_apperance(seg_num, gain_num_true, loss_num_true, tmp_CNVs, tmp_present_ssms,
		len(true_lin), true_lin, tmp_ssm_infl_cnv_same_lineage, evaluation_param=True)

	# get CN changes of inferred reconstruction
	gain_num_inf = []
	loss_num_inf = []
	tmp_cnvs = []
	tmp_present_ssms = []
	tmp_ssm_infl_cnv_same_lineage = []
	get_CN_changes_SSM_apperance(seg_num, gain_num_inf, loss_num_inf, tmp_CNVs, tmp_present_ssms,
		len(inf_lin), inf_lin, tmp_ssm_infl_cnv_same_lineage, evaluation_param=True)

	# compute CNC differences
	gain_overestimation, gain_underestimation = compute_cnc_differences(gain_num_true, gain_num_inf, seg_num)
	loss_overestimation, loss_underestimation = compute_cnc_differences(loss_num_true, loss_num_inf, seg_num)

	return gain_overestimation, gain_underestimation, loss_overestimation, loss_underestimation


# compares true and inferred CNCs 
def compute_cnc_differences(change_true, change_inf, seg_num):
	overestimation = 0.0
	underestimation = 0.0

	# check all segments
	for i in xrange(seg_num):
		difference = change_inf[i] - change_true[i]
		if difference > 0:
			overestimation += difference
		elif difference < 0:
			underestimation += difference

	# return average numbers
	return overestimation/seg_num, underestimation/seg_num

# computes the 1C score from the SMC-Het Scoring system
def getSMCHet1C(true_lin, test_lin, SMC_path):
	# get primary lists with information
	true_list = get1CLineageList(true_lin)
	test_list = get1CLineageList(test_lin)

	# get total number of mutations
	tot_mut_num = sum([true_list[i][1] for i in xrange(len(true_list))])

	# get random name
	random_name = random()
	# name of output file
	output_name = "{0}_output.txt".format(random_name)

	# write pseudo VCF file
	vcf_name = write_pseudo_SSM_file(tot_mut_num, random_name)

	# write predicted and true mutations files
	write_pseudo_SSM_file(tot_mut_num, random_name)
	pred_line = "\n".join(["\t".join(map(str, test_list[i])) for i in xrange(len(test_list))])
	true_line = "\n".join(["\t".join(map(str, true_list[i])) for i in xrange(len(true_list))])
	pred_name = "{0}_inferred.txt".format(random_name)
	true_name = "{0}_true.txt".format(random_name)
	with open(pred_name, "w") as f:
		f.write(pred_line)
	with open(true_name, "w") as f:
		f.write(true_line)
	
	# call the SMC-Het 1C score
	command = ("python {0}/SMCScoring_LINDA.py -c 1C --predfiles {1} --truthfiles {2} --vcf {3} -o {4}"
		">/dev/null".format(
		SMC_path, pred_name, true_name, vcf_name, output_name))
	subprocess.call(command, shell=True)

	# get 1C score
	with open(output_name, "r") as f:
		for line in f:
			score = float(line.rstrip("\n"))

	# delete used files
	remove(output_name)
	remove(pred_name)
	remove(true_name)
	remove(vcf_name)

	return score

def write_pseudo_SSM_file(tot_mut_num, random_name):
	header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttumour"
	lines = ["chr\t1\t.\tA\tA\t.\tPASS\tSOMATIC;VT=SNP\tGT:AD\t0:28,22\tTrue"] * tot_mut_num
	lines = "\n".join(lines)
	vcf = "{0}\n{1}".format(header, lines)
	vcf_name = "{0}.vcf".format(random_name)
	with open(vcf_name, "w") as f:
		f.write(vcf)

	return vcf_name

# computes the 2B score from the SMC-Het Scoring system
# soft mutations cluster assignment
def getSMCHet2B(true_lin, test_lin, SMC_path, tot_mut_num, co_clustering_true=None, co_clustering_test=None,
	keep_2b_matrices=False, random_name=None):
	# compute co-clustering matrix
	if co_clustering_true is None:
		co_clustering_true = get_co_clustering_matrix_from_lineages(true_lin)
	if co_clustering_test is None:
		co_clustering_test = get_co_clustering_matrix_from_lineages(test_lin)

	# get random name
	if random_name is None:
		random_name = random()
	# name of output file
	output_name = "{0}_output.txt".format(random_name)

	# write pseudo VCF file
	vcf_name = write_pseudo_SSM_file(tot_mut_num, random_name)

	# write predicted and true co-clustering file
	pred_name = "{0}_inferred.txt".format(random_name)
	np.savetxt(pred_name, co_clustering_test, delimiter="\t")
	true_name = "{0}_true.txt".format(random_name)
	np.savetxt(true_name, co_clustering_true, delimiter="\t")
	
	# call the SMC-Het 2B score
	command = ("python {0}/SMCScoring_LINDA.py -c 2B --predfiles {1} --truthfiles {2} --vcf {3} -o {4}"
		">/dev/null".format(
		SMC_path, pred_name, true_name, vcf_name, output_name))
	subprocess.call(command, shell=True)

	# get 2B score
	with open(output_name, "r") as f:
		for line in f:
			score = float(line.rstrip("\n"))

	# delete used files
	remove(output_name)
	remove(vcf_name)
	if keep_2b_matrices == False:
		remove(pred_name)
		remove(true_name)

	return score, co_clustering_true, co_clustering_test

# computes the 3B score from the SMC-Het Scoring system
# soft mutations assignment, lineage relationships
# true_lins and test_lins are lists with reconstructions
def getSMCHet3B(true_lins, test_lins, SMC_path, tot_mut_num, co_clustering_true=None, co_clustering_test=None,
	random_name=None, only_write_input=False, only_compute_SMCHet=False, adr_test=None,
	true_zmatrices=None, inferred_zmatrices=None, co_clustering_test_name=None,
	adr_test_name=None):

	if only_compute_SMCHet == False:
		# compute co-clustering matrix
		if co_clustering_true is None:
			co_clustering_true = get_co_clustering_matrix_from_lineages(true_lins[0])
		if co_clustering_test is None and co_clustering_test_name is None:
			co_clustering_test = get_co_clustering_matrix_from_lineages(test_lins[0])
		# compute ADR matrix
		adr_true = get_adr_matrix(true_lins, my_list=True, z_matrix_list_or_not=true_zmatrices)
		if adr_test is None and adr_test_name is None:
			adr_test = get_adr_matrix(test_lins, my_list=True, z_matrix_list_or_not=inferred_zmatrices)

		# get random name
		if random_name is None:
			random_name = random()
		# name of output file
		output_name = "{0}_output.txt".format(random_name)

		# write pseudo VCF file
		vcf_name = write_pseudo_SSM_file(tot_mut_num, random_name)

		# write predicted and true co-clustering file
		if co_clustering_test_name is None:
			pred_ccm_name = "{0}_ccm_inferred.txt".format(random_name)
			np.savetxt(pred_ccm_name, co_clustering_test, delimiter="\t")
		true_ccm_name = "{0}_ccm_true.txt".format(random_name)
		np.savetxt(true_ccm_name, co_clustering_true, delimiter="\t")
		# write predicted and true co-clustering file
		if adr_test_name is None:
			pred_adr_name = "{0}_adr_inferred.txt".format(random_name)
			np.savetxt(pred_adr_name, adr_test, delimiter="\t")
		true_adr_name = "{0}_adr_true.txt".format(random_name)
		np.savetxt(true_adr_name, adr_true, delimiter="\t")
	else:
		pred_ccm_name = "{0}_ccm_inferred.txt".format(random_name)
		pred_adr_name = "{0}_adr_inferred.txt".format(random_name)
		true_ccm_name = "{0}_ccm_true.txt".format(random_name)
		true_adr_name = "{0}_adr_true.txt".format(random_name)
		vcf_name = "{0}.vcf".format(random_name)
		output_name = "{0}_output.txt".format(random_name)

	if only_write_input:
		return
	
	# call the SMC-Het 3B score
	command = ("python {0}/SMCScoring_LINDA.py -c 3B --predfiles {1} {5} --truthfiles {2} {6} --vcf {3} -o {4}"
		">/dev/null".format(
		SMC_path, pred_ccm_name, true_ccm_name, vcf_name, output_name, pred_adr_name, true_adr_name))
	subprocess.call(command, shell=True)

	# get 3B score
	with open(output_name, "r") as f:
		for line in f:
			score = float(line.rstrip("\n"))

	# delete used files
	remove(output_name)
	remove(pred_ccm_name)
	remove(true_ccm_name)
	remove(pred_adr_name)
	remove(true_adr_name)
	remove(vcf_name)

	return score

# creates values needed for 1C Score: lineage identifier, number of mutations, lineage frequency
def get1CLineageList(my_lin):
	my_list = []
	i = 0
	for single_lin in my_lin:
		# don't use normal lineage
		if i > 0:
			# number of mutations
			mut_num = len(single_lin.ssms) + len(single_lin.ssms_a) + len(single_lin.ssms_b)
			# if a lineage has no mutation, it isn't added to the list of lineages
			if mut_num == 0:
				continue
			my_list.append([i, mut_num, single_lin.freq])
		i += 1
	return my_list

def getAverageCopyNumberMSE(true_average_cn, inferred_average_cn):
	""" compute mean squared error between true and inferred average copy numbers of all segments"""
	res = 0.0
	for seg_index in xrange(len(true_average_cn)):
		res += (true_average_cn[seg_index] - inferred_average_cn[seg_index])**2
	return (res/len(true_average_cn))

def getAverageCopyNumberMAE(true_average_cn, inferred_average_cn):
	""" compute mean average error between true and inferred average copy numbers of all segments"""
	res = 0.0
	for seg_index in xrange(len(true_average_cn)):
		res += abs(true_average_cn[seg_index] - inferred_average_cn[seg_index])
	return (res/len(true_average_cn))

def getLineageFrequencyMSE(true_lin, test_lin):
    """ compute mean squared error between true and test lineage frequencies """
    res = 0.0
    for x in xrange(len(true_lin)):
        res += (true_lin[x].freq-test_lin[x].freq)**2
    return (res/len(true_lin))

def getLineageFrequencyMAE(true_lin, test_lin):
    """ compute mean average error between true and test lineage frequencies """
    res = 0.0
    for x in xrange(len(true_lin)):
        res += abs(true_lin[x].freq-test_lin[x].freq)
    return (res/len(true_lin))

def constructLineageMtrx(lineage):
    res = [[0] * len(lineage) for _ in xrange(len(lineage))]

    for x in xrange(len(lineage)):
        for y in lineage[x].sublins:
            res[x][y] = 1
    return res
    
def getLineageReconstructionStats(true_lin_mtrx, test_lin_mtrx):
    """ get number of matching, compatible, and false lineage filiations """
  
    # matching, compatible, false heredities
    m = c = f = 0
    # start with 2nd row and line, because 1st is root and always the same, hence should
    # be ignored
    for x in xrange(1, len(true_lin_mtrx)):
        for y in xrange(1, len(true_lin_mtrx)):
	    if true_lin_mtrx[x][y] == test_lin_mtrx[x][y]:
                m += 1 
            elif true_lin_mtrx[x][y] - test_lin_mtrx[x][y] > 0:
                c += 1
            else:
                f += 1
    return m, c, f


def getSSMStats(true_lin, test_lin):
    """ compute TP, FP and FN between true and test lineage """
    TP = FP = FN = 0
    
    segment_ssms = set(map(lambda x: x.pos, chain(*map(lambda z: z.ssms +
        z.ssms_a + z.ssms_b, test_lin))))
    for x in xrange(len(true_lin)):
        # we collect all SSMs on either allele positions of ssms in true lineage
        true_ssms = segment_ssms.intersection(map(lambda z: z.pos,
            true_lin[x].ssms + true_lin[x].ssms_a + true_lin[x].ssms_b))
        # positions of ssms in test lineage
        test_ssms = set(map(lambda z: z.pos, test_lin[x].ssms +
            test_lin[x].ssms_a + test_lin[x].ssms_b))
        # true positives are those ssms that are in both lists
        TP += len(true_ssms.intersection(test_ssms))
        # false positives are those ssms that are only in the false lineage
        FP += len(test_ssms.difference(true_ssms))
        # false negatives are those ssms that are only in the true lineage 
        FN += len(true_ssms.difference(test_ssms))

    return TP, FP, FN


def print_table(TPs, FPs, FNs, FREQ_MSE, FREQ_MAE, CORRECT_LIN, COMPAT_LIN, FALSE_LIN,
        out, omit_header=False, output_file=False, SMCHET1C=[]):
    """ print out statistics """

    if output_file:
    	if isfile(output_file):
	    omit_header = True
	    f = open(output_file, "a")
	else:
	    omit_header = False
	    f = open(output_file, "w")

    if not omit_header:
    	header = '\t'.join(('SSMs_TP', 'SSMs_FP', 'SSMs_FN',
            'CORRECT_DESCENDANTS', 'COMPATIBLE_DESCENDANTS',
            'WRONG_DESCENDANTS', 'LIN_FREQUENCY_MSE', 'LIN_FREQUENCY_MAE'))
	if len(SMCHET1C) > 0:
		header = "{0}\t{1}".format(header, 'SMC-HET_1C')
	if output_file:
	    f.write("{0}\n".format(header))
	else:
	    print >> out, header

    line = '\t'.join(map(lambda x: str(sum(x)), (TPs, FPs, FNs,
        CORRECT_LIN, COMPAT_LIN, FALSE_LIN, FREQ_MSE, FREQ_MAE)))
    if len(SMCHET1C) > 0:
    	line = '{0}\t{1}'.format(line, str(sum(SMCHET1C)))
    if output_file:
    	f.write("{0}\n".format(line))
	f.close()
    else:
        print >> out, line

# given the true and an inferred Z-Matrix, the confusion matrix of the three relationships
# "presence" (1), "absence" (-1) and "ambiguity" (0) is build
def build_lineage_relationship_confusion_matrix(true_matrix, inferred_matrix):

	if len(true_matrix) != len(inferred_matrix):
		raise eo.MyException("Both matrices have to have the same size.")

	# rows: inferred, columns: ground truth
	# 0: presence, 1 : absence, 2: ambiguity
	confusion_matrix = np.zeros(9).reshape(3,3)

	# iterate through fields above the diagonal in Z-matrix
	for k in xrange(1, len(true_matrix)):
		for k_prime in xrange(k+1, len(true_matrix)):
			# true precense
			if true_matrix[k][k_prime] == 1:
				# inferred precense
				if inferred_matrix[k][k_prime] == 1:
					confusion_matrix[0][0] += 1
				# inferred absence
				elif inferred_matrix[k][k_prime] == -1:
					confusion_matrix[1][0] += 1
				# inferred ambiguity
				elif inferred_matrix[k][k_prime] == 0:
					confusion_matrix[2][0] += 1
			# true absence
			elif true_matrix[k][k_prime] == -1:
				# inferred precense
				if inferred_matrix[k][k_prime] == 1:
					confusion_matrix[0][1] += 1
				# inferred absence
				elif inferred_matrix[k][k_prime] == -1:
					confusion_matrix[1][1] += 1
				# inferred ambiguity
				elif inferred_matrix[k][k_prime] == 0:
					confusion_matrix[2][1] += 1
			# true ambiguity
			elif true_matrix[k][k_prime] == 0:
				# inferred precense
				if inferred_matrix[k][k_prime] == 1:
					confusion_matrix[0][2] += 1
				# inferred absence
				elif inferred_matrix[k][k_prime] == -1:
					confusion_matrix[1][2] += 1
				# inferred ambiguity
				elif inferred_matrix[k][k_prime] == 0:
					confusion_matrix[2][2] += 1

	return confusion_matrix
			
# given a list of Z-matrices without ambiguous entries, a new Z-matrix is created that contains ambiguities
# two cutoffs define how often a relationship has to be present to be classified to which relationship
def create_matrix_with_ambiguities(z_matrix_list, cutoff_absence, cutoff_ambiguity):

	matrix_num = len(z_matrix_list)
	lin_num = len(z_matrix_list[0])

	new_matrix = np.zeros(lin_num * lin_num).reshape(lin_num, lin_num)

	# Z-matrices need to be of type numpy, if not, they are converted to it
	if type(z_matrix_list[0]).__module__ != "numpy":
		z_matrix_list = [np.array(matrix) for matrix in z_matrix_list]

	# combine entries of all matrices
	for matrix in z_matrix_list:
		# filter entries in matrix, only keep the ones that are 1
		not_1 = matrix != 1
		tmp_matrix = deepcopy(matrix)
		tmp_matrix[not_1] = 0
		# add "1"s of current matrix to new matrix
		new_matrix += tmp_matrix
	# normalize entries
	new_matrix = new_matrix / float(matrix_num)

	# convert entries to 1, -1 and 0
	for k in xrange(lin_num):
		for k_prime in xrange(lin_num):
			if new_matrix[k][k_prime] <= cutoff_absence:
				new_matrix[k][k_prime] = -1
			elif new_matrix[k][k_prime] > cutoff_absence and new_matrix[k][k_prime] <= cutoff_ambiguity:
				new_matrix[k][k_prime] = 0
			elif new_matrix[k][k_prime] > cutoff_ambiguity:
				new_matrix[k][k_prime] = 1

	return new_matrix
	
    
#def main(omit_header, no_matching, true_lineage_file, inferred_lineage_file, output_file,
#	SMC_path=None):
#    # one instance to 
#    munkres = Munkres()
#
#    true_lin = read_result_file(true_lineage_file)
#    TPs, TNs, FPs, FNs = [list() for _ in xrange(4)]
#    FREQ_MSE = list()
#    FREQ_MAE = list()
#    CORRECT_LIN, COMPAT_LIN, FALSE_LIN  = [list() for _ in xrange(3)]
#    SMCHET1C = list()
#    for f in inferred_lineage_file:
#        test_lin = read_result_file(f)
#	current_true_lin = true_lin
#	# if no matching should be computed, the reconstructions must have the same number of 
#	# lineages
#        if no_matching:
#	    if len(test_lin) != len(true_lin):
#	    	raise Exception("Matching must be computed because reconstructions have different number"
#		    " of lineages.")
#        else:
#            match = getLineageMapping(munkres, true_lin, test_lin)
#	    (test_lin, current_true_lin) = change_labeling_of_lineage(match, test_lin, true_lin)
#        TP, FP, FN = getSSMStats(current_true_lin, test_lin)
#        TPs.append(TP)
#        FPs.append(FP)
#        FNs.append(FN)
#        FREQ_MSE.append(getLineageFrequencyMSE(current_true_lin, test_lin))
#        FREQ_MAE.append(getLineageFrequencyMAE(current_true_lin, test_lin))
#        m, c, f = getLineageReconstructionStats(constructLineageMtrx(current_true_lin),
#                constructLineageMtrx(test_lin))
#        CORRECT_LIN.append(m)
#        COMPAT_LIN.append(c)
#        FALSE_LIN.append(f)
#	if SMC_path:
#		# doesn't need the remodeled lineages after matching, original ones
#		# are fine because comparison is only done via lineage frequencies
#		SMCHET1C.append(getSMCHet1C(true_lin, test_lin, SMC_path))
#
#    print_table(TPs, FPs, FNs, FREQ_MSE, FREQ_MAE, CORRECT_LIN, COMPAT_LIN,
#            FALSE_LIN, stdout, omit_header, output_file, SMCHET1C=SMCHET1C)
#
#if __name__ == '__main__':
#    parser = ArgumentParser()
#    parser.add_argument('-o', '--omit_header', action='store_true', 
#            help='omit printing of the table header')
#    parser.add_argument('-n', '--no_matching', action='store_true', 
#            help='do not perform a matching between lineage SSMs, but ' + \
#                    'compare lineages only on the basis of their frequencies')
#    parser.add_argument('-f', '--output_file', type=str, 
#    	    help='file in which output is written')
#    parser.add_argument('-s', '--SMC_path', type=str, 
#    	    help='path to SMC-Het scoring script')
#    parser.add_argument('true_lineage_file', type=str, 
#            help='file containing information about the true lineage')
#    parser.add_argument('inferred_lineage_file', type=str, nargs='+',
#            help='file containing information about the inferred lineage')
#    args = parser.parse_args()
#    main(args.omit_header, args.no_matching, args.true_lineage_file, args.inferred_lineage_file,
#    	args.output_file, args.SMC_path)


