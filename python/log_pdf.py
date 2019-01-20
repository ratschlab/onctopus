from scipy.misc import comb
from scipy.special import betaln
from numpy import log
from numpy import pi
from numpy import sqrt
from numpy import ceil
from scipy.special import gammaln
from scipy.interpolate import UnivariateSpline
import numpy as np
from numpy.random import poisson
import matplotlib as mpl
mpl.use('Agg')
import exceptions_onctopus as eo
import logging
from numpy import amax
from numpy import argmax
from numpy import array
import constants as cons
from copy import deepcopy
import model

#TODO if lineages are read from file and the result file is not from Onctopus but parsed from other tool
#	where I don't know about the phasing, the phase of the SSMs shouldn't be set while reading the file
# use_given_CN: whether the in the segment file given CN should be used for the computation of the VAF
#	mathematically, this isn't correct but it's done like this in the optimization of Onctopus
def compute_llh(lineages, ssm_list, seg_list, overdispersion_parameter, phasing_not_known=False, 
	use_given_CN=False, compute_objective_value=False):
	# build Delta C matrices
	delta_C_A, delta_C_B = build_delta_C_from_reconstructions(lineages, len(seg_list), phasing_not_known)
	# build phi
	phi = [lineages[k].freq for k in xrange(len(lineages))]
	# compute all allele-specific average CNs
	as_avg_cn_inferred_A = [get_allele_specific_average_cn(delta_C_A, phi, i) for i in xrange(len(seg_list))]
	as_avg_cn_inferred_B = [get_allele_specific_average_cn(delta_C_B, phi, i) for i in xrange(len(seg_list))]

	# if reconstructions are not from Onctopus but PhyloWGS or Canopy, I don't have phasing information about the SSMs
	# so I just assign them all to A here
	if phasing_not_known:
		for my_ssm in ssm_list:
			if my_ssm.phase is None:
				my_ssm.phase = cons.A
			else:
				raise eo.MyException("Do I really know the phase here or did I just parse in result"
					" file wrong?")
	# get Z-matrix
	z_matrix = model.get_Z_matrix(lineages)[0]
	# compute VAF
	vaf = compute_vaf(ssm_list, lineages, delta_C_A, delta_C_B, z_matrix, as_avg_cn_inferred_A, as_avg_cn_inferred_B, 
		seg_list, use_given_CN)

	# compute LLH
	if compute_objective_value == False:
		llh_seg_a = [log_normal_complete_llh(as_avg_cn_inferred_A[i], seg_list[i].given_cn_A, 
			seg_list[i].standard_error_A)
			for i in xrange(len(seg_list))]
		llh_seg_b = [log_normal_complete_llh(as_avg_cn_inferred_B[i], seg_list[i].given_cn_B, 
			seg_list[i].standard_error_B)
			for i in xrange(len(seg_list))]
		llh_ssms = compute_llh_ssms(vaf, ssm_list, overdispersion_parameter)
	# compute objective value
	else:
		number_spline_points = 50
		(seg_splines_A, seg_splines_B, ssm_splines) = model.create_segment_and_mutation_splines(
			seg_list, None, ssm_list, number_spline_points, True)
		llh_seg_a = [seg_splines_A[i](as_avg_cn_inferred_A[i]) for i in xrange(len(seg_list))]
		llh_seg_b = [seg_splines_B[i](as_avg_cn_inferred_B[i]) for i in xrange(len(seg_list))]
		llh_ssms = [ssm_splines[i](vaf[i]) for i in xrange(len(ssm_list))]

	# if reconstructions are not from Onctopus but PhyloWGS or Canopy, the phasing information is not known
	# and also the CN influence in same lineage is not known
	# so all possible options are checked
	# already done for phasing to A and no influence of CN changes in same lineage
	if phasing_not_known:
		llh_ssms_a_no_infl = deepcopy(llh_ssms)
		llh_ssms_b_no_infl = change_SSM_info_and_get_llh(ssm_list, lineages, delta_C_A, delta_C_B, z_matrix, 
			as_avg_cn_inferred_A, as_avg_cn_inferred_B, overdispersion_parameter, phase=cons.B,
			infl_cnv_same_lin=False)
		llh_ssms_a_with_infl = change_SSM_info_and_get_llh(ssm_list, lineages, delta_C_A, delta_C_B, z_matrix, 
			as_avg_cn_inferred_A, as_avg_cn_inferred_B, overdispersion_parameter, phase=cons.A,
			infl_cnv_same_lin=True)
		llh_ssms_b_with_infl = change_SSM_info_and_get_llh(ssm_list, lineages, delta_C_A, delta_C_B, z_matrix, 
			as_avg_cn_inferred_A, as_avg_cn_inferred_B, overdispersion_parameter, phase=cons.B,
			infl_cnv_same_lin=True)

		# get best LLH for each SSM
		llh_ssms = [max([llh_ssms_a_no_infl[i], llh_ssms_b_no_infl[i], llh_ssms_a_with_infl[i],
			llh_ssms_b_with_infl[i]]) for i in xrange(len(ssm_list))]

	# return all likelihoods
	return sum(llh_seg_a) + sum(llh_seg_b) + sum(llh_ssms)


# changes the phase and the influence of CN gains in the same lineage for SSMs
# ssm_list: list with all SSMs
# lineages: the list with all lineages
# delta_C_A/delta_C_B: matrices with CN changes
# z_matrix: the Z-matrix
# as_avg_cn_inferred_A/as_avg_cn_inferred_B: allele-specific average CNs inferred by the method
# overdispersion_parameter: parameter to compute the beta binomial distribution
# phase: phase to which the SSMs should be changed
# infl_cnv_same_lin: whether SSMs should be influenced by CN gains in the same lineage
def change_SSM_info_and_get_llh(ssm_list, lineages, delta_CA, delta_CB, z_matrix, as_avg_cn_inferred_A, as_avg_cn_inferred_B,
	overdispersion_parameter, phase, infl_cnv_same_lin):
	# set phase and influence of CN gains in same lineage accordingly
	for my_ssm in ssm_list:
		my_ssm.phase = phase
		my_ssm.infl_cnv_same_lin = infl_cnv_same_lin

	# compute VAF
	vaf = compute_vaf(ssm_list, lineages, delta_CA, delta_CB, z_matrix, as_avg_cn_inferred_A, as_avg_cn_inferred_B)
	# compute LLH of all SSMs and return it
	new_llh = compute_llh_ssms(vaf, ssm_list, overdispersion_parameter)
	return new_llh

# computes the LLH of all SSMs
def compute_llh_ssms(vaf, ssm_list, overdispersion_parameter):
	return [beta_binomial(p=vaf[i], V=ssm_list[i].variant_count, D=ssm_list[i].variant_count+ssm_list[i].ref_count,
		s=overdispersion_parameter, compute_llh=True) for i in xrange(len(ssm_list))]

# computes the VAF of each SSM
# ssm_list: list with all SSMs
# lineages: the list with all lineages
# delta_C_A/delta_C_B: matrices with CN changes
# z_matrix: the Z-matrix
# as_avg_cn_inferred_A/as_avg_cn_inferred_B: allele-specific average CNs inferred by the method
# seg_list: list with segments and their information, if this is given, the average SSM copy number is divided
#	by the given CN, not the inferred one (as done in the Onctopus optimization)
def compute_vaf(ssm_list, lineages, delta_C_A, delta_C_B, z_matrix, as_avg_cn_inferred_A, as_avg_cn_inferred_B,
	seg_list=None, use_given_CN=False):
	avg_ssm_cn =  [compute_average_SSM_cn(my_ssm, lineages, delta_C_A, delta_C_B, z_matrix)
		for my_ssm in ssm_list]
	if use_given_CN == False:
		return [float(avg_ssm_cn[i])/float(as_avg_cn_inferred_A[ssm_list[i].seg_index] +
			as_avg_cn_inferred_B[ssm_list[i].seg_index]) for i in xrange(len(ssm_list))]
	else:
		return [float(avg_ssm_cn[i])/float(seg_list[ssm_list[i].seg_index].given_cn_A + 
			seg_list[ssm_list[i].seg_index].given_cn_B) for i in xrange(len(ssm_list))]

# computes the matrices Delta C_A and Delta C_B given a reconstruction
# lineages: the list with all lineages
# seg_num: number of segments
# phasing_not_known: if the phasing of the SSMs is not known
def build_delta_C_from_reconstructions(lineages, seg_num, phasing_not_known):
	lin_num = len(lineages)

	# empty matrices are creates
	delta_C_A = np.zeros(seg_num*lin_num).reshape(seg_num, lin_num)
	delta_C_B = np.zeros(seg_num*lin_num).reshape(seg_num, lin_num)

	# each lineage is checked for CNVS
	for k, my_lin in enumerate(lineages):
		# CN changes are entered into Delta C matrices
		[help_build_delta_C_from_reconstructions_allele(delta_C_A, k, my_cnv, phasing_not_known=phasing_not_known)
			for my_cnv in my_lin.cnvs_a]
		[help_build_delta_C_from_reconstructions_allele(delta_C_B, k, my_cnv, phasing_not_known=phasing_not_known)
			for my_cnv in my_lin.cnvs_b]

	return delta_C_A, delta_C_B
		
# sets the entry in the current Delta C matrix
# delta_C: current Delta C matrix
# lin_index: index of current lineage
# my_cnv: current CNV
# phasing_not_known: if the phasing of the CNAs is not known
def help_build_delta_C_from_reconstructions_allele(delta_C, lin_index, my_cnv, phasing_not_known):
	# problem if the phasing of CNAs is not known:
	# CNVs of Canopy and PhyloWGS are automatically assigned to allele A
	# this gets problematic if several CN changes appear in one segment because then I don't know
	# whether they belong to the same or to the other allele
	if phasing_not_known and delta_C[my_cnv.seg_index][lin_index] != 0:
		raise eo.MyException("Phasing not known and multiple changes per segment, don't know what to do.")
	# in general there should never be two CNVs that belong to the same entry in the Delta C matrix
	if delta_C[my_cnv.seg_index][lin_index] != 0:
		raise eo.MyException("There shouldn't be more than one CNV for an entry in the Delta C matrix.")
	
	# entry in current Delta C matrix is set according to CN change
	delta_C[my_cnv.seg_index][lin_index] = delta_C[my_cnv.seg_index][lin_index] + my_cnv.change

# computes the allele-specific average CN, based on the matrices
# 	1 + \sum_k \Delta C_{A/B_{i,k}} * phi_k
# delta_C: matrix with CN changes
# phi: vector with lineage frequencies
# seg_index: index of the current segment
def get_allele_specific_average_cn(delta_C, phi, seg_index):
	return 1 + sum([delta_C[seg_index][k] * phi[k] for k in xrange(len(phi))])

# computes the average CN of an SSM
# ssm: the current SSM
# lineages: the list with all lineages
# delta_C_A/delta_C_B: CN changes on allele A/B
# z_matrix: the Z matrix
def compute_average_SSM_cn(my_ssm, lineages, delta_C_A, delta_C_B, z_matrix):
	# frequency of the lineage the SSM is assigned to
	lineage_frequency = lineages[my_ssm.lineage].freq

	# if SSM is influenced by CN gain in same lineage, frequency of lineage is multiplied with amount of
	# phased CN gain
	cn_infl_same_lin = 0
	if my_ssm.infl_cnv_same_lin:
		if my_ssm.phase == cons.A and delta_C_A[my_ssm.seg_index][my_ssm.lineage] > 0:
			cn_infl_same_lin = delta_C_A[my_ssm.seg_index][my_ssm.lineage] * lineages[my_ssm.lineage].freq
		if my_ssm.phase == cons.B and delta_C_B[my_ssm.seg_index][my_ssm.lineage] > 0:
			cn_infl_same_lin = delta_C_B[my_ssm.seg_index][my_ssm.lineage] * lineages[my_ssm.lineage].freq

	# influence of CN changes in other lineages are computed
	cnv_infl_other_lin = 0
	# CN change on right allele is determined
	allele_specific_C = None
	if my_ssm.phase == cons.A:
		allele_specific_C = delta_C_A
	elif my_ssm.phase == cons.B:
		allele_specific_C = delta_C_B
	# if SSM is phased, it can be influenced by a CN change of a lineage that is a descendant of the lineage
	# it is assigned to
	if allele_specific_C is not None:
		cnv_infl_other_lin = sum([allele_specific_C[my_ssm.seg_index][k_prime] * lineages[k_prime].freq
			for k_prime in xrange(my_ssm.lineage+1, len(z_matrix))
			if z_matrix[my_ssm.lineage][k_prime] == 1])
	
	return lineage_frequency + cn_infl_same_lin + cnv_infl_other_lin

# Function to compute if two float variables are close enough to be considered
# as equal
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

#Compute log n choose k using a difference of log gamma's
def log_n_choose_k(n,k):
	return gammaln(n+1) - gammaln(k+1) - gammaln(n -k + 1)

# V = variant read count
# D = total read count
# s = overdispersion parameter
# p = probability of drawing a variant read

def beta_binomial(p,V,D,s, compute_llh=False):
	a = p * s
	b = (1-p) * s
	if compute_llh:
		if a == 0:
			a = cons.MIN_ALPHA_BETA
		if b == 0:
			b = cons.MIN_ALPHA_BETA
	# in the following case the normal computation would lead to "nan" because
	# "infinity - infinity" gets calculated
	# when "-infinity" is returned a new value for the final computation of the
	# splines is done in the function "check_and_change_boundary_elements"
	if (b == 0) and (D - V == 0):
		return -float('Inf')
	return log_n_choose_k(D,V) + betaln(V+a, D-V+b) - betaln(a,b)

# D = observed reads
# s = overdispersion parameter
# l = expected number of reads
def neg_binomial(l,D,s):
	r = s
	p = l/(r+l)

	if p == 0.0:
		return -float('Inf')
	if p == 1.0:
		return -float('Inf')

	return log_n_choose_k(D+r-1,D) + r*log(1-p) + D*log(p)

# computes the second part of the pdf of the normal distribution
# only second part is needed because only this part depends on the cn
#	that is infered by Onctopus
# logarithm is already applied
# cn_given: externally given cn, equivalent to the mean
def log_normal(cn_onctopus, cn_given, standard_error):
	return (float(-(cn_onctopus - cn_given)**2)/(2 * (standard_error**2)))

# computes the complete log likelihood of the normal distribution
# first part of subtraction: - log(x) = log()
def log_normal_complete_llh(cn_onctopus, cn_given, standard_error):
	return - log(sqrt(2 * pi * (standard_error**2))) - (float((cn_onctopus - cn_given))**2 / 
		float(2 * (standard_error**2)))

# inserts a point (xs_max, ys_max) into a list of coordinates (xs and ys)
# where the point is inserted according to its position on the x-axis
def insert_max_point(xs, xs_max, ys, ys_max):
	# check if xs_max is already in list xs
	for x_elem in xs:
		if isclose(x_elem, xs_max):
			return (xs, ys)
	# get first index where value is bigger than xs_max
	array_with_index = np.where(xs > xs_max)
	index = -1
	if len(array_with_index[0]) > 0:
		index = array_with_index[0][0]
	else:
		index = len(xs)
	# insert xs_max in xs and ys_max in ys
	xs = np.concatenate((xs[:index], np.array([xs_max]), xs[index:]))
	ys = np.concatenate((ys[:index], np.array([ys_max]), ys[index:]))

	return (xs, ys)

# creates an array with <points> points and transforms it into an array in the
# interval from xmin to xmax
# also computes the correspoding values using function <func>
def transform_array(points, xmin, xmax, func, *args):
	xs = compute_interval(points, xmin, xmax)
	# values for ys are computed with "func"
	ys = [func(x,*args) for x in xs]

	return (xs, ys)

# computes interval from xmin to xmax with points points
def compute_interval(points, xmin, xmax):
	# array with values 0 - "points" is created, length = "points" + 1
	xs = np.array(xrange(points + 1))
	# values in array are transformed, start with xmin, end with xmax, other
	#	values distributed equally between xmin and xmax
	xs = xs / float(len(xs) - 1) * (xmax - xmin) + xmin

	return xs

# if boundary points in ys are -inf they are recomputed because otherwise it can lead
# to problems in the spline computation
# the corresponding values on the x-axis of the y-values are changed and the y-values
# are recomputed
# also xmin and xmax are reset if values are recomputed
def check_and_change_boundary_elements(xmin, xmax, xs, ys, small_value, func, *args):
	# check that small_value is not too big
	while ((xs[1] - xs[0]) <= small_value) or ((xs[-1] - xs[-2]) <= small_value):
		small_value = float(small_value) / 2.0
	# check first point in ys
	if ys[0] == -float('Inf'):
		xs[0] = xs[0] + small_value
		if xs[0] > xs[1]:
			raise eo.MyException("Problem in check_and_change_boundary_elements, left boundary element can't be changed.")	
		ys[0] = func(xs[0], *args)
		xmin = xs[0]
	# check last point in ys
	if ys[-1] == -float('Inf'):
		xs[-1] = xs[-1] - small_value
		if xs[-1] < xs[-2]:
			raise eo.MyException("Problem in check_and_change_boundary_elements, right boundary element can't be changed.")
		ys[-1] = func(xs[-1], *args)
		xmax = xs[-1]
	return (xmin, xmax)

# computes and returns a small intervall in which the maxima of the function lies
def get_intervall_for_maxima(xmin, xmax, func, *args):
	around = -1
	middle_point = -1

	# snp/ssm and segment use different functions
	# snp/ssm function
	if func == beta_binomial:
		middle_point = float(args[0])/float(args[1])
		around = 0.05
	# segment function
	elif func == neg_binomial:
		middle_point = args[0]
		around = 5
	# segment function allele-specific
	elif func == log_normal:
		middle_point = args[0]
		around = 0.05
	
	# compute new values
	new_min = middle_point - around
	new_max = middle_point + around

	# check new values
	if new_min < xmin:
		new_min = xmin
	if new_max > xmax:
		new_max = xmax

	return (new_min, new_max)

# computes piecewise linear function
# xmin = start of intervall of the function
# xmax = end of intervall of the function
# npoints = number of points function is represented with
# func = function that should be approximated
# *args = arguments for function
def find_piecewise_linear(number_spline_points, xmin, xmax, npoints, func, *args):
	# finding the maxima
	(xmin_new_intervall, xmax_new_intervall) = get_intervall_for_maxima(xmin, xmax, func, *args)
	(xs, ys) = transform_array(npoints, xmin_new_intervall, xmax_new_intervall, func, *args)
	(xs_max, ys_max) = get_maximum_point(xs, ys)
	
	# smaller array for spline
	# transform_array creates arry with "number_spline_points" + 1 element + 1 elements
	(xs, ys) = transform_array(number_spline_points, xmin, xmax, func, *args)
	# add maximum point
	# if xs_max is already in xs, size of xs and ys stay at "number_spline_points" + 1
	# if xs_max is not already in xs, size of xs and ys get increased by 1: "number_spline_points" + 2
	(xs, ys) = insert_max_point(xs, xs_max, ys, ys_max)

	# check if boundary points in xs and ys have valid values, -inf is not valid
	# if values are not valid intervall boundaries in xs are reset and values in ys are recalculated
	small_value = 0.0000001
	(xmin, xmax) = check_and_change_boundary_elements(xmin, xmax, xs, ys, small_value, func, *args)

	spl = UnivariateSpline(xs,ys,bbox=(xmin,xmax),k=1,s=0)

	return (spl, get_maximum_point(xs, ys))

# computes piecewise linear function
# xmin = start of intervall of the function
# xmax = end of intervall of the function
# npoints = number of points function is represented with
# mutation_type = either copy number change or SSM
# func = function that should be approximated
# *args = arguments for function
def find_piecewise_linear_new(number_spline_points, xmin, xmax, npoints, mutation_type, func, *args):

	small_interval = 0.1
	if mutation_type == cons.CNV:
		larger_interval = 1.0
	else:
		larger_interval = 0.02

	# finding the maximum
	(xmin_new_intervall, xmax_new_intervall) = get_intervall_for_maxima(xmin, xmax, func, *args)
	(xs, ys) = transform_array(npoints, xmin_new_intervall, xmax_new_intervall, func, *args)
	(xs_max, ys_max) = get_maximum_point(xs, ys)
	
	# create knots around maximum
	xs_intervall = list(compute_interval((2*number_spline_points), xs_max-small_interval, xs_max+small_interval))
	# check whether max point is in list
	max_point_in_list = False
	for my_x in xs_intervall:
		if my_x == xs_max:
			max_point_in_list = True
			break

	# create more knots left and right of current interval if mutation is copy number change
	if mutation_type == cons.CNV:
		# create interval
		left_interval = list(compute_interval(number_spline_points, xs_max-larger_interval, xs_max-small_interval))
		right_interval = list(compute_interval(number_spline_points, xs_max+small_interval, xs_max+larger_interval))
		# add intervals
		xs_intervall = left_interval[:-1] + xs_intervall + right_interval[1:]

	# check that left border isn't smaller than 0
	if xs_intervall[0] < 0:
		xs_intervall = set_left_border_correct(xs_intervall)
	# if mutation is SSM, also check right border, must be smaller than 1
	if mutation_type == cons.SSM and xs_intervall[-1] > 1:
		xs_intervall = set_right_border_correct(xs_intervall)

	# if left border > 0, add more knots
	if xs_intervall[0] > 0:
		xs_intervall = add_knots_to_left(xs_intervall, larger_interval)
	# if maximal right border not in interval, add more knots
	if xs_intervall[-1] < xmax:
		xs_intervall = add_knots_to_right(xs_intervall, xmax, larger_interval)

	# compute ys values
	ys_intervall = [func(i, *args) for i in xs_intervall]

	# if max point not in list, insert
	# add maximum point
	if max_point_in_list == False:
		(xs_intervall, ys_intervall) = insert_max_point(xs_intervall, xs_max, ys_intervall, ys_max)

	# check if boundary points in xs and ys have valid values, -inf is not valid
	# if values are not valid intervall boundaries in xs_intervall are reset and values in ys_intervall are recalculated
	small_value = 0.0000001
	(xmin, xmax) = check_and_change_boundary_elements(xmin, xmax, xs_intervall, ys_intervall, small_value, func, *args)

	spl = UnivariateSpline(xs_intervall, ys_intervall, bbox=(xmin,xmax), k=1, s=0)

	return (spl, (xs_max, ys_max))

# if list starts with values < 0, first position is found where it is >= 0, this position is made the new start position
def set_left_border_correct(xs_intervall):	
	index_greater_0 = 0
	while xs_intervall[index_greater_0] < 0:
		index_greater_0 += 1
	# remove entries with values smaller than 0
	return xs_intervall[index_greater_0:]

# if list ends with values > 1, last position is found where it is <= 1, this position is made the new end position
def set_right_border_correct(xs_intervall):
	index_smaller_1 = -1
	while xs_intervall[index_smaller_1] > 1:
		index_smaller_1 -= 1
	# remove entries with values higher than 1
	if index_smaller_1 == -1:
		return xs_intervall
	return xs_intervall[:index_smaller_1+1]

# more knots are inserted to the left, starting with 0, stepsize is larger_interval, until original start of list is reached
def add_knots_to_left(xs_intervall, larger_interval):
	left_interval_length = xs_intervall[0]
	left_knots_to_insert = int(ceil(float(left_interval_length) / larger_interval))
	left_interval = [i * larger_interval for i in xrange(left_knots_to_insert)]
	if abs(left_interval[-1] - xs_intervall[0]) < 0.0001:
		return left_interval[:-1] + xs_intervall
	return left_interval + xs_intervall

# more knots are inserted to the right until xmax is reached, last knot is xmax, from there in stepsize larger_interval
# to original end of list
def add_knots_to_right(xs_intervall, xmax, larger_interval):
	right_knots_to_insert = int(ceil(float(xmax - xs_intervall[-1]) / larger_interval))
	right_interval = [xmax - (i * larger_interval) for i in xrange(right_knots_to_insert)]
	right_interval.reverse()
	if abs(right_interval[0] - xs_intervall[-1]) < 0.0001:
		return xs_intervall + right_interval[1:]
	return xs_intervall + right_interval

# checks if piecewise linear function represented by a spline spl is concave

def check_concavity(spl):
	xs = spl.get_knots()
	ys = spl.get_coeffs()
	last_slope = float('inf')
	for i in range(len(xs)-1):
		cur_slope = (ys[i+1] - ys[i]) / (xs[i+1]-xs[i])
		if cur_slope > last_slope:
			return False
		last_slope = cur_slope
	return True

# get boundaries for segment for piecewise linear function
# TODO: is dealing like this with boundaries the good way?
def get_xmin_xmax_for_seg(count, hm):
	#offset = 150
	#if (count < hm):
	#	return (0, hm + offset)
	#return (0, count + offset)
	return (0, 3 * hm)

# get boundaries for snp/ssm for piecewise linear function
def get_xmin_xmax_for_snp_ssm(variant, total):
	#offset = 0.1
	#
	#prob = float(variant) / float(total)
	#
	#xmin = prob - offset
	#if (xmin < 0):
	#	xmin = 0
	#
	#xmax = prob + offset
	#if (xmax > 1):
	#	xmax = 1

	#return (xmin, xmax)
	return (0, 1)

# computes the piecewise lineaer functions for a list of segments
# who are considered allele-specific
# seg_points: with how many points the curve of the pmf should be computed
# number_spline_points: how many points should be in the spline 
def compute_piecewise_linear_for_seg_list_allele_specific(seg_list, seg_points,
	number_spline_points):
	seg_splines_A = []
	seg_splines_B = []
	total_knots = 0
	for seg_index, seg in enumerate(seg_list):
		# intervalls in which the likelihood for the copy numbers
		# is computed
		(xmin_A, xmax_A) = (0, seg.given_cn_A + 5)
		(xmin_B, xmax_B) = (0, seg.given_cn_B + 5)
		# compute splines
		(spl_A, (xs_max_A, ys_max_A)) = find_piecewise_linear_new(number_spline_points,
			xmin_A, xmax_A, seg_points, cons.CNV,
			log_normal, seg.given_cn_A, seg.standard_error_A)
		(spl_B, (xs_max_B, ys_max_B)) = find_piecewise_linear_new(number_spline_points,
			xmin_B, xmax_B, seg_points, cons.CNV,
			log_normal, seg.given_cn_B, seg.standard_error_B)
		# check if spline is concave
		if (check_concavity(spl_A)):
			seg_splines_A.append(spl_A)
		else:
			raise eo.ConcavityException("Segment A with index {0} has no concave spline.".format(
				seg_index))
		if (check_concavity(spl_B)):
			seg_splines_B.append(spl_B)
		else:
			raise eo.ConcavityException("Segment B with index {0} has no concave spline.".format(
				seg_index))
	return (seg_splines_A, seg_splines_B)

# computes the piecewise lineaer functions for a list of segments
# seg_points: with how many points the curve of the pmf should be computed
# number_spline_points: how many points should be in the spline 
def compute_piecewise_linear_for_seg_list(seg_list, seg_overdispersion, seg_points,
	number_spline_points):
	seg_splines  = []
	total_knots = 0
	for seg_index, seg in enumerate(seg_list):
		(xmin, xmax) = get_xmin_xmax_for_seg(seg.count, seg.hm)
		(spl, (xs_max, ys_max)) = find_piecewise_linear(number_spline_points,
			xmin, xmax, seg_points,
			neg_binomial, seg.count, seg_overdispersion)
		(spl_x_max, spl_y_max) = get_maximum_point(spl.get_knots(), spl(spl.get_knots()))
		if (check_concavity(spl)):
			seg_splines.append(spl)
		else:
			raise eo.ConcavityException("Segment with index {0} has no concave spline.".format(
				seg_index))
	return seg_splines

# computes the piecewise lineaer functions for a list of snps or ssms
def compute_piecewise_linear_for_snp_ssm_list(mut_list, mut_overdispersion, mut_points,
	number_spline_points):
	mut_splines  = []
	total_knots = 0
	for mut_index, mut in enumerate(mut_list):
		total = mut.variant_count + mut.ref_count
		(xmin, xmax) = get_xmin_xmax_for_snp_ssm(mut.variant_count, total)
		(spl, (xs_max, ys_max)) = find_piecewise_linear_new(number_spline_points, xmin, xmax, 
			mut_points, cons.SSM,
			beta_binomial, mut.variant_count, total, mut_overdispersion)
		(spl_x_max, spl_y_max) = get_maximum_point(spl.get_knots(), spl(spl.get_knots()))
		if (check_concavity(spl)):
			mut_splines.append(spl)
		else:
			raise eo.ConcavityException("{0} with index {1} has no concave spline.".format(mut.type, mut_index))
	return mut_splines

# given to lists with x and y coordinates, returnes the maximal points
def get_maximum_point(xs, ys):
	ys_array = array(ys)
	ys_max = amax(ys_array)
	ys_argmax = argmax(ys_array)
	xs_max = xs[ys_argmax]
	return (xs_max, ys_max)
