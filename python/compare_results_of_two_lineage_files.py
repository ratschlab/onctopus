import onctopus_io as oio
import model
import sys

def evaluate(lin_true_file, lin_cplex_file, file_name=None, no_test=False):

	lineages_true = oio.read_result_file(lin_true_file)
	lineages_cplex = oio.read_result_file(lin_cplex_file)

	(freq_difference, found_ssm_percentage, found_cnv_percentage) = (
		model.compare_results(lineages_true, lineages_cplex))
	
	header = "lin_true_file: {0}\n lin_cplex_file: {1}".format(lin_true_file, lin_cplex_file)
	freq_difference = "freq_difference: {0}".format(freq_difference)
	found_ssm_percentage = "found_ssm_percentage: {0}".format(found_ssm_percentage)
	found_cnv_percentage = "found_cnv_percentage: {0}".format(found_cnv_percentage)

	# results should be written to file
	if file_name:
		if no_test:
			oio.raise_if_file_exists(file_name)
		with open(file_name, 'w') as f:
			f.write(header + "\n" + freq_difference + "\n" + found_ssm_percentage + "\n"
				+ found_cnv_percentage)
	# results should just be printed 
	else:
		print header
		print freq_difference
		print found_ssm_percentage
		print found_cnv_percentage


if __name__ == '__main__':
	(prog, lin_true_file, lin_cplex_file) = sys.argv
	evaluate(lin_true_file, lin_cplex_file)

