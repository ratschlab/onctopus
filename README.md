# Onctopus

​

This repository contains the code of the software Onctopus that is the implementation of a new lineage-based subclonal reconstruction method working with simple somatic mutation<sup>1</sup> (SSM) and copy number aberration (CNA) information.

​

<sup>1</sup>: SSMs comprise single nucleotide variants and small insertions and deletions.

​

## Input Files

---

Onctopus takes SSM and CNA information data as input, which are provided as tab-delimited text files.

​

The SSM data file contains the following information:

- chromosome ID

- position on Chromosome

- variant read counts

- reference read counts

​

The CNA information data file contains the following information:

- chromosome ID

- start position of segment

- end position of segment

- average allele-specific copy number of Allele A

- standard error for allele A

- average allele-specific copy number of Allele B

- standard error for allele B

​

## Installing Onctopus

---

​

You need to install the following dependencies for Onctopus:

* Python 2 versions of [NumPy](http://www.numpy.org/)

* Python 2 versions of[SciPy](https://www.scipy.org/)

* Python 2 versions of [scikit-learn](https://scikit-learn.org)

​

Also, you need to install the [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/products/ilog-cplex-optimization-studio?mhq=cplex&mhsrc=ibmsearch_a). An explanation how to set up the Python API can be found [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

​

## Running Onctopus

---

​

To run Onctopus, use the following command:

```

python2 python/main.py <cn_data.txt> <ssm_data.txt> <lineage_number> <output_file.txt> --allele_specific True --simple_CN_changes False --max_x_CN_changes 2 --only_one_loss False --only_gains_losses_LOH True --lineage_divergence_rule --dont_break_z_symmetry

```

​

### Trying Onctopus

---

​

Try Onctopus on the simualted data provided in `python/testdata`, e.g. with

​

```

python2 python/main.py python/testdata/cn_data.txt python/testdata/ssm_data.txt 4 output_file.txt --allele_specific True --simple_CN_changes False --max_x_CN_changes 2 --only_one_loss False --only_gains_losses_LOH True --lineage_divergence_rule --dont_break_z_symmetry

```

​

More testdatas can be found in `python/testdata/unittests/mini_test_cases`. The unit test in `python/test.py` make use of this data. You can execute the unit tests with 

```

python2 python/test.py

```

​

​

