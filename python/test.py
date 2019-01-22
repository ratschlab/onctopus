import unittest
import test_onctopus_io
import test_model
import test_log_pdf
import test_optimization
import test_mini_test_cases
import test_mini_test_cases_2

if __name__ == '__main__':
	suite_onctopus_io = test_onctopus_io.suite()
	suite_model = test_model.suite()
	suite_logpdf = test_log_pdf.suite()
	suite_optimization = test_optimization.suite()
	suite_mini_test_cases = test_mini_test_cases.suite()
	suite_mini_test_cases_2 = test_mini_test_cases_2.suite()

	print ""

	unittest.TextTestRunner(verbosity=2).run(suite_onctopus_io)
	#unittest.TextTestRunner(verbosity=2).run(suite_model) 
	#unittest.TextTestRunner(verbosity=2).run(suite_logpdf)
	#unittest.TextTestRunner(verbosity=2).run(suite_optimization)
	#unittest.TextTestRunner(verbosity=2).run(suite_mini_test_cases)
	#unittest.TextTestRunner(verbosity=2).run(suite_mini_test_cases_2)

	print ""
