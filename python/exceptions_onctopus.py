class  MyException(Exception):
	
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)
	
class AddingException(MyException):
	pass

class ConcavityException(MyException):
	pass

class FileExistsException(MyException):
	pass

class FileDoesNotExistException(MyException):
	pass

class SegmentException(MyException):
	pass

class SegmentAssignmentException(MyException):
	pass

class SSMAssignmentException(MyException):
	pass

class BAFComputationException(MyException):
	pass

class UnallowedNameException(MyException):
	pass

class NoGradientException(MyException):
	pass

class NoSolutionWithLineSearchException(MyException):
	pass

class CiLineSearchEpsilonPlateau(MyException):
	pass

class ReadCountsUnavailableError(MyException):
	pass

class SSMNotFoundException(MyException):
	pass

class ZMatrixPhisInfeasibleException(MyException):
	pass
class NoRootException(MyException):
	pass
class TooMuchChangeException(MyException):
	pass
class NotProperLOH(MyException):
	pass
class ZInconsistence(MyException):
	pass
class no_CNVs(MyException):
	pass
class ADRelationNotPossible(MyException):
	pass
class ZUpdateNotPossible(MyException):
	pass
class ParameterException(MyException):
	pass
class DifferentDimensionFormat(MyException):
	pass
class NoReconstructionWithGivenLineageNumber(MyException):
	pass
class FixPhiIncompatibleException(MyException):
	pass
class LineageWith0FreqMutations(MyException):
	pass
class ZMatrixNotNone(MyException):
	pass
