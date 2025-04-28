"""
Test runner for ARC DSL tests.

This script runs all the tests for the ARC DSL.
"""
import unittest
import sys
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_primitives import TestGridTransformations, TestColorOperations, TestObjectOperations, TestGridManipulations
from test_program import TestProgramExecution, TestComplexPrograms
from test_search import TestSearchAlgorithms, TestVerifier


def run_tests():
    """Run all tests."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases from test_primitives.py
    suite.addTest(unittest.makeSuite(TestGridTransformations))
    suite.addTest(unittest.makeSuite(TestColorOperations))
    suite.addTest(unittest.makeSuite(TestObjectOperations))
    suite.addTest(unittest.makeSuite(TestGridManipulations))
    
    # Add test cases from test_program.py
    suite.addTest(unittest.makeSuite(TestProgramExecution))
    suite.addTest(unittest.makeSuite(TestComplexPrograms))
    
    # Add test cases from test_search.py
    suite.addTest(unittest.makeSuite(TestSearchAlgorithms))
    suite.addTest(unittest.makeSuite(TestVerifier))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
