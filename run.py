# run.py (simplified)
from framework.main import TestingFramework

# RUNS THE FRAMEWORK IN THE TERMINAL ("python run.py")
# MAINLY USED FOR NEW FEATURE/EVALUATOR TESTING

if __name__ == "__main__":

    framework = TestingFramework()
    framework.run_menu()