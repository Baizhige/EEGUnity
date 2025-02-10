import os
import subprocess
def run_tests(directory=None):
    # Use the provided directory if given, otherwise use the current directory
    if directory:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return
        current_dir = directory
    else:
        current_dir = os.getcwd()

    # Get all .py files in the directory, excluding this script itself
    current_script = os.path.basename(__file__)
    py_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != current_script]
    print(py_files)
    failed_tests = []

    # Execute each .py file
    for py_file in py_files:
        print(f"Running test: {py_file}")
        try:
            # Execute the .py script and capture the output
            result = subprocess.run(['/gpfs/work/int/chengxuanqin21/conda_envs/EEG/bin/python', os.path.join(current_dir, py_file)], capture_output=True, text=True)

            # If the script returns a non-zero exit code, consider it a failure
            if result.returncode != 0:
                print(f"Test failed: {py_file}")
                print(result.stdout)
                print(result.stderr)
                failed_tests.append(py_file)
            else:
                print(f"Test passed: {py_file}")

        except Exception as e:
            print(f"Error running {py_file}: {e}")
            failed_tests.append(py_file)

    # Summary of failed tests
    if failed_tests:
        print("\nSummary of failed tests:")
        for failed_test in failed_tests:
            print(failed_test)
    else:
        print("\nAll tests passed successfully!")

if __name__ == "__main__":
    # Specify the test directory, or press Enter to use the current directory
    test_directory = '/gpfs/work/int/chengxuanqin21/science_works/EEGUnity_V0/CI'
    run_tests(test_directory if test_directory.strip() else None)
