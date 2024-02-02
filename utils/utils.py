import os
def assert_env_variables_set(self, variables):
        for var in variables:
            if var not in os.environ:
                raise AssertionError(f"Environment variable '{var}' is not set.")