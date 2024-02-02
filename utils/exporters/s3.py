from utils.utils import assert_env_variables_set
import subprocess
import os

# We create an s3 client that can load and save files to s3 that also manages opening and closing processes. 

class S3():
    def __init__(self) -> None:
        # If we are ever able to upgrade to python 3.8, we can use the Boto3 client, and better
        # deal with upload times, for now we simply have to use an aws bash script.
        self.Running = False
        assert_env_variables_set(['S3_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_BUCKET', 'S3_REGION'])

    def export(self, file_path):
        
        print(f"Save {file_path} from s3")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        command = f"sh {project_root}/aws.sh {file_path}"
        subprocess.run(command, shell=True)

        # Example usage