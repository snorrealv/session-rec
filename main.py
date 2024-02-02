from utils.notifier import Notifyer
from utils.utils import assert_env_variables_set
from run_preprocessing import preprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Session Rec lets you train and evaluate sequense based algoritms.')

    parser.add_argument('action', choices=['preprocess', 'run'], type=str, help='Choose between "config" or "run"')

    # Add a flag to specify a file
    parser.add_argument('-f', '--config', type=str, help='Path to the config file')

    args = parser.parse_args()

    # Your application logic using the arguments
    try:
        assert_env_variables_set(variables=["JOB", "CONFIG"])
    # Your application logic using the arguments
        if args.action == 'preprocess':
            if args.config:
                preprocess(args.config)
            print("preprocess")
        elif args.action == 'run':
            print("run")

        if args.config:
            print(f"Config specified: {args.config}")

    except Exception as e:
        n = Notifyer(mode="slack")
        n.send_exception("Error")

if __name__ == "__main__":
    main()
