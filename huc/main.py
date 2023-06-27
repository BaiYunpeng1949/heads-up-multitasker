from RL import RL
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    # Run the RL pipeline with the given configurations.
    rl = RL(config_file='config.yaml')
    rl.run()


if __name__ == "__main__":
    main()