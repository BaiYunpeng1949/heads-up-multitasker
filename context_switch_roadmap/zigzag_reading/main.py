from RL import RL


def main():
    # Run the RL pipeline with the given configurations.
    rl = RL(config_file='config.yaml')
    rl.run()


if __name__ == "__main__":
    main()
