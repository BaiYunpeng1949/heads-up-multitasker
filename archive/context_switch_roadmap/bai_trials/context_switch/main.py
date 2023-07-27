from RLPipeline import RLPipeline


def main():
    # Run the RL pipeline with the given configurations.
    rl_pipeline = RLPipeline(
        config_file='config.yaml'
    )
    rl_pipeline.run()


if __name__ == "__main__":
    main()
