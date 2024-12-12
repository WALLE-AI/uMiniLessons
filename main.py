import loguru

from vlm.fine_tune_paliGemma import execute_train


if __name__ == "__main__":
    loguru.logger.info(f"start paligemma fine tune")
    execute_train()