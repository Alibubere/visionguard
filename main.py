import logging
import os

def setup_logging():
    """
    Set Up logging for the Entire Pipeline 
    """

    log_dir = "logs"
    os.makedirs(log_dir,exist_ok=True)
    log_file = os.path.join(log_dir,"pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format= "%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file),logging.StreamHandler()],
    )
    logging.info("Logging successfully Initialized")



def main():
    setup_logging()
    


if __name__ == "__main__":

    main()