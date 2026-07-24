import os


def create_output_folders():
   
    folders = [
        "outputs",
        "outputs/models",
        "outputs/figures",
        "outputs/metrics"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Output folders are ready.")

if __name__ == "__main__":
    create_output_folders()
    

    