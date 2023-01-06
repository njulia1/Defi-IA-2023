# Defi-IA-2023
This repository contains all codes, data and information regarding the AI challenge 2023. The goal of the challenge was to predict prices hotels.

## Repository
This github contains the following files:

- `data_analysis.ipynb` which introduce the features of our dataset, and performs a preliminary analysis on the data,
- `train.py` which allows to train a random Forest or a Boosting model,
- `adversarial_validation.ipynb` and its python file `adversarial_validation.py`containing the code we used to check the similary between our trainset and the test set of Kaggle,
- `interpretability.ipynb` which helps us to understand the most importance features of our model,
- `gradio_app.py`, which launch the gradio application,
- `test.py`, which test the model on Kaggle test set,
- `Dockerfile`, which contains the information of the Dockerfile,
- a folder `data` containing datasets for train, improved dataset, the test set and the hotel features. 

## Instruction to run the code

To run the code, you need to follow the instructions below:
1. Open a terminal 
2. Clone the Github repository : `git clone https://github.com/njulia1/Defi-IA-2023.git `
3. Go to the repository: `cd Defi-IA-2023` 
4. Download the pre-trained model (a bit long, so take): `wget 'https://filesender.renater.fr/download.php?token=8aaf1fbe-c8b4-4041-9901-c7825e275ec5&files_ids=20724435' -O model/model.pkl`
(Link for downloading the model : https://filesender.renater.fr/download.php?token=8aaf1fbe-c8b4-4041-9901-c7825e275ec5&files_ids=20724435' -O model/model.pkl )
6. Create the docker image: `sudo docker build -t image1 ./docker`
7. Create a docker container in which we clone the whole Github: `sudo docker run -it --name container1 image1`
8. Exit the container: `exit` 
9. Copy the Github into the container: `docker cp .`
10. Start the container: `sudo docker start container1`
11. Go to the container: `sudo docker attach container1`
12. Go to the IA folder: `cd ia`

From there, you have to options:
1. you can launch the model to be trained on different data: either the whole dataset (containing more than 600 000 lines) or an improved dataset (after adversarial_network.py) : `python3 ./train.py --data PATH_TO_CSV --model MODEL`
where MODEL is 1 for Random Forest or 2 for Boosting
To launch the code on the whole dataset for Boosting algorithm, use the command: `python3 ./train.py --data 'data/data.csv' --model 2`
To launch the code on the optimized dataset for Random Forest algorithm, use the command: `python3 ./train.py --data 'data/data_improved.csv' --model 1`
2. you can also launch the gradio application : `python3 gradio_app.py`

## Additional codes and instructions
