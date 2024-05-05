# Futures-Prediction-System


<!-- ABOUT THE PROJECT -->
## About The Project

This is a project for my dissertation - Index Futures Trading with Stable Profits using Deep Learning Models, which aims to get stable profits when trading futures and reduce the risk of index futures trading.


<!-- GETTING STARTED -->
## Getting Started

### Requirements

* Python 3.10
* Pytorch 1.13.1
* Pytorch-cuda 11.6

### Installation

* Install required packages
    ```sh
    pip install -r requirements.txt
    ```
    or
    ```sh
    bash ./run_requirements.sh
    ```

### Structure

Inroduce the structure of the project.
* contracts
    * Create the daily contracts with the needed format by executing preprocess_raw_contracts.py
    * The contracts are used for simulation
* cronjobs
    * The settings of a monthly task
    * The settings are similiar to example directory below
* example
    * The settings of a whole year task with simulation
    * Read more [here](./example/README.md)
* futuresoperations
    * The operation of training data and predicting monthly index value
* raw_contracts
    * Since MTX data cannot be obtained from Yahoo Finance, put raw MTX data here
* tradingsimulation
    * Simulatate the trading process, which should be cooperated with futuresoperations
* utils
    * Check if the format of the settings are correct


<!-- USAGE EXAMPLES -->
## Usage

* A whole year simulation
    1. Preprocess raw contracts for simulation
        * Run `python preprocess_raw_contracts.py <target>`
    2. Run example
        * Run a specific example
            * Run `bash run_specific_examples.sh -t <target> -m <model> -y <year>`
        * Run a example whose model is used for predicting a whole year
            * Run `python main.py example/<target>/example_<model>_<year>.yaml`
        * Run a example with finetuning
            * Run `python main.py example/<target>/example_<model>_<year>.yaml`
            * Then run `python finetune_when_trade.py example/<target>/example_<model>_<year>.yaml`
    3. Results are in models directory
* Monthly updating prediction
    1. Run `cronjob.py cronjobs/<target>.yaml`
    2. Predictions are in models directory, which are store in prediction.csv
