# multilingual transfer learning using CPC 

This project focuses on training a self-supervised deep learning model on speech datasets from two different languages: libri-speech for English and lohita puhetta for Finnish.  After pre-training, the model is further fine-tuned on separate datasets to optimize its performance. the downstream tasks are speech emotion recognition and speaker recognition.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation and Setup

1. Clone the repository:

```bash
git clone git@github.com:praburocking/CPC.git
```
   
2. Navigate to the directory:

```bash
cd CPC
```

3. Install the required packages:

```bash
conda env create -f environment.yml
conda activate mlenv
```

4. directories to re-configure:
- Data pre-processing 
  - ```CPC/split_to_utterence.py```
  - ```CPC/wav2raw.py```
Both the file contains similar functions that take the input wave file, chunks them into the fixex length and convert them into log mel format and store them into HDF(.h5) and pickle(.pkl) format, where .h5 files contains all the data and .pkl files contains only the name of the files. the output directory is also configurable. 

- training
  - ```CPC/new_conf.py```
it is a sample training configuration file, Which is divided into modes. The current modes are ```["up_stream","down_stream_fine_tune","down_stream_train","test"]``` which could be choosen with ```mode_index``` variable. The common variable and mode specific variables can be added to this file.
  - ```CPC/main.py```

   ```conf_file="CPC/new_conf.py"``` and the configuration file path.

5. run the model:

  ```python3 main.py <experiment_name> <experiment_description> ```

## Datasets

- **English Dataset**: 
  - Description: [Briefly describe the dataset, number of samples, duration, etc.]
- **Finnish Dataset**: 
  - Description: [Briefly describe the dataset, number of samples, duration, etc.]

## Model Architecture

[Provide a brief description or diagram of the model architecture]

## Usage

### Pre-training

```bash
python pretrain.py --language [english/finnish]
```

### Fine-tuning

```bash
python finetune.py --language [english/finnish] --dataset [fine_tune_dataset_name]
```

## Results

| Dataset | Language | Pre-training Accuracy | Fine-tuning Accuracy |
|---------|----------|-----------------------|----------------------|
| Dataset1 | English  | XX%                   | YY%                  |
| Dataset2 | Finnish  | XX%                   | YY%                  |
| ...     | ...      | ...                   | ...                  |

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

[Your License Name]

## Acknowledgements

- [Any acknowledgements, references, or datasets you've used]

---

Feel free to customize this template based on your project's specific needs. Make sure to replace placeholders (like `[YOUR_REPOSITORY_LINK]`) with the actual information relevant to your project.


Implementation of Contrastive Predictive coding as a up stream task and simple classifier as the down stream task to classify the finnish emotion recognition.</n>


Finnish Speech class distribution
 class 0 --- 971
 class 1 --- 457
 class 2 --- 984
 class 3 --- 917
 class 4 --- 736

pretraining details:
test-Librispeech.pkl ...5559
dev-Librispeech.pkl ...5567
train_500-Librispeech.pkl ...148688
train_360-Librispeech.pkl ...104014
train_100-Librispeech.pkl ...28539
total 292367