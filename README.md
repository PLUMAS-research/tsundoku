# 📚 tsundoku

`tsundoku` is a Python toolkit to analyze X/Twitter data, following the methodology published in:

> Graells-Garrido, E., Baeza-Yates, R., & Lalmas, M. (2020, July). [Every colour you are: Stance prediction and turnaround in controversial issues](https://dl.acm.org/doi/abs/10.1145/3394231.3397907). In 12th ACM Conference on Web Science (pp. 174-183).

## Development Setup

We use `mamba` to install all necessary packages. First, you'll need to install Mamba:

```bash
# Install Mamba if you haven't already
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

After installing Mamba:

```bash
# Clone repository
git clone http://github.com/PLUMAS-research/tsundoku

# Move into folder
cd tsundoku

# Create environment, install dependencies and activate it
make mamba-create-env

# Activate the environment 
mamba activate tsundoku

# make the tsundoku module available in your environment
make install-package
```

Optionally, you may opt to analyze the data generated by `tsundoku` in a Jupyter environment. In that case, you will need to install a kernel:

```bash
# install kernel for use within Jupyter
make install-kernel
```

If you want to test your installation, you may execute:

```bash
python -m tsundoku.utils.test
```

## Environment Configuration

Create an `.env` file in the root of this repository with the following structure:

```
TSUNDOKU_PROJECT_PATH=./softwarex
JSON_TWEET_PATH=./test_data/sample_public
TWEET_PATH=./test_data/parquet_public
```

This is the meaning of each option:

* `TSUNDOKU_PROJECT_PATH`: path to your project configuration (this is explained below).
* `JSON_TWEET_PATH`: directory where you stored the tweets. This code assumes that you crawl tweets using the Streaming API. These tweets are stored in JSON format, one tweet per line, in files compressed using gzip. Particularly, we assume that each file contains **10 minutes of tweets**. The system assumes that those tweets were pre-processed by flattening the structure. These files may be related to any project.
* `TWEET_PATH`: folder where the system stores tweets in Apache Parquet format.

Files present in `test_data/sample_public` have already been flattened. They are in JSON format, one tweet per line. For analysis, you will need to convert those files to Parquet. You can do so with the following command:

```bash
python -m tsundoku.data.parse_json_to_parquet 20220501
```
## Project Configuration

The `TSUNDOKU_PROJECT_PATH` folder defines a project. It contains the following files and folders:

- `config.toml`: project configuration.
- `groups/*.toml`: classifier configuration for several groups of users. This is arbitrary, you can define your own groups. The mandatory one is called `relevant.toml`.
- `experiments.toml`: experiment definition and classifier hyper-parameters. Experiments enable analysis in different periods (for instance, first and second round of a presidential election).
- `keywords.txt` (optional): set of keywords to filter tweets. For instance, presidential candidate names, relevant hashtags, etc.
- `stopwords.txt` (optional): list of stop words.

Please see the example in the `softwarex` folder, which contains a full project that uses the data in `test_data`.

In `config.toml` there you will need to configure at least the following attribute:

```toml
[project.settings]
data_path = "/home/USERNAME/path_to_project/data"
```

The `data_path` attribute states where the imported data will be stored after filtering with your specified keywords.

## Data and Projects

`tsundoku` has three folders within the project data folder: `raw`, `interim`, and `processed`.

The `raw` folder contains a subfolder for each day you aim to analyze. The format is `YYYY-MM-DD`. 

The following command imports a specific date from `TWEET_PATH`:

```sh
$ ./tsundoku-cli import_date 20220501
```

This imports that specific day into the project. 

For every day of data you can compute features, such as document-term matrices:

```sh
$ ./tsundoku-cli compute_features 20220501
```

You may import multiple days using the `--days n` parameter (with n being an integer).

In the experiments file you defined experiments such as:

```toml
[experiments]
[experiments.workers_day]
key = 'workers_day'
folder_start = '2022-05-01'
folder_end = '2022-05-01'
discussion_only = 1
discussion_directed = 0
```

In this case, there is a single experiment, of key value `workers_day`. You can perform analysis through the following commands:

1. `$ ./tsundoku-cli prepare_experiment workers_day`: this will prepare the features for the specific experiment. For instance, a experiment has start/end dates, so it consolidates the data between those dates only.
2. `$ ./tsundoku-cli classify_users workers_day relevance`: this command predicts whether a user profile is relevant or not (noise) for the experiment. It uses a XGB classifier.
3. `$ ./tsundoku-cli classify_users workers_day stance`: this command predicts groups within users. The sample configuration includes _stance_. You can define as many groups as you want. Note that for each group you must define categories in the corresponding `.toml` file. In this file, if a category is called _noise_, it means that users who fall in the category will be discarding when consolidating results.
4. `$ ./tsundoku-cli consolidate_analysis workers_day stance`: this command takes the result from the classification and consolidates the analysis with respect to interaction networks, vocabulary, and other features. It requires a reference group to base the analysis (for instance, _stance_ allows you to characterize the supporters of a political position).
5. `$ ./tsundoku-cli generate_report workers_day stance`: this command generates a summary report for the `workers_day` experiment in HTML format, having _stance_ as a reference group.
6. `$ ./tsundoku-cli open_report workers_day`: this command opens a Web browser to display the corresponding report.

## About the name

_Tsundoku_ is a Japanese word (積ん読) that means "to pile books without reading them" (see more in [Wikipedia](https://en.wikipedia.org/wiki/Tsundoku)). It is common to crawl data  continuously and do nothing with them later. So, `tsundoku` provides a way to work with all those piled-up datasets (mainly in the form of *tweets*).