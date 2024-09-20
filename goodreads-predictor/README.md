# goodreads-predictor

## Overview

This project was built using the kedro templating and pipelineing framework.

Documentation on this framework can be found here: [Kedro documentation](https://docs.kedro.org)

Documentation and a report on the work completed in this project can be found [here](https://dstisas-my.sharepoint.com/:w:/g/personal/ismail_ben-abdelkader_edu_dsti_institute/EQDTfbDYkZVItOKpr2nIKPUB9Vvy9EeEmQ-UvoXkKhiT2A?e=SSqWXA).

## Installing Dependencies

Dependencies are all declared in the requirements.txt file

To install them, run:

```
pip install -r requirements.txt
```

Additionally, the main input dataset is not provided in this repo. You will either need to as for a credentials file from one of the project owners (see contact details below) or you will need to store a copy of the books.csv file in the data/01_raw/ directory and repoint the [catalog entry](conf/base/catalog.yml) to the local file.

## Running the Kedro Project

You can run this Kedro project from the terminal with:

```
kedro run
```

## Running the tests

Tests can be viewed in the [tests folder](src/goodreads_predictor/tests/).

Tests can be run using the following command which will also show a coverage report.

```
pytest
```

Note: At the time of writing, the only tests created are for the factor lumper class which is in the utils section of the tests folder.

## Getting Started
We recommend beginning with the [project summary notebook](notebooks/project_summary.ipynb). This covers the main elements of the project and provides links to the different parts of the code in a manner that makes sense and follows the project steps chronologically.

## Project Layout
To understand the project layout you can either run `kedro viz` from the terminal, or 
