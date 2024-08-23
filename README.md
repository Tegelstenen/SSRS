<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Tegelstenen/SSRS">
    <img src="images/readme_logo.png" alt="Logo">
  </a>

  <h3 align="center">AI for impact X SSRS</h3>

  <p align="center">
    A summer internship aimed to assist the Swedish Sea Resuce Society (SSRS) with predictive maintenance of their vessels and GPS anomalies.
    <br />
 

[![Filip Sjöstrand](https://img.shields.io/badge/Filip_Sjöstrand-blue?logo=linkedin)](https://www.linkedin.com/in/filipsjostrand​) [![Axel Karlsson](https://img.shields.io/badge/Axel_Karlsson-blue?logo=linkedin)](https://www.linkedin.com/in/axel-karlsson-16251871​) [![Theo Malmstedt](https://img.shields.io/badge/Theo_Malmstedt-blue?logo=linkedin)](https://www.linkedin.com/in/theo-larsson-malmstedt/)

</div>

  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">
  <a href="https://github.com/Tegelstenen/SSRS">
    <img src="images/Screenshot 2024-08-16 at 13.47.50.png" alt="dahsboard">
  </a>

</div>

For non profit organizations provide vital services to society as a whole. Their existence is often depended upon their members volunteering and donations. These organization are driven by the passion of doing good and usually lack the continuous development found in most companies. As a consequence, many of these organizations lack the resources to maintain and catch up with the latest technological advancements. AI for impact is a program funded by Google.org that aims to bridge this gap by helping non-profits implement AI solutions to their operational problems.

This project is in partnership with the Swedish Sea Rescue Society (SSRS) and AI Sweden, driven by three university students. Given the time constraints, the project has focused on analysing engine data for predictive maintenance as well as analysis of geographical data to identify potential GPS jammings and spoofings.


### Built With
<div align="center">

![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![streamlite](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![influxdb](https://img.shields.io/badge/InfluxDB-22ADF6?style=for-the-badge&logo=InfluxDB&logoColor=white) ![keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)

</div>

<!-- GETTING STARTED -->
## Getting Started

The project is divide into three main components with each having its own `main.py`.
```sh
SSRS/
├── dashboard/
│   ├── ...
│   └── main.py
├── models/
│   ├── ...
│   └── main.py
├── pipeline/
│   ├── ...
│   └── main.py
└── .../
```

Begin by creating a virtual environment and installing the dependencies.

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Also, create the necessary `.env` file with the following variables:
```
TOKEN='influxDB token'
API_KEY_STADIA='Stadia API key'
ACCESS_KEY='AWS access key'
SECRET_KEY='AWS secret key'
```

<!-- USAGE EXAMPLES -->
## Usage

#### Pipeline
The pipeline is responsible for ingesting data from the SSRS database and preprocess it for the models. To run the pipeline, run the following command:

```sh
python pipeline/main.py start_date end_date directory_to_save_data
```

For example:
 ```sh
python pipeline/main.py 2024-01-01 2024-01-02 models/data
```

<details><summary style="font-size: 14px; font-weight: bold;">Extended Description</summary>

Everythin is orchestrated under the `main.py` file. The steps taken are:

1. Querying to the database using `db_query.py`
    * First we find the sequence where RPM is greater or equal to 0 to find where the boats are actually moving.
    * This is used to determining boat trips for each boat and then we make the queries on all of the data based of these trips
    * This saves a lot of different csv files for each boat, each trip, and each variable, like:
    ```
    data/
    └── c710dd10-8987-4216-bf82-6fef6cf5225c/
        ├── Trip1/
        │   ├── RPM.csv
        │   ├── COG.csv
        │   ├── ...
        └── Trip2/
            ├── ...
    ```
2. Then we make some initial processing inside `process_raw.py` of the individual csv files to create two merged csv files `engine_data.csv` `geo_data.csv`
    * `engine_data.csv` contains the data pertaining to the engine (i.e., they have a value to `signal_instance`)
        $\longrightarrow$ `RPM`
        $\longrightarrow$`ENGTEMP`
        $\longrightarrow$`ENGHOURS`
        $\longrightarrow$`ENGLOAD`
        $\longrightarrow$`FUELRATE`
        $\longrightarrow$`ENGINE_LOAD`
    * `geo_data.csv` contains the data pertaining to the GPS (i.e., they lack a value to `signal_instance`)
        $\longrightarrow$`COG`
        $\longrightarrow$`SOG`
        $\longrightarrow$`LON`
        $\longrightarrow$`LAT`


        
        DataCleaner()

3. We follow with `widen.py` which pivots to a wide format keeping `node_name`, `date`, and `signal_instance` as index.

4. The `sequence.py` ensure that our time series data is sequential.
    * Some of the data points are collected at different time steps.
    We utilise an averaged value for each 5th second to ensure no NaN values and linear observations.

5. Next, we impute the missing values with `impute.py`. Here we group by `boat` and `TRIP_ID` and then use a forward fill.
    * We are assuming not much is happening within potential gaps in the 5 second windows created in the previous step.
5. Then we create a merged dataframe with `merge.py`
6. To make better use of the geographical data we add weather conditions using `weather.py`
    * This script fetches weather data from the Open-Meteo API and maps the nearest weather reading to nearby boats
    * The result is `wind_velocity` and `alignment_factor` features added to the dataframe
7. The final `final_cleanup.py` does minimal alteration to the data to fit what we have in the models.

</details>



#### Models

The model is responsible for training and inferring with a trained model. To run the model, run the following command:

```sh
python models/main.py --mode mode --model model
```

For example:

 ```sh
python models/main.py --mode lstm --model infer
```

There are two modes: `train` and `infer`. The `train` mode is used to train the model and the `infer` mode is used to infer with a trained model.

There are two models: `lstm` and `autoencoder`. The `lstm` is an LSTM autoencoder model and 'autoencoder' is a simple autoencoder model.

<details><summary style="font-size: 14px; font-weight: bold;">Extended Description</summary>

The LSTM autoencoder is the best performing and primary model in our code. Its code is structured as follows:

```sh
models/
└── modules/
 ├── lstm
 │ ├── model.py
 │ ├── train.py
 │ └── infer.py
 └── autoencoder
 │ ├── ...
 │ ├── ...
 │ └── ...
```

The model is defined in the model.py file as follows:

<div align="center">
  <a href="https://github.com/Tegelstenen/SSRS">
    <img src="images/lstm_model.jpg" alt="GPS Flowchart">
  </a>
</div>




</details>

#### Dashboard

The dashboard is responsible for visualizing the data and the inference results. To run the dashboard locally, run the following command:

```sh
streamlit run dashboard/main.py
```

Otherwise, the dashboard is hosted on [Streamlit Cloud](https://ssrs-anomaly.streamlit.app/) behind a password.

<details><summary style="font-size: 14px; font-weight: bold;">Extended Description</summary>
 
BLA BLA BLA

</details>

### GPS Disruptions

The dashboard also visualises potetial GPS disruptions such as jamming och spoofing around the coast of Sweden. The logic behind this part of the dashborad is all being run from `main.py` and `GPSjam.py`

This is a flowchart of the dashboard once you press the button GPS Disruptions.

<div align="center">
  <a href="https://github.com/Tegelstenen/SSRS">
    <img src="images/GPSFlowchartApp.png" alt="GPS Flowchart">
  </a>
</div>

A static display of a certain time period will occur when pressing update date or aggregate map. Update date will show a map for a specific selected hour and aggregate map will show an aggregated map over 24 hours. If the button now is pressed the website will update every hour and show the latest GPS disruptions. 

<details><summary style="font-size: 14px; font-weight: bold;">Extended Description</summary>

#### Main & Gpsjam
These files control the logic behind the GPS Disruptions page on the dashboard. The flowchart above gives a good picture of the logic of this part of the program. Good to note however is that no other commands need to be used except for the buttons on the screen. 

#### Model
Model - The model is an Autoencoder trained on timeseries data just like the models for anomaly detection in engine behaviour. The model inputs here a sequences of 60 timesteps (roughly 60s) with values for LAT, LON, SOG, COG & RPM. It has later been finetuned with labelled data and converted to a binary classifier wich is the model used at this moment, `fine_tune_test.keras` is the model currently implemented in the script. The scripts/notebooks used to train this model are not available in this directory. 

#### Pipeline
Pipeline - The datapipeline used is simliar to the one described for engine anomaly detection, for GPS disruptions the relavant files can be found under `dashboard\modules\GpsDataPipeline`.Initial processing is done with `process_raw.py` in the GpsDataPipeline folder, final stage of processing is done with `manipulate.py`. After both files are run the data is in the same format as the model is trained on and stored under `dashboard\data\tmpfiles` as `geo_data.csv`. This file will be replaced each time now is pressed or an automatic update is triggered. 

</details>

<!-- CONTACT -->
## Contact

Feel free to reach out to any of the team members for more information. We are always open to discuss new ideas and collaborations.

[Filip Sjöstrand](mailto:f.w.sjostrand@gmail.com?subject=Questions%20regarding%20AI%20for%20impact%20-%20SSRS&body=Hello%2C%20I%20found%20your%20contact%20in%20the%20SSRS%20GitHub%20repo.%0D%0A%0D%0A...) — [Axel Karlsson](mailto:axeka624@student.liu.se?subject=Questions%20regarding%20AI%20for%20impact%20-%20SSRS&body=Hello%2C%20I%20found%20your%20contact%20in%20the%20SSRS%20GitHub%20repo....) - [Theo Malmstedt](mailto:theo.js.malmstedt@gmail.com?subject=Questions%20regarding%20AI%20for%20impact%20-%20SSRS&body=Hello%2C%20I%20found%20your%20contact%20in%20the%20SSRS%20GitHub%20repo....) 
