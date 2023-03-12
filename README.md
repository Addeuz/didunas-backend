# DIDUNASMathDifficultyPredictor

This repository contains a machine learning model for predicting students' risk of having difficulty with math based on their performance on a set of problem solving tasks. The model has been deployed on a Flask server and can be accessed through an API. The API can route requests to a Linode server where an app is hosted, and obtain the students' performance data to make predictions.

## Requirements
- Python 3.9
- Flask
- scikit-learn
- pandas
- numpy

## Usage
To use the model, simply send a POST request to the API with the following parameters:

- `er_0`: The student's error ratre of the 1st task (float)
- `er_1`: The student's error ratre of the 2nd task (float)
- `er_2`: The student's error ratre of the 3rd task (float)
- `er_3`: The student's error ratre of the 4th task (float)
- `er_4`: The student's error ratre of the 5th task (float)
- `er_5`: The student's error ratre of the 6th task (float)
- `er_6`: The student's error ratre of the 7th task (float)
- `er_7`: The student's error ratre of the 8th task (float)
- `er_8`: The student's error ratre of the 9th task (float)
- `er_9`: The student's error ratre of the 10th task (float)

The API will return a JSON object with the predicted risk of math difficulty (0 or 1).

## Deployment
To deploy the Flask app, you can use a cloud service like AWS, GCP or Azure, or a Linode server. You can use NGINX or another reverse proxy to route requests to the Flask server.

## License
This project is licensed under the Apache License 2.0. The Apache License 2.0 is also compatible with the EUPL (European Union Public Licence). See the LICENSE file for details.

## Authors
- Andreas Johansson
- Han Fan

## Acknowledgements
This project was funded by the EU project Digital Identification and Support of Under-Archiving Students (DIDUNAS). We would like to thank the project participants for their contributions.
