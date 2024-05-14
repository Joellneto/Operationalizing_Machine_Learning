# Operationalizing Machine Learning Project

This project has the following goals:
- Use the Bank Marketing dataset, that contains data about direct marketing campaigns (phone calls) of a Portuguese banking institution.
- Create an AutoML experiment, deploy the best model as a webservice in Azure Container Instance, enable logging and Application Insights in the endpoint.
- Consume the model REST endpoint using authentication and inspect the Swagger documentation.
- Create and publish an AutoML pipeline programmatically, with a python notebook.
- Export the best model in ONNX format, and use the ONNX Runtime in .NET in a Windows Console and Android Xamarin mobile application to make inferences in a local copy of the model.

## Architectural Diagram

Using the symbology from "Artificial intelligence (AI) architecture design" -  https://learn.microsoft.com/en-us/azure/architecture/ai-ml, the diagram of the project is:

![project diagram](images/diagram.png)

The first step is to import de Bankmarketing tabular data and make it available as a dataset.
The second step is to create a AutoML experiment using the dataset, initially manually in AzureML web interface, and then programmatically, using a Jupyter notebook that runs in a compute instance and creates a automatic pipeline to perform the model training.
The pipeline is published to a REST URL, so that it can be retrained from calls to this endpoint.
The best AutoML model is deployed in a webservice on ACI, the Swagger endpoint documentation is inspected, and a python script is used to make calls do the REST URL of the model, performing inferences on input data, provided in JSON format.
The Application Insights are enabled for this endpoint, allowing to view statistics about the endpoint consumption and running a script which displays the endpoint log.
Finally, the best model is exported to ONNX format, and a console application on Windows and a mobile application on Android were built to run inferences on the local copy of the model.

## Key Steps

1.	Authentication

Not necessary in this case, since I am not using my own Azure account, and I am not authorized to create a security principal in Udacity Azure account.

2.	Automated ML Experiment

This step consists in create the BankMarketing dataset, by importing the data file in CSV format, and running an AutoML experiment on a compute cluster with 6 nodes, maximum execution time of 1 hour and allowing early termination.

![dataset Bankmarketing](images/Dataset.png)

The execution rans for 28 minutes, and was finished by the early termination condition (scores not improving for 20 iterations):

![AutoML Experiment completed](images/AutoMLExperiment.png)

The best model is a Voting Ensemble of 10 classifiers with AUC weighted score of 0.94791:

![AutoML Best Model](images/AutoMLBestModel.png)

3.	Deploy the best model

The best model is deployed in Azure Container Instance (ACI) with authentication enabled:

![AutoML Deploy Best Model](images/AutoMLDeployBestModel.png)

4.	Enable logging

A new virtual environment was created to run a python script that enables Application Insights. The package "azureml-core" was installed to support the execution of the following script:

![Enable Application Insights Script](images/EnableApplicationInsightsScript.png)

The published model in a "Healthy" state with application insights enabled can be seen in the screen bellow:

![Model deployed with Application Insights enabled](images/ModelDeployedApplicationInsights.png)

The execution of logs.py script returns the following output:

![Model deployed Logs](images/ModelDeployedLogs.png)

5.	Swagger Documentation

The swagger documentation  of the deployed model, including the inputs and outputs specification for the POST endpoint "/score" can be seen in the screen below:

![Model deployed Swagger](images/ModelDeployedSwagger.png)


6.	Consume model endpoints

To consume the model REST endpoint, the following Python script was used, with the REST URL and authentication key adjusted to match the published model:

![REST endpoint consume script](images/RESTEndpointConsumeScript.png)

The script contains two records in the "data" JSON array, one to be classified as "yes" and the other as "no", so the output returned by executing the script is:

![REST endpoint consume output](images/RESTEndpointConsumeOutput.png)

To better understand the expected response time for the endpoint, the Apache Benchmark tool was used for 10 requests and returns an average time per request of 101.966 ms:

![REST endpoint consume output](images/RESTEndpointABExecution.png)


7.	Create and publish a pipeline

To create and publish a pipeline, the notebook "aml-pipelines-with-automated-machine-learning-step.ipynb" was executed in a compute instance, with a small change in AutoML step configuration to enable ONNX compatible models, allowing to export the best model in this format later:

![Pipeline notebook allow ONNX models](images/PipelineNotebookAllowONNXModels.png)

The published pipeline can been seen below:

![Pipeline created](images/PipelineCreated.png)

And the pipeline endpoints available:

![Pipeline endpoints](images/PipelineEndpoints.png)

Clicking over the created pipeline, is possible to see the pipeline diagram, showing the Bankmarketing dataset and the AutoML module:

![Pipeline diagram](images/PipelineDiagram.png)

The "published pipeline overview" shows the pipeline REST endpoint and a status of ACTIVE:

![Pipeline endpoint published](images/PipelineEndpointPublished.png)

In the notebook, the execution of "Run Details Widget" showing the steps run:

![Pipeline Run Details Widget](images/PipelineRunWidget.png)

And finally, the ML Studio web interface showing the scheduled run of the pipeline (now called "Pipeline Jobs"):

![Pipeline Run Details Widget - 2](images/PipelineScheduleRun.png)

## How to improve

The project can be improved in several ways:

- change some AutoML parameters, like increase the experiment timeout, not allow early termination, customize the featurization or mark the "Enable deep learning" option (requires a GPU).
- including treatment to handle class imbalance and overfiting, as described in "Prevent overfitting and imbalanced data with Automated ML" -  https://learn.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls?view=azureml-api-2 , as using resampling the data to even the class imbalance (manually or automatically) and using cross validation to prevent overfiting (not used in these experiments).
- including data cleaning steps in the pipeline.

## Screen Recording

Additionally, a screencast was provided with the following main points of the project:
- The dataset Bankmarketing.
- The AutoML experiment.
- The Pipeline created by the notebook.
- The AutoML model endpoint deployed.
- API call to model REST endpoint.
- The pipeline endpoint.

The screencast does not contain audio, but it contain subtitles in English describing each project steps above.

The screencast is available at the link:

https://vimeo.com/946184298?share=copy

To enable the subtitles, please click in icon "CC" and select "English". To change the resolution to 1080p, click in the gear icon and select "quality"="1080p".


## Standout Suggestions

To extend the project, were explore the use of ONNX format, an open standard for representing machine learning models and facilitating their sharing and exchange, and the use of Onnx Runtime to run the model locally on a Windows machine and an Android smartphone.
First, the best AutoML model was exported using this notebook (based on https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb):

![ONXX export best AutoML model notebook](images/onnx_export_best_model_notebook.png)

Using the tool Netron  - https://netron.app/ , is possible to visualize the ONNX model, including the input attributes, the output and the multiples ensemble classifiers that composes the AutoML best model:

![ONXX model diagram](images/onnx_model_netron.png)

After that, the ONNX model file was downloaded locally, and using the source code in the article "Make predictions with an AutoML ONNX model in .NET" -  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automl-onnx-model-dotnet?view=azureml-api-2 , a simple console .NET project was created in Visual Studio .NET 2022 containing:

- a class to represent the input attributes of the model:

![ONXX input class](images/OnnxInputClass.png)

- a class to represent the output and a code in the main method to run the inference on the model:

![ONXX output class and main code](images/OnnxOutputClassAndMain.png)

The execution of the code as a console application run the classifier for a given input and returns the predicted class (no in this case):

![Console app output](images/ConsoleAppOnnx.png)

The same source code was adapted with small modifications to work in a Android application using the Xamarin Framework, in this case in a graphical interface that permits to change the attribute values and check the output class:

![Xamarin app output](images/XamarinAppOnnx.png)

According to attributes values, the prediction is changed from no to yes:

![Xamarin app predictions](images/XamarinAppPredictions.png)

## Conclusions

The contribution of this project is to demonstrate the ease of automating machine learning pipelines in Azure, consuming a REST endpoint of the published model in an Azure Container Instance and also the versatility of exporting the best model in the ONNX format, which allows to run the model locally on a PC or in a smartphone with some lines of code.
