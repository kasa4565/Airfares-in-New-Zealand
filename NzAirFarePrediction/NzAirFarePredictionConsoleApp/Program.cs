using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using PLplot;
using NzAirFarePrediction.DataStructures;
using static Microsoft.ML.Transforms.NormalizingEstimator;

namespace NzAirFarePrediction
{
    internal static class Program
    {
        #region paths

        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../../Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/nz-airfares-train.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/nz-airfares-test.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/AirTravelFareModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        #endregion

        static void Main(string[] args) //If args[0] == "svg" a vector-based chart will be created instead a .png chart
        {
            //Create ML Context with seed for repeatable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // Create, Train, Evaluate and Save a model
            BuildTrainEvaluateAndSaveModel(mlContext);

            // Make a single test prediction loding the model from .ZIP file
            TestSinglePrediction(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            IDataView baseTrainingDataView =
                mlContext.Data.LoadFromTextFile<AirTravel>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView =
                mlContext.Data.LoadFromTextFile<AirTravel>(TestDataPath, hasHeader: true, separatorChar: ',');

            //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data 
            var cnt = baseTrainingDataView.GetColumn<float>(nameof(AirTravel.AirFare)).Count();
            IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView,
                nameof(AirTravel.AirFare), lowerBound: 30, upperBound: 1400);
            var cnt2 = trainingDataView.GetColumn<float>(nameof(AirTravel.AirFare)).Count();

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms
                .CopyColumns(outputColumnName: "Label", inputColumnName: nameof(AirTravel.AirFare))
                // TravelDate

                // DepartmentAirport
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DepartmentAirportEncoded",
                    inputColumnName: nameof(AirTravel.DepartmentAirport)))

                // DepartmentTime

                // ArrivalAirport
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ArrivalAirportEncoded",
                    inputColumnName: nameof(AirTravel.ArrivalAirport)))

                // ArrivalTime

                // Duration

                // Direct

                // Transit

                // Baggage

                // Airline
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AirlineEncoded",
                    inputColumnName: nameof(AirTravel.Airline)))
                .Append(mlContext.Transforms.Concatenate("Features", "DepartmentAirportEncoded",
                    "ArrivalAirportEncoded", "AirlineEncoded"));
            // TODO: Fill gaps

            // (OPTIONAL) Peek data (such as 5 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
            // ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);
            // ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline,
            //     5);
// TODO: Fix the error related to the exception
            // STEP 3: Set the training algorithm, then create and config the modelBuilder - Selected Trainer (SDCA Regression algorithm)                            
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            //The pipeline is trained on the dataset that has been loaded and transformed.
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // STEP 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics =
                mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            // Test: 18/12/2019,ZQN,10:20 AM,WLG,6:10 PM,7h 50m,(1 stop),4h 50m in AKL,,Air New Zealand,422
            var airTravelSample = new AirTravel
            {
                TravelDate = DateTime.ParseExact("18/12/2019", "dd/MM/yyyy", CultureInfo.InvariantCulture),
                DepartmentAirport = "ZQN",
                DepartmentTime = DateTime.ParseExact("10:20 AM", "h:mm tt", CultureInfo.InvariantCulture),
                ArrivalAirport = "WLG",
                ArrivalTime = DateTime.ParseExact("6:10 PM", "h:mm tt", CultureInfo.InvariantCulture),
                Duration = TimeSpan.ParseExact("7h 50m", "h\\h\\ mm\\m", CultureInfo.InvariantCulture),
                Direct = "(1 stop)",
                Transit = "4h 50m in AKL",
                Baggage = "",
                Airline = "Air New Zealand",
                AirFare = 422
            };

            ///
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<AirTravel, AirTravelFarePrediction>(trainedModel);

            //Score
            var resultprediction = predEngine.Predict(airTravelSample);
            ///

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 422");
            Console.WriteLine($"**********************************************************************");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}