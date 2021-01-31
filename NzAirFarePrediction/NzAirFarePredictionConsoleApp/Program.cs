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
using Microsoft.ML.Trainers;

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

        /// <summary>
        /// Start the program.
        /// Create ML Context with seed for repeatable/deterministic results.
        /// Create, Train, Evaluate and Save a model.
        /// Make a single test prediction loding the model from .ZIP file.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            BuildTrainEvaluateAndSaveModel(mlContext);
            TestSinglePrediction(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        /// <summary>
        /// Create, Train, Evaluate and Save a model.
        /// Common data loading configuration.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            SdcaRegressionTrainer trainer;
            EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline;

            IDataView baseTrainingDataView =
                mlContext.Data.LoadFromTextFile<AirTravel>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView =
                mlContext.Data.LoadFromTextFile<AirTravel>(TestDataPath, hasHeader: true, separatorChar: ',');
            IDataView trainingDataView = GetTrainingDataView(mlContext, baseTrainingDataView);

            var dataProcessPipeline = GetDataProcessPipeline(mlContext);
            SetTrainingAlgorithm(mlContext, dataProcessPipeline, out trainer, out trainingPipeline);
            var trainedModel = GetTrainedModel(trainingPipeline, trainingDataView);
            RegressionMetrics metrics = Evaluate(mlContext, testDataView, trainedModel);
            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);
            SaveModel(mlContext, trainingDataView, trainedModel);

            return trainedModel;
        }

        /// <summary>
        /// Save/persist the trained model to a .ZIP file.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainingDataView"></param>
        /// <param name="trainedModel"></param>
        private static void SaveModel(MLContext mlContext, IDataView trainingDataView, TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        /// <summary>
        /// Evaluate the model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="testDataView"></param>
        /// <param name="trainedModel"></param>
        /// <returns>Accuracy stats</returns>
        private static RegressionMetrics Evaluate(MLContext mlContext, IDataView testDataView, TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainedModel)
        {
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            return metrics;
        }

        /// <summary>
        /// Train the model fitting to the DataSet.
        /// The pipeline is trained on the dataset that has been loaded and transformed.
        /// </summary>
        /// <param name="trainingPipeline"></param>
        /// <param name="trainingDataView"></param>
        /// <returns></returns>
        private static TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> GetTrainedModel(EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline, IDataView trainingDataView)
        {
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            return trainedModel;
        }

        /// <summary>
        /// Set the training algorithm, then create and config the modelBuilder - Selected Trainer (SDCA Regression algorithm).
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataProcessPipeline"></param>
        /// <param name="trainer"></param>
        /// <param name="trainingPipeline"></param>
        private static void SetTrainingAlgorithm(MLContext mlContext, EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline, out SdcaRegressionTrainer trainer, out EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline)
        {
            trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            trainingPipeline = dataProcessPipeline.Append(trainer);
        }

        /// <summary>
        /// Common data process configuration with pipeline data transformations.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static EstimatorChain<ColumnConcatenatingTransformer> GetDataProcessPipeline(MLContext mlContext)
        {
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
            return dataProcessPipeline;
        }

        /// <summary>
        /// Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="baseTrainingDataView"></param>
        /// <returns></returns>
        private static IDataView GetTrainingDataView(MLContext mlContext, IDataView baseTrainingDataView)
        {
            var cnt = baseTrainingDataView.GetColumn<float>(nameof(AirTravel.AirFare)).Count();
            IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView,
                nameof(AirTravel.AirFare), lowerBound: 30, upperBound: 1400);
            var cnt2 = trainingDataView.GetColumn<float>(nameof(AirTravel.AirFare)).Count();
            return trainingDataView;
        }

        /// <summary>
        /// Make a single test prediction loding the model from .ZIP file.
        /// </summary>
        /// <param name="mlContext"></param>
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