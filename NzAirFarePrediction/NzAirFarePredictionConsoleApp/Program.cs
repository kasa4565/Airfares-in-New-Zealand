using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using NzAirFarePrediction.DataStructures;
using Microsoft.ML.Trainers;
using System.Collections.Generic;

namespace NzAirFarePrediction
{
    internal static class Program
    {

        #region paths

        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/nz-airfares-train.csv";
        private static readonly string TestDataRelativePath = $"{BaseDatasetsRelativePath}/nz-airfares-test.csv";

        private static readonly string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static readonly string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/AirTravelFareModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

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
            DoSamplePredictions(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static void DoSamplePredictions(MLContext mlContext)
        {
            var samples = GetSamples();

            foreach (var sample in samples)
            {
                TestSinglePrediction(mlContext, sample);
            }
        }

        private static IEnumerable<AirTravel> GetSamples()
        {
            var samples = new List<AirTravel>();

            samples.Add(TestAirTravels.Travel1);
            samples.Add(TestAirTravels.Travel2);
            samples.Add(TestAirTravels.Travel3);

            return samples;
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
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TravelDateEncoded",
                                inputColumnName: nameof(AirTravel.TravelDate)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DepartmentAirportEncoded",
                                inputColumnName: nameof(AirTravel.DepartmentAirport)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DepartmentTimeEncoded",
                                inputColumnName: nameof(AirTravel.DepartmentTime)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ArrivalAirportEncoded",
                                inputColumnName: nameof(AirTravel.ArrivalAirport)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ArrivalTimeEncoded",
                                inputColumnName: nameof(AirTravel.ArrivalTime)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DurationEncoded",
                                inputColumnName: nameof(AirTravel.Duration)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DirectEncoded",
                                inputColumnName: nameof(AirTravel.Direct)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TransitEncoded",
                                inputColumnName: nameof(AirTravel.Transit)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "BaggageEncoded",
                                inputColumnName: nameof(AirTravel.Baggage)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AirlineEncoded",
                                inputColumnName: nameof(AirTravel.Airline)))
                            .Append(mlContext.Transforms.Concatenate("Features", "TravelDateEncoded", "DepartmentAirportEncoded", "DepartmentTimeEncoded",
                                "ArrivalAirportEncoded", "ArrivalTimeEncoded", "DurationEncoded", "DirectEncoded", "TransitEncoded", "BaggageEncoded", "AirlineEncoded"));

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
        /// Create prediction engine related to the loaded trained model.
        /// </summary>
        /// <param name="mlContext"></param>
        private static void TestSinglePrediction(MLContext mlContext, AirTravel sample)
        {
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<AirTravel, AirTravelFarePrediction>(trainedModel);
            var resultprediction = predEngine.Predict(sample);

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