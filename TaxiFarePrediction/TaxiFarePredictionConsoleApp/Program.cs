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
using Regression_TaxiFarePrediction.DataStructures;
using static Microsoft.ML.Transforms.NormalizingEstimator;

namespace Regression_TaxiFarePrediction
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

            // Paint regression distribution chart for a number of elements read from a Test DataSet file
            // PlotRegressionChart(mlContext, TestDataPath, 100, args);
            // TODO: Fix the error related to the exception

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

        private static void PlotRegressionChart(MLContext mlContext,
            string testDataSetPath,
            int numberOfRecordsToRead,
            string[] args)
        {
            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            // Create prediction engine related to the loaded trained model
            var predFunction = mlContext.Model.CreatePredictionEngine<AirTravel, AirTravelFarePrediction>(trainedModel);

            string chartFileName = "";
            using (var pl = new PLStream())
            {
                // use SVG backend and write to SineWaves.svg in current directory
                if (args.Length == 1 && args[0] == "svg")
                {
                    pl.sdev("svg");
                    chartFileName = "AirRegressionDistribution.svg";
                    pl.sfnam(chartFileName);
                }
                else
                {
                    pl.sdev("pngcairo");
                    chartFileName = "AirRegressionDistribution.png";
                    pl.sfnam(chartFileName);
                }

                // use white background with black foreground
                pl.spal0("cmap0_alternate.pal");

                // Initialize plplot
                pl.init();

                // set axis limits
                const int xMinLimit = 0;
                const int xMaxLimit = 35; //Rides larger than $35 are not shown in the chart
                const int yMinLimit = 0;
                const int yMaxLimit = 35; //Rides larger than $35 are not shown in the chart
                pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

                // Set scaling for mail title text 125% size of default
                pl.schr(0, 1.25);

                // The main title
                pl.lab("Measured", "Predicted", "Distribution of Air Travel Fare Prediction");

                // plot using different colors
                // see http://plplot.sourceforge.net/examples.php?demo=02 for palette indices
                pl.col0(1);

                int totalNumber = numberOfRecordsToRead;
                var testData = new TaxiTripCsvReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();

                //This code is the symbol to paint
                char code = (char) 9;

                // plot using other color
                //pl.col0(9); //Light Green
                //pl.col0(4); //Red
                pl.col0(2); //Blue

                double yTotal = 0;
                double xTotal = 0;
                double xyMultiTotal = 0;
                double xSquareTotal = 0;

                for (int i = 0; i < testData.Count; i++)
                {
                    var x = new double[1];
                    var y = new double[1];

                    //Make Prediction
                    var FarePrediction = predFunction.Predict(testData[i]);

                    x[0] = testData[i].AirFare;
                    y[0] = FarePrediction.FareAmount;

                    //Paint a dot
                    pl.poin(x, y, code);

                    xTotal += x[0];
                    yTotal += y[0];

                    double multi = x[0] * y[0];
                    xyMultiTotal += multi;

                    double xSquare = x[0] * x[0];
                    xSquareTotal += xSquare;

                    double ySquare = y[0] * y[0];

                    Console.WriteLine($"-------------------------------------------------");
                    Console.WriteLine($"Predicted : {FarePrediction.FareAmount}");
                    Console.WriteLine($"Actual:    {testData[i].AirFare}");
                    Console.WriteLine($"-------------------------------------------------");
                }

                // Regression Line calculation explanation:
                // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

                double minY = yTotal / totalNumber;
                double minX = xTotal / totalNumber;
                double minXY = xyMultiTotal / totalNumber;
                double minXsquare = xSquareTotal / totalNumber;

                double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);

                double b = minY - (m * minX);

                //Generic function for Y for the regression line
                // y = (m * x) + b;

                double x1 = 1;
                //Function for Y1 in the line
                double y1 = (m * x1) + b;

                double x2 = 39;
                //Function for Y2 in the line
                double y2 = (m * x2) + b;

                var xArray = new double[2];
                var yArray = new double[2];
                xArray[0] = x1;
                yArray[0] = y1;
                xArray[1] = x2;
                yArray[1] = y2;

                pl.col0(4);
                pl.line(xArray, yArray);

                // end page (writes output to disk)
                pl.eop();

                // output version of PLplot
                pl.gver(out var verText);
                Console.WriteLine("PLplot version " + verText);
            } // the pl object is disposed here

            // Open Chart File In Microsoft Photos App (Or default app, like browser for .svg)

            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }

    public class TaxiTripCsvReader
    {
        public IEnumerable<AirTravel> GetDataFromCsv(string dataLocation, int numMaxRecords)
        {
            IEnumerable<AirTravel> records =
                File.ReadAllLines(dataLocation)
                    .Skip(1)
                    .Select(x => x.Split(','))
                    .Select(x => new AirTravel
                    {
                        TravelDate = DateTime.ParseExact(x[0], "dd/MM/yyyy", CultureInfo.InvariantCulture),
                        DepartmentAirport = x[1],
                        DepartmentTime = DateTime.ParseExact(x[2], "h:mm tt", CultureInfo.InvariantCulture),
                        ArrivalAirport = x[3],
                        ArrivalTime = DateTime.ParseExact(x[4], "h:mm tt", CultureInfo.InvariantCulture),
                        Duration = TimeSpan.ParseExact(x[5], "h\\h\\ mm\\m", CultureInfo.InvariantCulture),
                        Direct = x[6],
                        Transit = x[7],
                        Baggage = x[8],
                        Airline = x[9],
                        AirFare = float.Parse(x[10])
                    })
                    .Take<AirTravel>(numMaxRecords);

            return records;
        }
    }
}