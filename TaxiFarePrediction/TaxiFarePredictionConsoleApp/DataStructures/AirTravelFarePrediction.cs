using Microsoft.ML.Data;

namespace Regression_TaxiFarePrediction.DataStructures
{
    public class AirTravelFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}