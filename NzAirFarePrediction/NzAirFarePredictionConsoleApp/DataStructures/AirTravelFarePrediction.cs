using Microsoft.ML.Data;

namespace NzAirFarePrediction.DataStructures
{
    public class AirTravelFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}