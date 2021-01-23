using System;
using Microsoft.ML.Data;

namespace Regression_TaxiFarePrediction.DataStructures
{
    public class AirTravel
    {
        [LoadColumn(0)]
        public DateTime TravelDate;
        
        
        [LoadColumn(1)]
        public string DepartmentAirport;
        
        
        [LoadColumn(2)]
        public DateTime DepartmentTime;
        
        
        [LoadColumn(3)]
        public string ArrivalAirport;
        
        
        [LoadColumn(4)]
        public DateTime ArrivalTime;
        
        
        [LoadColumn(5)]
        public TimeSpan Duration;
        
        
        [LoadColumn(6)]
        public string Direct;
        
        
        [LoadColumn(7)]
        public string Transit;
        
        
        [LoadColumn(8)]
        public string Baggage;
        
        
        [LoadColumn(9)]
        public string Airline;

        [LoadColumn(10)]
        public float AirFare;
    }
}