using System;
using System.Globalization;

namespace NzAirFarePrediction.DataStructures
{
    public class TestAirTravels
    {
        /// <summary>
        /// Test: 18/12/2019,ZQN,9:35 AM,WLG,6:10 PM,8h 35m,(1 stop),5h 35m in AKL,,Air New Zealand,422
        /// </summary>
        internal static readonly AirTravel Travel1 = new AirTravel
        {
            TravelDate = "18/12/2019",
            DepartmentAirport = "ZQN",
            DepartmentTime = "9:35 AM",
            ArrivalAirport = "WLG",
            ArrivalTime = "6:10 PM",
            Duration = "8h 35m",
            Direct = "(1 stop)",
            Transit = "5h 35m in AKL",
            Baggage = "",
            Airline = "Air New Zealand,",
            AirFare = 422
        };

        /// <summary>
        /// Test: 18/12/2019,ZQN,10:20 AM,WLG,6:40 PM,8h 20m,(1 stop),5h 20m in AKL,,Air New Zealand,422
        /// </summary>
        internal static readonly AirTravel Travel2 = new AirTravel
        {
            TravelDate = "18/12/2019",
            DepartmentAirport = "ZQN",
            DepartmentTime = "10:20 AM",
            ArrivalAirport = "WLG",
            ArrivalTime = "6:40 PM",
            Duration = "8h 20m",
            Direct = "(1 stop)",
            Transit = "5h 20m in AKL",
            Baggage = "",
            Airline = "Air New Zealand",
            AirFare = 422
        };
    }
}