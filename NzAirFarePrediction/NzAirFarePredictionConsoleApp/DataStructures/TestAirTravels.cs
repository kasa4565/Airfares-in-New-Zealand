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
            TravelDate = DateTime.ParseExact("18/12/2019", "dd/MM/yyyy", CultureInfo.InvariantCulture),
            DepartmentAirport = "ZQN",
            DepartmentTime = DateTime.ParseExact("9:35 AM", "h:mm tt", CultureInfo.InvariantCulture),
            ArrivalAirport = "WLG",
            ArrivalTime = DateTime.ParseExact("6:10 PM", "h:mm tt", CultureInfo.InvariantCulture),
            Duration = TimeSpan.ParseExact("8h 35m", "h\\h\\ mm\\m", CultureInfo.InvariantCulture),
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
            TravelDate = DateTime.ParseExact("18/12/2019", "dd/MM/yyyy", CultureInfo.InvariantCulture),
            DepartmentAirport = "ZQN",
            DepartmentTime = DateTime.ParseExact("10:20 AM", "h:mm tt", CultureInfo.InvariantCulture),
            ArrivalAirport = "WLG",
            ArrivalTime = DateTime.ParseExact("6:40 PM", "h:mm tt", CultureInfo.InvariantCulture),
            Duration = TimeSpan.ParseExact("8h 20m", "h\\h\\ mm\\m", CultureInfo.InvariantCulture),
            Direct = "(1 stop)",
            Transit = "5h 20m in AKL",
            Baggage = "",
            Airline = "Air New Zealand",
            AirFare = 422
        };
    }
}