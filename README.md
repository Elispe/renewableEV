# Towards the Decarbonization of the Mobility Sector: Promoting Renewable-Based Charging in Green Ride-Sharing

Python code that implements a Gauss-Seidel algorithm to promote renewable-based charging in a 100% -electrified fleet of vehicles for ride-hailing or ride-sharing services, as described in [1]. 

# Required Software

Python 3.8 or earlier: https://www.python.org/downloads/

cvxpy: https://www.cvxpy.org/install/

PuLP: https://pypi.org/project/PuLP/

# Data

Data for the ride requests can be downloaded from the Manhattan Taxi and Limousine Commission website https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page, selecting 2022 >> March >> Yellow Taxi Trip Records (PARQUET).

The .parquet file must be saved in the same folder as the main files, listed below, under the name <em>yellow_tripdata_2022-03.parquet</em>.

# Execution

The cases described in [1] are modeled by the main files below:

Business-as-usual case: <em>businessAsUsual_main.py</em>

Case 1 and Case 2: <em>rideSharing_main.py</em>

Fossil-fuel vehicles case: <em>fossilFuel_main.py</em>

The same folder that contains the main files above, should also contain:
<ul>
  <li>tripData.py: code to prepare the ride request data from the <em>yellow_tripdata_2022-03.parquet</em> file</li>
  <li>Request.py: request class</li>
  <li>Vehicle.py: vehicle class</li>
  <li>taxiZones.cvs: taxi zone lookup table</li>
  <li>PV_norm.cvs: PV generation data for sunny, cloudy morning, and cloudy afternoon scenarios</li>
</ul>

The results presented in [1] are obtained averaging over 100 iterations considering random initial SOC. 
To run 1 iteration, comment seed = int(sys.argv[1]) and set a seed instead, e.g. seed = 1.
Results will be automatically saved keeping track of the seed number. Change seed to obtain different initial SOC.

To repeat the experiments, for different weather conditions or initial SOC:

In <em>businessAsUsual_main.py</em>:

·        Lines 57-58: comment to set the initial SOC conditions to “fully charged”

In <em>rideSharing_main.py</em>:

·        Line 19: set to 0.75 for the case where the willingness to ride-share is 75%, or 1 for 100%, 0.5 for 50%, and 0.25 for 25%

·        Line 26: set fleet size
         
·        Lines 29-31: uncomment “sunny_1day”, “cloud_am_1day” or “cloud_pm_1day” to select the weather scenario and then update charge_req_prob based on the simulation length

·        Lines 73-74: comment to set the initial SOC conditions to “fully charged”

·        Lines 113-115: uncomment if simulation period exceeds 24 hours

·        Lines 156-160: uncomment to charge idling EVs at night during grid off-peak hours

# Results and Plots

In <em>calculateAvgRes.py</em>:

·        Line 4: set case_name = 'case1or2' or case_name = 'businessAsUsal'

·        Line 7: enter the number of iterations to average over

In <em>calculateAvgRes.py</em>:

·        Line 7-9: set case_name equal to 'case1' or 'case2' or 'businessAsUsal'

To average results and make plots run the following files in this order:
<ul>
  <li>calculateAvgRes.py: produces averaged results </li>
  <li>avgResPlotLocal.py: produces and saves plots</li>
</ul>



# Documentation

[1] E. Perotti, A. M. Ospina, G. Bianchin, A. Simonetto, and E. Dall’Anese, “Towards the Decarbonization of the Mobility Sector: Promoting Renewable-Based Charging in Green Ride-Sharing”. 
