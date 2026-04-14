%Input Parameters

%Control Parameters for generating EV load profile 
Total_EVs=100; % Total number of EVs in circulation in the region
Pen_Level=65; % Penetration level, Number of EVs charging consuming grid power

%For Charging profile in Resedential Sector
TimeIntervals=[0,3,7,11,14,17,19,21,24];%Hrs of the day
CarDist=round([6,0,5,7,17,26,22,17]*Pen_Level*Total_EVs/(100*100));% Number of cars during the above intervals of the day. CAN BE p.u FOR ANY NUMBER OF CARS

timeScale=60;% 60 means, timeslot size is one second, and 60/15 means 15 mins size of timeslot
timeHorizon=24; % Daily charge load profile 24 hours
timeSlots_t=timeHorizon*timeScale*60; %Time slots in the time horizon

BC=[22,32,40,60]; % in KWh , Battery capacities range  

chargingPower=7;%kW Charging Station capacity
chargers_count=10;% total number of charging stations
max_wait=15*60; % maximum waiting time of the EV to wait for its turn to charge

GenerateRandomSchedule_new_ScenerioGenerator(TimeIntervals,CarDist,timeHorizon,timeSlots_t,BC,chargingPower,chargers_count,max_wait);
run('PlotBox_ScenerioGenerator.m');