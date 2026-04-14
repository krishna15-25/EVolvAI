function GenerateRandomSchedule_new_ScenerioGenerator(TimeIntervals,CarDist,timeHorizon,timeSlots_t,BC,chargingPower,chargers_count,max_wait)

carNumber=0;%for keeping the count of the cars used for indexing the cars.
IntervalCount = length(TimeIntervals) - 1;
startTime = zeros(IntervalCount,1);
endTime = zeros(IntervalCount,1);	

exp_data_each=zeros(sum(CarDist),timeSlots_t); % to save the expected charging profile of each EV

for i=1:length(CarDist)% loop for distributing the cars during each interval.
	startTime(i)=TimeIntervals(i)*4;%% scaling it to the timeslots of 15 min, i.e. 1hr is of 4 timeslots. therefore 96 timeslots over the day. But have been updated to each second in the later part of the code.
	endTime(i)=(TimeIntervals(i+1)*4)-1;
	timeSlots=endTime(i)-startTime(i)+1;
	numCars=CarDist(i);%total number of cars to be distributed during the interval
	
	lb=0;% my upper and lower limits for random numbers to be generated within this range
	ub=5;
	if numCars==0
		numberOfCars=zeros(timeSlots);
	else
		if ub>numCars 
			ub=numCars-1;
		end
		%%finds timeSlots random ints in [lb,ub] with sum of numCars
		hit = 0;
		while ~hit
			total=0;
			count = 0;
			numberOfCars = [];
			while total < numCars && count < timeSlots
				r = randi([lb,ub]);
				total = total+r;
				count = count+1;
				numberOfCars=[numberOfCars;r];
				if total == numCars && count == timeSlots
					hit = 1;
				end
			end
		end
	end
	
	Index=0;
	for timeIndex=startTime(i)+1:endTime(i)+1 %%for each interval save the data for each car i.e. starting, ending of its charging period.
		Index=timeIndex-startTime(i);%% for renewing the count of timeslots within the interval as 1 2 3..
		
		if (numberOfCars(Index)~=0) %% there could be no car at some times.
			for carIndex=1:numberOfCars(Index)
				carNumber=carNumber+1; %%keep the total count of the cars.
				EV(carNumber).number=carNumber;
				EV_startTime=timeIndex*15*60+ceil(rand()*15*60);%converting the start time in second scale, randomly at any second it may approach rather than only at the multiples of 15.
				EV_startTime(EV_startTime>timeSlots_t)=timeSlots_t;
				
				%%% (Battery Characteristics of EV) Choosing battery capacity, SOCmin, SOCmax
				
				EV(carNumber).BC=BC(randi(length(BC)));% will use rand to generate any number from the given list, while there exist 12kwh to 100kwh battery capcities depending on car type
				EV(carNumber).SOCact=randi([5,80],1,1);
				SOCact=EV(carNumber).SOCact;minchar=5;% 10 percent minimum charging
				EV(carNumber).SOCmax=randi([SOCact+minchar,100],1,1);% will use rand to generate any number between given limit
				SOCmax=EV(carNumber).SOCmax;
				
				%%%calculate continuous charging duration now
				EV(carNumber).powerConsumed=chargingPower;%in kW
				EV(carNumber).Esocmax=(SOCmax-SOCact)*(EV(carNumber).BC)*3600/100;% Energy calculation from battery capacity and required SoC in kWs
				EV(carNumber).tsocmax=fix(EV(carNumber).Esocmax/(EV(carNumber).powerConsumed));% time duration ('CONTINUOUS') required to charge till SOCmin, hours to seconds conversion
				
				EV(carNumber).startTime=EV_startTime;
				EV(carNumber).endTime=EV_startTime+EV(carNumber).tsocmax; %% Total time required to charge EV to its max SOC
				
				EV(carNumber).expected_duration=EV(carNumber).tsocmax; %% Expected charging duration after scheduling
				EV(carNumber).real_duration=EV(carNumber).tsocmax; %% Actually charging duration after scheduling
				EV(carNumber).powerConsumed=chargingPower;
			end
		end
	end
end

[~,b]=sort([EV.startTime].');
EV=EV(b);
save('randomEVschedule.mat', 'EV');

EV_count = sum(CarDist);
data = zeros (1,timeSlots_t);
for i =1:EV_count
	wait=1;counter=0;
	duration =EV(i).real_duration;
	time_start=EV(i).startTime;
	while(data(1,time_start) == chargers_count)%% checking if all the chargers are occupied then wait
		if counter>(max_wait)%% checking if the car has to wait for more than 15 mins then car leaves
			EV(i).endTime=EV(i).startTime+max_wait;%% keep the record of leaving cars (arrival time and leaving time without charging)
			EV(i).real_duration=0;
			EV(i).powerConsumed=[];
			wait=0;
			break
		else
			if (time_start+1>=timeSlots_t)%Keeping the time horizon constraint as defined, EVs whose charging activity lying in this time are considered 
				
				EV(i).endTime=timeSlots_t;%% keep the record of leaving cars (arrival time and leaving time without charging)
				EV(i).real_duration=0;
				EV(i).powerConsumed=[];
				wait=0;
			else
				time_start=time_start+1;
				counter=counter+1;
			end
		end
	end
	if wait==1
		time_end = time_start+duration-1;
		time_end(time_end>timeSlots_t)=timeSlots_t;
		EV(i).startTime=time_start;
		EV(i).endTime=time_end;
		data(1,time_start:time_end)=data(1,time_start:time_end)+1;
		exp_data_each(i,time_start:time_end)=chargingPower; %%% to plot each EV charging
	end
end

%%% Generated schedule saved in a mat file named below
save('randomEVschedule.mat', 'EV','data','exp_data_each');

slot=0.5;
samples=timeHorizon/slot;
I= [EV.real_duration].' > 0 ;
startTime=[EV.startTime];
endTime=[EV.endTime];
c= fix(startTime(I)'/(timeSlots_t/samples));
d=fix(endTime(I)'/(timeSlots_t/samples));
y=zeros(samples,1);
z=zeros(samples,1);
for j=1:1:samples
	for i=1:1:length(c)
		if c(i)==j
			y(j)=y(j)+1;
		end
		if d(i)==j
			z(j)=z(j)+1;
		end
	end
end
x= 0:slot:timeHorizon-slot;
figure();
bar(x,y,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
ylim([0,10]);
xlabel('Arrival Time (Day hours)');
ylabel('Number of EVs');


hold;
figure();
bar(x,z,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
ylim([0,10]);
xlabel('Departure Time (Day hours)');
ylabel('Number of EVs');


s= seconds([EV.real_duration].');
s.Format='hh:mm:ss';
q=1:1:sum(CarDist);
figure();
bar(q,s,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
xlabel('Electric Vehicles');
ylabel('Charging Duration in hh:mm:ss');

soc=([EV.SOCmax].'-[EV.SOCact].');
figure();
bar(q,soc,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
xlabel('Electric Vehicles');
ylabel('SOC demand in %');

period=15;
d1= 1+fix([EV.real_duration].'/(period*60));
a1=period:period:120;
y1=histcounts(d1,8);
figure;
bar(a1,y1,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
xlabel('Charging Duration in minutes');
ylabel('Number of EVs');
[y1_sorted,i_sort]=sort(y1,'descend');
figure;
bar(y1_sorted,'FaceColor',[0.50,0.50,0.50],'EdgeColor',[0.50,0.50,0.50]);
xticks(1:120/period);
for i=1:120/period
	xLabels{i}=strcat(num2str((i-1)*period),' -> ',num2str(i*period));
end
xticklabels(xLabels(i_sort));
xtickangle(45)
xlabel('Range of charging times of EVs in minutes');
ylabel('Number of EVs');
end

