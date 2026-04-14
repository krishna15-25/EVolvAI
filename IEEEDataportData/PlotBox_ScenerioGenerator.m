fig = figure();
clf;
hold on;

temp=load('randomEVschedule.mat');
x= round((1:length(temp.data))/3600,2);
y= (temp.data)*7;
yyaxis right
plot(x,y,'Color',[0.5,0.5,0.5],'LineWidth',1);
legend({'Total EVs Charging Consumption'});


tmp = load('randomEVschedule.mat');
EV = tmp.EV;
CarCount = length(EV);
Car1 = 1;
Car2 = CarCount;

% Draw charging durations of Individual EVs
for CarIndex = Car1 : Car2
	ts	= EV(CarIndex).startTime;
	te	= EV(CarIndex).endTime;
	x = round([ts te]/3600,2);
	y = [CarIndex CarIndex];
	yyaxis left
	plot(x, y,'-','Color', [0.47,0.67,0.19], 'LineWidth',2);
	
end
hold off;
legend({'Individual EV Charging Time'},'Location','northwest');

ax = gca;
ax.XTick = (0:2:24)';
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';
yyaxis left
axis([0 24 0 70]);






