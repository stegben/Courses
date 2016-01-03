%% this function plot the EMD of the data
%% 
%% type the following in command line interface:
%%      plotOrigData(a , b)  
%% then a figure of the signal chosen will jump out

%% a is the No. of video, ranging from 1~40

%% b is different channels of bio signal:
%% b	biosignal
%% 1	zEMG (Zygomaticus Major EMG, zEMG1 - zEMG2)
%% 2	tEMG (Trapezius EMG, tEMG1 - tEMG2)
%% 3	GSR (values from Twente converted to Geneva format (Ohm))
%% 4	Respiration belt
%% 5	Plethysmograph
%% 6	Temperature


function y = plotIMF( video , channel)
	data = evalin('base' , 'imfs');
	imfNumber = length(data{1,1}(1,:));
	
	figure(1); % separate imfs
	for k=1:imfNumber
		subplot(imfNumber,1,k);
		temp=data{video,channel};
		plot( temp(:,k) );
	end
	
	figure(2); % all imf in one plot
	plot(data{video,channel});
end