%% Author: Benjamin Chu, Kevin Huang, Racoon Wu, Jocelyn Hsieh

%% Project conductor: Homer Chen, Ding Ding

%% BioExperiment Final Project

%% this m file runs the experiment


%---------setting input format
X1 = [ones(40,1) maxResp];
X2 = [ones(40,1) maxResp];

%---------start Linear Regression
[aroCoe cin r] = regress(arousal , X1);
[valCoe cin r] = regress(valence , X2);

%---------visualize linear result
