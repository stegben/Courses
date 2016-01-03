%% this m file is aim to create several feature vectors


%-----------output---------
valence   = labels(:,1);
arousal   = labels(:,2);
dominance = labels(:,3);
liking    = labels(:,4);

%-----------input----------
index = 1:40;
% zEMG

% tEMG

% GSR

% Respiration belt
maxResp = max(origData(:,4,:),[],3 );
% Plethysmorgraph

% Tempture
