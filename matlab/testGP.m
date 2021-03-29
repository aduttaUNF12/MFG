% testGP.m
clear all;

%figure(1); GP = generateGP(5,4,1,1);
%figure(1); GP = generateGP(5,4,1,5);
%figure(1); GP = generateGP(5,4,1,0.2);
%figure(1); GP = generateGP(10,8,1,1); 
%figure(1); GP = generateGP(15,12,1,1);
%figure(1); 
GP = generateGP(12,12,1,0.5); % Info-theoretic metrics crash...
%figure(1); GP = generateGP(40,32,1,1); % Info-theoretic metrics crash...

%figure(2); 
Z = sampleGP(GP);

obs = randperm(length(Z),floor(length(Z))); % Observe half the cells
RMSE = nan(length(obs)+1,1);
ZHat = GP.Value; ZHat(isnan(ZHat)) = GP.Mu; RMSE(1) = norm(Z-ZHat);
for k = 1:length(obs)
  %figure(3); 
  GP = updateGP(GP,[obs(k) Z(obs(k))]);
  ZHat = GP.Value; ZHat(isnan(ZHat)) = GP.Mu; RMSE(k+1) = norm(Z-ZHat);
  pause(1);
end
%figure(4); stairs(0:length(obs),RMSE);
xlabel('number of observations'); ylabel('RMSE'); 
title('Prediction Error vs. Number of Observations');

