function Z = sampleGP(GP)
%function Z = sampleGP(GP)
% Generates a sample Z from a Gaussian process GP (see generateGP.m)

Z = GP.Value; A = find(isnan(Z)); n = length(A); % Identify hidden components
Z(A) = GP.Mu + chol(GP.Sigma)*randn(n,1);

% if 1
%   ldSig = log(det(GP.Sigma)); 
%   llZ = -0.5*(ldSig + mrdivide((Z(A)-GP.Mu)',GP.Sigma)*(Z(A)-GP.Mu) + n*log(2*pi));
%   llM = -0.5*(ldSig + n*log(2*pi));
%   disp(['Log-likelihood of Z = ' num2str(llZ) ' (relative to Mu of ' num2str(llM) ')'])
% 
%   subplot(1,1,1); plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
%   axis equal; xlabel('x'); ylabel('y'); 
%   title(['Sample: L(Z)=' num2str(llZ) ' (while L(\mu)=' num2str(llM) ')']);
%   delta = diff(GP.Coord(1:2,1));
%   xMax = max(GP.Coord(:,1))+delta/2; yMax = max(GP.Coord(:,2))+delta/2; 
%   xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
%   ZBar = exp(Z) ./ (1+exp(Z));
%   for i = 1:length(ZBar)
%     theCol = [1 0 1] - [1 0 0]*2*max(0.5-ZBar(i),0) - [0 0 1]*2*max(ZBar(i)-0.5,0);
%     patch([GP.Coord(i,1)-delta/2,GP.Coord(i,1)-delta/2,GP.Coord(i,1)+delta/2,GP.Coord(i,1)+delta/2],...
%           [GP.Coord(i,2)-delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)-delta/2],theCol);
%     if ~isnan(GP.Value(i)), hold on; plot(GP.Coord(i,1),GP.Coord(i,2),'k.'); hold off; end
%   end
% end
