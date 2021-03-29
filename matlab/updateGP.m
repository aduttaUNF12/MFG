function GPA = updateGP(GP,zB)
%function GPA = updateGP(GP,zB)
% Computes Gaussian posterior GPA for given observation vector
%   GP -- Gaussian prior (see generateGP.m)
%   zB -- a p-by-2 matrix with first column containing the indices of the 
%         components being observed and second column containing the 
%         respective observed values
% Calling this function when supplying only the first column of zB updates 
% only the posterior covariance. Components in zB that the GP has observed
% in a prior update are ignored.

n = size(GP.Value,1);
assert(all(zB(:,1)<=n&zB(:,1)>=1),'Invalid index in observation vector');
zB = sortrows(zB,1);

GPA.Param = GP.Param; GPA.Coord = GP.Coord;

A0 = find(isnan(GP.Value)); % Identify hidden components pre-update
[B,Iz,IB] = intersect(zB(:,1),A0); zB = zB(Iz,:); % Ignore observations seen before
[~,IA] = setdiff(A0,B); % Identify hidden components post-update
TauAB = mrdivide(GP.Sigma(IA,IB),GP.Sigma(IB,IB));
GPA.Sigma = GP.Sigma(IA,IA) - TauAB*GP.Sigma(IB,IA);

if size(zB,2) == 2
  GPA.Mu = GP.Mu(IA) + TauAB*(zB(:,2)-GP.Mu(IB));
  GPA.Value = GP.Value; GPA.Value(B) = zB(:,2); 
else
  GPA.Mu = GP.Mu(IA);
  GPA.Value = GP.Value; GPA.Value(B) = GP.Mu(IB);
end

% Connection to information-theoretic metrics and predictions
if 1
  n0 = length(A0);
  prioH = 0.5*(n0 + n0*log(2*pi) + log(det(GP.Sigma)));
  disp(['Prior Differential Entropy H(Z) = ' num2str(prioH) ' nats'])
  m0 = n0 - size(zB,1); 
  margH = 0.5*(m0 + m0*log(2*pi) + log(det(GP.Sigma(IA,IA))));
  disp(['Marginal Differential Entropy H(ZA) = ' num2str(margH) ' nats'])
  postH = 0.5*(m0 + m0*log(2*pi) + log(det(GPA.Sigma)));
  disp(['Posterior Differential Entropy H(ZA|ZB) = ' num2str(postH) ' nats'])
  disp(['--> Mutual Information I(ZA,ZB) = ' num2str(margH-postH) ' nats'])
  
%   subplot(2,2,1); plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
%   axis equal; xlabel('x'); ylabel('y'); 
%   title('Prior Prediction');
%   delta = diff(GP.Coord(1:2,1));
%   xMax = max(GP.Coord(:,1))+delta/2; yMax = max(GP.Coord(:,2))+delta/2; 
%   xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
%   ZHat = GP.Value; ZHat(isnan(ZHat)) = GP.Mu; ZHatBar = exp(ZHat) ./ (1+exp(ZHat));
%   for i = 1:length(ZHatBar)
%     theCol = [1 0 1] - [1 0 0]*2*max(0.5-ZHatBar(i),0) - [0 0 1]*2*max(ZHatBar(i)-0.5,0);
%     patch([GP.Coord(i,1)-delta/2,GP.Coord(i,1)-delta/2,GP.Coord(i,1)+delta/2,GP.Coord(i,1)+delta/2],...
%           [GP.Coord(i,2)-delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)-delta/2],theCol);
%     if ~isnan(GP.Value(i)), hold on; plot(GP.Coord(i,1),GP.Coord(i,2),'k.'); hold off; end
%   end
  %subplot(2,2,2); plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
%   axis equal; xlabel('x'); ylabel('y'); 
%   title('Posterior Prediction');
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %Visualization uses the following variables above, but via GP are
  %  M = GP.Param(1); N = GP.Param(2); sigma = GP.Param(3); ell = GP.Param(4); 
  %  xC = reshape(GP.Coord(:,1),M,N); yC = reshape(GP.Coord(:,2),M,N);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
%   ZHat = GPA.Value; ZHat(isnan(ZHat)) = GPA.Mu; ZHatBar = exp(ZHat) ./ (1+exp(ZHat));
%   for i = 1:length(ZHatBar)
%     theCol = [1 0 1] - [1 0 0]*2*max(0.5-ZHatBar(i),0) - [0 0 1]*2*max(ZHatBar(i)-0.5,0);
%     patch([GP.Coord(i,1)-delta/2,GP.Coord(i,1)-delta/2,GP.Coord(i,1)+delta/2,GP.Coord(i,1)+delta/2],...
%           [GP.Coord(i,2)-delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)-delta/2],theCol);
%     if ~isnan(GPA.Value(i)), hold on; plot(GP.Coord(i,1),GP.Coord(i,2),'k.'); hold off; end
%   end
  %iptsetpref('ImshowAxesVisible','on');
  %n = length(GP.Value); S0 = zeros(n,n);
  %A0 = find(isnan(GP.Value)); S0(A0,A0) = GP.Sigma;
  %subplot(2,2,3); imshow(S0,[]); colorbar; cAx1 = caxis;
  %S1 = zeros(n,n); A1 = find(isnan(GPA.Value)); S1(A1,A1) = GPA.Sigma; 
  %subplot(2,2,4); imshow(S1,[]); colorbar; cAx2 = caxis;
  %cAx = [min([cAx1(1) cAx2(1)]) max([cAx1(2) cAx2(2)])];
  %subplot(2,2,3); imshow(S0,cAx); title(['Prior Covariance: H(Z)=' num2str(prioH)]); colorbar;
  %subplot(2,2,4); imshow(S1,cAx); title(['Posterior Covariance: I(ZA,ZB)=' num2str(margH-postH)]); colorbar;
end
