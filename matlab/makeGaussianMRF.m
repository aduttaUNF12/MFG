function makeGaussianMRF(filename,M,N,sigma,ell,mumix)
%function makeGaussianMRF(FILENAME,M,N,SIGMA,LENGTH,MUMIX)
% Input FILENAME is always necessary, which stores (or will store) all 
% variables that define the specified random field. These are
%   M = number of cells in the vertical dimension
%   N = number of cells in the horizontal dimension
% All cells are square, sized and positioned so that any square grid 
% occupies the 2-d region [0,1]x[0,1]; non-square grids are scaled so that
% the dimension with the greatest number of (still-square) cells occupies 
% [0,1] and the other dimension occupies [0,f] with f proportionally in 
% (0,1) based on the ratio of M and N. The specific 2-D point locations of 
% all cell centers are defined by
%   xC = a length-M row vector of locations in the vertical direction 
%   yC = a length-N row vector of locations in the horizontal direction
% These resulting n = N*M jointly Gaussian random variables are
% defined by an n-by-1 mean vector Mu and an n-by-n covariance matrix S.
%
% Inputs M and N are necessary only if generating a specific random field 
% for the first time, which means FILENAME.mat will not yet exist. Inputs 
% SIGMA & LENGTH are optional (with default values of 1) and denote the 
% variance & length parameters in the spatially-isotropic exponential 
% kernel function defining the covariance between any two points iPos = 
% [xC(i); yC(i)] and jPos = [xC(j); yC(j)] according to: 
%            S(i,j) = SIGMA^2*exp(-norm(iPos-jPos)/LENGTH)
% Input MUMIX is also optional (with default value of []) and denotes the 
% Gaussian mixture parameters to construct a non-zero mean-field (i.e., 
% mean vector Mu). Specifically, each r'th row of MUMIX corresponds to a 
% component bivariate Gaussian PDF over the 2-D region
%   -- centered at mean vector MUMIX(r,2:3)';
%   -- rotated by covariance matrix [MUMIX(r,4), MUMIX(r,6); MUMIX(r,6), MUMIX(r,5)]; and 
%   -- weighted by MUMIX(r,1)
% The mean field becomes the sum of these weighted bell curves.
%
% The function will always generate and visualize at least one new sample 
% from the process, accepting user input on whether to generate a next
% sample. A set of D generated samples is saved in variable X within 
% FILENAME.mat as a 3-dimensional array of dimensions M, N and D.
%
% Visualizing each d'th sample outside this function is accomplished by the
% commands
% >> load FILENAME.mat; 
% >> surf(yC,xC,X(:,:,d),'EdgeColor','none'); view([0 90]);
% >> xlabel(['y (in ' num2str(N) ' cells)']); 
% >> ylabel(['x (in ' num2str(M) ' cells)']);
% >> title(['Sample ' num2str(d) ' from ' num2str(M) '-by-' num2str(N) ' MRF']);
% >> axis equal; axis([0 yMax 0 xMax]);
% >> set(gca,'XTick',[0 yMax],'YTick',[0 xMax]); colorbar;
% It should be noted that MATLAB's surf command will interpret the positions
% as the patch boundaries, not the cell centers as in the GP. A final remark 
% concerns the indexing conventions into the spatial coordinates of each 
% cell relative to MATLAB's axes coordinates. Vertical coordinate xC(k) 
% increases as index k increases (moving from bottom to top along MATLAB's 
% y-axis), and similarly for horizontal coordinate yC(m) as index m increases
% (moving from left to right along MATLAB's x-axis). The samples associated 
% with vertical-horizontal position [xC(k),yC(m)] are X(k,m,:), which via 
% MATLAB's plot3 command is marked as a x-y-z point of
% >> hold on; plot3(yC(m),xC(k),X(k,m,d),'x'); hold off;

if alreadyProcessed(filename)
  disp('Loading previously constructed Gaussian process...');
  eval(['load ' filename '.mat;']); 
  n = size(S,1);
else
  if nargin < 6, mumix = zeros(0,6); end
  if nargin < 5, ell = 1; end
  if nargin < 4, sigma = 1; end
  if M > N
    xMax = 1; yMax = N/M;
  else
    yMax = 1; xMax = M/N;
  end
  n = M*N;
  xC = linspace(0,xMax,M+1); xC = xC(1:end-1) + diff(xC(1:2))/2;
  yC = linspace(0,yMax,N+1); yC = yC(1:end-1) + diff(yC(1:2))/2;
  disp(['Constructing covariance matrix of length-' num2str(n) ' Gaussian process...']);
  S = eye(n)*sigma^2;
  if ell > 0 && sigma > 0
    for i = 1:n
      x = 1 + floor((i-1)/N); y = i - (x-1)*N; iPos = [xC(x); yC(y)];
      for j = i:n
        x = 1 + floor((j-1)/N); y = j - (x-1)*N; jPos = [xC(x); yC(y)];
        S(i,j) = sigma^2*exp(-norm(iPos-jPos)/ell); S(j,i) = S(i,j);
%        S(i,j) = sigma^2*exp(-0.5*(iPos-jPos)'*(iPos-jPos)/ell^2); S(j,i) = S(i,j);
      end
    end
  end
  disp(['Constructing mean field of length-' num2str(n) ' Gaussian process...']);
  Mu = zeros(n,1);
  for c = 1:size(mumix,1)
    wc = mumix(c,1); mc = mumix(c,2:3)'; s12 = prod(mumix(c,4:6)); 
    Kc = inv(diag(mumix(c,4:5).^2) + s12*fliplr(eye(2)));
    for i = 1:n
      x = 1 + floor((i-1)/N); y = i - (x-1)*N; iPos = [xC(x); yC(y)];
      Mu(i) = Mu(i) + wc*exp(-0.5*(iPos-mc)'*Kc*(iPos-mc));
    end
  end

  X = nan(M,N,0); 
  eval(['save ' filename '.mat M N xMax yMax sigma ell mumix xC yC S Mu X;']);
end

if sigma > 0, R = chol(S); else, R = zeros(size(S)); end; option = 'n';
while option == 'n'
  X(:,:,end+1) = reshape(randn(1,n)*R + Mu',N,M)'; k = size(X,3);
  subplot(2,2,[1 2]); surf(yC,xC,X(:,:,k),'EdgeColor','none'); view([0 90]);
  xlabel(['y (in ' num2str(N) ' cells)']); ylabel(['x (in ' num2str(M) ' cells)']);
  title('Realization'); axis equal; axis([0 yMax 0 xMax]); 
  set(gca,'XTick',[0 yMax],'YTick',[0 xMax]); colorbar;
  subplot(2,2,4); surf(yC,xC,X(:,:,k)-reshape(Mu',N,M)','EdgeColor','none'); view([0 90]);
  xlabel(['y (in ' num2str(N) ' cells)']); ylabel(['x (in ' num2str(M) ' cells)']);
  title('Deviation Field'); axis equal; axis([0 yMax 0 xMax]); 
  set(gca,'XTick',[0 yMax],'YTick',[0 xMax]); colorbar;
  subplot(2,2,3); surf(yC,xC,reshape(Mu',N,M)','EdgeColor','none'); view([0 90]);
  xlabel(['y (in ' num2str(N) ' cells)']); ylabel(['x (in ' num2str(M) ' cells)']);
  title('Mean Field'); axis equal; axis([0 yMax 0 xMax]); 
  set(gca,'XTick',[0 yMax],'YTick',[0 xMax]); colorbar;
  if k == 1, kStr = '1st';
  elseif k == 2, kStr = '2nd';
  elseif k == 3, kStr = '3rd';
  else kStr = [num2str(k) 'th'];
  end
  option = input(['Generated the ' kStr ' sample...shall we stop? [y,n] '],'s');
end
eval(['save ' filename '.mat X -append;']);

end

function flag = alreadyProcessed(filename)

flag = 0; fid = fopen([filename '.mat'],'r');
if fid ~= -1, fclose(fid); flag = 1; end 

end