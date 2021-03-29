function GP = generateGP(M,N,sigma,ell)
%function GP = generateGP(M,N,sigma,ell)
% Returns a structure GP with fields
%   Param % a 1-by-4 row vector of the input arguments [M,N,sigma,ell],
%         % where M is the number of cells in the horizontal dimension and 
%         % N is the number of cells in the vertical dimension
%   Coord % an (M*N)-by-2 matrix of spatial coordinates (x,y) corresponding 
%         % to the cell centers; perfectly square grids occupy the 2-d 
%         % region [0,1]x[0,1], while non-square grids force the dimension 
%         % with the greater number of cells to occupy [0,1] and the other
%         % dimension to occupy [0,f] for some f in (0,1) in proportion to 
%         % the ratio of M and N.
%   Sigma % an (M*N)-by-(M*N) covariance matrix with pairwise covariances 
%         % defined by the exponential kernel function of sigma and ell
%   Mu    % a length-(M*N) vector with component means, initialized as zero
%   Value % a length-(M*N) vector of known cell values, initialized as NaN

% Set default values
if nargin < 4, ell = 1; end
if nargin < 3, sigma = 1; end
if nargin < 2, N = M; end

assert(M==floor(M)&&M>0&&N==floor(N)&&N>0,'Number of cells must be positive integer');
assert(sigma>0,'Per-cell variance must be positive');
assert(ell>=0,'Cross-cell correlation must be non-negative');

% Assign spatial coordinates (of cell centers) to the cell grid
if M > N % Non-square area-of-interest
  xMax = 1; yMax = N/M;
else
  yMax = 1; xMax = M/N;
end
xC = linspace(0,xMax,M+1); delta = diff(xC(1:2)); xC = xC(2:end) - delta/2; 
yC = linspace(0,yMax,N+1); yC = yC(2:end) - delta/2;
[xC,yC] = meshgrid(xC,yC); xC = xC'; yC = yC';

% Compute covariance matrix using exponential kernel function
n = M*N; Sigma = eye(n)*sigma^2;
if ell > 0
  disp(['Constructing covariance matrix of length-' num2str(n) ' Gaussian process...']);
  for i = 1:n
    iPos = [xC(i); yC(i)]; % linearly index into meshgrid
    for j = i+1:n
      jPos = [xC(j); yC(j)]; % linearly index into meshgrid
      Sigma(i,j) = sigma^2*exp(-norm(iPos-jPos)/ell); % exponential kernel 
      Sigma(j,i) = Sigma(i,j); % Covariance matrix is symmetric
    end
  end
end

GP.Param = [M,N,sigma,ell];
GP.Coord = [xC(:) yC(:)];
GP.Sigma = Sigma;
GP.Mu = zeros(n,1);
GP.Value = nan(n,1);
% % Some visualizations for debugging
% if 1
%   subplot(2,2,1); plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
%   axis equal; xlabel('x'); ylabel('y'); 
%   title('Spatial Coordinates (cell centers)');
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   %Visualization uses the following variables above, but via GP are
%   %  M = GP.Param(1); N = GP.Param(2); sigma = GP.Param(3); ell = GP.Param(4); 
%   %  xC = reshape(GP.Coord(:,1),M,N); yC = reshape(GP.Coord(:,2),M,N);
%   %  delta = diff(xC(1:2)); xMax = max(xC(:))+delta/2; yMax = max(yC(:))+delta/2; n = M*N;
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
%   text(xC(1,1),yC(1,1),'1','Color',[1 0 0],...
%        'HorizontalAlignment','center','VerticalAlignment','top');
%   text(xC(1,N),yC(1,N),num2str(n-N),'Color',[1 0 0],...
%        'HorizontalAlignment','center','VerticalAlignment','bottom');
%   text(xC(M,1),yC(M,1),num2str(M),'Color',[1 0 0],...
%        'HorizontalAlignment','center','VerticalAlignment','top');
%   text(xC(M,N),yC(M,N),num2str(n),'Color',[1 0 0],...
%        'HorizontalAlignment','center','VerticalAlignment','bottom');
%   subplot(2,2,3); d = linspace(0,min([1,5*ell]),1001);
%   plot(d,sigma^2*exp(-d/ell),'k-');
%   xlabel('relative distance'); ylabel('pairwise covariance'); 
%   title(['Exponential Kernel (\sigma=' num2str(sigma) ', L=' num2str(ell) ')']);
%   hold on; d0 = delta; 
%   while d0 < max(d), plot(d0*[1 1],ylim,'k--'); d0 = d0 + delta; end
%   hold off;
%   iptsetpref('ImshowAxesVisible','on');
%   subplot(2,2,2); imshow(GP.Sigma,[]); 
%   title('Covariance Matrix'); colorbar;
%   subplot(2,2,4); imshow(inv(GP.Sigma),[]); 
%   title('Concentration Matrix'); colorbar;
% end
