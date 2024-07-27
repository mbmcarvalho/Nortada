function [rho pval] = circ_corrcc_degrees(alpha1, alpha2)
%
% [rho pval ts] = circ_corrcc(alpha1, alpha2)
%   Circular correlation coefficient for two circular random variables.
%
%   Input:
%     alpha1	sample of angles in degrees
%     alpha2	sample of angles in degrees
%
%   Output:
%     rho     correlation coefficient
%     pval    p-value
%
% References:
%   Topics in circular statistics, S.R. Jammalamadaka et al., p. 176
%
% PHB 6/7/2008
%

if size(alpha1,2) > size(alpha1,1)
	alpha1 = alpha1';
end

if size(alpha2,2) > size(alpha2,1)
	alpha2 = alpha2';
end

if length(alpha1)~=length(alpha2)
  error('Input dimensions do not match.')
end

% compute mean directions
n = length(alpha1);
alpha1_bar = circ_mean_degrees(alpha1);
alpha2_bar = circ_mean_degrees(alpha2);

% compute correlation coeffcient from p. 176
num = sum(sind(alpha1 - alpha1_bar) .* sind(alpha2 - alpha2_bar));
den = sqrt(sum(sind(alpha1 - alpha1_bar).^2) .* sum(sind(alpha2 - alpha2_bar).^2));
rho = num / den;	

% compute pvalue
l20 = mean(sind(alpha1 - alpha1_bar).^2);
l02 = mean(sind(alpha2 - alpha2_bar).^2);
l22 = mean((sind(alpha1 - alpha1_bar).^2) .* (sind(alpha2 - alpha2_bar).^2));

ts = sqrt((n * l20 * l02)/l22) * rho;
pval = 2 * (1 - normcdf(abs(ts)));

