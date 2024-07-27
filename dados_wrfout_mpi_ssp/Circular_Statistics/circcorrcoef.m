function r = circcorrcoef(a,b)
%CIRCCORRCOEF Circular correlation coefficients.
%   R=CIRCCORRCOEF(X,Y) calculates the circular correlation coefficients
%   for the vectors X and Y in radians.

% evaristo@ensenada.net
% http://evaristor.blogspot.com
%
% Reference: Fisher, N.I. and A.J. Lee (1983) A correlation coefficient for
% circular data. Biometrika Trust. 70(2):327-332
% http://biomet.oxfordjournals.org/cgi/content/abstract/70/2/327

a=a(:);
b=b(:);
n=length(a);
r=4*(sum(cos(a).*cos(b))*sum(sin(a).*sin(b))-sum(cos(a).*sin(b))*sum(sin(a).*cos(b)))/sqrt((n.^2-sum(cos(2*a)).^2-sum(sin(2*a)).^2)*(n.^2-sum(cos(2*b)).^2-sum(sin(2*b)).^2));
