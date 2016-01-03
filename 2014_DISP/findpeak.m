function n = findpeak(x)
% Find peaks.
% n = findpeaks(x)

n1    = find(diff(diff(x) > 0) < 0);
u    = find(x(n1+1) > x(n1));
n1(u) = n1(u)+1;

if(x(1)>x(2)) 
    n = zeros(length(n1)+1,1);
    n(1)=1;
    for k=1:length(n1)
        n(k+1) = n1(k);
    end
else
    n = n1;
end
