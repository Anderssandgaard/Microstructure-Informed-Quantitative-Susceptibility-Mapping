function Y = sphericalHarmonics(v,lmax)
Y = zeros(size(v,1),lmax+1,lmax/2+1);
Y(:,1,1) = 1; % all spherical harmonics are multiplied by sqrt(4*pi)

theta = acos(v(:,3));
phi = atan2(v(:,2),v(:,1));

for l = 2:2:lmax
    for m = 0:l
        Y(:,1+m,1+l/2) = sqrt(4*pi)*harmonicY(l,m,theta,phi);
    end
end