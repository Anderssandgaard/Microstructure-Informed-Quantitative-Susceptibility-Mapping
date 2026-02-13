function Ymat = sphericalHarmonicsMatrix(v,lmax)
Ymat = sphericalHarmonics(v,lmax);
Ymat2 = zeros(size(Ymat,1),(lmax+3)*lmax/2);
    i = 1;
    for l = 2:lmax/2+1
        Ymat2(:,i) = real(Ymat(:,1,l));
        i = i+1;
        mmax = (l-1)*2+1;
        for m = 2:mmax
            Ymat2(:,i) = 2*real(Ymat(:,m,l));
            i = i+1;
        end
        for m = 2:mmax
            Ymat2(:,i) = -2*imag(Ymat(:,m,l));
            i = i+1;
        end
    end
Ymat = Ymat2(:,1:(lmax+3)*lmax/2);
clear Ymat2