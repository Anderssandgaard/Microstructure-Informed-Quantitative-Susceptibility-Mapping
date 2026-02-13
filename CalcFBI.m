function [OUT] = CalcFBI(Signal, bval_all, bvec_all, Mask, degree, b_FBI, D_intra)
% FBI: Estimate fODF and Mesoscopic Demagnetization Tensor
% Author: Anders Dyhr Sandgaard (Optimized Feb 2026)

%% 1. Input Normalization
% Ensure b-values are in ms/um^2 (SI units logic)
if max(bval_all) > 100
    bval_all = bval_all / 1000; 
end

% Identify indices matching target FBI b-value
index_fbi = find(round(bval_all) == round(b_FBI));
ndir_fbi  = length(index_fbi);

if isempty(index_fbi)
    error('No volumes found matching b_FBI = %f', b_FBI);
end

%% 2. Data Preparation (Vectorized)
dim = size(Signal);
numVoxels = prod(dim(1:3));

% Efficiently extract and reshape signal for target b-values
% Reshape to [Directions x Voxels]
S_temp = reshape(Signal, [numVoxels, dim(4)]);
Sb_reshape = S_temp(:, index_fbi).'; % [ndir_fbi x numVoxels]

%% 3. Spherical Harmonics Decomposition
% Get SH Basis Functions
GT_fbi = bvec_all(index_fbi, :);
Ymat = sphericalHarmonicsMatrix(GT_fbi, degree);

% Orthogonalize/Scale consistent with SHARD convention
Ymat = (1/sqrt(4*pi)) * [ones(ndir_fbi, 1), Ymat];

% Linear Least Squares for SH coefficients (GPU-friendly)
if isa(Sb_reshape, 'gpuArray')
    s_fbi_s2 = Ymat \ Sb_reshape;
else
    s_fbi_s2 = linsolve(Ymat'*Ymat, Ymat' * Sb_reshape);
end

% Map SH coefficients to full complex vector [l=0; l=2, m=-2:2]
s_fbi_s2_full = [ s_fbi_s2(1,:); ...
                  s_fbi_s2(4,:) - 1i*s_fbi_s2(6,:); ...
                 -s_fbi_s2(3,:) + 1i*s_fbi_s2(5,:); ...
                  s_fbi_s2(2,:); ...
                  s_fbi_s2(3,:) + 1i*s_fbi_s2(5,:); ...
                  s_fbi_s2(4,:) + 1i*s_fbi_s2(6,:) ];

%% 4. fODF and Mesoscopic Tensor Calculation
% Hypergeometric scaling factors
P20 = ((-1)^1 * factorial(2)) / (4^1 * factorial(1)^2);
G2  = (factorial(1) * (b_FBI*D_intra)^(1.5) / gamma(3.5) * hypergeom(1.5, 3.5, -b_FBI*D_intra)); 
G0  = ((b_FBI*D_intra)^(0.5) / gamma(1.5) * hypergeom(0.5, 1.5, -b_FBI*D_intra)); 

% Calculate plm coefficients
p2m      = s_fbi_s2_full * G0 .* (sqrt(4*pi) * P20 * s_fbi_s2_full(1,:) .* G2).^-1; 
p2m(1,:) = s_fbi_s2_full(1,:) .* (sqrt(4*pi) * s_fbi_s2_full(1,:)).^-1;

% Clean up p2m (handle potential divide by zeros)
p2m(~isfinite(p2m)) = 0;

%% 5. Output Construction (Tensor Components)
OUT.plm_FBI = reshape(p2m, [6, dim(1:3)]);
OUT.p2      = reshape(sqrt(sum(abs(p2m).^2, 1)) / sqrt(5/(4*pi)), dim(1:3));
OUT.p2(isnan(OUT.p2)) = 0;

% Helper for component extraction
P = OUT.plm_FBI;
c2 = sqrt(2*pi/15);
c4 = sqrt(pi/5);

% Vectorized tensor component calculation
OUT.nxx = Mask .* real( 1/3 - 2/3*c4*squeeze(P(4,:,:,:)) + c2*(squeeze(P(2,:,:,:)) + squeeze(P(6,:,:,:))) );
OUT.nyy = Mask .* real( 1/3 - 2/3*c4*squeeze(P(4,:,:,:)) - c2*(squeeze(P(2,:,:,:)) + squeeze(P(6,:,:,:))) );
OUT.nzz = Mask .* real( 1/3 + 4/3*c4*squeeze(P(4,:,:,:)) );
OUT.nxy = Mask .* real( 1i*c2*(squeeze(P(6,:,:,:)) - squeeze(P(2,:,:,:))) );
OUT.nxz = Mask .* real(    c2*(squeeze(P(3,:,:,:)) - squeeze(P(5,:,:,:))) );
OUT.nyz = Mask .* real(-1i*c2*(squeeze(P(5,:,:,:)) + squeeze(P(3,:,:,:))) );

% Global cleanup of non-finite values
fields = {'nxx','nyy','nzz','nxy','nxz','nyz'};
for f = 1:length(fields)
    OUT.(fields{f})(~isfinite(OUT.(fields{f}))) = 0;
end

%% 6. Vectorized Eigendecomposition
fprintf('--> Performing voxel-wise eigendecomposition...\n');
maskIdx = find(Mask(:));
N_eigvec = zeros(numVoxels, 3, 3, 'like', Signal);

% Loop only over masked voxels (much faster)
for i = 1:length(maskIdx)
    v = maskIdx(i);
    T = [OUT.nxx(v), OUT.nxy(v), OUT.nxz(v); ...
         OUT.nxy(v), OUT.nyy(v), OUT.nyz(v); ...
         OUT.nxz(v), OUT.nyz(v), OUT.nzz(v)];
    
    [V, D] = eig(T);
    [~, maxIdx] = max(abs(diag(D)));
    
    % Store the principal eigenvector or full matrix if needed
    N_eigvec(v,:,:) = V; 
end

OUT.N_eigvec = reshape(N_eigvec, [dim(1:3), 3, 3]);

end