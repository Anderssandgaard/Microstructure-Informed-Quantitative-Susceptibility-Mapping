%% Microstructure-Informed QSM (mu-QSM) Processing Pipeline
% -------------------------------------------------------------------------
% Author:   Anders Dyhr Sandgaard
% Date:     Feb 2026
% Project:  Standard_Model_Fitting
% Purpose:  Estimate fODF via FBI and perform Micro-QSM fitting to 
%           correct for mesoscopic demagnetization effects.
% -------------------------------------------------------------------------

clear; clc; close all;

%% 1. Environment & Path Setup
% Resolve Project Root (Assuming script is in /Run)
[currentDir] = fileparts(mfilename('fullpath'));
[rootPath, ~] = fileparts(currentDir);
addpath(genpath(rootPath));
%% 2. Load Data
load('DATA.mat'); 

% Check GPU Availability
if gpuDeviceCount == 0
    fprintf('No compatible GPU detected. This pipeline will run on CPU.');
else
    fprintf('GPU detected. This pipeline will run on GPU.');
    Dim = gpuArray(Dim);
end
%% 3. FBI Fitting (Fiber Basis Sets)
% Logic: Estimate fODF to find the tensor component along B0 (x-direction)
fprintf('--> Estimating fODF using SHARD_FBI...\n');

degree  = 2; % Spherical harmonics degree
D_intra = 2; % Intra-axonal diffusivity constant
b_FBI   = 1; % Target b-value for FBI approximation

% SHARD_FBI returns the mesoscopic orientation information
[FBI] = CalcFBI(Signal_dMRI, bvals, bvecs, Mask_MGE, degree, b_FBI, D_intra);

% Generate Mesoscopic Demagnetization Tensor (N_meso)
% Note: Using nxx because B0 is along the X-direction in test data.
% DOI: 10.1002/mrm.29867
N_Meso = cast(Mask_MGE .* (-0.5 .* Mask_WM .* real(FBI.nxx - 1/3)), 'like', Dim);

%% 4. Dipole Kernel Preparation (Frequency Domain)
fprintf('--> Generating 3D Dipole Kernel...\n');

% Zero-padding logic to avoid wrapping artifacts (circular convolution)
[ky, kx, kz] = meshgrid(-Dim(2):Dim(2)-1, -Dim(1):Dim(1)-1, -Dim(3):Dim(3)-1);

% Normalize and shift frequencies
kx = fftshift((kx / max(abs(kx(:)))) / voxelSize(1));
ky = fftshift((ky / max(abs(ky(:)))) / voxelSize(2));
kz = fftshift((kz / max(abs(kz(:)))) / voxelSize(3));
k2 = kx.^2 + ky.^2 + kz.^2 + eps;

% Kernel reflects B0 alignment (B0 || x)
Kernel        = single(1/3 - kx.^2 ./ k2); 
Kernel(1,1,1) = 0;
clear ky kx kz k2
%% 5. Data Preparation & Zero-Padding
fprintf('--> Preparing data for LSMR solver...\n');

% Demean Larmor frequency relative to Mask_MGE
Larmor_ref    = mean(Larmor_freq(Mask_MGE == 1));
Larmor_norm   = Mask_MGE .* (Larmor_freq - Larmor_ref);

% Pad all volumes to double size to match Kernel FFT dimensions
Larmorpadded  = cast(padarray(Larmor_norm, Dim, 'post'), 'like', Dim);
Maskpadded    = cast(padarray(Mask_MGE,    Dim, 'post'), 'like', Dim);
N_mesopadded  = cast(padarray(N_Meso,      Dim, 'post'), 'like', Dim);

% Solver Parameters
lambda = 0.1;   % Tikhonov regularization parameter
atol   = 1e-6;  % Tolerance
btol   = 1e-6;
show   = 1;     % Display solver progress
intlim = inf;   % Iteration limit

params        = struct; 
params.Demag  = Kernel;
params.Mask1  = Maskpadded;
params.Mask2  = Maskpadded;
params.Mask3  = Maskpadded;
params.N_meso = N_mesopadded; 

%% 6. QSM vs. mu-QSM Fitting
% --- Standard QSM (Ignores microstructure) ---
fprintf('--> Running Standard QSM...\n');
params.switch = 0;
[res_qsm] = lsmr_GPU(@LSMR_ISO_MANYDIRECTIONS, Larmorpadded(:), ...
                     lambda, atol, btol, [], intlim, [], show, params);

Chi_QSM = reshape(res_qsm, size(Maskpadded));
Chi_QSM = gather(Chi_QSM(1:Dim(1), 1:Dim(2), 1:Dim(3)));
% --- Micro-QSM (Incorporates N_meso) ---
fprintf('--> Running Micro-QSM...\n');
params.switch = 1;
[res_muqsm] = lsmr_GPU(@LSMR_ISO_MANYDIRECTIONS, Larmorpadded(:), ...
                        lambda, atol, btol, [], intlim, [], show, params);

Chi_muQSM = reshape(res_muqsm, size(Maskpadded));
Chi_muQSM = gather(Chi_muQSM(1:Dim(1), 1:Dim(2), 1:Dim(3)));

% Final Permute to match original data orientation
Chi_QSM   = permute(Chi_QSM, [2 3 1]);
Chi_muQSM = permute(Chi_muQSM, [2 3 1]);

%% 7. Visualization
slice = round(size(Chi_QSM, 3) / 2); % Middle slice
figure('Color', 'w', 'Name', 'QSM vs mu-QSM Comparison');

subplot(1,3,1);
imagesc(Chi_QSM(:,:,slice)); axis image off; colorbar;
title('Standard QSM');

subplot(1,3,2);
imagesc(Chi_muQSM(:,:,slice)); axis image off; colorbar;
title('{\mu}QSM (Microstructure Corrected)');

subplot(1,3,3);
imagesc(Chi_QSM(:,:,slice) - Chi_muQSM(:,:,slice)); axis image off; colorbar;
title('Correction Difference');

colormap gray;