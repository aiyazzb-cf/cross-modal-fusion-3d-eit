%% Demo: Dual-M SMP -> UREIT -> Voltage-to-Matrix (V2M)
% Requirements: MATLAB R2019b+ and EIDORS (http://eidors3d.sourceforge.net).
%
% The .mat files under ./FEMs contain EIDORS forward models (struct fmdl)
% generated for cylinder / head / thorax geometries with 8/16/32/64 electrodes.

clear; clc;

% --- load a 32-electrode cylinder model ---------------------------------
load(fullfile('FEMs', 'cylinder32.mat'));   % provides struct fmdl
sigma = ones(size(fmdl.elems, 1), 1);       % homogeneous background conductivity
nElectrodes = numel(fmdl.electrode);

% --- 1) Dual-M SMP: optimal stimulation/measurement pattern ---------------
tic;
smp = dual_m_smp(fmdl, sigma);
elapsed = toc;

fprintf('Dual-M SMP: %d independent channels, computed in %.1f s\n', ...
    size(smp, 1), elapsed);
fprintf('Row format: [s-  s+  m-  m+]\n');
disp(smp(1:min(5, end), :));

%% --- 2) UREIT: adaptive-uniform reconstruction matrix --------------------
% Sensitivity matrix evaluated on the Dual-M SMP channel set.
img = mk_image(fmdl, sigma);
img.fwd_model.stimulation = smc_list_to_stimulation(smp, 1e-2, nElectrodes);
jacobian = calc_jacobian(img);

% NOTE: for large models (> ~5e4 elements) each NF evaluation multiplies the
% reconstruction matrix by 500 noise vectors, so the lambda search can take
% minutes. Use 'noiseSamples', 100 for a quick first run, then the full 500
% for final results.
[reconMatrix, lambda] = ureit_reconstruction_matrix(fmdl, jacobian, 1, ...
    'verbose', true, 'randomSeed', 42, 'noiseSamples', 100);  % reproducible

fprintf('UREIT: lambda = %.6g, reconstruction matrix %d x %d\n', ...
    lambda, size(reconMatrix, 1), size(reconMatrix, 2));

% --- equivalent call with an explicit regularisation parameter -------------
% reconMatrix2 = ureit_reconstruction_matrix(fmdl, jacobian, 1, 'lambda', 1e-3);

%% --- 3) Voltage-to-matrix (V2M): map responses onto the electrode grid ----
% Simulate a conductivity perturbation and its voltage response, then map
% the response onto the (P x P) electrode-pair matrix for VIFI-Net.
cylinderRadius = 0.1;   cylinderHeight = 0.15;   % geometry of cylinder32
perturbRadius = 0.01;                            % perturbation radius
perturbCentre = [0.5 * cylinderRadius, 0, 0.5 * cylinderHeight];   % (x, y, z)
selectFcn = @(x, y, z) ((x - perturbCentre(1)).^2 + (y - perturbCentre(2)).^2 ...
                      + (z - perturbCentre(3)).^2 < perturbRadius^2);
perturbation = elem_select(fmdl, selectFcn);     % eidors function
y = jacobian * perturbation;                     % simulated voltage change

% Electrode positions = centroid of the electrode surface nodes.
electrodes = zeros(nElectrodes, 3);
for i = 1:nElectrodes
    nodeIdx = fmdl.electrode(i).nodes;
    electrodes(i, :) = mean(fmdl.nodes(nodeIdx, :), 1);
end

Y = voltage_to_matrix(smp, electrodes, y);       % (P x P x 1), single
fprintf('V2M: electrode-pair matrix %d x %d x %d\n', ...
    size(Y, 1), size(Y, 2), size(Y, 3));
