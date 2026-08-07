%% Demo: build the Dual-M SMP and the UREIT reconstruction matrix
% Requirements: MATLAB R2019b+ and EIDORS (http://eidors3d.sourceforge.net).
%
% The .mat files under ./FEMs contain EIDORS forward models (struct fmdl)
% generated for cylinder / head / thorax geometries with 8/16/32/64 electrodes.

clear; clc;

% --- load a 32-electrode cylinder model ---------------------------------
load(fullfile('FEMs', 'cylinder32.mat'));   % provides struct fmdl
sigma = ones(size(fmdl.elems, 1), 1);       % homogeneous background conductivity

% --- 1) Dual-M SMP: optimal stimulation/measurement pattern ---------------
tic;
smp = dual_m_smp(fmdl, sigma);
elapsed = toc;

fprintf('Dual-M SMP: %d independent channels, computed in %.1f s\n', ...
    size(smp, 1), elapsed);
fprintf('Row format: [s-  s+  m-  m+]\n');
disp(smp(1:min(5, end), :));
%%
% --- 2) UREIT: adaptive-uniform reconstruction matrix ----------------------
% Sensitivity matrix of the adjacent-stimulation pattern (or of the SMP).
img = mk_image(fmdl, sigma);
% img.fwd_model.stimulation = mk_stim_patterns(numel(fmdl.electrode), 1, ...
%     [0 1], '{ad}', {'meas_current'}, 1e-2);
img.fwd_model.stimulation = stim_meas_list(smp);
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
