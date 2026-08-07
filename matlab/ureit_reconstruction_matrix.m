function [reconMatrix, lambda] = ureit_reconstruction_matrix(fmdl, jacobian, noiseFigure, options)
%UREIT_RECONSTRUCTION_MATRIX Adaptive-uniform reconstruction matrix (UREIT).
%   [RM, LAMBDA] = UREIT_RECONSTRUCTION_MATRIX(FMDL, J, NF) builds the
%   spatially-compensated linear reconstruction operator
%       RM = D^c * J' * (J*J' + lambda*I)^{-1},   c = c_opt(lambda),
%   where D = diag(d_1..d_E), d_i = [J'J]_ii, and c_opt is the optimal
%   compensation exponent of the paper (Eq. c_opt) that homogenises the
%   inverse-problem sensitivity over the image domain.
%
%   The regularisation parameter LAMBDA is determined by a bisection search
%   in log-space so that the noise figure (NF) of the reconstruction equals
%   the target NF:
%       NF = (S/N)_out / (S/N)_in,
%   estimated from a unit-change signal response and Monte-Carlo noise
%   samples propagated through RM. This removes the influence of the model
%   scale on the order of magnitude of lambda.
%
%   Inputs:
%     FMDL   EIDORS forward model (fields .elems, .nodes, .electrode)
%     J      sensitivity (Jacobian) matrix, nMeas x nElements
%     NF     target noise figure (positive scalar)
%
%   Options (Name-Value):
%     lambda       ([])   explicit regularisation parameter; when given,
%                         the NF search is skipped and RM is built at lambda
%     noiseSamples (500)  number of Monte-Carlo noise vectors for the NF
%                         estimate
%     tolerance    ([])   relative NF-matching tolerance (default 1e-3,
%                         relaxed to 1e-2 for more than 3000 elements)
%     verbose      (false) print the search trace
%     randomSeed   ([])   seed for the noise vectors (reproducible; the
%                         global rng is restored on exit)
%
%   Outputs:
%     RM     nElements x nMeas reconstruction matrix
%     LAMBDA the regularisation parameter used
%
%   Requires MATLAB R2019b+ and EIDORS; no Statistics Toolbox needed.
%
%   See also ELEM_MEASURE, DUAL_M_SMP.

    arguments
        fmdl struct
        jacobian (:,:) double
        noiseFigure (1,1) double {mustBePositive}
        options.lambda double {mustBePositive} = []
        options.noiseSamples (1,1) double {mustBeInteger, mustBePositive} = 500
        options.tolerance double {mustBePositive} = []
        options.verbose (1,1) logical = false
        options.randomSeed double {mustBeInteger} = []
    end

    if ~isempty(options.randomSeed)
        prevRngState = rng(options.randomSeed);
        cleanupRng = onCleanup(@() rng(prevRngState));   % restore on exit
    end

    nMeas = size(jacobian, 1);
    nElements = size(jacobian, 2);

    % --- precomputed quantities (kept out of the search loop) ------------
    noiseSamples = randn(nMeas, options.noiseSamples);  % standard Gaussian
    elemMeasure  = elem_measure(fmdl.elems, fmdl.nodes);% 1 x nElements
    signalResponse = sum(jacobian, 2);                  % unit-change response
    diagonal     = sum(jacobian.^2, 1);                 % diag(J'J), 1 x nElements
    jacobianJac  = jacobian * jacobian.';               % J*J'

    tolerance = options.tolerance;
    if isempty(tolerance)
        tolerance = 0.001;
        if nElements > 3000
            tolerance = 0.01;
        end
    end

    % --- log-space bisection on lambda ----------------------------------
    lambdaLo = 1e-15;
    lambdaHi = 1e15;
    lambda = sqrt(lambdaLo * lambdaHi);                 % geometric midpoint
    useExplicitLambda = ~isempty(options.lambda);
    if useExplicitLambda
        lambda = options.lambda;
    end

    if options.verbose
        fprintf('[UREIT] start (target NF = %.4g)\n', noiseFigure);
    end

    while true
        reconMatrix = buildReconMatrix(lambda, jacobian, jacobianJac, diagonal);
        nf = computeNoiseFigure(reconMatrix, elemMeasure, signalResponse, noiseSamples);

        if options.verbose
            fprintf('[UREIT] NF = %.6g, lambda = %.6g\n', nf, lambda);
        end

        if useExplicitLambda ...
                || abs(nf - noiseFigure) < noiseFigure * tolerance ...
                || (lambdaHi - lambdaLo) < 1e-15
            break;
        end

        % Standard bisection in log-space (lambda is always the geometric
        % midpoint of [lambdaLo, lambdaHi] at the loop entry).
        if nf > noiseFigure
            lambdaHi = lambda;      % over-regularised -> move upper bound down
        else
            lambdaLo = lambda;      % under-regularised -> move lower bound up
        end
        lambda = sqrt(lambdaLo * lambdaHi);
    end

    if options.verbose
        fprintf('[UREIT] done (NF = %.6g, lambda = %.6g)\n', nf, lambda);
    end
end

% ========================================================================
function reconMatrix = buildReconMatrix(lambda, jacobian, jacobianJac, diagonal)
% Diagonal-compensated regularised reconstruction operator.
%   RM0 = J' (JJ' + lambda I)^{-1},  RM = D^c * RM0 (row scaling by d_i^c),
%   with c from the paper Eq. (c_opt):
%       c = ln((d_max+lambda)/(d_min+lambda)) / ln(d_max/d_min) - 1.
    dMax = max(diagonal);
    dMin = min(diagonal);
    assert(dMax > dMin && dMin > 0, ...
        'ureit_reconstruction_matrix:badDiagonal', ...
        'diag(J''J) must be strictly positive and non-constant.');

    compensation = log((dMax + lambda) / (dMin + lambda)) / log(dMax / dMin) - 1;
    reconMatrix0 = jacobian.' / (jacobianJac + lambda * eye(size(jacobianJac)));
    reconMatrix = (diagonal.^compensation).' .* reconMatrix0;
end

% ========================================================================
function nf = computeNoiseFigure(reconMatrix, elemMeasure, signalResponse, noiseSamples)
% Noise figure of the reconstruction operator:
%   nf = (S/N)_out / (S/N)_in.
% Signal and noise amplitudes are weighted by the relative element measure
% so that peripheral and central elements contribute by their volume.
    noiseImage  = reconMatrix * noiseSamples;    % nElements x nSamples
    signalImage = reconMatrix * signalResponse;  % nElements x 1

    relMeasure = elemMeasure.' / min(elemMeasure);   % nElements x 1

    % Signal: keep the dominant-polarity response only.
    if sum(signalImage) > 0
        signalImage(signalImage < 0) = 0;
    elseif sum(signalImage) < 0
        signalImage(signalImage > 0) = 0;
    end
    signalAmplitude = sum(abs(signalImage) .* relMeasure) / sum(relMeasure);

    % Noise: per-sample weighted RMS of the propagated noise images.
    noiseMean = sum(noiseImage .* relMeasure) / sum(relMeasure);   % 1 x nSamples
    noiseRms  = sqrt(sum(relMeasure .* (noiseImage - noiseMean).^2) / sum(relMeasure));
    noiseAmplitude = mean(noiseRms);

    % Input SNR reference.
    inputNoise  = mean(std(noiseSamples, 1));
    inputSignal = mean(abs(signalResponse));

    nf = (signalAmplitude / noiseAmplitude) / (inputSignal / inputNoise);
end
