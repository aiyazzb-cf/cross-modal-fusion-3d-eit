function yDense = interpolate_by_electrode_layout(ySparse, electrodeCoords, smcList)
%INTERPOLATE_BY_ELECTRODE_LAYOUT Complete a sparse electrode-pair matrix.
%   Y = INTERPOLATE_BY_ELECTRODE_LAYOUT(YS, COORDS, SMC) fills the zero
%   (missing) upper-triangular entries of the (P x P x C) electrode-pair
%   matrix YS by a weighted combination of the known entries. The weight of
%   a known entry (s, m) for a missing entry (u, v) is the product of the
%   similarity between the stimulation pairs u/s and the measurement pairs
%   v/m, each similarity being the product of three Gaussian/directional
%   kernels on the electrode-pair geometry:
%
%     S = exp(-d_center^2 / 2 sigma_d^2)            (position)
%       * cos(theta between pair vectors)           (direction, signed)
%       * exp(-|d_length|^2 / 2 sigma_L^2)          (length)
%
%   The kernel scales are automatic: sigma_d = 0.5 * median pairwise
%   distance between pair centres, sigma_L = 0.2 * mean pair length.
%   The matrix is symmetrised afterwards (reciprocity).
%
%   Inputs:
%     YS        (P x P x C) sparse matrix, 0 marks a missing entry
%     COORDS    (N x 2/3) electrode coordinates
%     SMC       (n x 4) electrode quadruples [s- s+ m- m+]; the union of
%               their stimulation and measurement pairs must equal the P
%               rows/columns of YS
%
%   Output:
%     Y         (P x P x C) completed dense matrix
%
%   See also VOLTAGE_TO_MATRIX.

    arguments
        ySparse (:,:,:) double
        electrodeCoords (:,:) double
        smcList (:,:) double {mustBeInteger}
    end

    [nPairs, ~, nChannels] = size(ySparse);

    % --- electrode pairs underlying the grid ------------------------------
    pairs = [smcList(:, [1 2]); smcList(:, [3 4])];
    uniquePairs = unique(pairs, 'rows');
    assert(size(uniquePairs, 1) == nPairs, ...
        'interpolate_by_electrode_layout:dimMismatch', ...
        'The unique pairs of SMC (%d) must match the size of ySparse (%d).', ...
        size(uniquePairs, 1), nPairs);

    % --- 1. geometric features of each pair (vectorised) ------------------
    centers = (electrodeCoords(uniquePairs(:, 1), :) ...
             + electrodeCoords(uniquePairs(:, 2), :)) / 2;   % (P x dim)
    vectors = electrodeCoords(uniquePairs(:, 2), :) ...
            - electrodeCoords(uniquePairs(:, 1), :);         % (P x dim)
    lengths = sqrt(sum(vectors.^2, 2));                      % (P x 1)
    lengths(lengths == 0) = 1e-10;                           % guard

    % --- 2. automatic kernel scales ---------------------------------------
    % Pairwise centre distances (hand-rolled; avoids the Statistics Toolbox
    % dependency of pdist/pdist2).
    centerDiff = reshape(centers, [nPairs, 1, size(centers, 2)]) ...
               - reshape(centers, [1, nPairs, size(centers, 2)]);
    distCent = sqrt(sum(centerDiff.^2, 3));                  % (P x P)
    upperTri = triu(true(nPairs), 1);
    sigmaD = 0.5 * median(distCent(upperTri));
    sigmaL = 0.2 * mean(lengths);

    % --- 3. similarity matrices -------------------------------------------
    sPos = exp(-distCent.^2 / (2 * sigmaD^2));               % position
    sDir = (vectors * vectors.') ./ (lengths * lengths.');   % direction (signed)
    lengthDiff = lengths - lengths.';
    sLen = exp(-lengthDiff.^2 / (2 * sigmaL^2));             % length

    % --- 4. known entries ------------------------------------------------
    nKnown = size(smcList, 1);
    [~, knownRow] = ismember(smcList(:, [1 2]), uniquePairs, 'rows');
    [~, knownCol] = ismember(smcList(:, [3 4]), uniquePairs, 'rows');
    knownVals = zeros(nKnown, nChannels);
    for k = 1:nKnown
        knownVals(k, :) = reshape(ySparse(knownRow(k), knownCol(k), :), [1 nChannels]);
    end

    % --- 5. missing upper-triangular entries ------------------------------
    if nChannels == 1
        missingMask = (ySparse == 0);
    else
        missingMask = all(ySparse == 0, 3);
    end
    [missingRows, missingCols] = find(missingMask);
    keep = missingRows < missingCols;                        % upper triangle
    missingRows = missingRows(keep);
    missingCols = missingCols(keep);

    % --- 6. weighted completion -------------------------------------------
    % For each missing entry (u, v): weights_k = S(u, s_k) * S(v, m_k);
    % the estimate is sum(w * y_k) / sum(|w|). Loop kept for memory
    % reasons (vectorising over all missing entries would need an
    % nMissing x nKnown matrix).
    yDense = ySparse;
    for m = 1:numel(missingRows)
        u = missingRows(m);
        v = missingCols(m);
        wStim = sPos(u, knownRow).' .* sDir(u, knownRow).' .* sLen(u, knownRow).';
        wMeas = sPos(v, knownCol).' .* sDir(v, knownCol).' .* sLen(v, knownCol).';
        weights = wStim .* wMeas;
        yDense(u, v, :) = sum(weights .* knownVals, 1) / sum(abs(weights));
    end

    % --- 7. symmetrisation (reciprocity) ----------------------------------
    for c = 1:nChannels
        yDense(:, :, c) = triu(yDense(:, :, c)) + triu(yDense(:, :, c), 1).';
    end
end
