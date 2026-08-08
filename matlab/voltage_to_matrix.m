function Y = voltage_to_matrix(smp, electrodeCoords, y)
%VOLTAGE_TO_MATRIX Voltage-to-matrix (V2M) mapping via the reciprocity theorem.
%   Y = VOLTAGE_TO_MATRIX(SMP, ELEC_COORDS, Y) maps the voltage vector y
%   (Dual-M SMP responses) onto a dense (P x P x T) matrix, where P is the
%   number of unique electrode pairs and T is the number of samples.
%
%   Each channel [s- s+ m- m+] carries the response of the electrode-pair
%   (stimulation, measurement); by reciprocity the response is stored
%   symmetrically at (s, m) and (m, s). The remaining entries of the
%   electrode-pair grid are completed by
%   INTERPOLATE_BY_ELECTRODE_LAYOUT, yielding a dense matrix that can be
%   fed to the VIFI-Net image branch.
%
%   Inputs:
%     SMP            (n x 4) electrode quadruples [s- s+ m- m+]
%     ELEC_COORDS    (N x 2/3) electrode coordinates (rows are electrode
%                    indices, matching SMP)
%     Y              (n x T) measured voltage responses
%
%   Output:
%     Y              (P x P x T) dense single-precision matrix
%
%   See also INTERPOLATE_BY_ELECTRODE_LAYOUT, DUAL_M_SMP, SMC_LIST_TO_STIMULATION.

    arguments
        smp (:,:) double {mustBeInteger}
        electrodeCoords (:,:) double
        y (:,:) double
    end

    nChannels = size(smp, 1);
    assert(size(y, 1) == nChannels, ...
        'voltage_to_matrix:dimMismatch', ...
        'y must have one row per SMP channel (%d), got %d.', nChannels, size(y, 1));

    % --- unique electrode pairs (stimulation + measurement) --------------
    pairs = [smp(:, [1 2]); smp(:, [3 4])];
    uniquePairs = unique(pairs, 'rows');
    nPairs = size(uniquePairs, 1);
    nSamples = size(y, 2);

    % Row/column index of each channel's stimulation and measurement pair.
    [~, pairIndex] = ismember(pairs, uniquePairs, 'rows');
    stimIdx = pairIndex(1:nChannels);
    measIdx = pairIndex(nChannels + 1:end);

    % --- symmetric sparse filling (vectorised linear indexing) ------------
    % Element (row, col, t) <-> row + nPairs*(col-1) + nPairs^2*(t-1).
    linIdx = stimIdx + nPairs * (measIdx - 1);
    linIdxSym = measIdx + nPairs * (stimIdx - 1);
    sampleOffset = nPairs^2 * (0:nSamples - 1);       % (1 x T)

    Y = zeros(nPairs, nPairs, nSamples);
    Y(linIdx + sampleOffset) = y;                      % broadcast (n x T)
    Y(linIdxSym + sampleOffset) = y;                   % reciprocity symmetry

    % --- complete the missing entries by electrode-layout interpolation ---
    Y = single(interpolate_by_electrode_layout(Y, electrodeCoords, smp));
end
