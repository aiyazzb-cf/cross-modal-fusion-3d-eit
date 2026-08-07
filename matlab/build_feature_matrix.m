function zMatrix = build_feature_matrix(smcData, featureMatrix, nElectrodes)
%BUILD_FEATURE_MATRIX Map a set of SMCs into the random-feature matrix Z.
%   Z = BUILD_FEATURE_MATRIX(SMC, E, N) builds the feature matrix
%       Z = (S * E) .* (M * E)
%   where S and M are the stimulation / measurement incidence matrices
%   (each row holds +1 / -1 / 0 per electrode) and E is the random
%   feature-mapping matrix from RANDOM_FEATURE_MATRIX. This implements
%   Eq. (z) of the paper:
%       z_k = (E_{s+,:} - E_{s-,:}) .* (E_{m+,:} - E_{m-,:})
%   The Hadamard product encodes the reciprocity theorem (commutativity of
%   the two differences); the row-vector differences encode the
%   superposition theorem. Rank(Z) equals the number of independent
%   channels of the input SMC set.
%
%   SMC accepts either
%     * an M x 4 numeric array of electrode quadruples
%           [s-  s+  m-  m+]
%       (sink, source, negative and positive measurement electrodes), or
%     * an EIDORS stimulation struct array (with fields stim_pattern and
%       meas_pattern), which also supports differential measurements
%       involving more than two electrodes.
%
%   N is the number of electrodes; it is inferred from the data when empty.
%
%   See also RANDOM_FEATURE_MATRIX, RANK_SMC_SET, FIND_INDEPENDENT_SMCS.

    arguments
        smcData
        featureMatrix (:,:) double
        nElectrodes double {mustBeInteger} = []
    end

    [stimIncidence, measIncidence] = smcIncidenceMatrices(smcData, nElectrodes);
    zMatrix = (stimIncidence * featureMatrix) .* (measIncidence * featureMatrix);
end

% ------------------------------------------------------------------------
function [stimIncidence, measIncidence] = smcIncidenceMatrices(smcData, nElectrodes)
% Convert SMC data into (nSMC x N) incidence matrices with entries in {-1,0,1}.
% Sign convention (consistent with EIDORS): for stimulation, +1 is the current
% source, -1 is the sink; for measurement, +1 is the positive voltage terminal.
    if isstruct(smcData)
        % --- EIDORS stimulation struct array ---------------------------
        stimCounts = arrayfun(@(s) size(s.meas_pattern, 1), smcData);
        nTotal = sum(stimCounts);
        if isempty(nElectrodes)
            nElectrodes = size(smcData(1).stim_pattern, 1);
        end
        stimIncidence = zeros(nTotal, nElectrodes);
        measIncidence = zeros(nTotal, nElectrodes);
        rowStart = 1;
        for k = 1:numel(smcData)
            nMeas = stimCounts(k);
            block = rowStart:(rowStart + nMeas - 1);
            stimIncidence(block, :) = repmat(full(smcData(k).stim_pattern).', nMeas, 1);
            measIncidence(block, :) = full(smcData(k).meas_pattern);
            rowStart = rowStart + nMeas;
        end
    elseif isnumeric(smcData)
        % --- M x 4 quadruple list [s- s+ m- m+] ------------------------
        nSmc = size(smcData, 1);
        if isempty(nElectrodes)
            nElectrodes = max(smcData(:));
        end
        stimIncidence = zeros(nSmc, nElectrodes);
        measIncidence = zeros(nSmc, nElectrodes);
        % Column-major linear indexing: element (i, j) <-> i + (j-1)*nSmc.
        stimLinIdx = (smcData(:, [1 2]) - 1) * nSmc + (1:nSmc).';
        measLinIdx = (smcData(:, [3 4]) - 1) * nSmc + (1:nSmc).';
        stimIncidence(stimLinIdx) = repmat([-1 1], nSmc, 1);
        measIncidence(measLinIdx) = repmat([-1 1], nSmc, 1);
    else
        error('build_feature_matrix:badInput', ...
            'smcData must be an M x 4 numeric array or an EIDORS stimulation struct.');
    end
end
