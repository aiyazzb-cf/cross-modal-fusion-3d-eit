function [refResult, indepMask, transformMatrix] = find_independent_smcs(smcData, nElectrodes, featureMatrix)
%FIND_INDEPENDENT_SMCS Select the maximal independent SMC subset via rref.
%   [R, MASK] = FIND_INDEPENDENT_SMCS(SMC, N, E) determines which channels
%   of the input SMC set are mutually independent (with respect to the
%   superposition and reciprocity theorems) and which are redundant.
%
%   R is the reduced row echelon form of Z' (the transposed feature matrix)
%   and MASK is a logical column vector of length size(Z,1) marking the
%   independent channels (pivot columns of Z').
%
%   [R, MASK, T] = FIND_INDEPENDENT_SMCS(...) additionally returns T, an
%   (nRedundant x nIndependent) matrix whose k-th row gives the coefficients
%   expressing the k-th redundant channel as a linear combination of the
%   independent ones. This is the exact form of the linear dependency.
%
%   SMC accepts either an M x 4 electrode-quadruple array or an EIDORS
%   stimulation struct array (see BUILD_FEATURE_MATRIX). N is the number of
%   electrodes and E the random feature-mapping matrix; both are inferred
%   when omitted.
%
%   See also BUILD_FEATURE_MATRIX, RANDOM_FEATURE_MATRIX, RANK_SMC_SET.

    arguments
        smcData
        nElectrodes double {mustBeInteger} = []
        featureMatrix (:,:) double = []
    end

    if isempty(smcData)
        refResult = [];
        indepMask = false(0, 1);
        transformMatrix = [];
        return;
    end

    if isempty(nElectrodes)
        if isstruct(smcData)
            nElectrodes = size(smcData(1).stim_pattern, 1);
        else
            nElectrodes = max(smcData(:));
        end
    end
    if isempty(featureMatrix)
        featureMatrix = random_feature_matrix(nElectrodes);
    end

    zMatrix = build_feature_matrix(smcData, featureMatrix, nElectrodes);

    % rref(Z') : columns of Z' are the channel feature vectors; pivot
    % columns index the independent channels.
    [refResultFull, pivotCols] = rref(zMatrix.');
    indepMask = false(size(zMatrix, 1), 1);
    indepMask(pivotCols) = true;

    % Round the reported rref for readability; independence decisions are
    % taken before rounding and are unaffected.
    refResult = round(refResultFull, 3);

    if nargout >= 3
        nIndep = nnz(indepMask);
        % First nIndep rows of the rref carry the pivots; take the
        % non-pivot columns (redundant channels) of these rows. Kept at
        % FULL precision: the coefficients feed numerical routines (e.g.
        % the matrix completion in UREIT) where rounding to 3 decimals
        % would introduce ~1e-2 errors.
        transformMatrix = refResultFull(1:nIndep, ~indepMask).';
    end
end
