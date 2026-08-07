function featureMatrix = random_feature_matrix(nElectrodes, seed)
%RANDOM_FEATURE_MATRIX Random feature-mapping matrix E used in independence screening.
%   E = RANDOM_FEATURE_MATRIX(N) returns an N x M_max matrix whose entries are
%   independent standard normal variates, where
%       M_max = N*(N-3)/2
%   is the maximum number of linearly independent channels of an N-electrode
%   EIT system (superposition + reciprocity theorems). The matrix is resampled
%   until it has full row rank (rank(E) = N); for a continuous distribution
%   this happens with probability 1, so the loop is purely defensive.
%
%   E = RANDOM_FEATURE_MATRIX(N, SEED) fixes the global random generator seed
%   while E is drawn and restores the previous generator state on return, so
%   the call is reproducible without disturbing the caller's random stream.
%
%   See also BUILD_FEATURE_MATRIX, RANK_SMC_SET, DUAL_M_SMP.

    arguments
        nElectrodes (1,1) double {mustBeInteger, mustBeNonnegative}
        seed double {mustBeInteger} = []
    end

    if ~isempty(seed)
        prevRngState = rng(seed);
        cleanupRng = onCleanup(@() rng(prevRngState));   % restore on exit
    end

    maxIndep = nElectrodes * (nElectrodes - 3) / 2;
    featureMatrix = randn(nElectrodes, maxIndep);

    while rank(featureMatrix) < nElectrodes
        featureMatrix = randn(nElectrodes, maxIndep);
    end
end
