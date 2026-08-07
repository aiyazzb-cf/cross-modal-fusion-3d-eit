function [numIndependent, zMatrix] = rank_smc_set(smcData, featureMatrix, nElectrodes)
%RANK_SMC_SET Number of independent channels of a set of SMCs.
%   R = RANK_SMC_SET(SMC, E, N) returns the rank of the feature matrix Z
%   built by BUILD_FEATURE_MATRIX, i.e. the number of linearly independent
%   channels in the input SMC set. R == N*(N-3)/2 means the set is a
%   maximum independent SMP (Max-I SMP).
%
%   [R, Z] = RANK_SMC_SET(...) also returns the feature matrix Z.
%
%   See also BUILD_FEATURE_MATRIX, RANDOM_FEATURE_MATRIX, DUAL_M_SMP.

    arguments
        smcData
        featureMatrix (:,:) double
        nElectrodes double {mustBeInteger} = []
    end

    zMatrix = build_feature_matrix(smcData, featureMatrix, nElectrodes);
    numIndependent = rank(zMatrix);
end
