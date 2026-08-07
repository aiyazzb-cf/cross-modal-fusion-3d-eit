function smp = dual_m_smp(fmdl, sigma, options)
%DUAL_M_SMP Construct the Dual-M SMP: the Max-I SMP of maximal total MS.
%   SMP = DUAL_M_SMP(FMDL, SIGMA) builds the Dual-M SMP for the EIDORS
%   forward model FMDL with homogeneous background conductivity SIGMA. The
%   returned SMP is an (M_max x 4) matrix whose rows are electrode
%   quadruples [s-  s+  m-  m+] and M_max = N*(N-3)/2 is the maximum number
%   of independent channels (superposition + reciprocity theorems).
%
%   The pipeline follows the paper:
%     1. Enumerate the candidate SMC universe (optionally reduced by the
%        reciprocity theorem).
%     2. Rank candidates by measurement sensitivity S = |J_c * x_ROI|
%        (computed either from the adjacent-stimulation pattern SMP_ads by
%        superposition, or by direct batched Jacobian solves).
%     3. Run the matroid greedy selection: process candidates in
%        non-increasing order of S and keep exactly those that preserve
%        linear independence, judged on the random feature matrix Z, until
%        M_max channels are selected (Max-I SMP). By the Rado-Edmonds
%        theorem this maximizes the total MS among all Max-I SMPs.
%
%   Options (Name-Value):
%       currentAmp           (1e-2)    stimulation current amplitude [A]
%       independenceMethod   ("gaussian") independence test:
%                                   "gaussian" - batched rank test with
%                                                bisection (fast)
%                                   "rank"     - per-channel rank test
%                                                (textbook greedy)
%       forwardMethod        ("adjacent") sensitivity computation:
%                                   "adjacent" - superposition synthesis
%                                                from SMP_ads (fast)
%                                   "direct"   - batched direct Jacobian
%                                                solves
%       measurementSet       ("reciprocity") candidate SMC universe:
%                                   "reciprocity" - M_total = C(N,2)*C(N-2,2)/2
%                                   "permutation" - all polarities, A(N,2)*A(N-2,2)
%       randomSeed           ([])       seed for the random feature matrix E
%                                       (reproducible, restores rng on exit)
%
%   Requires EIDORS (http://eidors3d.sourceforge.net) and MATLAB R2019b+.
%
%   See also RANDOM_FEATURE_MATRIX, BUILD_FEATURE_MATRIX, RANK_SMC_SET,
%   FIND_INDEPENDENT_SMCS, CHANNEL_SENSITIVITY, SMC_LIST_TO_STIMULATION.

    arguments
        fmdl struct
        sigma (:,1) double
        options.currentAmp (1,1) double {mustBePositive} = 1e-2
        options.independenceMethod (1,1) string ...
            {mustBeMember(options.independenceMethod, ["gaussian", "rank"])} = "gaussian"
        options.forwardMethod (1,1) string ...
            {mustBeMember(options.forwardMethod, ["adjacent", "direct"])} = "adjacent"
        options.measurementSet (1,1) string ...
            {mustBeMember(options.measurementSet, ["reciprocity", "permutation"])} = "reciprocity"
        options.randomSeed double {mustBeInteger} = []
    end

    % ---------------------------------------------------------------- basic
    nElectrodes = numel(fmdl.electrode);
    nElements   = size(fmdl.elems, 1);

    roiMask = true(nElements, 1);
    if isfield(fmdl, 'ROI')
        roiMask = logical(fmdl.ROI);
    end

    maxIndep = nElectrodes * (nElectrodes - 3) / 2;   % M_max

    % ---------------------------------------------------------------- step 1
    % Candidate SMC universe.
    candidateSmcs = generateCandidateSmcs(nElectrodes, options.measurementSet);
    assert(size(candidateSmcs, 1) >= maxIndep, ...
        'dual_m_smp:tooFewCandidates', ...
        'The candidate set (%d SMCs) cannot contain a Max-I SMP (needs >= %d).', ...
        size(candidateSmcs, 1), maxIndep);

    % ---------------------------------------------------------------- step 2
    % Measurement sensitivity S for every candidate channel.
    img = mk_image(fmdl, sigma);
    sensitivityValues = evaluateChannelSensitivity(img, candidateSmcs, ...
        nElectrodes, roiMask, options);

    [~, order] = sort(sensitivityValues, 'descend');
    candidateSmcs = candidateSmcs(order, :);

    % ---------------------------------------------------------------- step 3
    % Matroid greedy selection of the maximum-weight basis (Dual-M SMP).
    featureMatrix = random_feature_matrix(nElectrodes, options.randomSeed);

    if options.independenceMethod == "gaussian"
        smp = selectSmcsBatched(candidateSmcs, featureMatrix, nElectrodes, maxIndep);
    else
        smp = selectSmcsGreedy(candidateSmcs, featureMatrix, nElectrodes, maxIndep);
    end

    % Canonical ordering of the output channels.
    smp = sortrows(smp);
end

% ========================================================================
function candidateSmcs = generateCandidateSmcs(nElectrodes, measurementSet)
% Enumerate all stimulation/measurement electrode pairs, excluding the
% stimulation electrodes from measurement, then apply the requested
% reduction (reciprocity) or expansion (all polarities).
    stimPairs = nchoosek(1:nElectrodes, 2);
    nStim = size(stimPairs, 1);

    measPairs = cell(nStim, 1);
    nMeasPerStim = zeros(nStim, 1);
    for i = 1:nStim
        measPairs{i} = nchoosek(setdiff(1:nElectrodes, stimPairs(i, :)), 2);
        nMeasPerStim(i) = size(measPairs{i}, 1);
    end

    candidateSmcs = zeros(sum(nMeasPerStim), 4);
    rowStart = 1;
    for i = 1:nStim
        nMeas = nMeasPerStim(i);
        candidateSmcs(rowStart:rowStart + nMeas - 1, :) = ...
            [repmat(stimPairs(i, :), nMeas, 1), measPairs{i}];
        rowStart = rowStart + nMeas;
    end

    if measurementSet == "reciprocity"
        % Keep exactly one channel of each reciprocity pair
        % [s- s+ m- m+] <-> [m- m+ s- s+].
        mirrored = candidateSmcs(:, [3 4 1 2]);
        [~, mirrorRow] = ismember(candidateSmcs, mirrored, 'rows');
        keep = mirrorRow > (1:size(candidateSmcs, 1)).';
        candidateSmcs = candidateSmcs(keep, :);
    else
        % Expand each (stimulation pair, measurement pair) into the four
        % polarity combinations: (id, swap stim, swap meas, swap both).
        permutations = [1 2 3 4; 2 1 3 4; 1 2 4 3; 2 1 4 3];
        candidateSmcs = cell2mat(arrayfun(@(r) candidateSmcs(:, permutations(r, :)), ...
            1:4, 'UniformOutput', false));
        candidateSmcs = sortrows(candidateSmcs);
    end
end

% ========================================================================
function sensitivityValues = evaluateChannelSensitivity(img, candidateSmcs, ...
        nElectrodes, roiMask, options)
% Measurement sensitivity of every candidate channel.
%   "adjacent": precompute the SMP_ads Jacobian once (N x N measurement
%   lattice) and synthesise any channel by superposition; see the paper's
%   acceleration method (2).
%   "direct"  : solve the forward problem in batches of BATCH_SIZE channels
%   and take |J_c * x_ROI| per channel.
    nCandidates = size(candidateSmcs, 1);
    sensitivityValues = zeros(nCandidates, 1);

    if options.forwardMethod == "adjacent"
        stimAds = mk_stim_patterns(nElectrodes, 1, [0 1], '{ad}', ...
            {'meas_current'}, options.currentAmp);
        img.fwd_model.stimulation = stimAds;
        jacobianAds = calc_jacobian(img);
        assert(size(jacobianAds, 1) == nElectrodes^2, ...
            'dual_m_smp:unexpectedAdJacobian', ...
            'SMP_ads Jacobian has %d rows; expected N^2 = %d rows. Check the EIDORS AD stimulation pattern.', ...
            size(jacobianAds, 1), nElectrodes^2);

        jacobianAds(:, ~roiMask) = [];
        adsSensitivity = zeros(nElectrodes, nElectrodes);
        adsSensitivity(:) = sum(jacobianAds, 2);

        % Superposition: any channel (s-,s+; m-,m+) is the block sum of the
        % ads lattice over rows m-..(m+-1) and columns s-..(s+-1).
        block = false(nElectrodes);
        for i = 1:nCandidates
            block(:) = false;
            sPair = candidateSmcs(i, [1 2]);
            mPair = candidateSmcs(i, [3 4]);
            block(mPair(1):(mPair(2) - 1), sPair(1):(sPair(2) - 1)) = true;
            sensitivityValues(i) = sum(adsSensitivity(block));
        end
        sensitivityValues = abs(sensitivityValues);
    else
        batchSize = 10000;
        nBatches = ceil(nCandidates / batchSize);
        for b = 1:nBatches
            batchIdx = (b - 1) * batchSize + (1:batchSize);
            batchIdx = batchIdx(batchIdx <= nCandidates);
            stimBatch = smc_list_to_stimulation(candidateSmcs(batchIdx, :), ...
                options.currentAmp, nElectrodes);
            img.fwd_model.stimulation = stimBatch;
            jacobianBatch = single(calc_jacobian(img));
            sensitivityValues(batchIdx) = channel_sensitivity(jacobianBatch, roiMask);
        end
    end
end

% ========================================================================
function smp = selectSmcsBatched(candidateSmcs, featureMatrix, nElectrodes, maxIndep)
% Fast independence screening: grow the candidate window by 10*M_max until
% the rank is full, then shrink it by M_max until it drops, leaving the
% smallest window of full rank; finally keep the independent channels of
% that window. Equivalent to SELECTSMCSGREEDY but with far fewer rank calls.
    nCandidatesTotal = size(candidateSmcs, 1);
    bigStep   = 10 * maxIndep;
    smallStep = maxIndep;

    nCandidates = 0;
    while true
        nCandidates = min(nCandidates + bigStep, nCandidatesTotal);
        subset = candidateSmcs(1:nCandidates, :);
        if rank_smc_set(subset, featureMatrix, nElectrodes) == maxIndep
            break;
        end
        assert(nCandidates < nCandidatesTotal, ...
            'dual_m_smp:noFullRankWindow', ...
            'Full rank was not reached within the candidate set.');
    end

    while true
        subset = candidateSmcs(1:nCandidates, :);
        if rank_smc_set(subset, featureMatrix, nElectrodes) == maxIndep
            nCandidates = nCandidates - smallStep;
        else
            break;
        end
    end
    minFullRankSize = nCandidates + smallStep;

    [~, indepMask] = find_independent_smcs( ...
        candidateSmcs(1:minFullRankSize, :), nElectrodes, featureMatrix);
    smp = candidateSmcs(1:minFullRankSize, :);
    smp = smp(indepMask, :);
end

% ========================================================================
function smp = selectSmcsGreedy(candidateSmcs, featureMatrix, nElectrodes, maxIndep)
% Textbook matroid greedy: scan candidates in the given (sensitivity
% non-increasing) order and retain a channel iff it increases the rank of
% the current selection. Stop as soon as M_max channels are selected.
    smp = candidateSmcs(1, :);
    currentRank = 1;
    for i = 2:size(candidateSmcs, 1)
        candidate = [smp; candidateSmcs(i, :)];
        newRank = rank_smc_set(candidate, featureMatrix, nElectrodes);
        if newRank > currentRank
            smp = candidate;
            currentRank = newRank;
        end
        if currentRank == maxIndep
            break;
        end
    end
end
