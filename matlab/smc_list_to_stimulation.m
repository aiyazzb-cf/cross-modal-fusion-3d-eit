function stimulation = smc_list_to_stimulation(smcList, currentAmp, nElectrodes)
%SMC_LIST_TO_STIMULATION Convert an SMC quadruple list into EIDORS stimulations.
%   STIM = SMC_LIST_TO_STIMULATION(SMC, IAMP, N) turns each row
%       [s-  s+  m-  m+]
%   of SMC into an EIDORS stimulation struct with a single differential
%   measurement: current IAMP is injected through s+ and sunk at s-, and the
%   voltage difference v(m+) - v(m-) is measured. This lets the Jacobian of
%   an arbitrary SMC set be computed in one EIDORS forward solve.
%
%   The resulting struct array can be assigned to img.fwd_model.stimulation
%   before calling CALC_JACOBIAN. N is the number of electrodes; it is
%   inferred as max(SMC(:)) when empty.
%
%   See also DUAL_M_SMP.

    arguments
        smcList (:,:) double {mustBeInteger}
        currentAmp (1,1) double {mustBePositive} = 1e-2
        nElectrodes double {mustBeInteger} = []
    end

    if isempty(nElectrodes)
        nElectrodes = max(smcList(:));
    end
    nSmc = size(smcList, 1);

    stimulation = struct('stimulation', cell(1, nSmc), ...
                         'stim_pattern', cell(1, nSmc), ...
                         'meas_pattern', cell(1, nSmc));

    for i = 1:nSmc
        % Stimulation pattern: s- is the sink, s+ is the source.
        stimPattern = sparse(nElectrodes, 1);
        stimPattern(smcList(i, 1)) = -currentAmp;
        stimPattern(smcList(i, 2)) =  currentAmp;
        % Measurement pattern: v(m+) - v(m-).
        measPattern = sparse(1, nElectrodes);
        measPattern(smcList(i, 3)) = -1;
        measPattern(smcList(i, 4)) =  1;

        stimulation(i).stimulation = 'Amp';
        stimulation(i).stim_pattern = stimPattern;
        stimulation(i).meas_pattern = measPattern;
    end
end
