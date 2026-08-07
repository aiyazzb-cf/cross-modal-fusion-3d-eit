function sensitivityVector = channel_sensitivity(jacobianMatrix, roiMask)
%CHANNEL_SENSITIVITY Per-channel measurement sensitivity S = |J_c * x_ROI|.
%   S = CHANNEL_SENSITIVITY(J, ROI) computes, for every measurement channel,
%   the sensitivity to a unit conductivity change over the region of
%   interest, where J is the Jacobian (nChannels x nElements) and ROI is a
%   logical mask of length nElements marking the ROI elements. Columns
%   outside the ROI are discarded and the absolute row sum is returned:
%       S(i) = | sum_{e in ROI} J(i, e) |
%   This equals Eq. (1) of the paper for a unit perturbation x_ROI = 1.
%
%   See also DUAL_M_SMP.

    arguments
        jacobianMatrix (:,:) double
        roiMask (:,1) logical = true(size(jacobianMatrix, 2), 1)
    end

    if numel(roiMask) ~= size(jacobianMatrix, 2)
        error('channel_sensitivity:dimMismatch', ...
            'roiMask must have one entry per finite element (columns of J).');
    end

    jacobianMatrix(:, ~roiMask) = [];
    sensitivityVector = abs(sum(jacobianMatrix, 2));
end
