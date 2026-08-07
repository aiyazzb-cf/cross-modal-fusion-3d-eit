function measure = elem_measure(elems, nodes)
%ELEM_MEASURE Measure (area / volume) of every finite element.
%   M = ELEM_MEASURE(ELEMS, NODES) returns a row vector of element measures:
%   2D triangulation  -> polygon area (computed via polyshape);
%   3D tetrahedral mesh -> tetrahedron volume |det([1 x y z])| / 6.
%
%   ELEMS is an nElements x k node-index table (zero-padded rows are
%   accepted; trailing zeros are ignored). NODES is an nNodes x 2 (2D) or
%   nNodes x 3 (3D) coordinate matrix.
%
%   See also UREIT_RECONSTRUCTION_MATRIX.

    arguments
        elems (:,:) double {mustBeInteger}
        nodes (:,:) double
    end

    nElements = size(elems, 1);
    measure = zeros(1, nElements);

    switch size(nodes, 2)
        case 2
            % --- 2D: element area via polygon ---------------------------
            for e = 1:nElements
                nodeIdx = elems(e, :);
                nodeIdx(nodeIdx == 0) = [];
                polygon = polyshape(nodes(nodeIdx, 1), nodes(nodeIdx, 2));
                measure(e) = polygon.area;
            end

        case 3
            % --- 3D: tetrahedron volume --------------------------------
            for e = 1:nElements
                nodeIdx = elems(e, :);
                nodeIdx(nodeIdx == 0) = [];
                assert(numel(nodeIdx) == 4, ...
                    'elem_measure:unsupportedElement', ...
                    '3D elements must be tetrahedra (4 nodes); got %d node(s).', ...
                    numel(nodeIdx));
                basis = [ones(4, 1), nodes(nodeIdx, :)];
                measure(e) = abs(det(basis) / 6);
            end

        otherwise
            error('elem_measure:dim', ...
                'nodes must have 2 (2D) or 3 (3D) columns; got %d.', size(nodes, 2));
    end
end
