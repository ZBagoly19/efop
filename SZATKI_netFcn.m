function [output, state] = netFcn(input, params, varargin)
%NETFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 9
%
% Variable names in this function are taken from the original ONNX file.
%
% [OUTPUT] = netFcn(INPUT, PARAMS)
%			- Evaluates the imported ONNX network NETFCN with input(s)
%			INPUT and the imported network parameters in PARAMS. Returns
%			network output(s) in OUTPUT.
%
% [OUTPUT, STATE] = netFcn(INPUT, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = netFcn(INPUT, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% INPUT
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  INPUT:		[batch_size, 1, 43]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% OUTPUT
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  OUTPUT:		[batch_size, 1]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[input, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(input, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'input'}, {input}, [3]);
% Call the top-level graph function:
[output, NumDims.output, state] = torch_jit_exportGraph1000(input, NumDims.input, Vars, NumDims, Training, params.State);
% Postprocess the output data
[output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [output, outputNumDims1012, state] = torch_jit_exportGraph1000(input, inputNumDims1011, Vars, NumDims, Training, state)
% Function implementing the graph 'torch_jit_exportGraph1000'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.input = input;
NumDims.input = inputNumDims1011;

% Execute the operators:
% Reshape:
[shape, NumDims.x12] = prepareReshapeArgs(Vars.input, Vars.x11, NumDims.input);
Vars.x12 = reshape(Vars.input, shape{:});

% Gemm:
[A, B, C, alpha, beta, NumDims.x13] = prepareGemmArgs(Vars.x12, Vars.fc1_weight, Vars.fc1_bias, Vars.Gemmalpha1001, Vars.Gemmbeta1002, 0, 1, NumDims.fc1_bias);
Vars.x13 = alpha*B*A + beta*C;

% Relu:
Vars.x14 = relu(Vars.x13);
NumDims.x14 = NumDims.x13;

% Gemm:
[A, B, C, alpha, beta, NumDims.x15] = prepareGemmArgs(Vars.x14, Vars.fc2_weight, Vars.fc2_bias, Vars.Gemmalpha1003, Vars.Gemmbeta1004, 0, 1, NumDims.fc2_bias);
Vars.x15 = alpha*B*A + beta*C;

% Relu:
Vars.x16 = relu(Vars.x15);
NumDims.x16 = NumDims.x15;

% Gemm:
[A, B, C, alpha, beta, NumDims.x17] = prepareGemmArgs(Vars.x16, Vars.fc3_weight, Vars.fc3_bias, Vars.Gemmalpha1005, Vars.Gemmbeta1006, 0, 1, NumDims.fc3_bias);
Vars.x17 = alpha*B*A + beta*C;

% Relu:
Vars.x18 = relu(Vars.x17);
NumDims.x18 = NumDims.x17;

% Gemm:
[A, B, C, alpha, beta, NumDims.x19] = prepareGemmArgs(Vars.x18, Vars.fc4_weight, Vars.fc4_bias, Vars.Gemmalpha1007, Vars.Gemmbeta1008, 0, 1, NumDims.fc4_bias);
Vars.x19 = alpha*B*A + beta*C;

% Relu:
Vars.x20 = relu(Vars.x19);
NumDims.x20 = NumDims.x19;

% Gemm:
[A, B, C, alpha, beta, NumDims.output] = prepareGemmArgs(Vars.x20, Vars.fc_last_weight, Vars.fc_last_bias, Vars.Gemmalpha1009, Vars.Gemmbeta1010, 0, 1, NumDims.fc_last_bias);
Vars.output = alpha*B*A + beta*C;

% Set graph output arguments from Vars and NumDims:
output = Vars.output;
outputNumDims1012 = NumDims.output;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(input, numDataOutputs, params, varargin)
% Function to validate inputs to netFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'input', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, input, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,1);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [input, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(input, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(input, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {input}));
% Make the input variables into unlabelled dlarrays:
input = makeUnlabeledDlarray(input);
% Permute inputs if requested:
input = permuteInputVar(input, inputDataPerms{1}, 3);
% Check input size(s):
checkInputSize(size(input), {'batch_size' 1 43}, "input");
end

function [output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    output = extractdata(output);
end
% Permute outputs if requested:
output = permuteOutputVar(output, outputDataPerms{1}, 2);
end


%% dlarray functions implementing ONNX operators:

function [A, B, C, alpha, beta, numDimsY] = prepareGemmArgs(A, B, C, alpha, beta, transA, transB, numDimsC)
% Prepares arguments for implementing the ONNX Gemm operator
if transA
    A = A';
end
if transB
    B = B';
end
if numDimsC < 2
    C = C(:);   % C can be broadcast to [N M]. Make C a col vector ([N 1])
end
numDimsY = 2;
% Y=B*A because we want (AB)'=B'A', and B and A are already transposed.
end

function [DLTShape, numDimsY] = prepareReshapeArgs(X, ONNXShape, numDimsX)
% Prepares arguments for implementing the ONNX Reshape operator
ONNXShape = flip(extractdata(ONNXShape));            % First flip the shape to make it correspond to the dimensions of X.
% In ONNX, 0 means "unchanged", and -1 means "infer". In DLT, there is no
% "unchanged", and [] means "infer".
DLTShape = num2cell(ONNXShape);                      % Make a cell array so we can include [].
% Replace zeros with the actual size
if any(ONNXShape==0)
    i0 = find(ONNXShape==0);
    DLTShape(i0) = num2cell(size(X, numDimsX - numel(ONNXShape) + i0));  % right-align the shape vector and dims
end
if any(ONNXShape == -1)
    % Replace -1 with []
    i = ONNXShape == -1;
    DLTShape{i} = [];
end
if numel(DLTShape)==1
    DLTShape = [DLTShape 1];
end
numDimsY = numel(ONNXShape);
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

% The input dimensions have been reversed; flip them back to compare to the
% expected ONNX shape.
inputShape = fliplr(inputShape);

% Check whether the expected shape is 0 or 1D. If so, expand the expected size.
if isempty(expectedShape)
    expectedShape = {1, 1};
elseif numel(expectedShape)==1
    expectedShape{2} = 1;
end

% If the expected shape has fewer dims than the input shape, error.
if numel(expectedShape) < numel(inputShape)
    expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
    error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
end

% Prepad the input shape with trailing ones up to the number of elements in
% expectedShape
inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);

% Find the number of variable size dimensions in the expected shape
numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));

% Find the number of input dimensions that are not in the expected shape
% and cannot be represented by a variable dimension
nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;

expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
inputSizeStr = strjoin(["[", strjoin(string(inputShape), ","), "]"], "");
if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
    % The actual and expected input dimensions match, but in
    % a different order. The input needs to be permuted.
    error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
elseif numNonMatchingInputDims > 0
    % The actual and expected input sizes do not match.
    error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
end

end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if ~(isa(X,'single') || isa(X,'double'))
        % Make ints double so they can combine with anything without
        % reducting precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif onnxNDims == 0
    return;
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
