function use_net_block(block)
%use_net_block Using loaded neural network
%
%Level-2 M-file S-function

%%
%% The setup method is used to setup the basic attributes of the
%% S-function such as ports, parameters, etc. Do not add any other
%% calls to the main body of the function.
%%
setup(block);

%endfunction

%% Function: setup ===================================================
function setup(block)

% % Simulink passes an instance of the Simulink.MSFcnRunTimeBlock class 
% to the setup method in the input argument "block". This is known as 
% the S-function block's run-time object.

% Register original number of input ports based on the S-function
% parameter values

block.NumInputPorts = 43;
block.NumOutputPorts = 1;

% Setup port properties to be inherited or dynamic
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;

% Override input port properties
block.InputPort(1).DatatypeID  = 0;  % double
block.InputPort(1).Complexity  = 'Real';
block.InputPort(2).DatatypeID  = 0;  % double
block.InputPort(2).Complexity  = 'Real';
block.InputPort(3).DatatypeID  = 0;  % double
block.InputPort(3).Complexity  = 'Real';
block.InputPort(4).DatatypeID  = 0;  % double
block.InputPort(4).Complexity  = 'Real';
block.InputPort(5).DatatypeID  = 0;  % double
block.InputPort(5).Complexity  = 'Real';
block.InputPort(6).DatatypeID  = 0;  % double
block.InputPort(6).Complexity  = 'Real';
block.InputPort(7).DatatypeID  = 0;  % double
block.InputPort(7).Complexity  = 'Real';
block.InputPort(8).DatatypeID  = 0;  % double
block.InputPort(8).Complexity  = 'Real';
block.InputPort(9).DatatypeID  = 0;  % double
block.InputPort(9).Complexity  = 'Real';
block.InputPort(10).DatatypeID  = 0;  % double
block.InputPort(10).Complexity  = 'Real';
block.InputPort(11).DatatypeID  = 0;  % double
block.InputPort(11).Complexity  = 'Real';
block.InputPort(12).DatatypeID  = 0;  % double
block.InputPort(12).Complexity  = 'Real';
block.InputPort(13).DatatypeID  = 0;  % double
block.InputPort(13).Complexity  = 'Real';
block.InputPort(14).DatatypeID  = 0;  % double
block.InputPort(14).Complexity  = 'Real';
block.InputPort(15).DatatypeID  = 0;  % double
block.InputPort(15).Complexity  = 'Real';
block.InputPort(16).DatatypeID  = 0;  % double
block.InputPort(16).Complexity  = 'Real';
block.InputPort(17).DatatypeID  = 0;  % double
block.InputPort(17).Complexity  = 'Real';
block.InputPort(18).DatatypeID  = 0;  % double
block.InputPort(18).Complexity  = 'Real';
block.InputPort(19).DatatypeID  = 0;  % double
block.InputPort(19).Complexity  = 'Real';
block.InputPort(20).DatatypeID  = 0;  % double
block.InputPort(20).Complexity  = 'Real';
block.InputPort(21).DatatypeID  = 0;  % double
block.InputPort(21).Complexity  = 'Real';
block.InputPort(22).DatatypeID  = 0;  % double
block.InputPort(22).Complexity  = 'Real';
block.InputPort(23).DatatypeID  = 0;  % double
block.InputPort(23).Complexity  = 'Real';
block.InputPort(24).DatatypeID  = 0;  % double
block.InputPort(24).Complexity  = 'Real';
block.InputPort(25).DatatypeID  = 0;  % double
block.InputPort(25).Complexity  = 'Real';
block.InputPort(26).DatatypeID  = 0;  % double
block.InputPort(26).Complexity  = 'Real';
block.InputPort(27).DatatypeID  = 0;  % double
block.InputPort(27).Complexity  = 'Real';
block.InputPort(28).DatatypeID  = 0;  % double
block.InputPort(28).Complexity  = 'Real';
block.InputPort(29).DatatypeID  = 0;  % double
block.InputPort(29).Complexity  = 'Real';
block.InputPort(30).DatatypeID  = 0;  % double
block.InputPort(30).Complexity  = 'Real';
block.InputPort(31).DatatypeID  = 0;  % double
block.InputPort(31).Complexity  = 'Real';
block.InputPort(32).DatatypeID  = 0;  % double
block.InputPort(32).Complexity  = 'Real';
block.InputPort(33).DatatypeID  = 0;  % double
block.InputPort(33).Complexity  = 'Real';
block.InputPort(34).DatatypeID  = 0;  % double
block.InputPort(34).Complexity  = 'Real';
block.InputPort(35).DatatypeID  = 0;  % double
block.InputPort(35).Complexity  = 'Real';
block.InputPort(36).DatatypeID  = 0;  % double
block.InputPort(36).Complexity  = 'Real';
block.InputPort(37).DatatypeID  = 0;  % double
block.InputPort(37).Complexity  = 'Real';
block.InputPort(38).DatatypeID  = 0;  % double
block.InputPort(38).Complexity  = 'Real';
block.InputPort(39).DatatypeID  = 0;  % double
block.InputPort(39).Complexity  = 'Real';
block.InputPort(40).DatatypeID  = 0;  % double
block.InputPort(40).Complexity  = 'Real';
block.InputPort(41).DatatypeID  = 0;  % double
block.InputPort(41).Complexity  = 'Real';
block.InputPort(42).DatatypeID  = 0;  % double
block.InputPort(42).Complexity  = 'Real';
block.InputPort(43).DatatypeID  = 0;  % double
block.InputPort(43).Complexity  = 'Real';

% Override output port properties
block.OutputPort(1).DatatypeID  = 0; % double
block.OutputPort(1).Complexity  = 'Real';

% Register parameters. In order:
% -- If the upper bound is off (1) or on and set via a block parameter (2)
%    or input signal (3)
% -- The upper limit value. Should be empty if the upper limit is off or
%    set via an input signal
% -- If the lower bound is off (1) or on and set via a block parameter (2)
%    or input signal (3)
% -- The lower limit value. Should be empty if the lower limit is off or
%    set via an input signal
% block.NumDialogPrms     = 5;
% block.DialogPrmsTunable = {'Nontunable','Tunable','Nontunable', ...
%     'Tunable', 'Nontunable'};

% Register continuous sample times [0 offset]
block.SampleTimes = [0 0];

%% -----------------------------------------------------------------
%% Options
%% -----------------------------------------------------------------
% Specify if Accelerator should use TLC or call back into
% M-file
block.SetAccelRunOnTLC(false);

%% -----------------------------------------------------------------
%% Register methods called during update diagram/compilation
%% -----------------------------------------------------------------

block.RegBlockMethod('CheckParameters',      @CheckPrms);
block.RegBlockMethod('ProcessParameters',    @ProcessPrms);
block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
block.RegBlockMethod('Outputs',              @Outputs);
block.RegBlockMethod('Terminate',            @Terminate);
%end setup function

%% Function: CheckPrms ===================================================
function CheckPrms(block)

% lowMode = block.DialogPrm(1).Data;
% lowVal  = block.DialogPrm(2).Data;
% upMode  = block.DialogPrm(3).Data;
% upVal   = block.DialogPrm(4).Data;
% 
% % The first and third dialog parameters must have values of 1-3
% if ~any(upMode == [1 2 3]);
%     error('The first dialog parameter must be a value of 1, 2, or 3');
% end
% 
% if ~any(lowMode == [1 2 3]);
%     error('The first dialog parameter must be a value of 1, 2, or 3');
% end
% 
% % If the upper or lower bound is specified via a dialog, make sure there
% % is a specified bound. Also, check that the value is of type double
% if isequal(upMode,2),
%     if isempty(upVal),
%         error('Enter a value for the upper saturation limit.');
%     end
%     if ~strcmp(class(upVal), 'double')
%         error('The upper saturation limit must be of type double.');
%     end
% end
% 
% if isequal(lowMode,2),
%     if isempty(lowVal),
%         error('Enter a value for the lower saturation limit.');
%     end
%     if ~strcmp(class(lowVal), 'double')
%         error('The lower saturation limit must be of type double.');
%     end
% end
% 
% % If a lower and upper limit are specified, make sure the specified
% % limits are compatible.
% if isequal(upMode,2) && isequal(lowMode,2),
%     if lowVal >= upVal,
%         error('The lower bound must be explicitly less than the upper bound.');
%     end
% end

%end CheckPrms function

%% Function: ProcessPrms ===================================================
function ProcessPrms(block)

%% Update run time parameters
block.AutoUpdateRuntimePrms;

%end ProcessPrms function

%% Function: DoPostPropSetup ===================================================
function DoPostPropSetup(block)

%% Register all tunable parameters as runtime parameters.
block.AutoRegRuntimePrms;

%end DoPostPropSetup function

%% Function: Outputs ===================================================
function Outputs(block)

ang000 = block.InputPort(1).Data;
ang005 = block.InputPort(2).Data;
ang010 = block.InputPort(3).Data;
ang015 = block.InputPort(4).Data;
ang020 = block.InputPort(5).Data;
ang025 = block.InputPort(6).Data;
ang030 = block.InputPort(7).Data;
ang035 = block.InputPort(8).Data;
ang040 = block.InputPort(9).Data;
ang045 = block.InputPort(10).Data;
ang050 = block.InputPort(11).Data;
ang055 = block.InputPort(12).Data;
ang060 = block.InputPort(13).Data;
ang065 = block.InputPort(14).Data;
ang070 = block.InputPort(15).Data;
ang075 = block.InputPort(16).Data;
ang080 = block.InputPort(17).Data;
ang085 = block.InputPort(18).Data;
ang090 = block.InputPort(19).Data;
ang095 = block.InputPort(20).Data;
ang100 = block.InputPort(21).Data;
ang105 = block.InputPort(22).Data;
ang110 = block.InputPort(23).Data;
ang115 = block.InputPort(24).Data;
ang120 = block.InputPort(25).Data;
ang125 = block.InputPort(26).Data;
ang130 = block.InputPort(27).Data;
ang135 = block.InputPort(28).Data;
ang140 = block.InputPort(29).Data;
ang145 = block.InputPort(30).Data;
ang150 = block.InputPort(31).Data;
ang155 = block.InputPort(32).Data;
ang160 = block.InputPort(33).Data;
ang165 = block.InputPort(34).Data;
ang170 = block.InputPort(35).Data;
ang175 = block.InputPort(36).Data;
ang180 = block.InputPort(37).Data;
ang185 = block.InputPort(38).Data;
ang190 = block.InputPort(39).Data;
ang195 = block.InputPort(40).Data;
ang200 = block.InputPort(41).Data;
dist = block.InputPort(42).Data;
vel    = block.InputPort(43).Data;

% result = [ang000 ang005 ang010 ang015]

% % Check upper saturation limit
% if isequal(upMode,2), % Set via a block parameter
%     upVal = block.RuntimePrm(2).Data;
% elseif isequal(upMode,3), % Set via an input port
%     upVal = block.InputPort(2).Data;
%     lowPortNum = 3; % Move lower boundary down one port number
% else
%     upVal = inf;
% end
% 
% % Check lower saturation limit
% if isequal(lowMode,2), % Set via a block parameter
%     lowVal = block.RuntimePrm(1).Data;
% elseif isequal(lowMode,3), % Set via an input port
%     lowVal = block.InputPort(lowPortNum).Data;
% else
%     lowVal = -inf;
% end
% 
% % Assign new value to signal
% if sigVal > upVal,
%     sigVal = upVal;
% elseif sigVal < lowVal,
%     sigVal=lowVal;
% end

% modelfile = "C:\CM_Projects\mate_measurements\src_cm4sl\multiple_road_sensors_1500epoch_1_0.005_sched_400_08_bn_no_reg_no_net_DevAngsDistVel_WA___64_4_1399.onnx";
% params = importONNXFunction(modelfile, 'netFcn')

current_input = [ang000, ang005, ang010, ang015, ang020, ang025, ang030, ang035, ang040, ang045, ang050, ang055, ang060, ang065, ang070, ang075, ang080, ang085, ang090, ang095, ang100, ang105, ang110, ang115, ang120, ang125, ang130, ang135, ang140, ang145, ang150, ang155, ang160, ang165, ang170, ang175, ang180, ang185, ang190, ang195, ang200, dist, vel];
result = netFcn(current_input, params);

block.OutputPort(1).Data = result;

%end Outputs function

%% Function: Terminate ===================================================
function Terminate(block)
%end Terminate function