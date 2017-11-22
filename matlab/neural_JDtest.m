%% Very simple and intuitive neural network implementation
%
%  Carl L�ndahl, 2008
%  email: carl(dot)londahl(at)gmail(dot)com
%  Feel free to redistribute and/or to modify in any way

function m = neural(  )
% DATA SETS; demo file
[Attributes, Classifications] = mendez;

n = 2.6;
nbrOfNodes = 8;
nbrOfEpochs = 800;

% Initialize matrices with random weights 0-1
Weight1 = rand(nbrOfNodes, length(Attributes(1,:)));
Weight2 = rand(length(Classifications(1,:)),nbrOfNodes);

m = 0; figure; hold on; e = size(Attributes);

while m < nbrOfEpochs

    % Increment loop counter
    m = m + 1;

    % Iterate through all examples
    for i=1:e(1)
        % Input data from current example set
        Input = Attributes(i,:).';
        Output = Classifications(i,:).';

        % Propagate the signals through network
        Activations1 = f(Weight1*Input);
        Activations2 = f(Weight2*Activations1);

        % Output layer error
        delta_2 = Activations2.*(1-Activations2).*(Output-Activations2);
        delta_1 = Activations1.*(1-Activations1).*(Weight2.'*delta_2);

        % Adjust weights in matrices sequentially
        Weight2 = Weight2 + n.*delta_2*(Activations1.');
        Weight1 = Weight1 + n.*delta_1*(Input.');
    end

    RMS_Err = 0;

    % Calculate RMS error
    for i=1:e(1)
        Output = Classifications(i,:).';
        Input = Attributes(i,:).';
        RMS_Err = RMS_Err + norm(Output-f(Weight2*f(Weight1*Input)),2);
    end
    
    y = RMS_Err/e(1);
    plot(m,log(y),'*');

end


function x = f(x)
x = 1./(1+exp(-x));