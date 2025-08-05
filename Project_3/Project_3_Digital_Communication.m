% clc; clear;
close all;

%% Parameters
numBits = 600000;
EbNo_dB = -4:1:14; %SNR
EbNo = 10.^(EbNo_dB/10);
sqrtEbNo = sqrt(EbNo);

% Initialize BER arrays
berSim = zeros(6, length(EbNo_dB));     % 1: BPSK, 2: QPSK Gray, 3: QPSK non-Gray, 4: 8PSK, 5: 16QAM
berTheory = zeros(6, length(EbNo_dB));

% Generate random data
dataBits = randi([0 1], 1, numBits);

%% ----- BPSK -----
bpskSymbols = 2 * dataBits - 1;
for i = 1:length(EbNo)
    noise = randn(1, numBits) / sqrt(2*EbNo(i));
    received = bpskSymbols + noise;
    detected = received > 0;
    berSim(1,i) = mean(dataBits ~= detected);
end
berTheory(1,:) = qfunc(sqrt(2)*sqrtEbNo);

%% ----- QPSK Gray -----
dataBitsGray = dataBits;
if mod(numBits, 2) ~= 0
    dataBitsGray = [dataBitsGray, 0];
    numBits = length(dataBitsGray);
end
numSymbols = numBits/2;
dataQPSK = reshape(dataBitsGray, 2, numSymbols).';
% QPSK Gray Mapping according to specified constellation:
% 11 to 45° (1+j)/sqrt(2)
% 01 to 135° (-1+j)/sqrt(2)
% 00 to 225° (-1-j)/sqrt(2)
% 10 to 315° (1-j)/sqrt(2)
map_gray = [(1+1j)/sqrt(2), (-1+1j)/sqrt(2), (-1-1j)/sqrt(2), (1-1j)/sqrt(2)];
symbols = zeros(1, numSymbols);

for i = 1:numSymbols
    dibit = dataQPSK(i,:);
    if isequal(dibit, [1 1])
        symbols(i) = map_gray(1);  % 45° 11
    elseif isequal(dibit, [0 1])
        symbols(i) = map_gray(2);  % 135° 01
    elseif isequal(dibit, [0 0])
        symbols(i) = map_gray(3);  % 225°  00
    elseif isequal(dibit, [1 0])
        symbols(i) = map_gray(4);  % 315° 10
    end
end
for i = 1:length(EbNo)
    %noise = (randn(1,length(numSymbols)) + 1j*randn(1,length(numSymbols))) / sqrt(2*EbNo(i));
    noise = (randn(1, numSymbols) + 1j*randn(1, numSymbols))/sqrt(2);
    noiseScaled = noise / sqrt(2*EbNo(i));
    rx = symbols  + noiseScaled;
    
    detectedBits = zeros(1, numBits);
    for j = 1:numSymbols
        distances = abs(rx(j) - map_gray);
        [~, idx] = min(distances);
        
        if idx == 1
            detectedBits(2*j-1:2*j) = [1 1];  % 45°
        elseif idx == 2
            detectedBits(2*j-1:2*j) = [0 1];  % 135°
        elseif idx == 3
            detectedBits(2*j-1:2*j) = [0 0];  % 225°
        elseif idx == 4
            detectedBits(2*j-1:2*j) = [1 0];  % 315°
        end
    end
    berSim(2,i) = sum(detectedBits ~= dataBits) / numBits;
    %berSim(2,i) = mean(dataBitsGray ~= reshape(detectedBits.', 1, []));
end

berTheory(2,:) = 0.5 * erfc(sqrtEbNo);
%berTheory(2,:) = qfunc(sqrt(2)*sqrtEbNo);

%% ----- QPSK Non-Gray -----
dataBitsBinary = dataBits;
if mod(numBits, 2) ~= 0
    dataBitsBinary = [dataBitsBinary, 0];
    numBits = length(dataBitsBinary);
end
numSymbols = numBits/2;
dataQPSK = reshape(dataBitsBinary, 2, numSymbols).';
% QPSK Gray Mapping according to specified constellation:
% 10 to 45° (1+j)/sqrt(2)
% 01 to 135° (-1+j)/sqrt(2)
% 00 to 225° (-1-j)/sqrt(2)
% 11 to 315° (1-j)/sqrt(2)
angles = [45, 135, 225, 315] * pi/180;
map_binary = cos(angles) + 1j*sin(angles);
symbols = zeros(1, numSymbols);

for i = 1:numSymbols
    dibit = dataQPSK(i,:);
    if isequal(dibit, [1 0])
        symbols(i) = map_binary(1);  % 45° 10
    elseif isequal(dibit, [0 1])
        symbols(i) = map_binary(2);  % 135° 01
    elseif isequal(dibit, [0 0])
        symbols(i) = map_binary(3);  % 225°  00
    elseif isequal(dibit, [1 1])
        symbols(i) = map_binary(4);  % 315° 11
    end
end
for i = 1:length(EbNo)
    %noise = (randn(1,length(numSymbols)) + 1j*randn(1,length(numSymbols))) / sqrt(2*EbNo(i));
    noise = (randn(1, numSymbols) + 1j*randn(1, numSymbols))/sqrt(2);
    noiseScaled = noise / sqrt(2*EbNo(i));
    rx = symbols  + noiseScaled;
    
    detectedBits = zeros(1, numBits);
    for j = 1:numSymbols
        distances = abs(rx(j) - map_binary);
        [~, idx] = min(distances);
        
        if idx == 1
            detectedBits(2*j-1:2*j) = [1 0];  % 45°
        elseif idx == 2
            detectedBits(2*j-1:2*j) = [0 1];  % 135°
        elseif idx == 3
            detectedBits(2*j-1:2*j) = [0 0];  % 225°
        elseif idx == 4
            detectedBits(2*j-1:2*j) = [1 1];  % 315°
        end
    end
    berSim(3,i) = sum(detectedBits ~= dataBits) / numBits;
    %berSim(2,i) = mean(dataBitsGray ~= reshape(detectedBits.', 1, []));
end

berTheory(3,:) = 0.5 * erfc(sqrtEbNo);
%berTheory(2,:) = qfunc(sqrt(2)*sqrtEbNo);

%% ----- 8PSK -----
%getting data ready
if mod(numBits, 3) ~= 0
    padding = 3 - mod(numBits, 3);
    dataBits = [dataBits, zeros(1, padding)];
    numBits = length(dataBits);
end

numSymbols = numBits/3;
data_8PSK = reshape(dataBits, 3, numSymbols).';

angles = [0, 45, 90, 135, 180, 225, 270, 315] * pi/180;
map_8psk = cos(angles) + 1j*sin(angles);

%mapping
symbols = zeros(1, numSymbols);
for i = 1:numSymbols
    tribit = data_8PSK(i,:);
    if isequal(tribit, [0 0 0])
        symbols(i) = map_8psk(1);      % 0°
    elseif isequal(tribit, [0 0 1])
        symbols(i) = map_8psk(2);      % 45°
    elseif isequal(tribit, [0 1 1])
        symbols(i) = map_8psk(3);      % 90°
    elseif isequal(tribit, [0 1 0])
        symbols(i) = map_8psk(4);      % 135°
    elseif isequal(tribit, [1 1 0])
        symbols(i) = map_8psk(5);      % 180°
    elseif isequal(tribit, [1 1 1])
        symbols(i) = map_8psk(6);      % 225°
    elseif isequal(tribit, [1 0 1])
        symbols(i) = map_8psk(7);      % 270°
    elseif isequal(tribit, [1 0 0])
        symbols(i) = map_8psk(8);      % 315°
    end
end

for i = 1:length(EbNo)
    % noise
    noise = (randn(1, numSymbols) + 1j*randn(1, numSymbols))/sqrt(2);
    noiseScaled = noise / sqrt(3*EbNo(i));
    
    % Add noise signal
    rx = symbols + noiseScaled;
    
    % Demapping 
    detectedBits = zeros(1, numBits);
    for j = 1:numSymbols
        distances = abs(rx(j) - map_8psk);
        [~, idx] = min(distances);%get min distance
        % Map signal
        if idx == 1
            detectedBits(3*j-2:3*j) = [0 0 0];      % 0°
        elseif idx == 2
            detectedBits(3*j-2:3*j) = [0 0 1];      % 45°
        elseif idx == 3
            detectedBits(3*j-2:3*j) = [0 1 1];      % 90°
        elseif idx == 4
            detectedBits(3*j-2:3*j) = [0 1 0];      % 135°
        elseif idx == 5
            detectedBits(3*j-2:3*j) = [1 1 0];      % 180°
        elseif idx == 6
            detectedBits(3*j-2:3*j) = [1 1 1];      % 225°
        elseif idx == 7
            detectedBits(3*j-2:3*j) = [1 0 1];      % 270°
        elseif idx == 8
            detectedBits(3*j-2:3*j) = [1 0 0];      % 315°
        end
    end
    
    % Calculate BER
    berSim(4,i) = sum(detectedBits ~= dataBits) / numBits;
end

berTheory(4,:)  = (1/3)*erfc(sqrt(3*EbNo)*sin(pi/8));


%% ----- 16-QAM -----
%edit data to be ready for 16QAM
if mod(numBits, 4) ~= 0
    padding = 4 - mod(numBits, 4);
    dataBits = [dataBits, zeros(1, padding)];
    numBits = length(dataBits);
end

numSymbols = numBits/4;
data16QAM = reshape(dataBits, 4, numSymbols).';

%axis values
I_values = [-3, -1, 1, 3];
Q_values = [3, 1, -1, -3];

qam_map = zeros(16, 1);
bit_patterns = zeros(16, 4);
idx = 1;

%mapper-------------------------
for q_idx = 1:4
    for i_idx = 1:4
        % Calculate coordinates
        i_val = I_values(i_idx);
        q_val = Q_values(q_idx);
        
        %map the bits to their points
        % First two bits (b0b1) determine column (I value)
        % Last two bits (b2b3) determine row (Q value)
        if i_idx == 1 %left column
            b0b1 = [0 0];
        elseif i_idx == 2
            b0b1 = [0 1];
        elseif i_idx == 3
            b0b1 = [1 1];
        else %last right column
            b0b1 = [1 0];
        end
        
        if q_idx == 1 % top row
            b2b3 = [1 0];
        elseif q_idx == 2
            b2b3 = [1 1];
        elseif q_idx == 3
            b2b3 = [0 1];
        else % bottom row
            b2b3 = [0 0];
        end
        
        % Store pattern
        bit_patterns(idx, :) = [b0b1, b2b3];
        
        %store conseltation points
        qam_map(idx) = (i_val + 1j*q_val);
        
        idx = idx + 1;
    end
end

% Normalize constellation points to have power equals 1
avg_power = mean(abs(qam_map).^2);
scale_factor = sqrt(avg_power);
qam_map = qam_map / scale_factor;


% Map 4-bit patterns to 16-QAM symbols
symbols = zeros(1, numSymbols);
for i = 1:numSymbols
    pattern = data16QAM(i, :);
    
    % Find the index of the pattern in our matrix
    for j = 1:16
        if all(pattern == bit_patterns(j, :))
            symbols(i) = qam_map(j);
            break;
        end
    end
end

%noise + demapper
for i = 1:length(EbNo)
    % Generate noise
    noise = (randn(1, numSymbols) + 1j*randn(1, numSymbols))/sqrt(2);
    noiseScaled = noise * sqrt((1/4)/EbNo(i));

    % Add noise
    rx = symbols + noiseScaled;
    
    % Demapping based on min distance
    detectedBits = zeros(1, numBits);
    for j = 1:numSymbols
        distances = abs(rx(j) - qam_map);
        [~, idx] = min(distances);

        % Map index back to its pattern
        detectedBits(4*j-3:4*j) = bit_patterns(idx, :);
    end
    
    %BER
    berSim(5,i) = sum(detectedBits ~= dataBits) / numBits;
end

berTheory(5,:) = (3/8)*erfc(sqrt(EbNo/2.5));


%% BFSK

%Parameters
BFSKnumBits = 200000;
BFSK_Eb=1; % assuming the energy per bit equals 1 
BFSK_EbNo_dB = -5:1:15;
BFSK_EbNo = 10.^(BFSK_EbNo_dB/10);
BFSK_Noise_PSD = BFSK_Eb./(10.^(BFSK_EbNo_dB/10)); 
BER_BFSK = zeros(1, length(BFSK_EbNo_dB)); 
Theoretical_BER_BFSK = zeros(1, length(BFSK_EbNo_dB)); 

% Generate random data
BFSK_binraydataBits = randi([0 1], 1, BFSKnumBits);
%-------------------------------------------

% BFSK Mapping 
BFSK_data=zeros(1,BFSKnumBits);
for i = 1 : (BFSKnumBits) 
    if BFSK_binraydataBits(i)  == 0 
        BFSK_data(i) = cos(0)+1i*sin(0); % mapping the '0' value to a symbol has phase = 0 degree
    else 
        BFSK_data(i) = cos(pi/2)+1i*sin(pi/2); % mapping the '1' value to a symbol has phase = 90 degree
    end 
end 
%-------------------------------------------

%Looping over the EbNo values and calc the BER at each value
for i = 1:length(BFSK_EbNo_dB) 
    % Generate Complex Noise has variance equals sqrt(No/2) in the I & Q components 
    BFSK_Noise = randn(1, BFSKnumBits) * sqrt(BFSK_Noise_PSD(i) / 2) + 1i .* randn(1,BFSKnumBits) * sqrt(BFSK_Noise_PSD(i) / 2); 
 
    % Add noise to transmitted signal 
    BFSK_Received_signal = BFSK_data + BFSK_Noise;
    
 
%% BFSK Demapper
 BFSK_Received_data= zeros(1, BFSKnumBits); 
    for j = 1:BFSKnumBits 
        % the region of the value '0' is betweem phases [45,-135]
        if (angle(BFSK_Received_signal(j)) >= -3*pi/4) && (angle(BFSK_Received_signal(j)) <= pi/4 )
            BFSK_Received_data(j) = 0; 
        else % the region of the value '1' is betweem phases [45,225]
            BFSK_Received_data(j) = 1; 
        end 
    end 

 
    %% BER of BFSK 
    

    % Calculate BFSK BER 
    BFSK_Error = abs(BFSK_Received_data - BFSK_binraydataBits(1:BFSKnumBits)); 
    BER_BFSK(i) = sum(BFSK_Error) / BFSKnumBits; %
 
    % Calculate Theoretical BFSK BER 
    Theoretical_BER_BFSK(i) = (1/2) * erfc(sqrt(1 / (2 * BFSK_Noise_PSD(i)))); 

    

end

%---------------------------------------------------------------------

%% Plotting BFSK BER 
figure;
semilogy(BFSK_EbNo_dB,BER_BFSK , '-o', 'linewidth', 2) ; 
hold on 
semilogy(BFSK_EbNo_dB, Theoretical_BER_BFSK ,'--','linewidth',2) ; 
xlabel('Eb/No'); 
ylabel('BER'); 
legend('BFSK BER' , 'Theoretical BER ') ; 
grid on 
title('BFSK BER');

%---------------------------------------------------------------------

%% Auto-Corr. of 100 BFSK Realizations each is 1000 bits
num_realizations = 100;  % Number of realizations 
bits_per_realization = 1000;     % Number of bits per realization
samples_per_bit=5; % Tb=5
upsampled_bfsk_data_length=bits_per_realization * samples_per_bit;



bfsk_binarydata = randi([0 1], 1, bits_per_realization);

bfsk_data_realizations = zeros(num_realizations, bits_per_realization);
bfsk_data_realizations(1,:)= bfsk_binarydata;  % First realization (no shift)

for r = 2:num_realizations
    % Apply a circular shift 
    shift_amount = randi([1, bits_per_realization-1]);
    circshifted_data = circshift(bfsk_binarydata, [0, shift_amount]);
    bfsk_data_realizations(r,:) = circshifted_data;
end


Tb = samples_per_bit;
f0 = 0;           % Normalized frequency for '0'
f1 = 1 / Tb;  % Normalized frequency (1/Tb) for '1'
t = (0:Tb-1);  % Time samples per bit
 symbol_energy=sqrt(2 * BFSK_Eb / Tb); 
mapped_bfsk_data = zeros(num_realizations, upsampled_bfsk_data_length);

for r = 1:num_realizations
    for c = 1:bits_per_realization
        % indexing the data by the length of Tb (upsampling by factor of 5)
        indx_start = (c-1)*Tb + 1;
        indx_end = c*Tb;
        
        if bfsk_data_realizations(r, c) == 0
            symbol = symbol_energy*(cos(2*pi*f0*t) + 1j*sin(2*pi*f0*t)); % mapping '0' to f0 
        else
            symbol = symbol_energy*(cos(2*pi*f1*t) + 1j*sin(2*pi*f1*t)); % mapping '1' to f1
        end

        mapped_bfsk_data(r, indx_start:indx_end) = symbol;
    end
end
BFSK_Auto_Corr = bfsk_autocorr_func(mapped_bfsk_data);
BFSK_Auto_Corr_flipped = [fliplr(conj(BFSK_Auto_Corr(2:end))) BFSK_Auto_Corr];

%---------------------------------------------------------------------

%% BFSK PSD
N = length(BFSK_Auto_Corr_flipped);
BFSK_PSD = abs(fftshift(fft(BFSK_Auto_Corr_flipped, N)));  
BFSK_PSD = BFSK_PSD / max(BFSK_PSD);   
fs = 5 * f1; % normalized by 5 as the mapped data is upsampled by factor of 5 due to that the Tb equals 5
f = linspace(-fs/2, fs/2, N); 
figure;
plot(f,BFSK_PSD, 'LineWidth', 1.5);
grid on;
xlabel('Normalized Frequency');
ylabel('BFSK Power');
title('PSD of BFSK');
ylim([0 0.05]);

figure;
plot(f,10*log(BFSK_PSD), 'LineWidth', 1.5);
grid on;
xlabel('Normalized Frequency');
ylabel('BFSK Power in dB');
title('PSD of BFSK in dB');
ylim([-200 10]);
xlim([-0.5 0.5]);





%----------------------------------
%% ----- Plot: Simulated BERs -----
figure;
semilogy(EbNo_dB, berSim(1,:), 'k-', 'LineWidth', 1.5); hold on;
semilogy(EbNo_dB, berSim(2,:), 'g-', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(3,:), 'y-', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(4,:), 'b-', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(5,:), 'r-', 'LineWidth', 1.5);
semilogy(BFSK_EbNo_dB,BER_BFSK,'m-','linewidth',1.5) ; 
xlabel('E_b/N_0 (dB)'); ylabel('BER'); grid on;
title('Simulated BER vs. E_b/N_0 for All Modulation Schemes');
legend('BPSK', 'QPSK Gray', 'QPSK Binary', '8PSK', '16QAM', 'BFSK');
set(gca, 'FontSize', 12);

%% ----- Plot: Theoretical BERs -----
figure;
semilogy(EbNo_dB, berTheory(1,:), 'k--', 'LineWidth', 1.5); hold on;
semilogy(EbNo_dB, berTheory(2,:), 'g--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berTheory(3,:), 'y--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berTheory(4,:), 'b--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berTheory(5,:), 'r--', 'LineWidth', 1.5);
semilogy(BFSK_EbNo_dB,Theoretical_BER_BFSK,'m--','linewidth',1.5) ; 
xlabel('E_b/N_0 (dB)'); ylabel('BER'); grid on;
title('Theoretical BER vs. E_b/N_0 for All Modulation Schemes');
legend('BPSK Theory', 'QPSK Gray Theory', 'QPSK Binary Theory', '8PSK Theory', '16QAM Theory', 'BFSK Theory');
set(gca, 'FontSize', 12);


%% ----- Plot: Simulated vs. Theoretical Combined -----
figure;
semilogy(EbNo_dB, berSim(1,:), 'k-', EbNo_dB, berTheory(1,:), 'k--', 'LineWidth', 1.5); hold on;
semilogy(EbNo_dB, berSim(2,:), 'g-', EbNo_dB, berTheory(2,:), 'g--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(3,:), 'y-', EbNo_dB, berTheory(3,:), 'y--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(4,:), 'b-', EbNo_dB, berTheory(4,:), 'b--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berSim(5,:), 'r-', EbNo_dB, berTheory(5,:), 'r--', 'LineWidth', 1.5);
semilogy(BFSK_EbNo_dB, BER_BFSK, 'm-',BFSK_EbNo_dB, Theoretical_BER_BFSK, 'm--', 'LineWidth', 1.5); 
xlabel('E_b/N_0 (dB)'); ylabel('BER'); grid on;
title('Simulated & Theoretical BER vs. E_b/N_0');
legend({'BPSK Sim', 'BPSK Theory', ...
        'QPSK Gray Sim', 'QPSK Gray Theory', ...
        'QPSK Binary Sim', 'QPSK Binary Theory', ...
        '8PSK Sim', '8PSK Theory', ...
        '16QAM Sim', '16QAM Theory', ...
        'BFSK Sim', 'BFSK Theory'}, ...
        'Location', 'southwest');
set(gca, 'FontSize', 12);

%% plot for every type separatly 
modNames = {'BPSK', 'QPSK Gray', 'QPSK Binary', '8PSK', '16QAM'};
colors = {'k', 'g', 'b', 'k', 'r'};

for m = 1:5
    figure;
    semilogy(EbNo_dB, berSim(m,:), [colors{m} '-'], 'LineWidth', 1.5); hold on;
    semilogy(EbNo_dB, berTheory(m,:), [colors{m} '--'], 'LineWidth', 1.5);
    xlabel('E_b/N_0 (dB)');
    ylabel('BER');
    title(['BER vs E_b/N_0 for ', modNames{m}]);
    legend([modNames{m} ' Sim'], [modNames{m} ' Theory'], 'Location', 'southwest');
    grid on;
    set(gca, 'FontSize', 12);
end

figure;
semilogy(EbNo_dB, berSim(2,:), 'g-', 'LineWidth', 1.5); hold on;
semilogy(EbNo_dB, berSim(3,:), 'y-', 'LineWidth', 1.5);
semilogy(EbNo_dB, berTheory(2,:), 'g--', 'LineWidth', 1.5);
semilogy(EbNo_dB, berTheory(3,:), 'y--', 'LineWidth', 1.5);

xlabel('E_b/N_0 (dB)');
ylabel('BER');
title('BER Comparison: QPSK Gray vs QPSK Binary');
legend('QPSK Gray Sim', 'QPSK Binary Sim', ...
       'QPSK Gray Theory', 'QPSK Binary Theory', ...
       'Location', 'southwest');
grid on;
set(gca, 'FontSize', 12);


%% BFSK Auto-Correlation Function
function autocorr_values = bfsk_autocorr_func(bfsk_data_matrix)

    first_col = bfsk_data_matrix(:, 1);
    num_realizations = size(bfsk_data_matrix, 1);

   
    num_samples = size(bfsk_data_matrix, 2);
    autocorr_values = zeros(1, num_samples);

    for col = 1:num_samples
        current_col = bfsk_data_matrix(:, col);
        dot_product = sum(current_col .* conj(first_col)); 
        autocorr_values(col) = dot_product / num_realizations; 
    end
end

