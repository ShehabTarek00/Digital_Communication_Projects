% Parameters
Ts = 1;                % Symbol duration in seconds
samples_per_symbol = 5;                % Samples per Ts (sampling frequency)
dt = 1/samples_per_symbol;             % Time between samples (200 ms)
N_bits = 10;           % Number of bits
% a) Generate Random Bits
bit_stream = randi([0 1], 1, N_bits);
% b) Map Bits to Symbols (+1 for 1, -1 for 0)
symbols = 2 * bit_stream - 1;
% c) Generate Impulse Signal (upsample by 5)
impulse_train = upsample(symbols, samples_per_symbol);
% Pulse Shaping Filter (normalized)
p = [5 4 3 2 1]/sqrt(55);
% Transmitted Signal (Pulse-Shaped)
tx_signal = conv(impulse_train, p);
t_tx = 0:dt:(length(tx_signal)-1)*dt;
% Plot Bitstream, Symbols, and Impulses
t_impulse = 0:dt:(length(impulse_train)-1)*dt;
figure;
subplot(3,1,1);
stem(t_impulse, impulse_train, 'filled');
title('Impulse Signal (Sampled Every 200 ms)');
xlabel('Time (s)'); ylabel('Amplitude'); ylim([-1.2 1.2]); grid on;

% Plot Transmitted Signal
figure;
plot(t_tx, tx_signal, 'LineWidth', 1.5);
title('Transmitted Signal y[n] (After Pulse Shaping)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%% Matched Filter
matched_filter = fliplr(p);
matched_output = conv(tx_signal, matched_filter);
t_matched = 0:dt:(length(matched_output)-1)*dt;
%.................................
% Correct total system delay after matched filtering
pulse_len = length(p);
total_delay = (pulse_len - 1);  % Transmit + matched filter

% Sampling offset: delay introduced by the filters
sample_offset = total_delay + 1;   % +1 because MATLAB indexing starts at 1
sample_indices = sample_offset : samples_per_symbol: sample_offset + (N_bits-1)*samples_per_symbol;

sampled_matched = matched_output(sample_indices);
t_samples = t_matched(sample_indices); % common time vector

%...............................
figure;
subplot(2,1,1);
plot(t_matched, matched_output, 'b-', 'LineWidth', 1.5); hold on;
stem(t_samples, sampled_matched, 'r^', 'filled', 'LineWidth', 1.2);
title('Output After Matched Filter ');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Matched Filter Output', 'Sampled Points');
hold off;
%% Rectangular Filter (ideal) — Energy Normalized
rect_filter = ones(1, samples_per_symbol)/sqrt(samples_per_symbol);
rect_output = conv(tx_signal, rect_filter);
t_rect = 0:dt:(length(rect_output)-1)*dt;
sampled_rect = rect_output(sample_indices);
figure;
subplot(2,1,1);
plot(t_rect, rect_output, 'b-', 'LineWidth', 1.5); hold on;
stem(t_samples, sampled_rect, 'r^', 'filled', 'LineWidth', 1.2);
title('Output After Rectangular Filter (Energy Normalized)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Rectangular Filter Output', 'Sampled Points');
hold off;


%% Compare Both Filters on Same Plot
figure;
subplot(2,1,1);
plot(t_matched, matched_output, 'b', 'LineWidth', 1.5); hold on;
plot(t_rect, rect_output, 'r--', 'LineWidth', 1.5);
legend('Matched Filter Output', 'Rectangular Filter Output');
title('Continuous-Time Output of Both Filters');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;



% Plot Sampled Values Only
subplot(2,1,2);
stem(t_samples, sampled_matched, 'bo', 'filled'); hold on;
stem(t_samples, sampled_rect, 'r^', 'filled');
legend('Matched Filter Samples', 'Rectangular Filter Samples');
title('Sampled Outputs at Symbol Timing Instants');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
%% correlator

% Correlator (continuous) - slide the pulse and take dot product
correlator_output = zeros(1, length(tx_signal));
for i = 5:5:length(tx_signal)
    for k=0:1:4 
    correlator_output(i-k) = sum(tx_signal(i-5+1:i-k) .* p(1:end-k));
end
end
% Pad correlator output to match length of matched filter output
corr_padded = [correlator_output, zeros(1, length(matched_output) - length(correlator_output))];

% Time vector (same as matched filter for visual alignment)
t_corr_padded = t_matched;
%t_corr = 0:dt:(length(correlator_output)-1)*dt;
sampled_correlator = corr_padded(sample_indices); 
 % trim to avoid index issues
t_corr_samples = t_matched(sample_indices(1:end));
% Plot both on same axis
figure;
subplot(2,1,1);
plot(t_matched, matched_output, 'b', 'LineWidth', 1.5); hold on;
stem(t_corr_samples, sampled_matched, 'bo', 'filled'); hold on;
plot(t_corr_padded, corr_padded, 'g', 'LineWidth', 1.5);
stem(t_corr_samples, sampled_correlator, 'gs', 'filled');

legend('Matched Filter Output', 'Correlator Output');
title('Matched Filter vs. Correlator Output (Continuous)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%%-------------------------------With Noise-----------------------

bits_for_noise = randi([0, 1], 1, 10000); % random 10000 bits
data_for_noise = 2 * bits_for_noise - 1;

%% Impulse Train for noise
impulse_train_for_noise = upsample(data_for_noise, samples_per_symbol); 
t_impulse_for_noise = (0:length(impulse_train_for_noise)-1) * (Ts/samples_per_symbol);

figure;
stem(t_impulse_for_noise, impulse_train_for_noise, 'filled', 'MarkerSize', 5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Impulse Train (10K bits)');
ylim([-2 2]);
xlim([0 30]); % plotting it from 0 to 30 not to 10000 for easy plotting 
grid on;
xticks(0:1:max(t_impulse_for_noise));


y_tx_for_noise = conv(impulse_train_for_noise, p, 'full');
t_tx_for_noise = 0:Ts/samples_per_symbol:(length(y_tx_for_noise)-1)*Ts/samples_per_symbol;


figure;
plot(t_tx_for_noise, y_tx_for_noise,'g','LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Transmitted Signal Pulse-Shaped (Convolved 10K bits)');
ylim([-5 5]);
xlim([0 50]);% plotting it from 0 to 50 not to 10000 for easy plotting 
grid on;
xticks(0:1:max(t_tx_for_noise));



% Generate and scale noise
N0 = 1/(10 ^ (-2/10));  % since Eb=1 & N0=1/(10^(SNR/10)),starting from Eb/N0 = -2 dB 
Noise_scaled = sqrt(N0/2) * randn(size(y_tx_for_noise));

figure;
plot(t_tx_for_noise, Noise_scaled,'k','LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Noise');
ylim([-5 5]);
xlim([0 30]);% plotting it from 0 to 30 not to 10000 for easy plotting 
grid on;
xticks(0:1:max(t_tx_for_noise));






% Add noise to signal
V = y_tx_for_noise + Noise_scaled; % The Transmitted pulse shaped signal added to noise
t_v = 0:Ts/samples_per_symbol:(length(V)-1)*Ts/samples_per_symbol; 

%% Matched Filter Noisy Output
y_matched_noisy= conv(V, matched_filter, 'full');
t_matched_noisy = 0:Ts/samples_per_symbol:(length(y_matched_noisy)-1)*Ts/samples_per_symbol;


%% Sampled Matched Filter (Symbol Rate Ts)
matched_noisy_sampled = y_matched_noisy(samples_per_symbol:samples_per_symbol:end);
t_matched_noisy_sampled = 0:Ts:(length(matched_noisy_sampled)-1)*Ts;

% Plot comparison using subplots
figure;

% Subplot 1: V signal and Continuous Matched Filter Output
subplot(2,1,1);
plot(t_v, V, 'b-', 'LineWidth', 1.5);
hold on;
plot(t_matched_noisy, y_matched_noisy, 'r-', 'LineWidth', 1.5); 
title('Noisy Signal (V) and Noisy Matched Filter Output (Continuous)');
legend('Noisy Signal (V)', 'Noisy MF Output Continuous', 'Location', 'SouthEast');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
xlim([0 50]);
ylim([-6 6]);
xticks(0:1:max(t_tx_for_noise));

% Subplot 2: V signal and Sampled Matched Filter Output
subplot(2,1,2);
plot(t_v, V, 'b-', 'LineWidth', 1.5);
hold on;
stem(t_matched_noisy_sampled, matched_noisy_sampled, 'ko', 'LineWidth', 1.5, 'MarkerSize', 5); 
plot(t_matched_noisy_sampled, matched_noisy_sampled, 'k--', 'LineWidth', 1.5);
title('Noisy Signal (V) and Sampled Noisy Matched Filter Output');
legend('Noisy Signal (V)', 'Noisy MF (Sampled)', 'Noisy MF (Continuous Samples)', 'Location', 'SouthEast');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
xlim([0 50]);
ylim([-6 6]);
xticks(0:1:max(t_tx_for_noise));

%% BER Calculation
N0_Values_linear = [1/(10 ^ (-2/10)),1/(10 ^ (-1/10)),1/(10 ^ (-0/10)),1/(10 ^ (1/10)),1/(10 ^ (2/10)),1/(10 ^ (3/10)),1/(10 ^ (4/10)),1/(10 ^ (5/10))];
EbN0_Values_linear = 1 ./ N0_Values_linear;
EbN0_Values_dB = [-2,-1,0,1,2,3,4,5];
BER_theoretical = 0.5 * erfc(sqrt(EbN0_Values_linear));


MF_BER_array = calculate_MF_BER(N0_Values_linear, y_tx_for_noise, bits_for_noise, matched_filter, samples_per_symbol);
% Plotting BER of Matched Filter
figure;
plot(EbN0_Values_dB, MF_BER_array, 'b-o', 'LineWidth', 1.5,'DisplayName','Simulated BER');  
hold on;
plot(EbN0_Values_dB, BER_theoretical, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');  
hold off;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER) ');
title('BER vs Eb/N0 for Matched Filter Output : Simulation vs Theoritical');
grid on;
legend('Location', 'southwest');
ylim([0 0.2]); 
yticks(0:(1e-2):0.2);  
xticks(EbN0_Values_dB); 




Rect_BER_array = calculate_Rect_BER(N0_Values_linear, y_tx_for_noise, bits_for_noise, rect_filter, samples_per_symbol);
% Plotting BER of Rect Filter
figure;
plot(EbN0_Values_dB, Rect_BER_array, 'b-o', 'LineWidth', 1.5,'DisplayName','Simulated BER');  
hold on;
plot(EbN0_Values_dB, BER_theoretical, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');  
hold off;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs Eb/N0 for Rect Filter Output : Simulation vs Theoritical');
grid on;
legend('Location', 'southwest');
ylim([0 0.2]); 
yticks(0:(1e-2):0.2);  
xticks(EbN0_Values_dB); 


% Combining MF BER & Rect BER Plots
figure;
plot(EbN0_Values_dB, MF_BER_array, 'm-o', 'LineWidth', 1.5, 'DisplayName', 'Matched Filter (Simulated)');
hold on;
plot(EbN0_Values_dB, Rect_BER_array, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Rect Filter (Simulated)');
plot(EbN0_Values_dB, BER_theoretical, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER');
hold off;

xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER Comparison: Matched Filter vs Rectangular Filter');
grid on;
legend('Location', 'southwest');
ylim([0, 0.2]); 
xticks(EbN0_Values_dB);  


%% ISI and Raised cosine
Data = randi([0, 1], 1, 100);%generation of 100 random bits data
Data_forISI = 2 * Data - 1;%mapping bits to 1 & -1
impulse_train_Data_forISI = upsample(Data_forISI, samples_per_symbol);


delay_values = [2, 8];%delay 
R_values = [0, 1];%roll-off factor



for R = R_values
   for delay = delay_values
       %rcos_filter = rcosine(1/Ts, samples_per_symbol, 'sqrt', R, delay);%generation of raisedcosine filter
       rcos_filter = rcosdesign(R, 2*delay, samples_per_symbol, 'sqrt');

       figure;%ploting the filter
       plot(rcos_filter);
       title("rcosine filter R: " + R + "Delay: " + delay);   
       xlabel("time_samples");
       ylabel("Amplitude");
        
       %passing signals through tx_filter then channel has no effect-------
      % A = filter(rcos_filter, 1, impulse_train_Data_forISI);
       A = conv(impulse_train_Data_forISI , rcos_filter , 'same');
       %then passing through rx filter-------------------------------
      % B = filter(rcos_filter, 1, A);
       B = conv(A , rcos_filter , 'same');
       
       %ploting  A,B signals---------------------------------------------
       fig_tit = ['R = ', num2str(R), ', Delay = ', num2str(delay)];
       figure;
       t_A = (0:length(A)-1) * (Ts/samples_per_symbol);
       plot(t_A, A);
       title(['Signal after transmit: ', fig_tit]);
       xlabel('Time (s)');
       ylabel('Amplitude');
       grid on;
       
       figure;
       t_B = (0:length(B)-1) * (Ts/samples_per_symbol);
       plot(t_B, B);
       title(['Signal after Receive : ', fig_tit]);
       xlabel('Time (s)');
       ylabel('Amplitude');
       grid on;
       
       
       %eye diagram------------------------------------------------------------------
       %ploting each diagram alone of A,B----------------
       figure;
       eyediagram(A, samples_per_symbol*2);
       title(['TX Filter Output : ', fig_tit]);

       figure;
       eyediagram(B, samples_per_symbol*2);
       title(['RX Filter Output : ', fig_tit]);
       %----------------------------------------------------
       %ploting eye diagram A,B together for the same R&delay-------
       %figure;
       %eye_fig = eyediagram( [A ; B]' , samples_per_symbol*2);
       %set(eye_fig,'Name',"eyediagram for R :" + R + " Delay: " + delay);
   end
end



















%%  Function for calculating the BER for different EbN0 after MF filtering 
function BER_array_MF = calculate_MF_BER(N0_Values_linear, y_tx_for_noise, bits_for_noise, matched_filter, samples_per_symbol)
    % Initialize BER array
    BER_array_MF = zeros(1, length(N0_Values_linear));
    
    % Loop over each EbN0 value 
    for i = 1:length(N0_Values_linear)
        
        Noise_scaled = sqrt(N0_Values_linear(i)/2) * randn(size(y_tx_for_noise));
        
        V = y_tx_for_noise + Noise_scaled;
        
        % Matched filtering
        y_matched_noisy = conv(V, matched_filter, 'full');
        
        % Sample at symbol rate
        matched_noisy_sampled = y_matched_noisy(samples_per_symbol:samples_per_symbol:end);
        
        matched_noisy_sampled = matched_noisy_sampled(1:length(bits_for_noise));
        % Decision at Threshold = 0 if >=0 → 1, else → 0
        detected_bits = matched_noisy_sampled >= 0;
        % errors counting
        num_errors = sum(detected_bits ~= bits_for_noise);
        % Calculating BER
        BER_array_MF(i) = num_errors / length(bits_for_noise);
        
        EbN0_dB = 10 * log10(1 / N0_Values_linear(i));
        fprintf('MF Out Bit Error Rate (BER) (@Eb/N0=%.0f dB) = %.5f\n', EbN0_dB, BER_array_MF(i));
    end
end




%%  Function for calculating the BER for different EbN0 after Rect filtering 
function BER_array_Rect = calculate_Rect_BER(N0_Values_linear, y_tx_for_noise, bits_for_noise, rect_filter, samples_per_symbol)
    % Initialize BER array
    BER_array_Rect = zeros(1, length(N0_Values_linear));
    % Loop over each EbN0 value 
    for i = 1:length(N0_Values_linear)
        
        Noise_scaled = sqrt(N0_Values_linear(i)/2) * randn(size(y_tx_for_noise));
        
        V = y_tx_for_noise + Noise_scaled;
        

        y_rect_noisy = conv(V, rect_filter, 'full');

        % Sample at symbol rate
        rect_noisy_sampled = y_rect_noisy(samples_per_symbol:samples_per_symbol:end);
        
        rect_noisy_sampled = rect_noisy_sampled(1:length(bits_for_noise));
        % Decision at Threshold = 0 if >=0 → 1, else → 0
        detected_bits = rect_noisy_sampled >= 0;
        % errors counting
        num_errors = sum(detected_bits ~= bits_for_noise);
        % Calculating BER
        BER_array_Rect(i) = num_errors / length(bits_for_noise);
        
        EbN0_dB = 10 * log10(1 / N0_Values_linear(i));
        fprintf('Rect Filter Bit Error Rate (BER) (@Eb/N0=%.0f dB) = %.5f\n', EbN0_dB, BER_array_Rect(i));
    end
end
