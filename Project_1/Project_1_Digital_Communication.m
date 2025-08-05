clc; clear; close all;
%define basic parameters
A = 4;
num_realization = 500;
num_bits = 100;
samples_per_bit = 7;
bitDuration = 70e-3; 
Fs = 1 / (bitDuration / samples_per_bit); 
total_samples = num_bits * samples_per_bit;
% Generate random binary data
Data = randi([0, 1], num_realization, num_bits);
% Generate random delay in the range of the bit samples
delay_samples = randi([1, 7], num_realization, 1);
% Time axis for plotting
time = (0:total_samples-1) / Fs;

%% 1. Polar NRZ (0 -> -A, 1 -> A)
%convert data to polar_nrz
Polar_NRZ = (2 * Data - 1) * A;
% Expand each bit by repeating its value 7 times(sampling)
Polar_NRZ_out = zeros(num_realization, total_samples);
for i = 1:num_realization
    Polar_NRZ_reshaped = repmat(Polar_NRZ(i, :), samples_per_bit, 1);
    Polar_NRZ_out(i, :) = reshape(Polar_NRZ_reshaped, 1, []);
end
% Apply delay to Polar NRZ using circular shift
Polar_NRZ_delayed = zeros(size(Polar_NRZ_out));
for i = 1:num_realization
    Polar_NRZ_delayed(i, :) = circshift(Polar_NRZ_out(i, :), delay_samples(i));  
end
% Plotting random realizations  against time
figure;
for i = 1:3
    subplot(3,1,i);
    plot(time, Polar_NRZ_delayed(i, :), 'b');
    title(['Polar NRZ - Realization ', num2str(i)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    xlim([0, 8]);
    ylim([-5, 5]);
end

% Compute means of PNRZ
stat_mean_NRZ = calculate_statistical_mean(Polar_NRZ_delayed, num_realization, total_samples);
theoretical_NRZ = 0; % Theoretical mean
time_mean_NRZ = Calculate_time_mean(Polar_NRZ_delayed, num_realization, total_samples);

% Plot ensamble mean for Polar NRZ
figure;
plot(time, stat_mean_NRZ, 'r', 'LineWidth', 2); hold on;
plot(time, theoretical_NRZ * ones(1, total_samples), 'b--', 'LineWidth', 2);
title('Mean - Polar NRZ');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Ensemble Mean', 'Theoretical Mean');
grid on;
xlim([0, 8]);
ylim([-2, 2]);
% Plot time mean across realizations for Polar NRZ
figure;
plot(1:num_realization, time_mean_NRZ, 'b', 'LineWidth', 2); 
title('Time Mean Across Realizations - Polar NRZ');
xlabel('Realization Index');
ylabel('Time Mean Value');
grid on;
ylim([-5, 5]); 

%% 2. Unipolar(0 -> 0, 1 -> A)
% Convert data to Uni-Polar 
Uni_Polar = Data * A;
% Expand each bit by repeating its value 7 times
Uni_Polar_out = zeros(num_realization, total_samples);
for i = 1:num_realization
    Uni_Polar_reshaped = repmat(Uni_Polar(i, :), samples_per_bit, 1);
    Uni_Polar_out(i, :) = reshape(Uni_Polar_reshaped, 1, []); 
end
% Apply delay to Uni_Polar using circular shift
Uni_Polar_delayed = zeros(size(Uni_Polar_out));
for i = 1:num_realization
    Uni_Polar_delayed(i, :) = circshift(Uni_Polar_out(i, :), delay_samples(i));  
end
% Plotting random realizations  against time
figure;
for i = 1:3
    subplot(3,1,i);
    plot(time, Uni_Polar_delayed(i, :), 'g');
    title(['Uni-Polar - Realization ', num2str(i)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    xlim([0, 8]);
     ylim([-5, 5]);
end
% Compute means of UPNRZ
stat_mean_Unipolar = calculate_statistical_mean(Uni_Polar_delayed, num_realization, total_samples);
theoretical_Unipolar = 0.5 * A; % Theoretical mean
time_mean_Unipolar = Calculate_time_mean(Uni_Polar_delayed, num_realization, total_samples);

% Plot ensamble mean for Unipolar
figure;
plot(time, stat_mean_Unipolar, 'r', 'LineWidth', 2); hold on;
plot(time, theoretical_Unipolar * ones(1, total_samples), 'b--', 'LineWidth', 2);
title('Mean - Unipolar');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Ensemble Mean', 'Theoretical Mean');
grid on;
xlim([0, 8]);
ylim([0, 4]);
% Plot time mean across realizations for Unipolar
figure;
plot(1:num_realization, time_mean_Unipolar, 'b', 'LineWidth', 2); 
title('Time Mean Across Realizations - Unipolar');
xlabel('Realization Index');
ylabel('Time Mean Value');
grid on;
ylim([0, 4]); 

%% 3. Polar RZ
% Convert to Polar RZ (0 -> -A, 1 -> A for 4 samples, then 0 for 3 samples) 
Polar_RZ = (2 * Data - 1) * A;

Polar_RZ_out = zeros(num_realization, total_samples);
% Iterate over realizations
for i = 1:num_realization
    % Expand each bit to cover the required samples
    Polar_RZ_expanded = repmat(Polar_RZ(i, :), samples_per_bit, 1);
    
    % Set last 3 samples of each bit to zero (Polar RZ format)
    for j = 1:num_bits
        Polar_RZ_expanded(5:end, j) = 0;  % First 4 samples remain, last 3 go to zero
    end
    
    % Flatten to a single row
    Polar_RZ_out(i, :) = reshape(Polar_RZ_expanded, 1, []);  
end


% Apply delay to Polar_RZ using circular shift
Polar_RZ_delayed = zeros(size(Polar_RZ_out));
for i = 1:num_realization
    Polar_RZ_delayed(i, :) = circshift(Polar_RZ_out(i, :), delay_samples(i));  
end
%plotting 
figure;
for i = 1:3
    subplot(3,1,i);
    plot(time, Polar_RZ_delayed(i, :), 'm');
    title(['Polar RZ - Realization ', num2str(i)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    xlim([0, 8]);
     ylim([-5, 5]);
end
% Compute means of PRZ
stat_mean_RZ = calculate_statistical_mean(Polar_RZ_delayed, num_realization, total_samples);
theoretical_RZ = 0; % Theoretical mean
time_mean_RZ = Calculate_time_mean(Polar_RZ_delayed, num_realization, total_samples);

% Plot ensamble mean for Polar RZ
figure;
plot(time, stat_mean_RZ, 'r', 'LineWidth', 2); hold on;
plot(time, theoretical_RZ * ones(1, total_samples), 'b--', 'LineWidth', 2);
title('Mean - Polar RZ');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Ensemble Mean', 'Theoretical Mean');
grid on;
xlim([0, 8]);
ylim([-2, 2]);
% Plot time mean across realizations for Polar RZ
figure;
plot(1:num_realization, time_mean_RZ, 'b', 'LineWidth', 2); 
title('Time Mean Across Realizations - Polar RZ');
xlabel('Realization Index');
ylabel('Time Mean Value');
grid on;
ylim([-5, 5]); 


%%---------------Theoritical AutoCorrelation---------------------
%theoritical PolarNRZ AutoCorrelation is (A^2) @tau=0 only and zero everywhere with slope 1/bit duration 
theoretical_auto_corr_PolarNRZ =@(tau) (A^2) * exp(-abs(tau) / 7);
%theoritical UniPolarNRZ AutoCorrelation is (A^2/2) @tau=0 only and (A^2/4) everywhere else
theoretical_auto_corr_UnipolarNRZ = @(tau) (A^2 / 4) + ((A^2 / 4) * exp(-abs(tau) / 7));

%theoritical PolarRZ AutoCorrelation is (A^2 * 4/7) @tau=0 only and zero everywhere else
theoretical_auto_corr_PolarRZ = @(tau) (A^2 * (4/7)) * exp(-abs(tau) / 7);

tau_values = 1:700; %% As they are 700 bits

%%Applying the tau values if the theoritical formulas of the line codes
PolarNRZ_Theoritical = arrayfun(theoretical_auto_corr_PolarNRZ, tau_values);
UnipolarNRZ_Theoritical = arrayfun(theoretical_auto_corr_UnipolarNRZ, tau_values);
PolarRZ_Theoritical = arrayfun(theoretical_auto_corr_PolarRZ, tau_values);

%%---------------------------------------------------------------



%%-------------------Auto Correlation simulation @ multiple values of tau-------------------------
%% 1.For Polar NRZ
pnrz_auto_corr_multiple_tau = autocorr_func_multiple_tau(Polar_NRZ_delayed,"Polar NRZ Auto Correlation @ tau = [0:699]");
plot_sim_and_theor(pnrz_auto_corr_multiple_tau,PolarNRZ_Theoritical,"Polar NRZ AutoCorrelation","Theoretical Polar NRZ AutoCorrelation","PolarNRZ AutoCorr. Simulated vs Theoritical");
%--------------------

%% 2.For Polar RZ
prz_auto_corr_multiple_tau = autocorr_func_multiple_tau(Polar_RZ_delayed,"Polar RZ Auto Correlation @ tau = [0:699]");
plot_sim_and_theor(prz_auto_corr_multiple_tau,PolarRZ_Theoritical,"Polar RZ AutoCorrelation","Theoretical Polar RZ AutoCorrelation","Polar RZ AutoCorr. Simulated vs Theoritical");
%--------------------

%% 3.For UniPolar NRZ
upnrz_auto_corr_multiple_tau = autocorr_func_multiple_tau(Uni_Polar_delayed,"UniPolar NRZ Auto Correlation @ tau = [0:699]");
plot_sim_and_theor(upnrz_auto_corr_multiple_tau,UnipolarNRZ_Theoritical,"Unipolar NRZ AutoCorr","Theoretical Unipolar NRZ AutoCorr","UnipolarNRZ AutoCorr. Simulated vs Theoritical");

%%----------------------------------------------------------------------------------------------------------------


%%-------------------Auto Correlation of Polar NRZ-------------------------
 %% 1.Auto Correlation @ tau = 0 for each RV Sample 
polar_nrz_corr_zero_shift=autocorr_func_zero_tau(Polar_NRZ_delayed,"Polar NRZ Auto Correlation @ tau = 0");
%% 2.Auto Correlation @ tau = 1 between each two RV Samples
pnrz_auto_corr_unity_tau=auto_corr_unity_tau(Polar_NRZ_delayed,"Polar NRZ Auto Correlation @ tau = 1");

%%-------------------Auto Correlation of Polar RZ-------------------------
 %% 1.Auto Correlation @ tau = 0 for each RV Sample 
polar_rz_corr_zero_shift=autocorr_func_zero_tau(Polar_RZ_delayed,"Polar RZ Auto Correlation @ tau = 0");
%% 2.Auto Correlation @ tau = 1 between each two RV Samples
prz_auto_corr_unity_tau=auto_corr_unity_tau(Polar_RZ_delayed,"Polar RZ Auto Correlation @ tau = 1");

%%-------------------Auto Correlation of UniPolar NRZ-------------------------
 %% 1.Auto Correlation @ tau = 0 for each RV Sample 
unipolar_nrz_corr_zero_shift=autocorr_func_zero_tau(Uni_Polar_delayed,"UniPolar NRZ Auto Correlation @ tau = 0");
%% 2.Auto Correlation @ tau = 1 between each two RV Samples
upnrz_auto_corr_unity_tau=auto_corr_unity_tau(Uni_Polar_delayed,"UniPolar NRZ Auto Correlation @ tau = 1");

%%----------------------------------------------------------------------------------------------------------------------




%definging an 1-D array aa a test values of tau to test the Ergodicity
ergodic_tau_values = 0:num_realization; % 500 values as they are 500 realizations



%%-------------Ergoticity Polar NRZ-------------------------------------------
%% 1.isergodic @tau=0 , Self AutoCorr for each Realization 
pnrz_isergodic=isergotic(Polar_NRZ_delayed,"Avg. Self-Correlation of all Polar NRZ Realizations @tau=0");
plot_autocorr(polar_nrz_corr_zero_shift,pnrz_isergodic,"Polar NRZ Auto Correlation @ tau = 0","Avg. Self-Correlation of all Polar NRZ Realizations");
%% 2.Ergodic AutoCorrelation @tau=0:500 , Auto Corr. between Realizations & between Samples RV's
pnrz_ergodic_auto_corr_arr=ergodic_auto_corr(Polar_NRZ_delayed,ergodic_tau_values,"AutoCorrelation across Time Vs AutoCorrelation across ensamble - Polar NRZ");


%%-------------Ergoticity Polar RZ--------------------------------------------
%% 1.isergodic @tau=0 , Self AutoCorr for each Realization 
prz_isergodic=isergotic(Polar_RZ_delayed,"Avg. Self-Correlation of all Polar RZ Realizations @tau=0");
plot_autocorr(polar_rz_corr_zero_shift,prz_isergodic,"Polar RZ Auto Correlation @ tau = 0","Avg. Self-Correlation of all Polar RZ Realizations");

%% 2.Ergodic AutoCorrelation @tau=0:500 , Auto Corr. between Realizations & between Samples RV's
prz_ergodic_auto_corr_arr=ergodic_auto_corr(Polar_RZ_delayed,ergodic_tau_values,"AutoCorrelation across Time Vs AutoCorrelation across ensamble - Polar RZ");



%%-------------Ergoticity UniPolar NRZ----------------------------------------
%% 1.isergodic @tau=0 , Self AutoCorr for each Realization 
upnrz_isergodic=isergotic(Uni_Polar_delayed,"Avg. Self-Correlation of all UniPolar NRZ Realizations @tau=0");
plot_autocorr(unipolar_nrz_corr_zero_shift,upnrz_isergodic,"UniPolar NRZ Auto Correlation @ tau = 0","Avg. Self-Correlation of all UniPolar NRZ Realizations");
%%-------------------------------------------------------------------------
%% 2.Ergodic AutoCorrelation @tau=0:500 , Auto Corr. between Realizations & between Samples RV's
upnrz_ergodic_auto_corr_arr=ergodic_auto_corr(Uni_Polar_delayed,ergodic_tau_values,"AutoCorrelation across Time Vs AutoCorrelation across ensamble - UniPolar NRZ");
%%-------------------------------------------------------------------------



%% ------------------ Compute PSD Using Fourier Transform of Autocorrelation ------------------

% Ensure FFT matches the length of the autocorrelation function
N = length(prz_auto_corr_multiple_tau);  

% Define frequency axis (centered at 0)
freq_axis = (-N/2:N/2-1) * (Fs / N);

% PSD is the Fourier Transform (FFT) of the autocorrelation function

PSD_Polar_NRZ = abs(fftshift(fft(pnrz_auto_corr_multiple_tau, N))); % Polar NRZ
PSD_Uni_Polar = abs(fftshift(fft(upnrz_auto_corr_multiple_tau, N))); % Unipolar NRZ
PSD_Polar_RZ = abs(fftshift(fft(prz_auto_corr_multiple_tau, N))); % Polar RZ

% Compute Theoretical PSD using the known mathematical formulas for each signal type

% Theoretical PSD of Polar NRZ
theoretical_PSD_Polar_NRZ = Fs*(A^2 * bitDuration) * (sinc(freq_axis * bitDuration).^2);

% Theoretical PSD of Unipolar NRZ
theoretical_PSD_Uni_Polar = Fs* (A^2 * bitDuration / 4) * (sinc(freq_axis * bitDuration).^2);

% Approximate the delta function at f = 0
delta_approx = 10*Fs* (A^2 / 4) * (abs(freq_axis) < (Fs / N));  
theoretical_PSD_Uni_Polar = theoretical_PSD_Uni_Polar + delta_approx;

% Theoretical PSD of Polar RZ
theoretical_PSD_Polar_RZ =Fs* (A^2 * bitDuration* 16 / 49) * (sinc(freq_axis * bitDuration*4 / 7).^2);

% ------------------ Plot Simulated PSD Only ------------------

figure;

% Plot Polar NRZ PSD
subplot(3,1,1);
plot(freq_axis, PSD_Polar_NRZ, 'b', 'LineWidth', 2);
ylim([0 200]);
title('Polar NRZ PSD (Simulated)');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Simulated'); grid on;

% Plot Unipolar NRZ PSD
subplot(3,1,2);
plot(freq_axis, PSD_Uni_Polar, 'b', 'LineWidth', 2);
ylim([0 100]);
title('Unipolar NRZ PSD (Simulated)');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Simulated'); grid on;

% Plot Polar RZ PSD
subplot(3,1,3);
plot(freq_axis, PSD_Polar_RZ, 'b', 'LineWidth', 2);
ylim([0 50]);
title('Polar RZ PSD (Simulated)');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Simulated'); grid on;

% ------------------ Plot Computed vs Theoretical PSD ------------------

figure;

% Plot Polar NRZ PSD
subplot(3,1,1);
plot(freq_axis, PSD_Polar_NRZ, 'b', 'LineWidth', 2); hold on;
plot(freq_axis, theoretical_PSD_Polar_NRZ, 'r--', 'LineWidth', 2);
ylim([0 200]);
title('Polar NRZ PSD');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Computed', 'Theoretical'); grid on;

% Plot Unipolar NRZ PSD
subplot(3,1,2);

plot(freq_axis, PSD_Uni_Polar, 'b', 'LineWidth', 2); hold on;
plot(freq_axis, theoretical_PSD_Uni_Polar, 'r--', 'LineWidth', 2);
ylim([0 100]);
title('Unipolar NRZ PSD');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Computed', 'Theoretical'); grid on;

% Plot Polar RZ PSD
subplot(3,1,3);
plot(freq_axis, PSD_Polar_RZ, 'b', 'LineWidth', 2); hold on;
plot(freq_axis, theoretical_PSD_Polar_RZ, 'r--', 'LineWidth', 2);
ylim([0 50]);
title('Polar RZ PSD');
xlabel('Frequency (Hz)'); ylabel('PSD');
legend('Computed', 'Theoretical'); grid on;

%% ------------------ Compute Theoretical Bandwidth ------------------
B_theoretical_Polar_NRZ = 1 / bitDuration;  % First null for Polar NRZ
B_theoretical_Uni_Polar = 1 / bitDuration;  % First null for Unipolar NRZ
B_theoretical_Polar_RZ  = 7 /(4*  bitDuration);  % First null for Polar RZ

%% ------------------ Compute Simulated Bandwidth Till First Null ------------------

% Define a strict threshold close to zero for polar_NRZ (0.1m% of max PSD)
threshold_polar_NRZ = max(PSD_Polar_NRZ) * 1e-6; 

% Find the first null (first zero-crossing or minimum point)
null_index_NRZ = find(PSD_Polar_NRZ(ceil(N/2):end) < threshold_polar_NRZ, 1, 'first') + ceil(N/2) - 1;
B_simulated_Polar_NRZ = abs(freq_axis(null_index_NRZ));

% Define a strict threshold close to zero for uni_polar_NRZ (0.1m% of max PSD)
threshold_uni_polar_NRZ = max(PSD_Uni_Polar) * 1e-6; 

null_index_Uni = find(PSD_Uni_Polar(ceil(N/2):end) < threshold_uni_polar_NRZ, 1, 'first') + ceil(N/2) - 1;
B_simulated_Uni_Polar = abs(freq_axis(null_index_Uni));


% Define a strict threshold close to zero for polar_RZ (0.1m% of max PSD)
threshold_polar_RZ= max(PSD_Polar_RZ) * 1e-6; 

null_index_prz = find(PSD_Polar_RZ(ceil(N/2):end) < threshold_polar_RZ, 1, 'first') + ceil(N/2) - 1;
B_simulated_polar_RZ = abs(freq_axis(null_index_prz));




%% ------------------ Display Bandwidth Results ------------------
disp('----------------------------------------------------');
disp('Bandwidth Till First Null (Hz)');
disp(['Theoretical Polar NRZ: ', num2str(B_theoretical_Polar_NRZ), ' Hz']);
disp(['Simulated Polar NRZ: ', num2str(B_simulated_Polar_NRZ), ' Hz']);
disp('----------------------------------------------------');
disp(['Theoretical Unipolar NRZ: ', num2str(B_theoretical_Uni_Polar), ' Hz']);
disp(['Simulated Unipolar NRZ: ', num2str(B_simulated_Uni_Polar), ' Hz']);
disp('----------------------------------------------------');
disp(['Theoretical Polar RZ: ', num2str(B_theoretical_Polar_RZ), ' Hz']);
disp(['Simulated Polar NRZ: ', num2str(B_simulated_polar_RZ), ' Hz']);

disp('----------------------------------------------------');
%%---------------------------------------------------------------------


%% Function Performing Self Auto Corr. for all Samples RV's (All Coloumns)
function autocorr_RVs = autocorr_func_zero_tau(matrix, plot_title)
    % Initialize output array
    autocorr_RVs = zeros(1, 700);
    
    % Loop over each RV
    for sample = 1:700
        sample_values = matrix(:, sample); % Extract the required column
        squared_values = sample_values .* sample_values; % Dot product
        autocorr_RVs(sample) = sum(squared_values) / 500;
    end
    
    % Plot the results
    figure; 
    plot(1:700, autocorr_RVs, 'b-', 'LineWidth', 1.5);
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude"); % Use the passed ylabel argument
    title(plot_title);
    grid on;
end

%% Function Performing Auto Corr. between each consecutive (tau = 1) Samples RV's (All Coloumns)
function dot_products = auto_corr_unity_tau(X, plot_title)
    [~, cols] = size(X); %storing the coloumns in a 2d array
    

    dot_products = zeros(1, cols - 1); % Initialize output array
    
    % Compute dot product for each consecutive column pair
    for i = 1:cols-1
        dot_products(i) = dot(X(:, i), X(:, i+1)) ;
        %the auto-correlation value is normalized to be within [0:1]
    end
    dot_products = dot_products/500;
    % Plot the results
    figure; 
    plot(1:length(dot_products), dot_products, 'b-', 'LineWidth', 1.5);
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude"); 
    title(plot_title);
    grid on;
end

%% Function Performing Auto Corr. between Samples RV's (All Coloumns) starting from t1 or t2 
function autocorr_from_t1_or_t2 = autocorr_func_multiple_tau(matrix, plot_title)
    
    autocorr_t1 = zeros(1, 700);
    autocorr_t2 = zeros(1, 700);
    
    
    for sample = 1:2
        autocorr_current = zeros(1, 700);
        for tau = 0:699
            sample_values1 = matrix(:, sample); % Extract the reference column
            if (tau == 699 && sample == 2) % Special case for circular shift 
                sample_values2 = matrix(:, sample-1);
            else
                sample_values2 = matrix(:, sample+tau);
            end
            squared_values = sample_values1 .* sample_values2; 
            autocorr_current(tau+1) = sum(squared_values) / 500;
        end
        
        if sample == 1
            autocorr_t1 = autocorr_current;
        else
            autocorr_t2 = autocorr_current;
        end
    end
    
    % Flip and mirror for plotting
    autocorr_flipped(1, :) = [fliplr(autocorr_t1), autocorr_t1];
    autocorr_flipped(2, :) = [fliplr(autocorr_t2), autocorr_t2];
    
    
    figure;
    plot(-699:700, autocorr_flipped(1, :), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(-699:700, autocorr_flipped(2, :), 'r-', 'LineWidth', 1.5);
    hold off;
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude");
    title([plot_title ' - Comparison']); 
    legend('Autocorr @ t=t1', 'Autocorr @ t=t2'); 
    grid on;

    
    figure;
    plot(-699:700, autocorr_flipped(1, :), 'b-', 'LineWidth', 1.5);
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude");
    title([plot_title ' - Starting from t1']);
    grid on;

    
    figure;
    plot(-699:700, autocorr_flipped(2, :), 'r-', 'LineWidth', 1.5);
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude");
    title([plot_title ' - Starting from t2']); 
    grid on;

    % Return the avg of the autocorrelation from t1 & t2 for PSD calcuations
    autocorr_from_t1_or_t2 = (autocorr_t1 + autocorr_t2) / 2;
end

%% Function Performing Self Auto Corr. (tau = 0 ) between each Realization (All Rows)
function ergodic_value = isergotic(matrix, plot_title)
    [rows, cols] = size(matrix);
    row_auto_corr = zeros(1, rows); 

    for r = 1:rows
        wavefrom = matrix(r, :); % Extract row
        squaredSum = sum(wavefrom .^ 2); % Squaring each element and get sum of them 
        row_auto_corr(r) = squaredSum / 700; % Averaging the sum of the squared elements
    end

    ergodic_value = sum(row_auto_corr) / rows; % Normalize by total rows

    % Plot the ergodic value
    figure;
    plot(1:cols, ones(1, cols) * ergodic_value, 'r-', 'LineWidth', 1.5); 
    xlabel('Time'); 
    ylabel("Autocorrelation Amplitude");
    yMin = floor(0 * 10) / 10; % Round down to nearest 0.1
    yMax = ceil(20 * 10) / 10;  % Round up to nearest 0.1
    ylim([yMin, yMax]);
    yticks(yMin:0.5:yMax); % Set ticks at intervals of 0.1
    title(plot_title); 
    grid on;
end

%% General Function Plotting Auto Corr. values with the values of lags (tau)
function plot_autocorr(autocorr_RVs, ergodic_value, title_autocorr, title_ergodic)
    cols = length(autocorr_RVs);
    time = 1:cols;
    
    figure;
    plot(time, autocorr_RVs, 'b-', 'LineWidth', 1.5);
    hold on;

    plot(time, ones(1, cols) * ergodic_value, 'r-', 'LineWidth', 1.5);

    xlabel('Time');
    ylabel('Autocorrelation Amplitude');
    title([title_autocorr, ' vs ', title_ergodic]);
    legend(title_autocorr, title_ergodic);
    grid on;
    hold off;
end



%% Function to Calculate Statistical Mean (Ensemble Mean)
function statistical_mean = calculate_statistical_mean(signal, num_realization, total_samples)
    Sum_signal = zeros(1, total_samples);
    for t = 1:total_samples
        for i = 1:num_realization
            Sum_signal(t) = Sum_signal(t) + signal(i, t);
        end
    end
    statistical_mean = Sum_signal /  num_realization;
end

%% Function to Calculate Time Mean
function time_mean = Calculate_time_mean(signal, num_realization, total_samples)
    Sum_time_signal = zeros(num_realization, 1);
    for i = 1:num_realization
        for t = 1:total_samples
            Sum_time_signal(i) = Sum_time_signal(i) + signal(i, t);
        end
    end
    time_mean = Sum_time_signal / total_samples;
end

%% General Function Plotting Any Two Matrices
function plot_sim_and_theor(matrix1, matrix2, title1, title2, title3)
    figure;
    hold on;
    
    x = -699:700;
    
    % Flip matrix and concatenate it with itself
    matrix1_flipped = flip(matrix1);
    matrix1_padded = [matrix1_flipped, matrix1]; 
    matrix2_flipped = flip(matrix2);
    matrix2_padded = [matrix2_flipped, matrix2]; 

    plot(x, matrix1_padded, 'b-', 'LineWidth', 1.5);
    plot(x, matrix2_padded, 'r--', 'LineWidth', 1.5);
    
    legend(title1, title2, 'Location', 'best');
    xlabel('tau');
    ylabel('AutoCorr Amplitude');
    title(title3);
    grid on;
    hold off;
end

%% Function Performing  Auto Corr. between all Samples RV's & All Realizations by the values of lag [0:500] to test Ergodicity
function ergodic_auto_corr_arr = ergodic_auto_corr(matrix2D, tau_values, plot_title)
    [rows, cols] = size(matrix2D);
    num_lags = length(tau_values);
    col_corr = zeros(1, num_lags);
    row_corr = zeros(1, num_lags);

    % Column dot product (Samples RV's Auto Corr.)
    for i = 1:num_lags
        tau = tau_values(i);
        if tau < cols
            col1 = matrix2D(:, 1);
            col2 = matrix2D(:, tau + 1);
            col_corr(i) = dot(col1, col2) / rows;
        else
            col_corr(i) = NaN; % If lag exceeds dimensions (500)
        end
    end

    % Row dot product (Realizations Auto Corr.)
    for i = 1:num_lags
        tau = tau_values(i);
        if tau < rows
            row1 = matrix2D(1, :);
            row2 = matrix2D(tau + 1, :);
            row_corr(i) = dot(row1, row2) / cols;
        else
            row_corr(i) = NaN; % If lag exceeds dimensions
        end
    end

    % Plot results
    figure;
    hold on;
    plot(tau_values, col_corr, 'r-', 'LineWidth', 2, 'DisplayName', 'Across Ensemble Correlation');
    plot(tau_values, row_corr, 'b-', 'LineWidth', 2, 'DisplayName', 'Across Time Correlation');
    legend;
    title(plot_title);
    xlabel('Lag (Tau)');
    ylabel('Normalized Dot Product');
    grid on;
    hold off;

    % Return the correlation results
    ergodic_auto_corr_arr = [col_corr; row_corr];
end


