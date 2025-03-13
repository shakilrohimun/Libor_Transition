%% Data importation
data = readtable('InterestRateData.csv');
if ~isdatetime(data.Date)
    data.Date = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd'); % Adjust format if needed
end

%% raw_data_fred --------------------------------------------------------------------------
requiredColumns = {'LIOR3M', 'LIOR3MUKM', 'SOFR', 'ECBESTRVOLWGTTRMDMNRT'};

% Plot
figure;

plot(data.Date, data.LIOR3M, '-x', 'LineWidth', 1.5, 'DisplayName', '3-Month USD LIBOR');
hold on;
plot(data.Date, data.LIOR3MUKM, '-x', 'LineWidth', 1.5, 'DisplayName', '3-Month GBP LIBOR');

plot(data.Date, data.SOFR, '-', 'LineWidth', 1.5, 'DisplayName', 'SOFR');
plot(data.Date, data.ECBESTRVOLWGTTRMDMNRT, '-', 'LineWidth', 1.5, 'DisplayName', 'Euro STR');
xlabel('Date');
ylabel('Rate (%)');
title('Raw Data of LIBOR (USD & GBP), SOFR, and Euro STR');
legend('Location','best');
grid on;
hold off;

%% libor_vs_sofr --------------------------------------------------------------------------
liborUSD = data.LIOR3M;
liborGBP = data.LIOR3MUKM;
sofr     = data.SOFR;

liborUSD = fillmissing(liborUSD, 'previous');
liborGBP = fillmissing(liborGBP, 'previous');

dates = data.Date;

% Volatility Calculation
rLiborUSD = diff(log(liborUSD + eps)); 
rLiborGBP = diff(log(liborGBP + eps));
rSofr     = diff(log(sofr + eps));

window = 30;

volLiborUSD = movstd(rLiborUSD, window, 'omitnan');
volLiborGBP = movstd(rLiborGBP, window, 'omitnan');
volSofr     = movstd(rSofr, window, 'omitnan');

volDates = dates(2:end);

% PLot
figure('Name','Evolution of Rates and Volatility','NumberTitle','off');

% Subplot 1: Evolution of the Rates
subplot(2,1,1)
plot(dates, liborUSD, 'b-', 'LineWidth', 1.5, 'DisplayName', '3-Month USD LIBOR');
hold on;
plot(dates, liborGBP, 'r-', 'LineWidth', 1.5, 'DisplayName', '3-Month GBP LIBOR');
plot(dates, sofr,     'g-', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Rate (%)');
title('Historical Trends of 3-Month USD LIBOR, 3-Month GBP LIBOR, and SOFR');
legend('Location','best');
grid on;

% Subplot 2: Evolution of the Volatility
subplot(2,1,2)
plot(volDates, volLiborUSD, 'b-', 'LineWidth', 1.5);
hold on;
plot(volDates, volLiborGBP, 'r-', 'LineWidth', 1.5);
plot(volDates, volSofr,     'g-', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Volatility (std of log returns)');
title('Historical Evolution of Volatility Over a 30-Day Rolling Window');
legend('Volatility of 3-Month USD LIBOR','Volatility of 3-Month GBP LIBOR','Volatility of SOFR','Location','best');
grid on;

%% sonia_volatility ----------------------------------------------------------------------
liborUSD = data.LIOR3M;     
liborGBP = data.LIOR3MUKM; 
sonia    = data.IUDSOIA;   

liborUSD = fillmissing(liborUSD, 'previous');
liborGBP = fillmissing(liborGBP, 'previous');
sonia    = fillmissing(sonia, 'previous');

dates = data.Date;

rLiborUSD = diff(log(liborUSD + eps));
rLiborGBP = diff(log(liborGBP + eps));
rSONIA    = diff(log(sonia + eps));

window = 30;

volLiborUSD = movstd(rLiborUSD, window, 'omitnan');
volLiborGBP = movstd(rLiborGBP, window, 'omitnan');
volSONIA    = movstd(rSONIA, window, 'omitnan');

volDates = dates(2:end);

% Plot
figure('Name','LIBOR & SONIA - Historical Trends and Volatility','NumberTitle','off');

% Subplot 1: Historical Trends 
subplot(2,1,1);
plot(dates, liborUSD, 'r-', 'LineWidth', 1.5, 'DisplayName', '3-Month USD LIBOR');
hold on;
plot(dates, liborGBP, 'b-', 'LineWidth', 1.5, 'DisplayName', '3-Month GBP LIBOR');
plot(dates, sonia, 'c--', 'LineWidth', 1.5, 'DisplayName', 'SONIA');
xlabel('Date');
ylabel('Rate (%)');
title('Historical Trends of 3-Month LIBOR (USD & GBP) and SONIA');
legend('Location','best');
grid on;
hold off;

% Subplot 2: Volatility 
subplot(2,1,2);
plot(volDates, volLiborUSD, 'r-', 'LineWidth', 1, 'DisplayName', 'Volatility of 3-Month USD LIBOR');
hold on;
plot(volDates, volLiborGBP, 'b-', 'LineWidth', 1, 'DisplayName', 'Volatility of 3-Month GBP LIBOR');
plot(volDates, volSONIA, 'c--', 'LineWidth', 1, 'DisplayName', 'Volatility of SONIA');
xlabel('Date');
ylabel('Volatility (std of log returns)');
title('Historical Evolution of Volatility Over a 30-Day Rolling Window');
legend('Location','best');
grid on;
hold off;

%% global_rates_transition ---------------------------------------------------------------
liborUSD = fillmissing(data.LIOR3M, 'previous');      
liborGBP = fillmissing(data.LIOR3MUKM, 'previous');    
sofr     = fillmissing(data.SOFR, 'previous');         
sonia    = fillmissing(data.IUDSOIA, 'previous');       
dates    = data.Date;

rLiborUSD = diff(log(liborUSD + eps));
rLiborGBP = diff(log(liborGBP + eps));
rSONIA    = diff(log(sonia + eps));
rSOFR     = diff(log(sofr + eps));

window = 30;
volLiborUSD = movstd(rLiborUSD, window, 'omitnan');
volLiborGBP = movstd(rLiborGBP, window, 'omitnan');
volSONIA    = movstd(rSONIA, window, 'omitnan');
volSOFR     = movstd(rSOFR, window, 'omitnan');

volDates = dates(2:end);

transitionDate = datetime('2021-07-01');
transitionLabel = '2021-07-01: SOFR replaces LIBOR USD';

% Plot
figure('Name','Historical Trends and Volatility','NumberTitle','off');

% Subplot 1: Historical Trends
subplot(2,1,1);
hold on;
plot(dates, liborUSD, 'b-', 'LineWidth', 1.5, 'DisplayName','3-Month USD LIBOR');
plot(dates, liborGBP, 'r-', 'LineWidth', 1.5, 'DisplayName','3-Month GBP LIBOR');
plot(dates, sonia, 'c--', 'LineWidth', 1.5, 'DisplayName','SONIA');
plot(dates, sofr, 'g-', 'LineWidth', 1.5, 'DisplayName','SOFR');

% Transition date
xline(transitionDate, 'k--', 'LineWidth', 1.2, 'DisplayName', transitionLabel);

xlabel('Date');
ylabel('Rate (%)');
title('Historical Trends of 3-Month LIBOR (USD & GBP), SONIA, and SOFR');
legend('Location','best');
grid on;
hold off;

% Subplot 2: Rolling Volatility
subplot(2,1,2);
hold on;
plot(volDates, volLiborUSD, 'b-', 'LineWidth', 1.5, 'DisplayName','Volatility of 3-Month USD LIBOR');
plot(volDates, volLiborGBP, 'r-', 'LineWidth', 1.5, 'DisplayName','Volatility of 3-Month GBP LIBOR');
plot(volDates, volSONIA, 'c--', 'LineWidth', 1.5, 'DisplayName','Volatility of SONIA');
plot(volDates, volSOFR, 'g-', 'LineWidth', 1.5, 'DisplayName','Volatility of SOFR');

xline(transitionDate, 'k--', 'LineWidth', 1.2, 'DisplayName', transitionLabel);

xlabel('Date');
ylabel('Volatility (std of log returns)');
title('Historical Evolution of Volatility Over a 30-Day Rolling Window');
legend('Location','best');
grid on;
hold off;

%% cap_floor_transition ------------------------------------------------------------------
liborUSD = fillmissing(data.LIOR3M, 'previous');  
sofr     = fillmissing(data.SOFR, 'previous');    
dates    = data.Date;

rLiborUSD = diff(log(liborUSD + eps)); 
rSofr     = diff(log(sofr + eps));

window = 30;
volLiborUSD = movstd(rLiborUSD, window, 'omitnan');
volSofr     = movstd(rSofr, window, 'omitnan');

sigmaLiborAnnual = volLiborUSD * sqrt(252);
sigmaSofrAnnual  = volSofr * sqrt(252);

K = 0.02;    % Strike rate (2%)
T = 1;       % Maturity in years
tau = 1;     % Accrual factor (for simplicity, set to 1)

% Use the rates starting from the second observation (because diff reduces one observation)
F_LIBOR = liborUSD(2:end) / 100; % Convert percentage to decimal
F_SOFR  = sofr(2:end) / 100;     
priceDates = dates(2:end);       % Adjusted dates for pricing

% Preallocate arrays for caplet prices
capletPrice_LIBOR = zeros(length(priceDates), 1);
capletPrice_SOFR  = zeros(length(priceDates), 1);

% Loop over each day and compute the caplet price using Black's formula
for i = 1:length(priceDates)
    % LIBOR-based caplet price using annualized volatility
    sigmaL = sigmaLiborAnnual(i);
    if sigmaL > 0 && F_LIBOR(i) > 0
        d1 = (log(F_LIBOR(i)/K) + 0.5*sigmaL^2*T) / (sigmaL*sqrt(T));
        d2 = d1 - sigmaL*sqrt(T);
        capletPrice_LIBOR(i) = tau * (F_LIBOR(i)*normcdf(d1) - K*normcdf(d2));
    else
        capletPrice_LIBOR(i) = 0;
    end
    
    % SOFR-based caplet price. If SOFR is missing (NaN), keep it as NaN.
    if isnan(F_SOFR(i)) || F_SOFR(i) <= 0 || sigmaSofrAnnual(i) <= 0
        capletPrice_SOFR(i) = NaN;
    else
        sigmaS = sigmaSofrAnnual(i);
        d1 = (log(F_SOFR(i)/K) + 0.5*sigmaS^2*T) / (sigmaS*sqrt(T));
        d2 = d1 - sigmaS*sqrt(T);
        capletPrice_SOFR(i) = tau * (F_SOFR(i)*normcdf(d1) - K*normcdf(d2));
    end
end

% Plot
figure('Name','Impact of Transition from LIBOR to SOFR on Cap and Floor Valuation','NumberTitle','off');
plot(priceDates, capletPrice_LIBOR, 'b-', 'LineWidth', 1.5); % LIBOR-based caplet price in blue
hold on;
plot(priceDates, capletPrice_SOFR, 'r-', 'LineWidth', 1.5);  % SOFR-based caplet price in red
xlabel('Date');
ylabel('Caplet Price');
title('Impact of Transition from LIBOR to SOFR on Cap and Floor Valuation');
legend('LIBOR-based Caplet','SOFR-based Caplet','Location','best');
grid on;

%% black_model_caplet ------------------------------------------------------------------
F = 0.05;        % Forward LIBOR rate
sigma = 0.2;     % Volatility
T = 1;           % Maturity in years
P = exp(-0.05*T);% Discount factor (assuming a constant rate of 5%)
K = linspace(0.01, 0.10, 100); % Strike varying from 1% to 10%

d1 = (log(F./K) + 0.5*sigma^2*T) ./ (sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);
price = P * (F*normcdf(d1) - K.*normcdf(d2));

figure;
plot(K, price, 'LineWidth', 2);
xlabel('Strike K');
ylabel('Caplet Price');
title('Black Model: Caplet Price vs. Strike');
grid on;

%% vasicek_simulation ------------------------------------------------------------------
kappa = 0.3;
theta = 0.05;
sigma = 0.02;
r0 = 0.03;
T = 10;    
dt = 0.01;
time = 0:dt:T;
N = length(time);
r = zeros(1, N);
r(1) = r0;

for i = 2:N
    dr = kappa*(theta - r(i-1))*dt + sigma*sqrt(dt)*randn;
    r(i) = r(i-1) + dr;
end

figure;
plot(time, r, 'LineWidth', 2);
xlabel('Time (years)');
ylabel('Short Rate');
title('Vasicek Model Simulation');
grid on;

%% cir_simulation ------------------------------------------------------------------
kappa = 0.5;
theta = 0.05;
sigma = 0.1;
r0 = 0.03;
T = 10;
dt = 0.01;
time = 0:dt:T;
N = length(time);
r = zeros(1, N);
r(1) = r0;

for i = 2:N
    dr = kappa*(theta - r(i-1))*dt + sigma*sqrt(max(r(i-1),0))*sqrt(dt)*randn;
    r(i) = max(r(i-1) + dr, 0);  % Ensure non-negative rates
end

figure;
plot(time, r, 'LineWidth', 2);
xlabel('Time (years)');
ylabel('Short Rate');
title('CIR Model Simulation');
grid on;

%% nelson_siegel_svensson --------------------------------------------------------
data = readtable('InterestRateData.csv');

% Identify columns that start with 'DGS'
varNames = data.Properties.VariableNames;
yieldCols = varNames(startsWith(varNames, 'DGS'));

% Plot of all DSG
figure('Name','DGS Time Series','NumberTitle','off');
hold on;
legendEntries = {};

if ~isdatetime(data.Date)
    data.Date = datetime(data.Date);
end

for i = 1:length(yieldCols)
    colName = yieldCols{i};
    plot(data.Date, data.(colName), 'LineWidth', 1.5);
    legendEntries{end+1} = colName; %#ok<AGROW>
end

xlabel('Date');
ylabel('Yield (%)');
title('Historical Time Series of DGS Columns');
legend(legendEntries, 'Location', 'best');
grid on;
hold off;

idx = [];
for i = height(data):-1:1
    row = data(i, :);
    rowValues = row{:, yieldCols};
    if any(~isnan(rowValues))
        idx = i;
        break;
    end
end

if isempty(idx)
    error('No row with valid DGS data found. All values are NaN.');
end

latestData = data(idx, :);
calibrationDate = latestData.Date;

% Extract Maturities and Yields
maturities = [];
yields = [];

for i = 1:length(yieldCols)
    colName = yieldCols{i};
    maturityStr = erase(colName, 'DGS');
    maturityValue = str2double(maturityStr);
    if ~isnan(maturityValue)
        currentYield = latestData.(colName);
        if ~isnan(currentYield)
            maturities(end+1) = maturityValue; %#ok<AGROW>
            yields(end+1) = currentYield;
        end
    end
end

if isempty(maturities)
    error('No valid DGS yields found in row %d. Check your data.', idx);
end

% Sort by ascending maturity
[maturities, sortIdx] = sort(maturities);
yields = yields(sortIdx);
maturities = maturities(:);  % column vector
yields = yields(:);           % column vector

% Definition of Nelson-Siegel and Svensson Model Functions
% Nelson-Siegel model
ns_model = @(b, T) b(1) ...
    + b(2)*((1 - exp(-T/b(4)))./(T/b(4))) ...
    + b(3)*(((1 - exp(-T/b(4)))./(T/b(4))) - exp(-T/b(4)));

% Svensson model
sv_model = @(b, T) b(1) ...
    + b(2)*((1 - exp(-T/b(5)))./(T/b(5))) ...
    + b(3)*(((1 - exp(-T/b(5)))./(T/b(5))) - exp(-T/b(5))) ...
    + b(4)*(((1 - exp(-T/b(6)))./(T/b(6))) - exp(-T/b(6)));

% Definition of Objective Functions (Sum of Squared Errors)
% Add a large penalty if tau parameters are non-positive
objective_ns = @(b) (b(4) <= 0)*1e10 + sum((ns_model(b, maturities) - yields).^2);
objective_sv = @(b) ((b(5) <= 0) + (b(6) <= 0))*1e10 + sum((sv_model(b, maturities) - yields).^2);

% Calibration
b0_ns = [mean(yields) + 0.2, -0.5, 0.2, 2];
b0_sv = [mean(yields) + 0.2, -0.5, 0.2, 0.1, 2, 6];

options = optimset('Display','iter','TolX',1e-8,'TolFun',1e-8,'MaxFunEvals',2000);
params_ns = fminsearch(objective_ns, b0_ns, options);
params_sv = fminsearch(objective_sv, b0_sv, options);

% Evaluate the Fitted Models on a Fine Grid
Tq = linspace(min(maturities), max(maturities), 100);
R_ns_fit = ns_model(params_ns, Tq);
R_sv_fit = sv_model(params_sv, Tq);

% Plot the Calibration Results (Figure 2)
figure('Name','Calibration Result','NumberTitle','off');
% Plot the actual yields as black crosses
plot(maturities, yields, 'xk', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName','Data');
hold on;
% Plot the Nelson-Siegel curve (red)
plot(Tq, R_ns_fit, 'r', 'LineWidth', 2, 'DisplayName', 'Nelson-Siegel');
% Plot the Svensson curve (green)
plot(Tq, R_sv_fit, 'g', 'LineWidth', 2, 'DisplayName', 'Svensson');
xlabel('Maturity (years)');
ylabel('Yield (%)');
title('Yield Curve Calibration');
legend('Location','best');
grid on;
hold off;

% Display the Calibrated Parameters
disp('--- Calibrated Nelson-Siegel Parameters ---');
disp(table(params_ns(1), params_ns(2), params_ns(3), params_ns(4), ...
    'VariableNames', {'beta0','beta1','beta2','tau'}));

disp('--- Calibrated Svensson Parameters ---');
disp(table(params_sv(1), params_sv(2), params_sv(3), params_sv(4), params_sv(5), params_sv(6), ...
    'VariableNames', {'beta0','beta1','beta2','beta3','tau1','tau2'}));

%% libor_sonia_gbp ----------------------------------------------------------------------
filename = 'InterestRateData.csv';
opts = detectImportOptions(filename);
opts = setvartype(opts, {'Date'}, 'datetime');
data = readtable(filename, opts);

filename = 'libord3m_gbp.csv';
opts = detectImportOptions(filename);
data_libor = readtable(filename, opts);
data_libor(:,2) = [];
data_libor.Properties.VariableNames{2} = 'LIBOR3M_GBP';
data_libor.Properties.VariableNames{1} = 'Date';

date_reference = datetime('2020-01-01','InputFormat','yyyy-MM-dd');
data_libor = data_libor(data_libor.Date >= date_reference, :);

disp("Preview of imported data:");
disp(head(data_libor));

% Plot LIBOR GBP
fig = figure;
plot(data_libor.Date, data_libor.LIBOR3M_GBP, '-', 'DisplayName', 'LIBOR GBP');
xlabel('Date'); ylabel('Interest Rate (%)'); title('LIBOR GBP Rate Evolution');
legend('Location', 'best'); grid on;

% Combine data
data_combined = innerjoin(data_libor, data, 'Keys', 'Date');
data_combined = data_combined(:, {'Date', 'LIBOR3M_GBP', 'IUDSOIA'});
data_combined.Properties.VariableNames{3} = 'SONIA_GBP';
data_combined.SONIA_GBP = fillmissing(data_combined.SONIA_GBP, 'spline');

disp('Combined table:');
disp(head(data_combined));

% Plot LIBOR GBP and SONIA
fig = figure;
plot(data_combined.Date, data_combined.LIBOR3M_GBP, 'b', 'LineWidth', 1.5);
hold on;
plot(data_combined.Date, data_combined.SONIA_GBP, 'r', 'LineWidth', 1.5);
xlabel('Date'); ylabel('Rate (%)'); title('Historical Trends of 3-Month USD LIBOR and SONIA');
legend('3-Month USD LIBOR', 'SONIA', 'Location', 'northwest'); grid on;
datetick('x', 'mmm yyyy', 'keepticks');

% Swap valuation
N = 1000000; rf = 0.015; delta_t = 0.25;
V_swap_libor = 0; V_swap_sonia = 0;
for i = 1:height(data_combined)
    r_discount = data_combined.SONIA_GBP(i);
    r_libor = data_combined.LIBOR3M_GBP(i);
    r_sonia = data_combined.SONIA_GBP(i);
    t_i = years(data_combined.Date(i) - data_combined.Date(1));
    V_swap_libor = V_swap_libor + N * exp(-r_discount * t_i) * (r_libor - rf) * delta_t;
    V_swap_sonia = V_swap_sonia + N * exp(-r_discount * t_i) * (r_sonia - rf) * delta_t;
end

fprintf('LIBOR swap value: %.2f GBP\n', V_swap_libor);
fprintf('SONIA swap value: %.2f GBP\n', V_swap_sonia);

