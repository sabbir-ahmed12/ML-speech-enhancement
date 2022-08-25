%% DATASET LOADING
datafolder = 'C:\Users\sabbir\Desktop\project\commonvoice';
adsTrain   = audioDatastore(fullfile(datafolder, 'train'), 'IncludeSubfolders',true);
adsTrain   = shuffle(adsTrain);
adsTrain   = subset(adsTrain, 1:1000);

% Reads the first file
[audio, adsTraininfo] = read(adsTrain);
sound(audio, adsTraininfo.SampleRate);

% Plotting the speech signal
figure
t = (1/adsTraininfo.SampleRate) * (0:numel(audio)-1);
plot(t, audio)
title("Example Speech Signal")
xlabel("Time (s)")
ylabel("Amplitude")
grid on

%% PARAMETERS FOR GENERATING TARGET AND PREDICTORS
windowLength = 256;
win          = hamming(windowLength, "periodic");
overlap      = round(0.75 * windowLength);
fftLength    = windowLength;
inputFs      = 48e3;
fs           = 8e3;
numFeatures  = (fftLength / 2) + 1;
numSegments  = 8;

%% Creating object using DSP System Toolbox for downsampling audio signals
src = dsp.SampleRateConverter("InputSampleRate", inputFs, ...
                              "OutputSampleRate", fs, ... 
                              "Bandwidth", 7920);
                          
audio = read(adsTrain);

%% Make the audio length multiple of the sample rate converter decimation factor
decimationFactor = inputFs / fs;
L                = floor(numel(audio) / decimationFactor);
audio            = audio(1:decimationFactor * L);

% Converting the audio signal into 8kHz
audio = src(audio);
reset(src);

% Loading washing machine noise
[noise, nFs] = audioread("WashingMachine-16-8-mono-1000secs.wav");

% Creating a random noise segment from the washing machine noise vector
randind      = randi(numel(noise) - numel(audio), [1 1]);
noiseSegment = noise(randind : randind + numel(audio) - 1);

% Adding noise to the speech signal to make the SNR equal to 0 dB
noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(audio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower / noisePower);
noisyAudio   = audio + noiseSegment;

%% Calculating STFT(col - time, row - frequency, nRow - fftLength)
cleanSTFT = stft(audio, 'Window', win, 'OverlapLength', overlap, 'FFTLength', fftLength);
cleanSTFT = abs(cleanSTFT(numFeatures-1:end, :));
noisySTFT = stft(noisyAudio, 'Window', win, 'OverlapLength', overlap, 'FFTLength', fftLength);
noisySTFT = abs(noisySTFT(numFeatures-1:end, :));

%% Generating 8 segment training predictor signals from the noisy STFT
noisySTFT = [noisySTFT(:, 1:numSegments - 1), noisySTFT];
stftSegments = zeros(numFeatures, numSegments, size(noisySTFT, 2) - numSegments + 1);
for index = 1:size(noisySTFT, 2) - numSegments + 1
    stftSegments(:, :, index) = (noisySTFT(:, index:index + numSegments - 1)); 
end

%% Dimension of target and predictor
targets = cleanSTFT;
size(targets)
predictors = stftSegments;
size(predictors)

%% Extract features using tall array
reset(adsTrain)
T = tall(adsTrain)

[targets, predictors] = cellfun(@(aud)HelperGenerateSpeechDenoisingFeatures(aud, noise, src), T, "UniformOutput", false);
[targets, predictors] = gather(targets, predictors);

% Compute the mean and standard deviation of the predictors and targets.
predictors = cat(3,predictors{:});
noisyMean = mean(predictors(:));
noisyStd = std(predictors(:));
predictors(:) = (predictors(:) - noisyMean)/noisyStd;

targets = cat(2,targets{:});
cleanMean = mean(targets(:));
cleanStd = std(targets(:));
targets(:) = (targets(:) - cleanMean)/cleanStd;

%% Reshape predictors and targets to the dimensions expected by the deep learning networks.
predictors = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targets = reshape(targets,1,1,size(targets,1),size(targets,2));

% Randomly split the data into training and validation sets.
inds = randperm(size(predictors,4));
L = round(0.99 * size(predictors,4));

trainPredictors = predictors(:,:,:,inds(1:L));
trainTargets = targets(:,:,:,inds(1:L));

validatePredictors = predictors(:,:,:,inds(L+1:end));
validateTargets = targets(:,:,:,inds(L+1:end));

%% Speech Denoising with Fully Connected Layers
layers = [
    imageInputLayer([numFeatures,numSegments])
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numFeatures)
    regressionLayer
    ];

miniBatchSize = 128;
options = trainingOptions("adam", ...
    "MaxEpochs",15, ...
    "InitialLearnRate",1e-5,...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "ValidationFrequency",floor(size(trainPredictors,4)/miniBatchSize),...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.9,...
    "LearnRateDropPeriod",1,...
    "ExecutionEnvironment","gpu",...
    "ValidationData",{validatePredictors,validateTargets});

denoiseNetFullyConnected = trainNetwork(trainPredictors,trainTargets,layers,options);


%% Speech Denoising using Convolutional Network
% layersConv = [imageInputLayer([numFeatures,numSegments])
%           convolution2dLayer([9 8],18,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer
%           
%           repmat( ...
%           [convolution2dLayer([5 1],30,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer
%           convolution2dLayer([9 1],8,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer
%           convolution2dLayer([9 1],18,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer],4,1)
%           
%           convolution2dLayer([5 1],30,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer
%           convolution2dLayer([9 1],8,"Stride",[1 100],"Padding","same")
%           batchNormalizationLayer
%           reluLayer
%           
%           convolution2dLayer([129 1],1,"Stride",[1 100],"Padding","same")
%           
%           regressionLayer
%           ];
% 
% optionsConv = trainingOptions("adam", ...
%     "MaxEpochs",3, ...
%     "InitialLearnRate",1e-5, ...
%     "MiniBatchSize",miniBatchSize, ...
%     "Shuffle","every-epoch", ...
%     "Plots","training-progress", ...
%     "Verbose",false, ...
%     "ValidationFrequency",floor(size(trainPredictors,4)/miniBatchSize), ...
%     "LearnRateSchedule","piecewise", ...
%     "LearnRateDropFactor",0.9, ...
%     "LearnRateDropPeriod",1, ...
%     "ValidationData",{validatePredictors,permute(validateTargets,[3 1 2 4])});
% 
% denoiseNetFullyConvolutional = trainNetwork(trainPredictors,permute(trainTargets,[3 1 2 4]),layersConv,optionsConv);


%% TESTING THE DENOISING NETWORK
adsTest = audioDatastore(fullfile(datafolder,'test'),'IncludeSubfolders',true);

[cleanAudio,adsTestInfo] = read(adsTest);

L = floor(numel(cleanAudio)/decimationFactor);
cleanAudio = cleanAudio(1:decimationFactor*L);

cleanAudio = src(cleanAudio);
reset(src);

randind = randi(numel(noise) - numel(cleanAudio), [1 1]);
noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);

noisePower = sum(noiseSegment.^2);
cleanPower = sum(cleanAudio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio = cleanAudio + noiseSegment;

% Generating STFT from noisy signals
noisySTFT = stft(noisyAudio,'Window',win,'OverlapLength',overlap,'FFTLength',fftLength);
noisyPhase = angle(noisySTFT(numFeatures-1:end,:));
noisySTFT = abs(noisySTFT(numFeatures-1:end,:));

% Generating 8 segment predictor signal
noisySTFT = [noisySTFT(:,1:numSegments-1) noisySTFT];
predictors = zeros( numFeatures, numSegments , size(noisySTFT,2) - numSegments + 1);
for index = 1:size(noisySTFT,2) - numSegments + 1
    predictors(:,:,index) = noisySTFT(:,index:index + numSegments - 1); 
end

predictors(:) = (predictors(:) - noisyMean) / noisyStd;

%% Predicting the denoised STFT
predictors = reshape(predictors, [numFeatures,numSegments,1,size(predictors,3)]);
STFTFullyConnected = predict(denoiseNetFullyConnected, predictors);
% STFTFullyConvolutional = predict(denoiseNetFullyConvolutional, predictors);

% Rescaling the signals
STFTFullyConnected(:) = cleanStd * STFTFullyConnected(:) + cleanMean;
% STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) + cleanMean;

% Calculating centered STFT
STFTFullyConnected = STFTFullyConnected.' .* exp(1j*noisyPhase);
STFTFullyConnected = [conj(STFTFullyConnected(end-1:-1:2,:)); STFTFullyConnected];
% STFTFullyConvolutional = squeeze(STFTFullyConvolutional) .* exp(1j*noisyPhase);
% STFTFullyConvolutional = [conj(STFTFullyConvolutional(end-1:-1:2,:)) ; STFTFullyConvolutional];

%% Computing denoised speech signals
denoisedAudioFullyConnected = istft(STFTFullyConnected,  ...
                                    'Window',win,'OverlapLength',overlap, ...
                                    'FFTLength',fftLength,'ConjugateSymmetric',true);
                                
% denoisedAudioFullyConvolutional = istft(STFTFullyConvolutional,  ...
%                                         'Window',win,'OverlapLength',overlap, ...
%                                         'FFTLength',fftLength,'ConjugateSymmetric',true);
                                    

%% Plotting the denoised time domain signal
t = (1/fs) * (0:numel(denoisedAudioFullyConnected)-1);

figure

subplot(3,1,1)
plot(t,cleanAudio(1:numel(denoisedAudioFullyConnected)))
title("Clean Speech")
xlabel('Time (s)')
grid on

subplot(3,1,2)
plot(t,noisyAudio(1:numel(denoisedAudioFullyConnected)))
title("Noisy Speech")
xlabel('Time (s)')
grid on

subplot(3,1,3)
plot(t,denoisedAudioFullyConnected)
title("Denoised Speech")
xlabel('Time (s)')
grid on

% subplot(4,1,4)
% plot(t,denoisedAudioFullyConvolutional)
% title("Denoised Speech (Convolutional Layers)")
% grid on
% xlabel("Time (s)")

%% Plotting the spectrogram
h = figure;

subplot(3,1,1)
spectrogram(cleanAudio,win,overlap,fftLength,fs);
title("Clean Speech")
grid on

subplot(3,1,2)
spectrogram(noisyAudio,win,overlap,fftLength,fs);
title("Noisy Speech")
grid on

subplot(3,1,3)
spectrogram(denoisedAudioFullyConnected,win,overlap,fftLength,fs);
title("Denoised Speech")
grid on

% subplot(4,1,4)
% spectrogram(denoisedAudioFullyConvolutional,win,overlap,fftLength,fs);
% title("Denoised Speech (Convolutional Layers)")
% grid on

p = get(h,'Position');
set(h,'Position',[p(1) 65 p(3) 800]);