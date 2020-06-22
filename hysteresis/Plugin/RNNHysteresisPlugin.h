#ifndef RNNHYSTERESISPLUGIN_H_INCLUDED
#define RNNHYSTERESISPLUGIN_H_INCLUDED

#include "JuceHeader.h"
#include "PluginBase.h"
#include "MLUtils/Model.h"
#include "DCBlocker.h"

class RNNHysteresis : public PluginBase<RNNHysteresis>
{
public:
    RNNHysteresis();

    static void addParameters (Parameters& params);
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (AudioBuffer<float>& buffer) override;

private:
    std::atomic<float>* driveParam;
    std::atomic<float>* satParam;
    std::atomic<float>* widthParam;
    std::atomic<float>* osParam;
    std::atomic<float>* smallParam;

    std::unique_ptr<Model<float>> rnn[2];
    float T = 1.0f / 44100.0f;

    std::unique_ptr<dsp::Oversampling<float>> oversampling[2];

    AudioBuffer<float> monoBuffer;
    DCBlocker dcBlocker[2];

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RNNHysteresis)
};

#endif // RNNHYSTERESISPLUGIN_H_INCLUDED
