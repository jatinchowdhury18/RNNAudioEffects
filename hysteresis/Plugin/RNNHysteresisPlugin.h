#ifndef RNNHYSTERESISPLUGIN_H_INCLUDED
#define RNNHYSTERESISPLUGIN_H_INCLUDED

#include "JuceHeader.h"
#include "PluginBase.h"
#include "MLUtils/Model.h"

class RNNHysteresis : public PluginBase<RNNHysteresis>
{
public:
    RNNHysteresis();

    static void addParameters (Parameters& params);
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (AudioBuffer<float>& buffer) override;

private:
    std::unique_ptr<Model<float>> rnn;
    float T = 1.0f / 44100.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RNNHysteresis)
};

#endif // RNNHYSTERESISPLUGIN_H_INCLUDED
