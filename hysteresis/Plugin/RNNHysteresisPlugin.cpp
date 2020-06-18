#include "RNNHysteresisPlugin.h"
#include "Json2RnnParser.h"

namespace
{
    const String driveTag = "drive";
    const String satTag = "sat";
    const String widthTag = "width";
}

RNNHysteresis::RNNHysteresis()
{
    MemoryInputStream jsonInput (BinaryData::hysteresis_full_json, BinaryData::hysteresis_full_jsonSize, false);

    Json2RnnParser parser;
    rnn.reset (parser.parseJson (jsonInput));

    driveParam = vts.getRawParameterValue (driveTag);
    satParam   = vts.getRawParameterValue (satTag);
    widthParam = vts.getRawParameterValue (widthTag);
}

void RNNHysteresis::addParameters (Parameters& params)
{
    params.push_back (std::make_unique<AudioParameterFloat> (driveTag, "Drive",      0.0f, 1.0f, 0.5f));
    params.push_back (std::make_unique<AudioParameterFloat> (satTag,   "Saturation", 0.0f, 1.0f, 0.5f));
    params.push_back (std::make_unique<AudioParameterFloat> (widthTag, "Width",      0.0f, 1.0f, 0.5f));
}

void RNNHysteresis::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    T = 1.0f / (float) sampleRate;
    rnn->reset();

    monoBuffer.setSize (1, samplesPerBlock);
}

void RNNHysteresis::releaseResources()
{

}

void RNNHysteresis::processBlock (AudioBuffer<float>& buffer)
{
    ScopedNoDenormals noDenormals;

    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();
    monoBuffer.setSize (1, numSamples, false, false, true);
    monoBuffer.clear();

    // sum to mono
    for (int ch = 0; ch < numChannels; ++ch)
        monoBuffer.addFrom (0, 0, buffer.getReadPointer (ch), numSamples, 1.0f / (float) numChannels);

    // process in mono
    auto* xPtr = monoBuffer.getWritePointer (0);
    for (int n = 0; n < numSamples; ++n)
    {
        float input[] = { xPtr[n], *driveParam, *satParam, *widthParam, T };
        xPtr[n] = rnn->forward (input);
    }

    // split back to stereo
    for (int ch = 0; ch < numChannels; ++ch)
        buffer.copyFrom (ch, 0, xPtr, numSamples);
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RNNHysteresis();
}
