#include "RNNHysteresisPlugin.h"
#include "Json2RnnParser.h"

namespace
{
    const String driveTag = "drive";
    const String satTag = "sat";
    const String widthTag = "width";
    const String osTag = "os";
    const String smallTag = "small";
}

RNNHysteresis::RNNHysteresis()
{
    MemoryInputStream jsonInput (BinaryData::hysteresis_full_json, BinaryData::hysteresis_full_jsonSize, false);
    MemoryInputStream jsonInputSmall (BinaryData::hysteresis_small_json, BinaryData::hysteresis_small_jsonSize, false);

    Json2RnnParser parser;
    rnn[0].reset (parser.parseJson (jsonInput));
    rnn[1].reset (parser.parseJson (jsonInputSmall));

    for (int i = 0; i < 2; i++)
        oversampling[i] = std::make_unique<dsp::Oversampling<float>> (1, i, dsp::Oversampling<float>::filterHalfBandPolyphaseIIR);

    driveParam = vts.getRawParameterValue (driveTag);
    satParam   = vts.getRawParameterValue (satTag);
    widthParam = vts.getRawParameterValue (widthTag);
    osParam    = vts.getRawParameterValue (osTag);
    smallParam = vts.getRawParameterValue (smallTag);
}

void RNNHysteresis::addParameters (Parameters& params)
{
    params.push_back (std::make_unique<AudioParameterFloat> (driveTag, "Drive",      0.0f, 1.0f, 0.5f));
    params.push_back (std::make_unique<AudioParameterFloat> (satTag,   "Saturation", 0.0f, 1.0f, 0.5f));
    params.push_back (std::make_unique<AudioParameterFloat> (widthTag, "Width",      0.0f, 1.0f, 0.5f));
    params.push_back (std::make_unique<AudioParameterBool> (osTag, "Oversample", false));
    params.push_back (std::make_unique<AudioParameterBool> (smallTag, "Small", true));
}

void RNNHysteresis::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    T = 1.0f / (float) sampleRate;

    rnn[0]->reset();
    rnn[1]->reset();

    monoBuffer.setSize (1, samplesPerBlock);

    for (int i = 0; i < 2; ++i)
        oversampling[i]->initProcessing (samplesPerBlock);

    dcBlocker[0].reset ((float) sampleRate);
    dcBlocker[1].reset ((float) sampleRate);
}

void RNNHysteresis::releaseResources()
{
    for (int i = 0; i < 2; ++i)
        oversampling[i]->reset();
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

    // upsample
    dsp::AudioBlock<float> block (monoBuffer);
    dsp::AudioBlock<float> osBlock;
    osBlock = oversampling[(int) *osParam]->processSamplesUp (block);
    T = 1.0f / ((float) getSampleRate() * oversampling[(int) *osParam]->getOversamplingFactor());

    // process in mono
    auto* xPtr = osBlock.getChannelPointer (0);
    for (int n = 0; n < osBlock.getNumSamples(); ++n)
    {
        float input[] = { xPtr[n], *driveParam, *satParam, *widthParam, T };
        xPtr[n] = rnn[(int) *smallParam]->forward (input);
    }

    oversampling[(int) *osParam]->processSamplesDown (block);

    // split back to stereo
    for (int ch = 0; ch < numChannels; ++ch)
        buffer.copyFrom (ch, 0, monoBuffer.getReadPointer (0), numSamples);

    // DC blocker
    for (int ch = 0; ch < numChannels; ++ch)
        dcBlocker[ch].processBlock (buffer.getWritePointer (ch), numSamples);
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RNNHysteresis();
}
