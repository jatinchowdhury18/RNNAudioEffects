#include "RNNHysteresisPlugin.h"
#include "Json2RnnParser.h"

RNNHysteresis::RNNHysteresis()
{
    MemoryInputStream jsonInput (BinaryData::hysteresis_fs_json, BinaryData::hysteresis_fs_jsonSize, false);

    Json2RnnParser parser;
    rnn.reset (parser.parseJson (jsonInput));
}

void RNNHysteresis::addParameters (Parameters& params)
{

}

void RNNHysteresis::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    T = 1.0f / (float) sampleRate;
    rnn->reset();
}

void RNNHysteresis::releaseResources()
{

}

void RNNHysteresis::processBlock (AudioBuffer<float>& buffer)
{
    ScopedNoDenormals noDenormals;

    for (int ch = 0; ch < 1; ++ch)
    {
        auto* xPtr = buffer.getWritePointer (ch);

        for (int n = 0; n < buffer.getNumSamples(); ++n)
        {
            float input[2] = { xPtr[n], T };
            xPtr[n] = rnn->forward (input);
        }
    }
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RNNHysteresis();
}
