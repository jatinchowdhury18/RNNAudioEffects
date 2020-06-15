# RNN Experiments

Experimental audio effects made with recurrent neural networks.

Goals:
- [ ] Hysteresis with 3 parameters
- [ ] Hysteresis with arbitrary sample rate (look at aliasing)
- [ ] Figure out how to adjust weights for any sample rate
- [ ] Time-varying effect (phaser?)
- [ ] Reverse Distortion (with params??)
- [ ] Restoration (high freq. loss, deadzone)

---

## Hysteresis (params)

- Train full model with sample rate
- Train full model with 3-params
- Train model with sample rate and 3-params
- Implement C++ hysteresi model plugin

## Reverse Distortion

- Working pretty well for single distortion curve
- Better results for blind mix of curves
- Try stateful distortion (WDF diode clipper)
- Compare two models: (Recurrent -> Dense)  vs (Time-Dist Dense -> Recurrent -> Dense)

## Restoration

- Poor results...
  - Try different net architectures
  - Try single degradation method at a time

## Time-varying

- success with vibrato
- Try with phaser mod section
- Try with phaser FB section
