# RNN Experiments

Experimental audio effects made with recurrent neural networks.

Goals:
- [ ] Hysteresis with 3 parameters
- [ ] Hysteresis with arbitrary sample rate (look at aliasing)
- [ ] Time-varying effect (phaser?)
- [ ] Reverse Distortion (with params??)
- [ ] Restoration (high freq. loss, deadzone)

---

## Hysteresis (params)

- Try with simple gain param, nothing else

## Reverse Distortion

- Working pretty well for single distortion curve
- Better results for blind mix of curves
- Try stateful distortion (WDF diode clipper)
- Compare two models: (Recurrent -> Dense)  vs (Time-Dist Dense -> Recurrent -> Dense)

## Restoration

- Poor results...
  - Try different net architectures
  - Try single degradation method at a time

## Phaser

- FB section not working very well
  - Try different architectures
  - Try different ways of inputing sine wave
  - Evetntually add FB param
- Try with mod section
