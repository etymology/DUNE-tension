To use, run the tension-server on the computer connected to the PLC, connect to 
the local area network (change the IP address as necessary) and configure the
sound device (we use a device called "USB PnP sound device" or similar, and the
pololu maestro servo controller (https://www.pololu.com/product/1350/resources)
The servo controller should have two subroutines on it, which move the servo up 
and down respectively. Alternatively, the servo can continuously pluck.
Measuring a sequence of wires takes the initial and final wire to be measured,
the layer the side and the direction to move. This can be horizontal vertical or
diagonal. Currently I do not have a way to navigate to wires on demand for the
diagonal layers. For the horizontal, this is trivial. Measurement takes the average
of several NN pitches with passing tension and confidence over a threshold. Tension
is calculated using a LUT for each layer.

## Building the harmonic comb trigger extension

The microphone-triggered audio capture now lives in a Rust crate located at
`rust/harmonic_comb`. Build and install the extension into your active Python
environment before launching the application:

```bash
cd rust/harmonic_comb
maturin develop
```

The build requires the system development headers for your audio backend (for
Linux this typically means installing `libasound2-dev`).
